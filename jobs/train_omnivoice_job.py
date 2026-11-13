# Runs ON KAGGLE (2 x T4). Fine-tunes an OmniVoice model (default: oddadmix/lahgtna-omnivoice-v2, the best open
# Sudanese voice in our blind test) on transcribed Sudanese clips, then speaks the benchmark with the result.
# Recipe: Lahgtna-OmniVoice examples/run_finetune.sh (pinned below): JSONL -> Higgs audio tokens -> accelerate train.
# T4 has no bf16 and no flex_attention: fp16 + the SDPA batching config.
# Inputs (flat, in SDTTS_IN):
#   job.json     {"init": "oddadmix/lahgtna-omnivoice-v2", "language_id": "sd", "steps": 1200, "lr": 1e-5,
#                 "batch_tokens": 2048, "grad_accum": 2, "save_steps": 400, "eval_steps": 200, "limit": 0, "bake": true}
#   train.dat + train.jsonl   clips (a tar named .dat, so Kaggle keeps it whole): (<speaker>/<id>.flac) and {"id", "text", ...} rows to train on
#   dev.dat + dev.jsonl       held-out clips for the dev loss (never trained on)
#   sentences.jsonl, refs.json, ref-*.wav   the benchmark, as for bake_job.py (optional, needs "bake": true)
# Outputs (/kaggle/working): model/ (final weights + config, about 2.5 GB), train_log.txt, losses.json,
#   bake/<id>.wav for every benchmark sentence, env.json
# Licence: OmniVoice weights are CC-BY-NC, so the fine-tuned model is non-commercial too.
import glob, json, os, re, subprocess, sys, tarfile, time

t0 = time.time()
IN = os.environ.get("SDTTS_IN", "/kaggle/input")
OUT = os.environ.get("SDTTS_OUTDIR", "/kaggle/working")
TMP = "/kaggle/temp/ft" if os.path.isdir("/kaggle") else "/tmp/sdtts-ft"
os.makedirs(TMP, exist_ok=True)
job = json.load(open(os.path.join(IN, "job.json")))
REPO, SHA = "https://github.com/Oddadmix/Lahgtna-OmniVoice", "c23421594e2d11c8b48a055619651412f18f7f35"


def run(cmd, **kw):
    print("+", " ".join(cmd) if isinstance(cmd, list) else cmd, flush=True)
    return subprocess.run(cmd, check=True, **kw)


src = os.path.join(TMP, "lahgtna-omnivoice")
if not os.path.isdir(src):
    run(["git", "clone", "-q", REPO, src])
    run(["git", "-C", src, "checkout", "-q", SHA])
run([sys.executable, "-m", "pip", "install", "-q", "-e", src, "bitsandbytes"])

# A 15 GB T4 cannot hold the full fine-tune (fp32 weights + grads + AdamW state = ~10 GB before activations).
# Patch the recipe's optimizer: freeze the 155 M-parameter text vocabulary table (speech is learned in the
# audio embeddings, transformer and audio heads) and keep the AdamW state in 8 bits (bitsandbytes).
trainer_py = os.path.join(src, "omnivoice/training/trainer.py")
code = open(trainer_py).read()
OLD = "        optimizer = torch.optim.AdamW(\n            self.model.parameters(),\n"
NEW = ("        for _n, _p in self.model.named_parameters():  # sdtts: low-memory patch\n"
       "            if _n.endswith(\"embed_tokens.weight\"):\n"
       "                _p.requires_grad_(False)\n"
       "        _train = [p for p in self.model.parameters() if p.requires_grad]\n"
       "        logger.info(\"trainable parameters: %d\", sum(p.numel() for p in _train))\n"
       "        import bitsandbytes as _bnb\n"
       "        optimizer = _bnb.optim.AdamW8bit(\n            _train,\n")
if job.get("lowmem", True) and "sdtts: low-memory patch" not in code:
    if OLD not in code:
        sys.exit("trainer.py changed upstream: low-memory patch does not apply")
    open(trainer_py, "w").write(code.replace(OLD, NEW, 1))
    print("trainer patched: frozen text embeddings + 8-bit AdamW", flush=True)
import torch

# ---- data: unpack clips, write the recipe's JSONL {id, audio_path, text, language_id}
data = os.path.join(TMP, "data")
os.makedirs(data, exist_ok=True)
manifests = {}
for split in ("train", "dev"):
    if not os.path.exists(os.path.join(IN, f"{split}.jsonl")):
        continue
    d = os.path.join(data, split)
    if not os.path.isdir(d):
        tarfile.open(os.path.join(IN, f"{split}.dat")).extractall(d, filter="data")
    paths = {os.path.splitext(os.path.basename(p))[0]: p for p in glob.glob(f"{d}/**/*.flac", recursive=True)}
    rows = [json.loads(x) for x in open(os.path.join(IN, f"{split}.jsonl"), encoding="utf-8") if x.strip()]
    rows = [r for r in rows if r["id"] in paths]
    if split == "train" and job.get("limit"):
        rows = rows[: job["limit"]]
    out = os.path.join(data, f"{split}_manifest.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({"id": r["id"], "audio_path": paths[r["id"]], "text": r["text"],
                                "language_id": job.get("language_id", "sd")}, ensure_ascii=False) + "\n")
    manifests[split] = out
    print(f"{split}: {len(rows)} clips", flush=True)

# ---- stage 0: audio -> Higgs tokens (WebDataset shards)
tok = os.path.join(TMP, "tokens")
for split, man in manifests.items():
    if os.path.exists(os.path.join(tok, split, "data.lst")):
        continue
    run([sys.executable, "-m", "omnivoice.scripts.extract_audio_tokens", "--input_jsonl", man,
         "--tar_output_pattern", f"{tok}/{split}/audios/shard-%06d.tar",
         "--jsonl_output_pattern", f"{tok}/{split}/txts/shard-%06d.jsonl",
         "--tokenizer_path", "eustlb/higgs-audio-v2-tokenizer", "--nj_per_gpu", "2", "--loader_workers", "4",
         "--min_num_shards", "16", "--skip_errors"], cwd=src)  # >= GPUs x loader workers, or a worker gets no shard
t_tok = time.time() - t0
print(f"tokenised in {t_tok:.0f}s", flush=True)

# ---- stage 1: fine-tune
cfg = json.load(open(os.path.join(src, "examples/config/train_config_finetune_sdpa.json")))
cfg.update({"init_from_checkpoint": job.get("init", "oddadmix/lahgtna-omnivoice-v2"), "mixed_precision": "fp16",
            "allow_tf32": False, "learning_rate": job.get("lr", 1e-5), "steps": job.get("steps", 1200),
            "batch_tokens": job.get("batch_tokens", 2048), "gradient_accumulation_steps": job.get("grad_accum", 2), "save_steps": job.get("save_steps", 400),
            "eval_steps": job.get("eval_steps", 200), "logging_steps": 10, "keep_last_n_checkpoints": 1,
            "num_workers": 2, "warmup_ratio": 0.05})
cfg_path = os.path.join(TMP, "train_config.json")
json.dump(cfg, open(cfg_path, "w"), indent=1)
dcfg = {"train": [{"manifest_path": [f"{tok}/train/data.lst"]}]}
if "dev" in manifests:
    dcfg["dev"] = [{"manifest_path": [f"{tok}/dev/data.lst"]}]
dcfg_path = os.path.join(TMP, "data_config.json")
json.dump(dcfg, open(dcfg_path, "w"))
exp = os.path.join(TMP, "exp")
ngpu = max(1, torch.cuda.device_count())
log_path = os.path.join(OUT, "train_log.txt")
t1 = time.time()
with open(log_path, "w") as log:
    p = subprocess.Popen(["accelerate", "launch", "--num_processes", str(ngpu), "--mixed_precision", "fp16",
                          *(["--multi_gpu"] if ngpu > 1 else []), "-m", "omnivoice.cli.train",
                          "--train_config", cfg_path, "--data_config", dcfg_path, "--output_dir", exp],
                         cwd=src, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         env=dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"))
    for line in p.stdout:
        log.write(line)
        if re.search(r"loss|step|Error|error|Traceback|OOM|out of memory", line):
            print(line.rstrip()[:300], flush=True)
    p.wait()
t_train = time.time() - t1
losses = []
for line in open(log_path):
    m = re.search(r"step[^0-9]*(\d+).*?(eval_)?loss[^0-9]*([0-9.]+)", line)
    if m:
        losses.append({"step": int(m.group(1)), "kind": "eval" if m.group(2) else "train", "loss": float(m.group(3))})
json.dump(losses, open(os.path.join(OUT, "losses.json"), "w"))
ckpts = sorted(glob.glob(f"{exp}/checkpoint-*"), key=lambda d: int(d.rsplit("-", 1)[1]))
if p.returncode != 0 or not ckpts:
    sys.exit(f"training failed (exit {p.returncode}); see train_log.txt")
final = ckpts[-1]
print(f"trained {cfg['steps']} steps in {t_train:.0f}s -> {final}", flush=True)

# keep only what inference needs (the optimizer state is 2-3x the model and never comes back)
model_out = os.path.join(OUT, "model")
os.makedirs(model_out, exist_ok=True)
for f in os.listdir(final):
    if f.endswith((".json", ".safetensors", ".jinja", ".txt")) and not f.startswith(("optimizer", "scheduler", "random_states")):
        run(["cp", os.path.join(final, f), model_out])

# ---- speak the benchmark with the fine-tuned model
done = 0
if job.get("bake", True) and os.path.exists(os.path.join(IN, "sentences.jsonl")):
    import numpy as np, soundfile as sf
    sys.path.insert(0, src)  # the editable install is not on this already-running interpreter's path
    from omnivoice import OmniVoice
    torch.cuda.empty_cache()
    model = OmniVoice.from_pretrained(model_out, device_map="cuda:0" if torch.cuda.is_available() else "cpu",
                                      dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
    refs = {r["id"]: dict(r, path=os.path.join(IN, r["file"])) for r in json.load(open(os.path.join(IN, "refs.json")))}
    os.makedirs(os.path.join(OUT, "bake"), exist_ok=True)
    sents = [json.loads(x) for x in open(os.path.join(IN, "sentences.jsonl"), encoding="utf-8") if x.strip()]
    for s in sents[: job["bake_limit"]] if job.get("bake_limit") else sents:
        try:
            torch.manual_seed(7)
            ref = refs[s["ref"]]
            wav = model.generate(text=s["text"], language=job.get("language_id", "sd"), ref_audio=ref["path"], ref_text=ref["text"])[0]
            sf.write(os.path.join(OUT, "bake", s["id"] + ".wav"), np.clip(np.asarray(wav, dtype=np.float32), -1, 1), 24000, subtype="PCM_16")
            done += 1
        except Exception as e:
            print(f"{s['id']}: FAILED {e!r}", flush=True)

json.dump({"init": cfg["init_from_checkpoint"], "steps": cfg["steps"], "gpus": ngpu, "torch": torch.__version__,
           "tokenise_s": round(t_tok), "train_s": round(t_train), "baked": done, "total_s": round(time.time() - t0),
           "gpu_peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if torch.cuda.is_available() else 0},
          open(os.path.join(OUT, "env.json"), "w"), indent=1)
print(f"done: {done} sentences baked, total {time.time() - t0:.0f}s", flush=True)
