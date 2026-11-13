# Runs ON KAGGLE (one T4). Fine-tunes lahgtna-chatterbox (MIT: Chatterbox Multilingual with Arabic dialect slots,
# Sudanese "sd" = the "pt" slot) on transcribed Sudanese clips, then speaks the benchmark with the result.
# It drives the upstream trainer (Oddadmix/lahgtna-chatterbox src/finetune.py, pinned) with run-time patches:
#   1. start from the lahgtna T3 weights: upstream reloads the ORIGINAL ResembleAI weights, which have no Sudanese
#   2. read our clips from plain JSONL rows, so nothing depends on the datasets library's audio decoding
#   3. current transformers: PretrainedConfig alias; skip Trainer.save_model (the T3 weights are saved right after it)
#   4. only for speaking afterwards: eager attention, as in bake_job.py
# Only the T3 (text -> speech tokens, 0.5 B parameters) is trained; voice encoder and S3Gen stay frozen (upstream default).
# Inputs (flat, in SDTTS_IN):
#   job.json   {"init": "oddadmix/lahgtna-chatterbox-v1", "language_id": "sd", "epochs": 3, "steps": 0, "lr": 2e-5,
#               "batch": 4, "grad_accum": 2, "eval_steps": 100, "bake": true, "bake_limit": 0}   (steps > 0 overrides epochs)
#   train.dat + train.jsonl, dev.dat + dev.jsonl   tars of <speaker>/<id>.flac (named .dat so Kaggle keeps them whole)
#   sentences.jsonl, refs.json, ref-*.wav          the benchmark, as for bake_job.py
# Outputs (/kaggle/working): model/t3_mtl23ls_v2.safetensors (about 2 GB), train_log.txt, bake/<id>.wav, env.json
import glob, json, os, re, shutil, subprocess, sys, tarfile, time

t0 = time.time()
IN = os.environ.get("SDTTS_IN", "/kaggle/input")
OUT = os.environ.get("SDTTS_OUTDIR", "/kaggle/working")
TMP = "/kaggle/temp/ft2" if os.path.isdir("/kaggle") else "/tmp/sdtts-ft2"
os.makedirs(TMP, exist_ok=True)
job = json.load(open(os.path.join(IN, "job.json")))
REPO, SHA = "https://github.com/Oddadmix/lahgtna-chatterbox", "433cb74200b55457bffa8ee6965a02ecab546a1c"
INIT = job.get("init", "oddadmix/lahgtna-chatterbox-v1")
LANG = job.get("language_id", "sd")
DEPS = ["resemble-perth", "conformer==0.3.2", "s3tokenizer", "diffusers", "omegaconf", "librosa", "safetensors",
        "pykakasi", "spacy-pkuseg", "transformers", "einops", "soundfile", "accelerate", "bitsandbytes", "datasets"]

HEAD = '''import transformers as _tf  # sdtts patches: see train_chatterbox_job.py
if not hasattr(_tf, "PretrainedConfig"):
    _tf.PretrainedConfig = _tf.PreTrainedConfig
'''
SHIM = '''

class _Rows(list):
    column_names = ["audio", "text", "language"]

    def remove_columns(self, cols):
        return self


class _Sets(dict):
    def cast_column(self, *a, **k):
        return self


def load_dataset(path, *a, **k):  # sdtts: plain JSONL rows {"audio": <file path>, "text", "language"}
    import json as _json
    import os as _os
    out = _Sets()
    for split in ("train", "validation"):
        f = _os.path.join(path, split + ".jsonl")
        if _os.path.exists(f):
            out[split] = _Rows(_json.loads(x) for x in open(f, encoding="utf-8") if x.strip())
    return out


def _sdtts_start(model):  # sdtts: begin from the dialect checkpoint, not the original weights
    import os as _os
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    model.t3.load_state_dict(load_file(hf_hub_download(_os.environ["SDTTS_INIT"], "t3_mtl23ls_v2.safetensors")))
    print("start weights:", _os.environ["SDTTS_INIT"], flush=True)

'''


def patch_trainer(code):
    """The upstream finetune.py with the patches listed at the top of this file. Fails loudly if upstream changed."""
    code, n1 = re.subn(r"^(from datasets import [^\n]*\n)", lambda m: m.group(1) + SHIM, code, count=1, flags=re.M)
    code, n2 = re.subn(r'^(\s*)(chatterbox_model = ChatterboxMultilingualTTS\.from_pretrained\(device="cpu"\))\n',
                       r"\1\2\n\1_sdtts_start(chatterbox_model)\n", code, flags=re.M)
    code, n3 = re.subn(r"^(\s*)trainer_instance\.save_model\(\)\n", r"\1pass  # sdtts: the T3 weights are saved just below\n",
                       code, flags=re.M)
    if not (n1 == 1 and n2 >= 1 and n3 >= 1):
        raise SystemExit(f"upstream finetune.py changed: patches matched {n1}/{n2}/{n3} times")
    return HEAD + code


def run(cmd, **kw):
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=True, **kw)


repo = os.path.join(TMP, "lahgtna")
if not os.path.isdir(repo):
    run(["git", "clone", "-q", REPO, repo])
    run(["git", "-C", repo, "checkout", "-q", SHA])
run([sys.executable, "-m", "pip", "install", "-q", *DEPS])
src = os.path.join(repo, "src")

# ---- data: unpack clips, write the rows the patched trainer reads
data = os.path.join(TMP, "data")
os.makedirs(data, exist_ok=True)
counts = {}
for split, name in (("train", "train"), ("dev", "validation")):
    if not os.path.exists(os.path.join(IN, f"{split}.jsonl")):
        continue
    d = os.path.join(data, split + "_audio")
    if not os.path.isdir(d):
        tarfile.open(os.path.join(IN, f"{split}.dat")).extractall(d, filter="data")
    paths = {os.path.splitext(os.path.basename(p))[0]: p for p in glob.glob(f"{d}/**/*.flac", recursive=True)}
    rows = [json.loads(x) for x in open(os.path.join(IN, f"{split}.jsonl"), encoding="utf-8") if x.strip()]
    rows = [r for r in rows if r["id"] in paths]
    with open(os.path.join(data, name + ".jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({"audio": paths[r["id"]], "text": r["text"], "language": LANG}, ensure_ascii=False) + "\n")
    counts[name] = len(rows)
print("clips:", counts, flush=True)

# ---- train (the upstream trainer in its own process, on one GPU)
patched = patch_trainer(open(os.path.join(src, "finetune.py"), encoding="utf-8").read())
compile(patched, "finetune_sdtts.py", "exec")
open(os.path.join(src, "finetune_sdtts.py"), "w", encoding="utf-8").write(patched)
exp = os.path.join(TMP, "exp")
steps = int(job.get("steps", 0))
cmd = [sys.executable, "finetune_sdtts.py", "--output_dir", exp, "--model_name_or_path", INIT, "--dataset_name", data,
       "--train_split_name", "train", "--eval_split_name", "validation", "--text_column_name", "text",
       "--audio_column_name", "audio", "--per_device_train_batch_size", str(job.get("batch", 4)),
       "--per_device_eval_batch_size", str(job.get("batch", 4)),
       "--gradient_accumulation_steps", str(job.get("grad_accum", 2)), "--learning_rate", str(job.get("lr", 2e-5)),
       "--warmup_steps", str(job.get("warmup", 30)), "--logging_steps", "10", "--eval_strategy", "steps",
       "--eval_steps", str(job.get("eval_steps", 100)), "--save_strategy", "no", "--fp16", "True",
       "--optim", "adamw_bnb_8bit", "--report_to", "none", "--dataloader_num_workers", "3",
       "--dataloader_pin_memory", "False", "--do_train", "--do_eval", "--label_names", "labels_speech",
       "--prediction_loss_only", "True"]
cmd += ["--max_steps", str(steps)] if steps else ["--num_train_epochs", str(job.get("epochs", 3))]
env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", SDTTS_INIT=INIT, TOKENIZERS_PARALLELISM="false",
           PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
log_path = os.path.join(OUT, "train_log.txt")
t1 = time.time()
print("+", " ".join(cmd), flush=True)
with open(log_path, "w") as log:
    p = subprocess.Popen(cmd, cwd=src, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        log.write(line)
        if re.search(r"loss|start weights|Error|error|Traceback|out of memory|Skipping", line):
            print(line.rstrip()[:300], flush=True)
    p.wait()
t_train = time.time() - t1
t3_path = os.path.join(exp, "t3_mtl23ls_v2.safetensors")
if p.returncode != 0 or not os.path.exists(t3_path):
    sys.exit(f"training failed (exit {p.returncode}); see train_log.txt")
os.makedirs(os.path.join(OUT, "model"), exist_ok=True)
shutil.copy(t3_path, os.path.join(OUT, "model"))
print(f"trained in {t_train:.0f}s -> model/t3_mtl23ls_v2.safetensors", flush=True)

# ---- speak the benchmark with the fine-tuned T3
done = 0
if job.get("bake", True) and os.path.exists(os.path.join(IN, "sentences.jsonl")):
    cfg = os.path.join(src, "chatterbox/models/t3/llama_configs.py")
    text = open(cfg).read()
    assert 'attn_implementation="sdpa"' in text, "lahgtna llama_configs.py changed"
    with open(cfg, "w") as f:  # its alignment analyzer asks for attention maps, which sdpa cannot return
        f.write(text.replace('attn_implementation="sdpa"', 'attn_implementation="eager"'))
    sys.path.insert(0, src)
    import numpy as np, soundfile as sf, torch
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    from safetensors.torch import load_file
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = ChatterboxMultilingualTTS.from_pretrained(device=dev)
    model.t3.load_state_dict(load_file(t3_path, device=dev))
    model.t3.to(dev).eval()
    refs = {r["id"]: dict(r, path=os.path.join(IN, r["file"])) for r in json.load(open(os.path.join(IN, "refs.json")))}
    sents = [json.loads(x) for x in open(os.path.join(IN, "sentences.jsonl"), encoding="utf-8") if x.strip()]
    if job.get("bake_limit"):
        sents = sents[: job["bake_limit"]]
    os.makedirs(os.path.join(OUT, "bake"), exist_ok=True)
    for s in sents:
        try:
            torch.manual_seed(7)
            wav = model.generate(s["text"], language_id=LANG, audio_prompt_path=refs[s["ref"]]["path"],
                                 exaggeration=0.5, temperature=0.8, cfg_weight=0.5)
            wav = np.clip(wav.squeeze().cpu().numpy().astype(np.float32), -1, 1)
            sf.write(os.path.join(OUT, "bake", s["id"] + ".wav"), wav, model.sr, subtype="PCM_16")
            done += 1
        except Exception as e:
            print(f"{s['id']}: FAILED {e!r}", flush=True)

json.dump({"init": INIT, "clips": counts, "steps": steps, "epochs": None if steps else job.get("epochs", 3),
           "lr": job.get("lr", 2e-5), "train_s": round(t_train), "baked": done, "total_s": round(time.time() - t0)},
          open(os.path.join(OUT, "env.json"), "w"), indent=1)
print(f"done: {done} sentences baked, total {time.time() - t0:.0f}s", flush=True)
