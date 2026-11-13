# Runs ON KAGGLE (free T4). Speaks every benchmark sentence with ONE open TTS model, zero-shot, cloning the short
# reference clip named by each sentence. One engine per kernel, because the engines need different packages.
# Inputs (flat, in SDTTS_IN): job.json, sentences.jsonl, refs.json, ref-<id>.wav
#   job.json        {"engine": "chatterbox|lahgtna|habibi|omnivoice|lahgtna-omnivoice", "variant": "text", "limit": 0, "seed": 7,
#                    "ckpt": "auto"}   (ckpt: lahgtna-omnivoice only, use a fine-tuned checkpoint mounted with --kernels)
#   sentences.jsonl {"id": "a01", "text": "...", "ref": "spk05", ...}   (job["variant"] names the text field to speak)
#   refs.json       [{"id": "spk05", "file": "ref-spk05.wav", "text": "<transcript of the clip>"}]
# Outputs (/kaggle/working): <id>.wav (mono PCM16 at the model's own rate), timings.json, env.json, errors.json
# Licences: chatterbox and lahgtna are MIT. habibi (CC-BY-NC-SA), omnivoice and lahgtna-omnivoice (CC-BY-NC) are
# non-commercial: research and evaluation only.
import json, os, subprocess, sys, time, traceback

t0 = time.time()
IN = os.environ.get("SDTTS_IN", "/kaggle/input")
OUT = os.environ.get("SDTTS_OUTDIR", "/kaggle/working")  # set SDTTS_OUTDIR to run off Kaggle
TMP = "/kaggle/temp" if os.path.isdir("/kaggle") else "/tmp/sdtts"
os.makedirs(TMP, exist_ok=True)
job = json.load(open(os.path.join(IN, "job.json")))
ENGINE = job["engine"]
FIELD = job.get("variant", "text")

# Pinned upstream code (commit SHAs read on 2026-11-10)
CHATTERBOX = "git+https://github.com/resemble-ai/chatterbox@5de7a54aa4e5e2baadb0182dde554908b48b85c2"
CHATTERBOX_DEPS = ["resemble-perth", "conformer==0.3.2", "s3tokenizer", "diffusers", "omegaconf", "librosa",
                   "safetensors", "pykakasi", "spacy-pkuseg", "transformers", "einops", "soundfile"]
LAHGTNA_REPO, LAHGTNA_SHA = "https://github.com/Oddadmix/lahgtna-chatterbox", "433cb74200b55457bffa8ee6965a02ecab546a1c"
LAHGTNA_WEIGHTS = ("oddadmix/lahgtna-chatterbox-v1", "t3_mtl23ls_v2.safetensors")
HABIBI = "git+https://github.com/SWivid/Habibi-TTS@9274498d8425b946866cfec1f6eee4841a9d80f3"
OMNIVOICE = "git+https://github.com/k2-fsa/OmniVoice@08be0b4ccbac3e13e374e86fbfead4b4cac343e2"
LAHGTNA_OMNI = "git+https://github.com/Oddadmix/Lahgtna-OmniVoice@c23421594e2d11c8b48a055619651412f18f7f35"


def pip(*args):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *args], check=True)


def load_chatterbox(dev):
    pip("--no-deps", CHATTERBOX)  # keep Kaggle's torch
    pip(*CHATTERBOX_DEPS)
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    model = ChatterboxMultilingualTTS.from_pretrained(device=dev, t3_model="v3")

    def speak(text, ref):
        wav = model.generate(text, language_id="ar", audio_prompt_path=ref["path"], exaggeration=0.5, cfg_weight=0.5)
        return wav.squeeze().cpu().numpy(), model.sr
    return speak


def load_lahgtna(dev):
    # Lahgtna ships its own patched copy of chatterbox (dialect text rules; "sd" rides the "pt" language slot)
    src = os.path.join(TMP, "lahgtna")
    subprocess.run(["git", "clone", "-q", LAHGTNA_REPO, src], check=True)
    subprocess.run(["git", "-C", src, "checkout", "-q", LAHGTNA_SHA], check=True)
    # its alignment analyzer sets config.output_attentions=True, which current transformers refuses under sdpa
    cfg = os.path.join(src, "src/chatterbox/models/t3/llama_configs.py")
    text = open(cfg).read()
    assert 'attn_implementation="sdpa"' in text, "lahgtna llama_configs.py changed"
    with open(cfg, "w") as f:
        f.write(text.replace('attn_implementation="sdpa"', 'attn_implementation="eager"'))
    pip(*CHATTERBOX_DEPS)
    sys.path.insert(0, os.path.join(src, "src"))
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    model = ChatterboxMultilingualTTS.from_pretrained(device=dev)
    model.t3.load_state_dict(load_file(hf_hub_download(*LAHGTNA_WEIGHTS), device=dev))
    model.t3.to(dev).eval()

    def speak(text, ref):
        wav = model.generate(text, language_id="sd", audio_prompt_path=ref["path"], exaggeration=0.5,
                             temperature=0.8, cfg_weight=0.5)
        return wav.squeeze().cpu().numpy(), model.sr
    return speak


def load_habibi(dev):
    pip(HABIBI)
    from importlib.resources import files
    from cached_path import cached_path
    from hydra.utils import get_class
    from omegaconf import OmegaConf
    from f5_tts.infer.utils_infer import load_model, load_vocoder, preprocess_ref_audio_text
    from habibi_tts.infer.utils_infer import infer_process
    from habibi_tts.model.utils import dialect_id_map
    cfg = OmegaConf.load(str(files("f5_tts").joinpath("configs/F5TTS_v1_Base.yaml")))
    voc = cfg.model.mel_spec.mel_spec_type
    ckpt = str(cached_path("hf://SWivid/Habibi-TTS/Unified/model_200000.safetensors"))
    vocab = str(cached_path("hf://SWivid/Habibi-TTS/Unified/vocab.txt"))
    model = load_model(get_class(f"f5_tts.model.{cfg.model.backbone}"), cfg.model.arch, ckpt,
                       mel_spec_type=voc, vocab_file=vocab, device=dev)
    vocoder = load_vocoder(vocoder_name=voc, is_local=False, device=dev)
    dialect = dialect_id_map[job.get("dialect", "SDN")]
    prepped = {}

    def speak(text, ref):
        if ref["id"] not in prepped:
            prepped[ref["id"]] = preprocess_ref_audio_text(ref["path"], ref["text"])
        audio, ref_text = prepped[ref["id"]]
        wav, sr, _ = infer_process(audio, ref_text, text, model, vocoder, mel_spec_type=voc, device=dev, dialect_id=dialect)
        return wav, sr
    return speak


def load_omnivoice(dev, package=OMNIVOICE, repo="k2-fsa/OmniVoice", language="apd"):
    pip(package)
    import torch
    from omnivoice import OmniVoice
    model = OmniVoice.from_pretrained(repo, device_map="cuda:0" if dev == "cuda" else "cpu",
                                      dtype=torch.float16 if dev == "cuda" else torch.float32)

    def speak(text, ref):
        audio = model.generate(text=text, language=language, ref_audio=ref["path"], ref_text=ref["text"])
        return audio[0], 24000
    return speak


def find_ckpt():
    """A fine-tuned OmniVoice mounted from another kernel's output: the folder with model.safetensors + config.json."""
    for root, _, files in os.walk("/kaggle/input"):
        if "model.safetensors" in files and "config.json" in files:
            if json.load(open(os.path.join(root, "config.json"))).get("model_type") == "omnivoice":
                return root
    print("mounted:", [r for r, _, _ in os.walk("/kaggle/input")][:30], flush=True)
    sys.exit("job.json asks for a fine-tuned checkpoint, but none is mounted (--sources <user>/<model dataset> or --kernels)")


def load_lahgtna_omnivoice(dev):
    # the fork keeps the package name "omnivoice" and adds dialect ids; "sd" = Sudanese.
    # job["ckpt"] == "auto": a fine-tuned checkpoint of it, mounted from the training kernel's output
    repo = find_ckpt() if job.get("ckpt") else "oddadmix/lahgtna-omnivoice-v2"
    print("weights:", repo, flush=True)
    return load_omnivoice(dev, LAHGTNA_OMNI, repo, "sd")


LOADERS = {"chatterbox": load_chatterbox, "lahgtna": load_lahgtna, "habibi": load_habibi,
           "omnivoice": load_omnivoice, "lahgtna-omnivoice": load_lahgtna_omnivoice}

refs = {r["id"]: dict(r, path=os.path.join(IN, r["file"])) for r in json.load(open(os.path.join(IN, "refs.json")))}
sents = [json.loads(x) for x in open(os.path.join(IN, "sentences.jsonl"), encoding="utf-8") if x.strip()]
sents = [s for s in sents if s.get(FIELD)]
if job.get("limit"):
    sents = sents[: job["limit"]]

import torch  # after reading inputs, so a bad job.json fails before any install
dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"engine={ENGINE} field={FIELD} sentences={len(sents)} device={dev} torch={torch.__version__}", flush=True)
speak = LOADERS[ENGINE](dev)
setup_s = time.time() - t0
print(f"ready in {setup_s:.0f}s", flush=True)

import numpy as np, soundfile as sf
timings, errors = [], []
for s in sents:
    t1 = time.time()
    try:
        torch.manual_seed(job.get("seed", 7))
        wav, sr = speak(s[FIELD], refs[s["ref"]])
        wav = np.asarray(wav, dtype=np.float32).reshape(-1)
        if not len(wav) or not np.isfinite(wav).all():
            raise ValueError("empty or non-finite audio")
        sf.write(os.path.join(OUT, s["id"] + ".wav"), np.clip(wav, -1, 1), int(sr), subtype="PCM_16")
        timings.append({"id": s["id"], "audio_s": round(len(wav) / sr, 2), "gen_s": round(time.time() - t1, 2), "sr": int(sr)})
        print(f"{s['id']}: {len(wav) / sr:.1f}s audio in {time.time() - t1:.1f}s", flush=True)
    except Exception as e:
        errors.append({"id": s["id"], "error": repr(e), "trace": traceback.format_exc()[-1500:]})
        print(f"{s['id']}: FAILED {e!r}", flush=True)

freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout.splitlines()
keep = ("torch", "transformers", "f5-tts", "f5_tts", "habibi", "omnivoice", "chatterbox", "numpy", "vocos", "s3tokenizer", "diffusers")
env = {"engine": ENGINE, "field": FIELD, "torch": torch.__version__,
       "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu",
       "gpu_peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if dev == "cuda" else 0,
       "packages": [p for p in freeze if p.lower().startswith(keep)], "setup_s": round(setup_s), "total_s": round(time.time() - t0)}
json.dump(timings, open(os.path.join(OUT, "timings.json"), "w"), indent=1)
json.dump(errors, open(os.path.join(OUT, "errors.json"), "w"), indent=1, ensure_ascii=False)
json.dump(env, open(os.path.join(OUT, "env.json"), "w"), indent=1)
print(f"done: {len(timings)} ok, {len(errors)} failed, total {time.time() - t0:.0f}s, gpu peak {env['gpu_peak_gb']} GB", flush=True)
if not timings:
    sys.exit("no audio was produced")
