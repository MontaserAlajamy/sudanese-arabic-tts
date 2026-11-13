# Runs ON KAGGLE (free T4). Scores speech clips against the text they should say and the voice they should copy.
# The same job scores real recordings ("human") and every TTS system, so each number has a human topline beside it.
# Inputs (flat, in SDTTS_IN): job.json, items.jsonl, clips.dat (a tar of the audio), text_norm.py, metrics.py
#   items.jsonl  {"clip": "habibi/a01.wav", "system": "habibi", "id": "a01", "text": "...", "ref": "refs/ref-spk05.wav"}
#   job.json     {"whisper": "large-v3", "lang": "ar"}
# Outputs (/kaggle/working): metrics.jsonl (one row per clip), summary.json (one row per system)
# Per clip: asr text, cer, wer (normalised Arabic, see text_norm.py); sim = cosine similarity of WavLM speaker
# embeddings between the clip and its reference; sim_gap = sim minus the clip's mean similarity to the OTHER reference
# voices (WavLM cosines sit high for any speech, so the gap is the readable part); utmos = predicted naturalness (trained on English and Chinese
# ratings: a rough proxy for Arabic); signal checks (length, letters per second, level, clipping, energy above 8 kHz).
# Whisper is weak on Sudanese Arabic: read every CER against the "human" row, never alone.
import ctypes, glob, json, os, subprocess, sys, tarfile, time

t0 = time.time()
IN = os.environ.get("SDTTS_IN", "/kaggle/input")
OUT = os.environ.get("SDTTS_OUTDIR", "/kaggle/working")  # set SDTTS_OUTDIR to run off Kaggle
TMP = "/kaggle/temp/eval" if os.path.isdir("/kaggle") else "/tmp/sdtts-eval"
job = json.load(open(os.path.join(IN, "job.json")))
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "faster-whisper"], check=True)
# CTranslate2 needs cuBLAS/cuDNN; Kaggle ships them as pip wheels (nvidia-*), so preload them by path
for pat in ("nvidia/cublas/lib/libcublas*.so*", "nvidia/cudnn/lib/libcudnn*.so*"):
    for so in sorted(sum((glob.glob(os.path.join(p, pat)) for p in sys.path if os.path.isdir(p)), [])):
        try:
            ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass
sys.path.insert(0, IN)

import numpy as np, soundfile as sf, torch
import torchaudio.functional as F
from faster_whisper import WhisperModel
from metrics import cer, corpus_rate, wer
from text_norm import normalize

os.makedirs(TMP, exist_ok=True)
tarfile.open(os.path.join(IN, "clips.dat")).extractall(TMP, filter="data")
items = [json.loads(x) for x in open(os.path.join(IN, "items.jsonl"), encoding="utf-8") if x.strip()]
dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"{len(items)} clips, device {dev}, setup {time.time() - t0:.0f}s", flush=True)

try:
    asr = WhisperModel(job.get("whisper", "large-v3"), device="cuda", compute_type="float16")
except Exception as e:
    print(f"GPU load failed ({e}); Whisper on CPU int8", flush=True)
    asr = WhisperModel(job.get("whisper", "large-v3"), device="cpu", compute_type="int8")

from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
fe = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
sv = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").to(dev).eval()
try:
    mos = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True, skip_validation=True).to(dev).eval()
except Exception as e:
    print(f"UTMOS unavailable: {e!r}", flush=True)
    mos = None


def load(path):
    x, sr = sf.read(os.path.join(TMP, path), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    return x, sr, F.resample(torch.from_numpy(np.ascontiguousarray(x)), sr, 16000).numpy()


_emb = {}


def embedding(key, x16):
    if key not in _emb:
        with torch.inference_mode():
            inp = fe([x16], sampling_rate=16000, return_tensors="pt", padding=True).to(dev)
            _emb[key] = torch.nn.functional.normalize(sv(**inp).embeddings, dim=-1)[0].cpu()
    return _emb[key]


def signal(x, sr, text):
    frame = int(0.02 * sr)
    db = np.array([10 * np.log10(np.mean(x[i:i + frame] ** 2) + 1e-10) for i in range(0, max(1, len(x) - frame), frame)])
    loud = np.where(db > -45)[0]
    spec = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(len(x), 1 / sr)
    hf = float(spec[freqs > 8000].sum() / (spec.sum() + 1e-12)) if sr > 16000 else 0.0
    n = len(normalize(text).replace(" ", ""))
    return {"dur": round(len(x) / sr, 2), "sr": sr, "lps": round(n / max(0.1, len(x) / sr), 2),
            "rms_db": round(float(10 * np.log10(np.mean(x ** 2) + 1e-10)), 1), "clip_frac": round(float(np.mean(np.abs(x) >= 0.999)), 5),
            "hf_db": round(10 * np.log10(hf + 1e-12), 1), "lead_sil": round(float(loud[0] * 0.02) if len(loud) else 0.0, 2),
            "trail_sil": round(float((len(db) - 1 - loud[-1]) * 0.02) if len(loud) else 0.0, 2)}


all_refs = sorted({it["ref"] for it in items if it.get("ref")})
rows = []
for n, it in enumerate(items):
    try:
        x, sr, x16 = load(it["clip"])
        segs, _ = asr.transcribe(x16, language=job.get("lang", "ar"), beam_size=5, vad_filter=False,
                                 condition_on_previous_text=False, temperature=0.0)
        hyp = " ".join(s.text.strip() for s in segs).strip()
        row = dict(it, asr=hyp, cer=round(cer(it["text"], hyp), 4), wer=round(wer(it["text"], hyp), 4), **signal(x, sr, it["text"]))
        if it.get("ref"):
            mine = embedding(it["clip"], x16)
            row["sim"] = round(float(mine @ embedding(it["ref"], load(it["ref"])[2])), 4)
            others = [r for r in all_refs if r != it["ref"]]  # floor: the same clip against the other voices
            if others:
                row["sim_other"] = round(float(np.mean([float(mine @ embedding(r, load(r)[2])) for r in others])), 4)
                row["sim_gap"] = round(row["sim"] - row["sim_other"], 4)
        if mos is not None:
            with torch.inference_mode():
                row["utmos"] = round(float(mos(torch.from_numpy(x16).unsqueeze(0).to(dev), 16000)[0]), 3)
        rows.append(row)
    except Exception as e:
        rows.append(dict(it, error=repr(e)))
        print(f"{it['clip']}: FAILED {e!r}", flush=True)
    if n % 25 == 0:
        print(f"{n + 1}/{len(items)} ({time.time() - t0:.0f}s)", flush=True)

with open(os.path.join(OUT, "metrics.jsonl"), "w", encoding="utf-8") as f:
    f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
mean = lambda v: round(float(np.mean(v)), 4) if len(v) else None
summary = []
for system in sorted({r["system"] for r in rows}):
    rs = [r for r in rows if r["system"] == system and "error" not in r]
    pairs = [(r["text"], r["asr"]) for r in rs]
    summary.append({"system": system, "n": len(rs), "failed": sum(1 for r in rows if r["system"] == system and "error" in r),
                    "cer": round(corpus_rate(pairs, "char"), 4), "wer": round(corpus_rate(pairs, "word"), 4),
                    "cer_median": round(float(np.median([r["cer"] for r in rs])), 4) if rs else None,
                    "sim": mean([r["sim"] for r in rs if "sim" in r]), "sim_gap": mean([r["sim_gap"] for r in rs if "sim_gap" in r]),
                    "utmos": mean([r["utmos"] for r in rs if "utmos" in r]),
                    "dur": mean([r["dur"] for r in rs]), "lps": mean([r["lps"] for r in rs]), "hf_db": mean([r["hf_db"] for r in rs]),
                    "clipped": sum(r["clip_frac"] > 0.001 for r in rs)})
json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
for s in summary:
    print(json.dumps(s), flush=True)
print(f"total {time.time() - t0:.0f}s", flush=True)
