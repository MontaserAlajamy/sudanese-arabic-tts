# Runs ON KAGGLE (free T4). Turns the Sudanese Arabic part (apd_Arab) of Meta's Omnilingual ASR Corpus (CC-BY-4.0:
# 12 speakers, about 10 h of spontaneous speech in long takes, human transcripts) into TTS-sized clips:
#   1. download the four parquet files from Hugging Face (not gated)
#   2. force-align every transcript to its audio, word by word (torchaudio MMS_FA + uroman romanisation)
#   3. cut each take into 3-14 s clips at pauses; hesitations, noises and word fragments are cut out, never spoken
#   4. write 24 kHz mono FLAC clips (one tar per split) + a manifest with text, speaker, timings and quality scores
# Inputs (SDTTS_IN): segment.py (sdtts/segment.py), job.json {"limit": 0, "splits": ["dev", "test"]}
#   (both optional: limit = first N takes per split, for a test; splits = only these)
# Off Kaggle: set SDTTS_OUTDIR, and SDTTS_APD_DIR if the parquet files are already on disk (a CPU does about 15 takes a minute).
# Outputs (/kaggle/working): apd_clips_<split>.tar, apd_segments.jsonl, apd_words.jsonl, stats.json
import io, json, os, subprocess, sys, tarfile, time

t0 = time.time()
IN = os.environ.get("SDTTS_IN", "/kaggle/input")
OUT = os.environ.get("SDTTS_OUTDIR", "/kaggle/working")  # set SDTTS_OUTDIR to run off Kaggle
job = json.load(open(os.path.join(IN, "job.json"))) if os.path.exists(os.path.join(IN, "job.json")) else {}
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uroman"], check=True)
sys.path.insert(0, IN)

import numpy as np, pyarrow.parquet as pq, soundfile as sf, torch, torchaudio
import torchaudio.functional as F
import uroman as ur
from huggingface_hub import hf_hub_download
from segment import PUNCT, clip_bounds, cut_points, letters, next_to_cut, runs_of, tokens_of

REPO = "facebook/omnilingual-asr-corpus"
FILES = {"train": ["train-00000-of-00002", "train-00001-of-00002"], "dev": ["dev-00000-of-00001"], "test": ["test-00000-of-00001"]}
SR_OUT, MIN_S = 24000, 3.0
dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"torch {torch.__version__} torchaudio {torchaudio.__version__} device {dev}", flush=True)

bundle = torchaudio.pipelines.MMS_FA
model = bundle.get_model().to(dev).eval()  # with the star token: it soaks up sounds that have no letters
DICT = bundle.get_dict()
STAR = "*"
roman = ur.Uroman()


def romanise(word):
    r = roman.romanize_string(word.strip(PUNCT), lcode="ara").lower()
    r = "".join(c for c in r if c in DICT and c not in "-*")
    return r or STAR


def align(wave16, toks):
    """Start, end (seconds) and mean letter probability for every token, by CTC forced alignment."""
    spell = [romanise(t["t"]) if t["kind"] == "w" else STAR for t in toks]
    with torch.inference_mode():
        emission, _ = model(torch.from_numpy(wave16).unsqueeze(0).to(dev))
    targets = torch.tensor([[DICT[c] for w in spell for c in w]], dtype=torch.int32, device=emission.device)
    aligned, scores = F.forced_align(emission, targets, blank=0)
    spans = F.merge_tokens(aligned[0], scores[0].exp())
    sec = len(wave16) / 16000 / emission.shape[1]
    i = 0
    for t, w in zip(toks, spell):
        sp = spans[i:i + len(w)]
        i += len(w)
        frames = sum(x.end - x.start for x in sp)
        t["s"], t["e"] = round(sp[0].start * sec, 3), round(sp[-1].end * sec, 3)
        t["p"] = round(float(sum(x.score * (x.end - x.start) for x in sp) / max(1, frames)), 3)


def clip_stats(x):
    frame = int(0.02 * SR_OUT)
    e = np.array([np.mean(x[i:i + frame] ** 2) for i in range(0, max(1, len(x) - frame), frame)]) + 1e-10
    db = 10 * np.log10(e)
    return {"rms_db": round(float(10 * np.log10(np.mean(x ** 2) + 1e-10)), 1), "peak": round(float(np.max(np.abs(x))), 3),
            "snr_db": round(float(np.percentile(db, 95) - np.percentile(db, 5)), 1)}


segs, takes = [], []
for split, names in FILES.items():
    if job.get("splits") and split not in job["splits"]:
        continue
    tar = tarfile.open(os.path.join(OUT, f"apd_clips_{split}.tar"), "w")
    done = 0
    for name in names:
        local = os.environ.get("SDTTS_APD_DIR")  # parquet files already on disk: skip the download
        path = os.path.join(local, name + ".parquet") if local else hf_hub_download(REPO, f"data/apd_Arab/{name}.parquet", repo_type="dataset")
        for batch in pq.ParquetFile(path).iter_batches(batch_size=16):
            for r in batch.to_pylist():
                if job.get("limit") and done >= job["limit"]:
                    continue
                done += 1
                utt = os.path.splitext(r["audio"]["path"])[0]
                wave, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32", always_2d=True)
                wave = torch.from_numpy(np.ascontiguousarray(wave.mean(axis=1)))
                total = len(wave) / sr
                toks = tokens_of(r["raw_text"])
                if not any(t["kind"] == "w" for t in toks):
                    continue
                try:
                    align(F.resample(wave, sr, 16000).numpy(), toks)
                except Exception as ex:
                    print(f"{utt}: alignment failed {ex!r}", flush=True)
                    continue
                takes.append({"utt": utt, "speaker": r["speaker_id"], "split": split, "dur": round(total, 2), "prompt_id": r["prompt_id"],
                              "words": [[t["t"], t["kind"], t["s"], t["e"], t["p"]] for t in toks]})
                wave24 = F.resample(wave, sr, SR_OUT).numpy()
                k = 0
                for run in runs_of(toks):
                    for a, b, hard in cut_points(run, min_s=MIN_S):
                        ws = run[a:b + 1]
                        s, e = clip_bounds(toks, ws[0], ws[-1], total)
                        if e - s < 1.0:
                            continue
                        x = wave24[int(s * SR_OUT):int(e * SR_OUT)]
                        text = " ".join(w["t"] for w in ws)
                        k += 1
                        cid = f"{utt}_{k:02d}"
                        buf = io.BytesIO()
                        sf.write(buf, x, SR_OUT, format="FLAC", subtype="PCM_16")
                        info = tarfile.TarInfo(f"{r['speaker_id']}/{cid}.flac")
                        info.size = buf.getbuffer().nbytes  # not tell(): the encoder seeks back to finish the header
                        buf.seek(0)
                        tar.addfile(info, buf)
                        gaps = [ws[j + 1]["s"] - ws[j]["e"] for j in range(len(ws) - 1)] or [0.0]
                        segs.append(dict(id=cid, utt=utt, speaker=r["speaker_id"], split=split, start=round(s, 3), end=round(e, 3),
                                         dur=round(e - s, 2), text=text, n_words=len(ws), cps=round(letters(text) / (e - s), 2),
                                         p_mean=round(float(np.mean([w["p"] for w in ws])), 3), p_min=min(w["p"] for w in ws),
                                         max_gap=round(max(gaps), 2), hard_cut=hard, edge=bool(next_to_cut(toks, ws[0], ws[-1])),
                                         prompt_id=r["prompt_id"], src_sr=sr, **clip_stats(x)))
    tar.close()
    ss = [s for s in segs if s["split"] == split]
    print(f"{split}: {done} takes -> {len(ss)} clips, {sum(s['dur'] for s in ss) / 3600:.2f} h ({time.time() - t0:.0f}s)", flush=True)

with open(os.path.join(OUT, "apd_segments.jsonl"), "w", encoding="utf-8") as f:
    f.writelines(json.dumps(s, ensure_ascii=False) + "\n" for s in segs)
with open(os.path.join(OUT, "apd_words.jsonl"), "w", encoding="utf-8") as f:
    f.writelines(json.dumps(w, ensure_ascii=False) + "\n" for w in takes)
durs = sorted(s["dur"] for s in segs) or [0]
pm = sorted(s["p_mean"] for s in segs) or [0]
q = lambda v, p: round(float(v[int(p * (len(v) - 1))]), 2)
stats = {"takes": len(takes), "clips": len(segs), "hours": round(sum(durs) / 3600, 2),
         "hours_in_takes": round(sum(w["dur"] for w in takes) / 3600, 2),
         "dur_p10_p50_p90": [q(durs, .1), q(durs, .5), q(durs, .9)], "p_mean_p10_p50_p90": [q(pm, .1), q(pm, .5), q(pm, .9)],
         "hard_cuts": sum(s["hard_cut"] for s in segs), "under_min": sum(s["dur"] < MIN_S for s in segs),
         "per_speaker_min": {k: round(sum(s["dur"] for s in segs if s["speaker"] == k) / 60, 1) for k in sorted({s["speaker"] for s in segs})},
         "torch": torch.__version__, "torchaudio": torchaudio.__version__, "seconds": round(time.time() - t0)}
json.dump(stats, open(os.path.join(OUT, "stats.json"), "w"), indent=1)
print(json.dumps(stats, ensure_ascii=False), flush=True)
