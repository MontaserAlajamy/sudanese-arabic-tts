"""Assemble the benchmark folder from the sentence list and the aligned held-out Omnilingual clips.

  python3 scripts/make_bench.py --apd <prep_apd output dir> [--out $SDTTS_WORK/bench] [--human 12]

Reads bench/sentences.jsonl and bench/refs.json (the chosen reference clip of each held-out speaker) and writes:
  <out>/sentences.jsonl   every sentence with the reference voice it should be spoken in (round robin per tag)
  <out>/refs.json         [{"id", "file", "text"}] and <out>/refs/ref-<id>.wav (24 kHz mono, -23 LUFS)
  <out>/human.jsonl       real clips of the same speakers (not the reference) = the topline for every metric
  <out>/human/<clip>.wav
Reference and human clips come from Meta's Omnilingual ASR Corpus (apd_Arab, CC-BY-4.0); none of those speakers is
in the train split, so a model fine-tuned on that split is still tested on unseen voices.
"""
import argparse, io, json, os, random, tarfile, warnings

import numpy as np
import pyloudnorm as pyln
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ap = argparse.ArgumentParser()
ap.add_argument("--apd", required=True, help="folder with apd_segments.jsonl and apd_clips_<split>.tar")
ap.add_argument("--out", default=os.path.join(os.environ.get("SDTTS_WORK", "out"), "bench"))
ap.add_argument("--human", type=int, default=12, help="real clips per speaker for the topline")
a = ap.parse_args()

sents = [json.loads(x) for x in open(os.path.join(ROOT, "bench/sentences.jsonl"), encoding="utf-8") if x.strip()]
refs = json.load(open(os.path.join(ROOT, "bench/refs.json")))
segs = {s["id"]: s for s in (json.loads(x) for x in open(os.path.join(a.apd, "apd_segments.jsonl"), encoding="utf-8"))}
meter = pyln.Meter(24000)


def level(x, target=-23.0):
    warnings.filterwarnings("ignore", "Possible clipped samples")  # the peak is capped just below
    y = pyln.normalize.loudness(x, meter.integrated_loudness(x), target)
    peak = np.max(np.abs(y))
    return y * (0.95 / peak) if peak > 0.95 else y


def clips(split):
    with tarfile.open(os.path.join(a.apd, f"apd_clips_{split}.tar")) as t:
        for m in t.getmembers():
            yield m.name.split("/")[-1][:-5], t.extractfile(m).read()


speakers = [r["id"] for r in refs]
ref_clip = {r["clip"]: r for r in refs}
rng = random.Random(7)
human = {}
for spk in speakers:  # real topline: fluent mid-length clips with confident alignment, never the reference itself
    pool = [s for s in segs.values() if s["speaker"] == spk and s["id"] not in ref_clip and 3 <= s["dur"] <= 12
            and not s["edge"] and not s["hard_cut"] and s["p_min"] > 0.2]
    for s in rng.sample(pool, min(a.human, len(pool))):
        human[s["id"]] = s

os.makedirs(os.path.join(a.out, "refs"), exist_ok=True)
os.makedirs(os.path.join(a.out, "human"), exist_ok=True)
out_refs, out_human = [], []
for split in sorted({s["split"] for s in segs.values()}):
    for cid, data in clips(split):
        if cid not in ref_clip and cid not in human:
            continue
        x, sr = sf.read(io.BytesIO(data), dtype="float32")
        assert sr == 24000
        if cid in ref_clip:
            r = ref_clip[cid]
            sf.write(os.path.join(a.out, "refs", f"ref-{r['id']}.wav"), level(x), sr, subtype="PCM_16")
            out_refs.append({"id": r["id"], "file": f"ref-{r['id']}.wav", "text": segs[cid]["text"], "clip": cid,
                             "dur": segs[cid]["dur"], "pitch": r["pitch"], "source": "facebook/omnilingual-asr-corpus apd_Arab (CC-BY-4.0)"})
        else:
            s = human[cid]
            sf.write(os.path.join(a.out, "human", cid + ".wav"), level(x), sr, subtype="PCM_16")
            out_human.append({"id": cid, "speaker": s["speaker"], "text": s["text"], "dur": s["dur"], "file": f"human/{cid}.wav"})
assert len(out_refs) == len(refs), "a reference clip was not found in the tars"

# reference voice per sentence: the listening subset gets exactly 3 sentences per voice, the rest a plain round robin
listen = [s for s in sents if s["listen"]]
rest = [s for s in sents if not s["listen"]]
for group in (listen, rest):
    for i, s in enumerate(group):
        s["ref"] = speakers[i % len(speakers)]

with open(os.path.join(a.out, "sentences.jsonl"), "w", encoding="utf-8") as f:
    f.writelines(json.dumps(s, ensure_ascii=False) + "\n" for s in sents)
json.dump(sorted(out_refs, key=lambda r: r["id"]), open(os.path.join(a.out, "refs.json"), "w"), ensure_ascii=False, indent=1)
with open(os.path.join(a.out, "human.jsonl"), "w", encoding="utf-8") as f:
    f.writelines(json.dumps(h, ensure_ascii=False) + "\n" for h in sorted(out_human, key=lambda h: h["id"]))
print(f"{len(sents)} sentences, {len(out_refs)} reference voices, {len(out_human)} human clips -> {a.out}")
print({spk: sum(s["ref"] == spk for s in sents) for spk in speakers})
