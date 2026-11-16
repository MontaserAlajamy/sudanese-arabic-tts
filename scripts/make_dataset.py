"""Build the open Sudanese TTS dataset (Hugging Face layout: parquet shards + README data card) from prep_apd_job.py output.

  python3 scripts/make_dataset.py --apd DIR [--out out/dataset] [--name sudanese-arabic-speech] [--owner HF_USER]

DIR holds apd_segments.jsonl, apd_words.jsonl and apd_clips_<split>.tar (24 kHz mono FLAC, one file per clip).
Writes <out>/data/<split>-00000-of-0000N.parquet (audio stored as FLAC bytes, the `datasets` Audio feature layout),
<out>/words.jsonl (word timings per original take), <out>/README.md (card with YAML header) and <out>/stats.json.
Every clip is kept; `tts_ready` marks the ones that pass the training filter (same rule as scripts/train.py).
Source: Omnilingual ASR Corpus, Sudanese Arabic (apd_Arab), Meta FAIR, CC-BY-4.0. Derived audio keeps CC-BY-4.0.
"""
import argparse, json, os, shutil, tarfile
from collections import defaultdict

import pyarrow as pa
import pyarrow.parquet as pq

SPLITS = ("train", "dev", "test")
MAX_SHARD = 400 * 1024 * 1024  # bytes of audio per parquet file


def tts_ready(s):  # keep in step with scripts/train.py ok()
    return s["dur"] >= 2.0 and not s["hard_cut"] and s["p_min"] >= 0.2 and 4 <= s["cps"] <= 20


FIELDS = [("id", pa.string()), ("speaker", pa.string()), ("utterance", pa.string()), ("prompt_id", pa.string()),
          ("text", pa.string()), ("duration", pa.float32()), ("start_in_take", pa.float32()), ("end_in_take", pa.float32()),
          ("n_words", pa.int32()), ("chars_per_second", pa.float32()), ("align_p_mean", pa.float32()),
          ("align_p_min", pa.float32()), ("hard_cut", pa.bool_()), ("snr_db", pa.float32()), ("rms_db", pa.float32()),
          ("tts_ready", pa.bool_())]
SCHEMA = pa.schema([("audio", pa.struct([("bytes", pa.binary()), ("path", pa.string())]))] + FIELDS)


def row(s, audio, path):
    return {"audio": {"bytes": audio, "path": path}, "id": s["id"], "speaker": s["speaker"], "utterance": s["utt"],
            "prompt_id": s.get("prompt_id"), "text": s["text"], "duration": s["dur"], "start_in_take": s["start"],
            "end_in_take": s["end"], "n_words": s["n_words"], "chars_per_second": s["cps"], "align_p_mean": s["p_mean"],
            "align_p_min": s["p_min"], "hard_cut": s["hard_cut"], "snr_db": s.get("snr_db"), "rms_db": s.get("rms_db"),
            "tts_ready": tts_ready(s)}


def write_split(split, segs, tar_path, out):
    by_id = {s["id"]: s for s in segs}
    shards, cur, size = [], [], 0
    with tarfile.open(tar_path) as t:
        for m in t:
            if not m.isfile() or not m.name.endswith(".flac"):
                continue
            cid = os.path.basename(m.name)[:-5]
            if cid not in by_id:
                raise SystemExit(f"{split}: clip {cid} has no manifest row")
            cur.append(row(by_id.pop(cid), t.extractfile(m).read(), f"{cid}.flac"))
            size += m.size
            if size >= MAX_SHARD:
                shards.append(cur); cur, size = [], 0
    if cur:
        shards.append(cur)
    if by_id:
        raise SystemExit(f"{split}: {len(by_id)} manifest rows have no audio, e.g. {next(iter(by_id))}")
    names = []
    for i, rows in enumerate(shards):
        rows.sort(key=lambda r: r["id"])
        name = f"data/{split}-{i:05d}-of-{len(shards):05d}.parquet"
        pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), os.path.join(out, name), compression="zstd")
        names.append(name)
    return names


def card(stats, name, owner="<owner>"):
    sp = "\n".join(f"| {k} | {v['speakers']} | {v['clips']:,} | {v['hours']:.2f} | {v['tts_ready']:,} ({v['tts_ready_hours']:.2f} h) |"
                   for k, v in stats["splits"].items())
    t = stats["total"]
    return f"""---
license: cc-by-4.0
language:
- apd
- ar
pretty_name: Sudanese Arabic Speech (TTS-ready clips)
task_categories:
- text-to-speech
- automatic-speech-recognition
size_categories:
- 1K<n<10K
tags:
- sudanese-arabic
- apd
- dialect
- speech-synthesis
configs:
- config_name: default
  data_files:
""" + "".join(f"  - split: {k}\n    path: data/{k}-*.parquet\n" for k in stats["splits"]) + f"""---

# {name}: Sudanese Arabic speech cut for text-to-speech

`apd` is the ISO 639-3 code for Sudanese Arabic; the source corpus files it under `apd_Arab`.

{t['clips']:,} clips ({t['hours']:.2f} h) of spontaneous **Sudanese Arabic** from {t['speakers']} speakers, cut into
1-14 s pieces at natural pauses (median about 8 s) by word-level forced alignment, 24 kHz mono FLAC, each with its transcript.
As far as we could find (November 2026), it is the only Sudanese speech with a clear open licence that allows
commercial use (other sets state no licence or are non-commercial), and it was not in a TTS-ready form before.

## Source and licence
Derived from the Sudanese Arabic (`apd_Arab`) part of the **Omnilingual ASR Corpus** by the Omnilingual ASR Team,
Meta FAIR (2025), licensed **CC-BY-4.0** (`facebook/omnilingual-asr-corpus` on Hugging Face). This dataset keeps
CC-BY-4.0. Credit both: "Omnilingual ASR Team, Meta FAIR (2025); clips by the sudanese-tts project".
Transcripts are the corpus's human transcripts, unchanged except where a take was cut. Speaker IDs are anonymous
(`spk01`-`spk12`), as in the source; do not try to identify the speakers or clone a voice to impersonate anyone.

## Splits (same speakers as the source splits, so dev/test voices are never trained on)
| split | speakers | clips | hours | `tts_ready` |
|---|---|---|---|---|
{sp}

## How the clips were made
1. Each recording's transcript was aligned word by word to its audio (torchaudio MMS_FA, uroman romanisation).
2. Takes were cut at pauses into 1-14 s clips. Hesitations, noises and word fragments were cut out, never spoken
   in a clip; a clip whose end had no pause to cut at is marked `hard_cut`.
3. Audio resampled 48 → 24 kHz mono, FLAC.
Code: `jobs/prep_apd_job.py` and `sdtts/segment.py` in the project repository (https://github.com/MontaserAlajamy/sudanese-arabic-tts).

## Fields
`audio` (FLAC bytes + file name), `id`, `speaker`, `utterance` (source take), `prompt_id`, `text`, `duration`,
`start_in_take` / `end_in_take` (seconds in the source take), `n_words`, `chars_per_second`,
`align_p_mean` / `align_p_min` (forced-alignment confidence), `hard_cut`, `snr_db`, `rms_db`, and
`tts_ready`: duration ≥ 2 s, no hard cut, `align_p_min` ≥ 0.2, 4-20 letters per second (the filter our
fine-tunes used). `words.jsonl` holds the word timings of every source take.

## Known limits
- Spontaneous speech: false starts, laughter and background sound remain in some clips.
- Transcripts follow the source's spelling, which is not standardised (e.g. ده / دا, الريحه / الريحة).
- {t['speakers']} speakers is small; most clips are a few speakers' voices. The dev/test speakers are 4 of the 12.
- Not checked clip by clip by a native listener. Alignment confidence is the only per-clip quality signal.

## Load
```python
from datasets import load_dataset
ds = load_dataset("{owner}/{name}", split="train").filter(lambda r: r["tts_ready"])
```
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apd", required=True)
    ap.add_argument("--out", default="out/dataset")
    ap.add_argument("--name", default="sudanese-arabic-speech")
    ap.add_argument("--owner", default="<owner>", help="Hugging Face account, for the card's load example")
    a = ap.parse_args()
    segs = [json.loads(l) for l in open(os.path.join(a.apd, "apd_segments.jsonl"), encoding="utf-8")]
    by_split = defaultdict(list)
    for s in segs:
        by_split[s["split"]].append(s)
    shutil.rmtree(a.out, ignore_errors=True)
    os.makedirs(os.path.join(a.out, "data"))
    stats = {"splits": {}, "files": []}
    for split in SPLITS:
        if not by_split[split]:
            continue
        ss = by_split[split]
        stats["files"] += write_split(split, ss, os.path.join(a.apd, f"apd_clips_{split}.tar"), a.out)
        ok = [s for s in ss if tts_ready(s)]
        stats["splits"][split] = {"speakers": len({s["speaker"] for s in ss}), "clips": len(ss),
                                  "hours": sum(s["dur"] for s in ss) / 3600, "tts_ready": len(ok),
                                  "tts_ready_hours": sum(s["dur"] for s in ok) / 3600}
        print(split, stats["splits"][split], flush=True)
    stats["total"] = {"speakers": len({s["speaker"] for s in segs}), "clips": len(segs),
                      "hours": sum(s["dur"] for s in segs) / 3600}
    shutil.copy(os.path.join(a.apd, "apd_words.jsonl"), os.path.join(a.out, "words.jsonl"))
    json.dump(stats, open(os.path.join(a.out, "stats.json"), "w"), indent=1)
    open(os.path.join(a.out, "README.md"), "w", encoding="utf-8").write(card(stats, a.name, a.owner))
    print(stats["total"], "->", a.out)


if __name__ == "__main__":
    main()
