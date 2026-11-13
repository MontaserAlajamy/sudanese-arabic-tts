"""Count what the gated Kejue corpus holds for one dialect, reading only its metadata columns (never the audio).

  HF_TOKEN=... python3 scripts/census_kejue.py [--config ar-sd] [--out $SDTTS_WORK/census] [--files N]

Each parquet file is opened over HTTP and only the small columns are read (pyarrow column projection + range
requests), so the 38 GB of audio in ar-sd never comes down. Resumes per file. Writes:
  <out>/kejue_<config>/<file>.parquet   the metadata rows of each source file (cache)
  <out>/kejue_<config>.parquet          all rows
  <out>/kejue_<config>.summary.json     hours by licence, source, origin dataset, quality; speakers; channels
The dataset is gated (research use only, access approved by its maintainers). Until access is granted, file reads
return 403 and the script stops with that message.
"""
import argparse, collections, json, os, sys, time

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

REPO = "datasets/Arabic-STT/kejue-common-voice"
COLS = ["id", "speaker", "speaker_id", "channel", "video_id", "source", "source_kind", "hf_dataset", "license",
        "dialect", "sub_dialect", "code_switching", "duration_ms", "avg_confidence", "speech_ratio", "quality_ok",
        "quality_flags", "denoised", "sample_rate", "text"]

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="ar-sd")
ap.add_argument("--out", default=os.path.join(os.environ.get("SDTTS_WORK", "out"), "census"))
ap.add_argument("--files", type=int, default=0, help="only the first N files (a quick test)")
a = ap.parse_args()
token = os.environ.get("HF_TOKEN")
if not token:
    sys.exit("set HF_TOKEN (a read token of an account that was granted access to the dataset)")

fs = HfFileSystem(token=token)
files = sorted(fs.glob(f"{REPO}/{a.config}/*.parquet"))
if a.files:
    files = files[: a.files]
cache = os.path.join(a.out, f"kejue_{a.config}")
os.makedirs(cache, exist_ok=True)
print(f"{len(files)} files in {a.config}", flush=True)

t0 = time.time()
for n, f in enumerate(files, 1):
    dst = os.path.join(cache, os.path.basename(f))
    if os.path.exists(dst):
        continue
    for attempt in range(4):
        try:
            with fs.open(f, "rb", block_size=2**20) as fh:
                pf = pq.ParquetFile(fh)
                have = [c for c in COLS if c in pf.schema_arrow.names]
                table = pf.read(columns=have)
            break
        except Exception as e:
            msg = str(e)
            if "403" in msg or "gated" in msg.lower() or "Access to dataset" in msg:
                sys.exit("access denied (403): the dataset's maintainers have not granted this account access yet.\n"
                         "Request it at https://huggingface.co/datasets/Arabic-STT/kejue-common-voice and run again.")
            if attempt == 3:
                raise
            time.sleep(5 * (attempt + 1))
    table = table.append_column("file", pa.array([os.path.basename(f)] * table.num_rows))
    pq.write_table(table, dst + ".tmp")
    os.replace(dst + ".tmp", dst)
    if n % 25 == 0 or n == len(files):
        print(f"{n}/{len(files)} files ({time.time() - t0:.0f}s)", flush=True)

tables = [pq.read_table(os.path.join(cache, x)) for x in sorted(os.listdir(cache)) if x.endswith(".parquet")]
rows = pa.concat_tables(tables, promote_options="default").to_pylist() if tables else []
pq.write_table(pa.Table.from_pylist(rows), os.path.join(a.out, f"kejue_{a.config}.parquet"))

H = lambda rs: round(sum((r.get("duration_ms") or 0) for r in rs) / 3.6e6, 2)


def by(field, rs, top=30):
    g = collections.defaultdict(list)
    for r in rs:
        g[str(r.get(field))].append(r)
    return [{"value": k, "clips": len(v), "hours": H(v)} for k, v in sorted(g.items(), key=lambda kv: -H(kv[1]))[:top]]


clean = [r for r in rows if r.get("quality_ok") is True]
spk = collections.defaultdict(list)
for r in clean:
    spk[r.get("speaker_id")].append(r)
spk_h = sorted((H(v) for v in spk.values()), reverse=True)
cps = sorted(len((r.get("text") or "").replace(" ", "")) / max(0.1, (r.get("duration_ms") or 0) / 1000) for r in rows) or [0]
q = lambda v, p: round(float(v[int(p * (len(v) - 1))]), 2)
summary = {
    "config": a.config, "files": len(tables), "clips": len(rows), "hours": H(rows),
    "card": {"clips": 239635, "hours": 305.1} if a.config == "ar-sd" else None,
    "quality_ok": {"clips": len(clean), "hours": H(clean)},
    "by_license": by("license", rows), "by_source_kind": by("source_kind", rows), "by_hf_dataset": by("hf_dataset", rows),
    "by_source": by("source", rows, 15), "by_sub_dialect": by("sub_dialect", rows, 15),
    "code_switching": by("code_switching", rows), "by_sample_rate": by("sample_rate", rows),
    "speakers_clean": {"count": len(spk), "with_10min": sum(h >= 1 / 6 for h in spk_h), "with_30min": sum(h >= 0.5 for h in spk_h),
                       "with_1h": sum(h >= 1 for h in spk_h), "top10_hours": spk_h[:10]},
    "channels_clean_top": by("channel", clean, 25),
    "avg_confidence_p10_p50_p90": [q(sorted(r.get("avg_confidence") or 0 for r in rows), p) for p in (.1, .5, .9)] if rows else [],
    "chars_per_s_p5_p50_p95": [q(cps, p) for p in (.05, .5, .95)],
    "seconds": round(time.time() - t0),
}
json.dump(summary, open(os.path.join(a.out, f"kejue_{a.config}.summary.json"), "w"), ensure_ascii=False, indent=1)
print(json.dumps({k: summary[k] for k in ("clips", "hours", "card", "quality_ok", "by_license", "by_source_kind")}, ensure_ascii=False))
