"""Score the baked clips of every system, plus the real held-out clips, in one Kaggle job (or locally with --local).

  python3 scripts/eval.py [--systems chatterbox,habibi,...] [--bench DIR] [--bake DIR] [--name eval] [--local]

Each system folder in <bake> (default $SDTTS_WORK/bake/<system>) holds <sentence id>.wav. The real clips in
<bench>/human/ are scored as system "human" against their own transcripts and their speaker's reference clip, so
every number has a human topline. Results: $SDTTS_WORK/eval/<name>/{metrics.jsonl, summary.json, table.md}
"""
import argparse, json, os, shutil, subprocess, sys, tarfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
W = os.environ.get("SDTTS_WORK", "out")
ap = argparse.ArgumentParser()
ap.add_argument("--systems", default="", help="comma list; default: every folder in --bake without -smoke")
ap.add_argument("--bench", default=os.path.join(W, "bench"))
ap.add_argument("--bake", default=os.path.join(W, "bake"))
ap.add_argument("--name", default="eval")
ap.add_argument("--whisper", default="large-v3")
ap.add_argument("--no-human", action="store_true")
ap.add_argument("--local", action="store_true", help="run the job here (CPU) instead of on Kaggle")
a = ap.parse_args()

sents = {s["id"]: s for s in (json.loads(x) for x in open(os.path.join(a.bench, "sentences.jsonl"), encoding="utf-8"))}
refs = {r["id"]: r for r in json.load(open(os.path.join(a.bench, "refs.json")))}
systems = [s for s in a.systems.split(",") if s] or sorted(d for d in os.listdir(a.bake) if not d.endswith("-smoke"))
stage = os.path.abspath(os.path.join(W, "kaggle-in", a.name))
shutil.rmtree(stage, ignore_errors=True)
os.makedirs(stage)
items, files = [], {}
for r in refs.values():
    files[f"refs/{r['file']}"] = os.path.join(a.bench, "refs", r["file"])
for system in systems:
    for sid, s in sents.items():
        wav = os.path.join(a.bake, system, sid + ".wav")
        if os.path.exists(wav):
            files[f"{system}/{sid}.wav"] = wav
            items.append({"clip": f"{system}/{sid}.wav", "system": system, "id": sid, "text": s["text"],
                          "ref": f"refs/{refs[s['ref']]['file']}", "tags": s.get("tags", []), "speaker": s["ref"]})
if not a.no_human:
    for x in open(os.path.join(a.bench, "human.jsonl"), encoding="utf-8"):
        h = json.loads(x)
        files[h["file"]] = os.path.join(a.bench, h["file"])
        items.append({"clip": h["file"], "system": "human", "id": h["id"], "text": h["text"],
                      "ref": f"refs/{refs[h['speaker']]['file']}", "tags": [], "speaker": h["speaker"]})
with tarfile.open(os.path.join(stage, "clips.dat"), "w") as t:  # .dat: Kaggle keeps it as one file
    for arc, src in files.items():
        t.add(src, arcname=arc)
with open(os.path.join(stage, "items.jsonl"), "w", encoding="utf-8") as f:
    f.writelines(json.dumps(i, ensure_ascii=False) + "\n" for i in items)
json.dump({"whisper": a.whisper, "lang": "ar"}, open(os.path.join(stage, "job.json"), "w"))
for m in ("text_norm.py", "metrics.py"):
    shutil.copy(os.path.join(ROOT, "sdtts", m), stage)
print(f"{len(items)} clips from {systems + ([] if a.no_human else ['human'])}", flush=True)

out = os.path.abspath(os.path.join(W, "eval", a.name))
os.makedirs(out, exist_ok=True)
if a.local:
    env = dict(os.environ, SDTTS_IN=stage, SDTTS_OUTDIR=out)
    subprocess.run([sys.executable, os.path.join(ROOT, "jobs/eval_job.py")], env=env, check=True)
else:
    job = f"eval-{a.name}"
    r = subprocess.run([os.path.join(ROOT, "kaggle/run.sh"), job, os.path.join(ROOT, "jobs/eval_job.py"),
                        *[os.path.join(stage, f) for f in os.listdir(stage)]])
    res = os.path.join(os.environ.get("SDTTS_OUT", os.path.join(W, "kaggle")), job, "out")
    for f in ("metrics.jsonl", "summary.json"):
        if os.path.exists(os.path.join(res, f)):
            shutil.copy(os.path.join(res, f), out)
    if r.returncode:
        sys.exit(f"eval kernel did not complete (see {res}/*.txt)")

summary = json.load(open(os.path.join(out, "summary.json")))
cols = ["system", "n", "failed", "cer", "wer", "cer_median", "sim", "sim_gap", "utmos", "dur", "lps", "hf_db", "clipped"]
fmt = lambda v: "" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))
table = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
table += ["| " + " | ".join(fmt(s.get(c)) for c in cols) + " |" for s in sorted(summary, key=lambda s: s["cer"])]
open(os.path.join(out, "table.md"), "w").write("\n".join(table) + "\n")
print("\n".join(table))
