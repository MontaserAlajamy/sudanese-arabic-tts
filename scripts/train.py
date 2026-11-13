"""Prepare the training set and launch a fine-tune on Kaggle.

  python3 scripts/train.py --engine omnivoice|lahgtna --apd-train DIR --apd-heldout DIR [--name ft1] [--smoke]
  python3 scripts/train.py --engine omnivoice|lahgtna [--name ft1] [--smoke] --collect

--engine omnivoice  jobs/train_omnivoice_job.py: OmniVoice family (default init oddadmix/lahgtna-omnivoice-v2), CC-BY-NC
--engine lahgtna    jobs/train_chatterbox_job.py: lahgtna-chatterbox (MIT), init oddadmix/lahgtna-chatterbox-v1
--apd-train / --apd-heldout: prep_apd_job.py outputs (apd_segments.jsonl + apd_clips_<split>.tar); may be the same folder.
Training clips: the train split, minus clips under 2 s, forced cuts, weak alignment (p_min < 0.2) and speeds
outside 4-20 letters per second. Dev loss: 120 clips of the dev split (never trained on).
--smoke: 60 clips, 30 steps, 2 benchmark sentences: proves the job installs, trains, saves and speaks.
The job runs detached (training takes about an hour); --collect waits for it and copies its outputs to
$SDTTS_WORK/train/<name>/ and the benchmark clips to $SDTTS_WORK/bake/<name>/.
"""
import argparse, json, os, random, shutil, subprocess, sys, tarfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
W = os.path.abspath(os.environ.get("SDTTS_WORK", "out"))
ENGINES = {"omnivoice": dict(job="train_omnivoice_job.py", init="oddadmix/lahgtna-omnivoice-v2", lr=1e-5, name="ft1", accum=2),
           "lahgtna": dict(job="train_chatterbox_job.py", init="oddadmix/lahgtna-chatterbox-v1", lr=2e-5, name="ft2", accum=4)}
ap = argparse.ArgumentParser()
ap.add_argument("--engine", choices=sorted(ENGINES), default="omnivoice")
ap.add_argument("--apd-train")
ap.add_argument("--apd-heldout")
ap.add_argument("--name")
ap.add_argument("--init")
ap.add_argument("--lr", type=float)
ap.add_argument("--steps", type=int, default=1200, help="omnivoice: optimizer steps")
ap.add_argument("--epochs", type=float, default=3, help="lahgtna: passes over the training clips")
ap.add_argument("--batch-tokens", type=int, default=2048, help="omnivoice, per GPU; 4096 runs out of memory on a 15 GB T4")
ap.add_argument("--batch", type=int, default=2, help="lahgtna: clips per step; 4 ran out of memory on a 15 GB T4 at step ~420 (clips up to 14.4 s)")
ap.add_argument("--grad-accum", type=int, help="default: omnivoice 2, lahgtna 4 (effective batch 8)")
ap.add_argument("--bench", default=os.path.join(W, "bench"))
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--collect", action="store_true")
a = ap.parse_args()
E = ENGINES[a.engine]
name = (a.name or E["name"]) + ("-smoke" if a.smoke else "")
job = f"train-{name}"
runner = os.path.join(ROOT, "kaggle/run.sh")
res = os.path.join(os.environ.get("SDTTS_OUT", os.path.join(W, "kaggle")), job, "out")

if a.collect:
    r = subprocess.run([runner, "--collect", job])
    dst = os.path.join(W, "train", name)
    os.makedirs(dst, exist_ok=True)
    for f in ("train_log.txt", "losses.json", "env.json"):
        if os.path.exists(os.path.join(res, f)):
            shutil.copy(os.path.join(res, f), dst)
    if os.path.isdir(os.path.join(res, "bake")):
        b = os.path.join(W, "bake", name)
        shutil.rmtree(b, ignore_errors=True)
        shutil.copytree(os.path.join(res, "bake"), b)
        print(f"{len(os.listdir(b))} benchmark clips -> {b}")
    print("model weights stay in", os.path.join(res, "model"))
    sys.exit(r.returncode)


def ok(s):
    return s["dur"] >= 2.0 and not s["hard_cut"] and s["p_min"] >= 0.2 and 4 <= s["cps"] <= 20


def pack(src_dir, split, keep, tar_out):
    with tarfile.open(os.path.join(src_dir, f"apd_clips_{split}.tar")) as t, tarfile.open(tar_out, "w") as o:
        for m in t.getmembers():
            if os.path.basename(m.name)[:-5] in keep:
                o.addfile(m, t.extractfile(m))


stage = os.path.join(W, "kaggle-in", job)
shutil.rmtree(stage, ignore_errors=True)
os.makedirs(stage)
rng = random.Random(7)
train = [s for s in map(json.loads, open(os.path.join(a.apd_train, "apd_segments.jsonl"), encoding="utf-8")) if s["split"] == "train" and ok(s)]
dev = [s for s in map(json.loads, open(os.path.join(a.apd_heldout, "apd_segments.jsonl"), encoding="utf-8")) if s["split"] == "dev" and ok(s)]
if a.smoke:
    train = rng.sample(train, 60)
dev = rng.sample(dev, min(30 if a.smoke else 120, len(dev)))
for split, rows, src in (("train", train, a.apd_train), ("dev", dev, a.apd_heldout)):
    with open(os.path.join(stage, f"{split}.jsonl"), "w", encoding="utf-8") as f:
        f.writelines(json.dumps({"id": s["id"], "text": s["text"], "speaker": s["speaker"]}, ensure_ascii=False) + "\n" for s in rows)
    pack(src, split, {s["id"] for s in rows}, os.path.join(stage, f"{split}.dat"))  # .dat: Kaggle would unpack a .tar
    print(f"{split}: {len(rows)} clips, {sum(s['dur'] for s in rows) / 3600:.2f} h, {len({s['speaker'] for s in rows})} speakers")
cfg = {"init": a.init or E["init"], "language_id": "sd", "lr": a.lr or E["lr"], "grad_accum": a.grad_accum or E["accum"],
       "bake": True, "bake_limit": 2 if a.smoke else 0}
if a.engine == "omnivoice":
    cfg.update(steps=30 if a.smoke else a.steps, batch_tokens=a.batch_tokens,
               save_steps=30 if a.smoke else 400, eval_steps=15 if a.smoke else 200)
else:
    cfg.update(steps=30 if a.smoke else 0, epochs=a.epochs, batch=a.batch, eval_steps=15 if a.smoke else 100,
               warmup=5 if a.smoke else 30)
json.dump(cfg, open(os.path.join(stage, "job.json"), "w"))
bench = os.path.abspath(a.bench)
shutil.copy(os.path.join(bench, "sentences.jsonl"), stage)
shutil.copy(os.path.join(bench, "refs.json"), stage)
for r in json.load(open(os.path.join(bench, "refs.json"))):
    shutil.copy(os.path.join(bench, "refs", r["file"]), stage)
print("job:", json.dumps(cfg))
files = [os.path.join(stage, f) for f in sorted(os.listdir(stage))]
sys.exit(subprocess.run([runner, "--detach", job, os.path.join(ROOT, "jobs", E["job"]), *files]).returncode)
