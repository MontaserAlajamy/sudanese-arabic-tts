"""Build the blind listening page: the listening subset of the benchmark, every system's clip under an opaque name,
in a shuffled order per sentence, all at the same loudness.

  python3 scripts/make_page.py --systems chatterbox,habibi,... [--extra name=dir,...] [--round r1] [--bench DIR]

<bake>/<system>/<sentence id>.wav are the clips (default $SDTTS_WORK/bake). --extra adds folders outside <bake>
(e.g. a paid service used only as a private yardstick). Writes $SDTTS_WORK/blind/<round>/:
  page.html      the page (publish it with clips/ as supporting files)
  clips/<k>.mp3  64 kbps mono at -20 LUFS, end silences trimmed to ~0.15 s, k = an opaque hash
  key.json       which system each k is: PRIVATE, never published with the page
"""
import argparse, hashlib, json, os, random, secrets, shutil, subprocess, tempfile

import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
W = os.environ.get("SDTTS_WORK", "out")
ap = argparse.ArgumentParser()
ap.add_argument("--systems", required=True)
ap.add_argument("--extra", default="", help="name=dir,name=dir")
ap.add_argument("--round", default="r1")
ap.add_argument("--bench", default=os.path.join(W, "bench"))
ap.add_argument("--bake", default=os.path.join(W, "bake"))
ap.add_argument("--reuse", action="store_true", help="keep this round's salt: same clip names and order")
ap.add_argument("--repeat", type=int, default=0, help="play N random clips twice (different letters) to measure rater consistency")
a = ap.parse_args()


def trimmed(wav, keep=0.15, floor_db=35.0):
    """The clip with silence at either end cut to `keep` seconds (silence = 20 ms frames more than floor_db below the
    loudest frame). Silence at the edges would identify a system on a blind page: one engine pads its starts."""
    x, sr = sf.read(wav, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    hop = int(0.02 * sr)
    n = max(1, len(x) // hop)
    db = 10 * np.log10(np.square(x[: n * hop]).reshape(n, hop).mean(axis=1) + 1e-12)
    on = np.where(db > db.max() - floor_db)[0]
    if not len(on):
        return x, sr
    start = max(0, on[0] * hop - int(keep * sr))
    end = min(len(x), (on[-1] + 1) * hop + int(keep * sr))
    return x[start:end], sr


dirs = {s: os.path.join(a.bake, s) for s in a.systems.split(",") if s}
for e in filter(None, a.extra.split(",")):
    name, d = e.split("=", 1)
    dirs[name] = d
out = os.path.join(W, "blind", a.round)
os.makedirs(os.path.join(out, "clips"), exist_ok=True)
old = os.path.join(out, "key.json")
salt = json.load(open(old))["salt"] if a.reuse and os.path.exists(old) else secrets.token_hex(8)
rng = random.Random(salt)
sents = [json.loads(x) for x in open(os.path.join(a.bench, "sentences.jsonl"), encoding="utf-8")]
page, key, missing = [], {}, []
listen_ids = [s["id"] for s in sents if s["listen"]]
repeat_ids = set(random.Random(salt + ":r").sample(listen_ids, min(a.repeat, len(listen_ids))))
for s in (s for s in sents if s["listen"]):
    clips = []
    for system, d in dirs.items():
        wav = os.path.join(d, s["id"] + ".wav")
        if not os.path.exists(wav):
            missing.append(f"{system}/{s['id']}")
            continue
        k = hashlib.sha256(f"{salt}:{system}:{s['id']}".encode()).hexdigest()[:12]
        x, sr = trimmed(wav)
        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            sf.write(tmp.name, x, sr, subtype="PCM_16")
            subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", tmp.name, "-af", "loudnorm=I=-20:TP=-1.5:LRA=11",
                            "-ac", "1", "-ar", "24000", "-b:a", "64k", os.path.join(out, "clips", k + ".mp3")], check=True)
        key[k] = {"system": system, "id": s["id"], "ref": s["ref"]}
        clips.append({"key": k, "src": f"clips/{k}.mp3"})
    if s["id"] in repeat_ids and clips:  # the same audio again under a new name: rater consistency
        orig = clips[rng.randrange(len(clips))]["key"]
        k2 = hashlib.sha256(f"{salt}:repeat:{s['id']}".encode()).hexdigest()[:12]
        shutil.copy(os.path.join(out, "clips", orig + ".mp3"), os.path.join(out, "clips", k2 + ".mp3"))
        key[k2] = dict(key[orig], repeat_of=orig)
        clips.append({"key": k2, "src": f"clips/{k2}.mp3"})
    rng.shuffle(clips)
    page.append({"id": s["id"], "text": s["text"], "tags": s["tags"], "clips": clips})
data = json.dumps({"round": a.round, "sentences": page}, ensure_ascii=False).replace("</", "<\\/")
html = open(os.path.join(ROOT, "sdtts/blind_page.html"), encoding="utf-8").read().replace("__DATA__", data)
open(os.path.join(out, "page.html"), "w", encoding="utf-8").write(html)
json.dump({"round": a.round, "salt": salt, "systems": list(dirs), "clips": key}, open(os.path.join(out, "key.json"), "w"), indent=1)
print(f"{len(page)} sentences, {len(key)} clips from {list(dirs)} -> {out}")
if missing:
    print("missing:", " ".join(missing))
