"""Copy a model's clips of the 12 listening sentences into a release folder as MP3 and write the sample table.

  python3 scripts/make_samples.py --clips <bake dir of the model> --out release/<model>/samples [--bench DIR]

Clips are loudness-matched to -20 LUFS (as on the blind page), 96 kbps mono MP3. Prints a Markdown table
(sentence, voice, file) to paste into the model card, and writes it to <out>/samples.md.
"""
import argparse, json, os, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ap = argparse.ArgumentParser()
ap.add_argument("--clips", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--bench", default=os.path.join(os.environ.get("SDTTS_WORK", "out"), "bench"))
a = ap.parse_args()

bench = {}
for line in open(os.path.join(a.bench, "sentences.jsonl"), encoding="utf-8"):
    s = json.loads(line)
    bench[s["id"]] = s
os.makedirs(a.out, exist_ok=True)
rows = ["| # | sentence | voice | audio |", "|---|---|---|---|"]
n = 0
for line in open(os.path.join(ROOT, "bench", "sentences.jsonl"), encoding="utf-8"):
    s = json.loads(line)
    if not s.get("listen"):
        continue
    src = os.path.join(a.clips, s["id"] + ".wav")
    if not os.path.exists(src):
        raise SystemExit(f"missing clip {src}")
    dst = os.path.join(a.out, s["id"] + ".mp3")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src, "-af", "loudnorm=I=-20:TP=-1.5:LRA=11", "-ac", "1",
                    "-ar", "24000", "-b:a", "96k", dst], check=True)
    n += 1
    rows.append(f"| {n} | {s['text']} | {bench.get(s['id'], {}).get('ref', '')} | [{s['id']}.mp3](samples/{s['id']}.mp3) |")
table = "\n".join(rows) + "\n"
open(os.path.join(a.out, "samples.md"), "w", encoding="utf-8").write(table)
print(table)
print(f"{n} samples -> {a.out}")
