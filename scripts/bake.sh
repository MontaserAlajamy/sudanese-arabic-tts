#!/usr/bin/env bash
# Speak the benchmark with one open TTS model on a free Kaggle GPU (zero-shot, cloning the benchmark's reference clips).
#   scripts/bake.sh <engine> [--smoke] [--variant text|tashkeel] [--bench DIR] [--tag NAME] [--kernels user/train-kernel]
# --kernels / --sources mount a training kernel's output or a model dataset and speaks with its fine-tuned checkpoint (lahgtna-omnivoice only).
# engines: chatterbox | lahgtna | habibi | omnivoice | lahgtna-omnivoice
# <bench> holds sentences.jsonl, refs.json and refs/<file>.wav (default $SDTTS_WORK/bench, written by scripts/make_bench.py).
# --smoke speaks only the first 2 sentences (a quick check that the engine installs and runs).
# Output: $SDTTS_WORK/bake/<engine>[-<variant>][-smoke]/<id>.wav + timings.json, env.json, errors.json
set -euo pipefail
[ $# -ge 1 ] || { sed -n 2,7p "$0"; exit 1; }
HERE=$PWD; abs() { case "$1" in /*) echo "$1" ;; *) echo "$HERE/$1" ;; esac; }  # paths are relative to the caller
cd "$(dirname "$0")/.."
ENGINE=$1; shift
W=$(abs "${SDTTS_WORK:-out}"); export SDTTS_OUT=$(abs "${SDTTS_OUT:-$W/kaggle}")
BENCH=$W/bench VARIANT=text LIMIT=0 TAG= KERNELS= SOURCES= CKPT=false
while [ $# -gt 0 ]; do case "$1" in
  --smoke) LIMIT=2; shift ;; --variant) VARIANT=$2; shift 2 ;; --bench) BENCH=$(abs "$2"); shift 2 ;; --tag) TAG=$2; shift 2 ;;
  --kernels) KERNELS=$2; CKPT=true; shift 2 ;;
  --sources) SOURCES=$2; CKPT=true; shift 2 ;;
  *) echo "unknown arg $1"; exit 1 ;; esac; done
case "$ENGINE" in chatterbox|lahgtna|habibi|omnivoice|lahgtna-omnivoice) ;; *) echo "unknown engine $ENGINE"; exit 1 ;; esac
for f in sentences.jsonl refs.json; do [ -f "$BENCH/$f" ] || { echo "missing $BENCH/$f"; exit 1; }; done
NAME=$ENGINE; [ "$VARIANT" = text ] || NAME=$NAME-$VARIANT; [ -z "$TAG" ] || NAME=$NAME-$TAG; [ $LIMIT = 0 ] || NAME=$NAME-smoke
T=$W/kaggle-in/bake-$NAME; rm -rf "$T"; mkdir -p "$T"
cp "$BENCH/sentences.jsonl" "$BENCH/refs.json" "$T/"
python3 - "$BENCH" "$T" <<'PY'
import json, shutil, sys
bench, out = sys.argv[1:3]
for r in json.load(open(f"{bench}/refs.json")):
    shutil.copy(f"{bench}/refs/{r['file']}", f"{out}/{r['file']}")
PY
printf '{"engine": "%s", "variant": "%s", "limit": %s, "seed": 7, "ckpt": %s}\n' "$ENGINE" "$VARIANT" "$LIMIT" "$CKPT" > "$T/job.json"
t0=$(date +%s)
kaggle/run.sh ${KERNELS:+--kernels "$KERNELS"} ${SOURCES:+--sources "$SOURCES"} "bake-$NAME" jobs/bake_job.py "$T"/* || echo "kernel did not complete (see $SDTTS_OUT/bake-$NAME/out/*.txt)"
R=$SDTTS_OUT/bake-$NAME/out; O=$W/bake/$NAME
n=$(ls "$R"/*.wav 2>/dev/null | wc -l)
[ "$n" -gt 0 ] || { echo "no audio came back for $NAME"; exit 1; }
rm -rf "$O"; mkdir -p "$O"; cp "$R"/*.wav "$R"/*.json "$O/" 2>/dev/null || true
echo "$NAME: $n clips in $(( $(date +%s) - t0 ))s -> $O"
