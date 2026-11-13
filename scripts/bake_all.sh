#!/usr/bin/env bash
# Speak the whole benchmark with each engine, one Kaggle job at a time, then score everything in one eval job.
#   scripts/bake_all.sh [engine ...]        (default: all five; a finished engine with 50 clips is skipped)
# Progress lines go to stdout; each engine's full log to $SDTTS_WORK/logs/bake-<engine>.log.
# Kaggle's GPU queue can hold a job for an hour, so run this detached:  setsid nohup scripts/bake_all.sh > log 2>&1 &
set -uo pipefail
HERE=$PWD; abs() { case "$1" in /*) echo "$1" ;; *) echo "$HERE/$1" ;; esac; }
export SDTTS_WORK=$(abs "${SDTTS_WORK:-out}")
cd "$(dirname "$0")/.."
ENGINES=("$@"); [ ${#ENGINES[@]} -gt 0 ] || ENGINES=(chatterbox habibi omnivoice lahgtna lahgtna-omnivoice)
N=$(grep -c . "$SDTTS_WORK/bench/sentences.jsonl")
mkdir -p "$SDTTS_WORK/logs"
ok=()
for e in "${ENGINES[@]}"; do
  have=$(ls "$SDTTS_WORK/bake/$e"/*.wav 2>/dev/null | wc -l)
  if [ "$have" -ge "$N" ]; then echo "$(date -u +%H:%M) $e: already done ($have clips)"; ok+=("$e"); continue; fi
  echo "$(date -u +%H:%M) $e: start"
  if scripts/bake.sh "$e" > "$SDTTS_WORK/logs/bake-$e.log" 2>&1; then ok+=("$e"); fi
  echo "$(date -u +%H:%M) $e: $(tail -1 "$SDTTS_WORK/logs/bake-$e.log")"
done
[ ${#ok[@]} -gt 0 ] || { echo "no engine produced audio"; exit 1; }
echo "$(date -u +%H:%M) eval: ${ok[*]} + human"
python3 scripts/eval.py --systems "$(IFS=,; echo "${ok[*]}")" > "$SDTTS_WORK/logs/eval.log" 2>&1
tail -12 "$SDTTS_WORK/logs/eval.log"
echo "$(date -u +%H:%M) ALL DONE"
