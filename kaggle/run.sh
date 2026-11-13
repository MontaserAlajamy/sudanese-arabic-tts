#!/usr/bin/env bash
# Run a Python script as a private Kaggle kernel (free GPU or CPU) and bring back what it wrote to /kaggle/working.
#   kaggle/run.sh [--cpu] [--detach] [--sources user/dataset,...] [--kernels user/kernel,...] <job> <script.py> [input files...]
#   kaggle/run.sh --collect <job>     wait for a job pushed with --detach, then pull its outputs
# Input files go up as a private dataset <user>/sdtts-in-<job> (flat, a new version each run); the script finds them in
# the folder named by env SDTTS_IN. Outputs land in $SDTTS_OUT/<job>/ (default ./out/kaggle), the kernel log beside them.
# --sources / --kernels mount extra Kaggle datasets or another kernel's output (checkpoints, prepared data) under /kaggle/input.
# Needs the kaggle CLI, KAGGLE_USER (your Kaggle username) and a Kaggle API token (KAGGLE_API_TOKEN or ~/.kaggle).
# Kaggle allows two GPU sessions at once: run one job at a time and keep the other free. Exit 0 only if the kernel completed.
set -euo pipefail
GPU=true DETACH=0 COLLECT=0 SOURCES= KERNELS=
while [ $# -gt 0 ]; do case "$1" in
  --cpu) GPU=false; shift ;; --detach) DETACH=1; shift ;; --collect) COLLECT=1; shift ;;
  --sources) SOURCES=$2; shift 2 ;; --kernels) KERNELS=$2; shift 2 ;;
  -h|--help) sed -n 2,10p "$0"; exit 0 ;; *) break ;; esac; done
[ $# -ge 1 ] || { sed -n 2,10p "$0"; exit 1; }
U=${KAGGLE_USER:?set KAGGLE_USER to your Kaggle username}
slug() { echo "$1" | tr 'A-Z' 'a-z' | tr -c 'a-z0-9\n' '-' | head -c "$2"; }
JOB=$1; shift
SLUG=sdtts-$(slug "$JOB" 40) DS=sdtts-in-$(slug "$JOB" 36)
D=${SDTTS_OUT:-out/kaggle}/$JOB

if [ $COLLECT = 0 ]; then
  [ $# -ge 1 ] || { echo "missing <script.py>"; exit 1; }
  SCRIPT=$1; shift
  [ -f "$SCRIPT" ] || { echo "no such script: $SCRIPT"; exit 1; }
  rm -rf "$D"; mkdir -p "$D/kernel" "$D/dataset"
  SRC=()
  if [ $# -gt 0 ]; then
    for f in "$@"; do
      [ -f "$f" ] || { echo "no such input: $f"; exit 1; }
      [ ! -e "$D/dataset/$(basename "$f")" ] || { echo "two inputs share the name $(basename "$f")"; exit 1; }
      cp "$f" "$D/dataset/"
    done
    RUN_ID="$(date -u +%Y%m%dT%H%M%S)-$$-$RANDOM"; echo "$RUN_ID" > "$D/dataset/_run_id"
    printf '{"title":"%s","id":"%s/%s","licenses":[{"name":"other"}],"isPrivate":true}\n' "$DS" "$U" "$DS" > "$D/dataset/dataset-metadata.json"
    if kaggle datasets status "$U/$DS" >/dev/null 2>&1; then
      kaggle datasets version -p "$D/dataset" -m "$(date -u +%FT%TZ)" -r zip -q >/dev/null
    else
      kaggle datasets create -p "$D/dataset" -r zip -q >/dev/null
    fi
    # "ready" can still describe the previous version: wait until the new version's own _run_id comes back
    CHK=$(mktemp -d); OK=
    for _ in $(seq 1 90); do
      rm -f "$CHK/_run_id"; kaggle datasets download "$U/$DS" -f _run_id -p "$CHK" -q --force >/dev/null 2>&1 || true
      if [ -f "$CHK/_run_id.zip" ]; then (cd "$CHK" && unzip -oq _run_id.zip && rm -f _run_id.zip) || true; fi
      [ "$(cat "$CHK/_run_id" 2>/dev/null)" = "$RUN_ID" ] && { OK=1; break; }; sleep 10
    done; rm -rf "$CHK"
    [ -n "$OK" ] || { echo "input dataset $U/$DS never served the new version ($RUN_ID); not starting the kernel"; exit 1; }
    SRC+=("\"$U/$DS\"")
  fi
  IFS=, read -ra extra <<<"$SOURCES"; for s in "${extra[@]}"; do [ -n "$s" ] && SRC+=("\"$s\""); done
  KSRC=(); IFS=, read -ra extra <<<"$KERNELS"; for s in "${extra[@]}"; do [ -n "$s" ] && KSRC+=("\"$s\""); done
  join() { local IFS=,; echo "$*"; }
  { cat <<PY
import os  # the input dataset can be mounted at /kaggle/input/<slug> or deeper: find it by name
_hits = [r for r, d, f in os.walk("/kaggle/input") if os.path.basename(r) == "$DS"]
os.environ["SDTTS_IN"] = _hits[0] if _hits else "/kaggle/input"
print("SDTTS_IN =", os.environ["SDTTS_IN"], flush=True)
_rid = open(os.path.join(os.environ["SDTTS_IN"], "_run_id")).read().strip() if os.path.exists(os.path.join(os.environ["SDTTS_IN"], "_run_id")) else None
if "${RUN_ID:-}" and _rid != "${RUN_ID:-}":
    raise SystemExit(f"stale inputs: mounted run {_rid}, expected ${RUN_ID:-}")
PY
  cat "$SCRIPT"; } > "$D/kernel/script.py"
  SHAPE=; [ $GPU = true ] && SHAPE='"machine_shape": "NvidiaTeslaT4", '
  cat > "$D/kernel/kernel-metadata.json" <<EOF
{"id": "$U/$SLUG", "title": "$SLUG", "code_file": "script.py", "language": "python", "kernel_type": "script",
 "is_private": true, "enable_gpu": $GPU, "enable_internet": true, $SHAPE
 "dataset_sources": [$(join "${SRC[@]:-}")], "competition_sources": [], "kernel_sources": [$(join "${KSRC[@]:-}")]}
EOF
  until out=$(kaggle kernels push -p "$D/kernel" 2>&1) && ! grep -qi "maximum.*session" <<<"$out"; do
    grep -qi "maximum.*session" <<<"$out" || { echo "$out"; exit 1; }
    echo "GPU sessions busy (2 max); retrying in 2 min"; sleep 120
  done
  echo "$out" | head -1
  if [ $DETACH = 1 ]; then echo "detached: collect with  kaggle/run.sh --collect $JOB"; exit 0; fi
fi

mkdir -p "$D/out"
while true; do
  st=$(kaggle kernels status "$U/$SLUG" 2>&1 | tail -1)
  case "$st" in *COMPLETE*|*ERROR*|*CANCEL*) break ;; esac
  sleep 30
done
rm -rf "$D/out"; mkdir -p "$D/out"
kaggle kernels output "$U/$SLUG" -p "$D/out" -o >/dev/null 2>&1 || true
python3 - "$D/out" <<'PY'
import glob, json, sys
for f in glob.glob(sys.argv[1] + "/*.log"):
    try: t = "".join(x.get("data", "") for x in json.load(open(f)))
    except Exception: t = open(f).read()
    open(f[:-4] + ".txt", "w").write(t)
    print("\n".join(t.strip().splitlines()[-15:]))
PY
echo "$st"
case "$st" in *COMPLETE*) exit 0 ;; *) exit 1 ;; esac
