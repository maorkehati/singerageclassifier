#!/usr/bin/env bash
set -euo pipefail

RAD_ROOT=/home/maork/Projects/rad_sandbox
REPO_ROOT=/home/maork/Projects/rad_sandbox/Sandbox/singerclassifier
PYTHON=/home/maork/Projects/rad_sandbox/Sandbox/SSL_Tabular/.venv/bin/python
SPLIT_CSV=/home/maork/Projects/rad_sandbox/Sandbox/data/singerclassifier/processed/damp_sag_splits.csv
AUDIO_CACHE_DIR=/home/maork/Projects/rad_sandbox/Sandbox/data/singerclassifier/cache/audio_22050_mono

LIMIT_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --limit)
      if [[ $# -lt 2 ]]; then
        echo "ERROR: --limit requires a value" >&2
        exit 1
      fi
      LIMIT_ARGS=(--limit "$2")
      shift 2
      ;;
    --limit=*)
      LIMIT_ARGS=(--limit "${1#*=}")
      shift
      ;;
    -h|--help)
      echo "Usage: $0 [--limit N]"
      echo
      echo "Precompute decoded mono 22.05 kHz waveform cache for Phase 6."
      echo "Use --limit N for a small smoke run (e.g. --limit 5)."
      exit 0
      ;;
    *)
      echo "ERROR: Unknown argument: $1" >&2
      echo "Usage: $0 [--limit N]" >&2
      exit 1
      ;;
  esac
done

if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: Python not found: $PYTHON" >&2
  exit 1
fi

if [[ ! -f "$SPLIT_CSV" ]]; then
  echo "ERROR: Split CSV not found: $SPLIT_CSV" >&2
  echo "Generate splits first with prepare_splits or remote submission preflight." >&2
  exit 1
fi

mkdir -p "$AUDIO_CACHE_DIR"

echo "Phase 6 audio cache precompute"
echo "=============================="
echo "Split CSV:   $SPLIT_CSV"
echo "Cache dir:   $AUDIO_CACHE_DIR"
if [[ ${#LIMIT_ARGS[@]} -gt 0 ]]; then
  echo "Limit:       ${LIMIT_ARGS[1]}"
else
  echo "Limit:       (none — full split CSV)"
fi
echo
echo "This may take a while on first run. Progress is printed every 25 files."
echo

cd "$RAD_ROOT"
export PYTHONPATH="$RAD_ROOT:${PYTHONPATH:-}"

"$PYTHON" -m Sandbox.singerclassifier.scripts.precompute_audio_cache \
  --split-csv "$SPLIT_CSV" \
  --cache-dir "$AUDIO_CACHE_DIR" \
  --sample-rate 22050 \
  "${LIMIT_ARGS[@]}"

CACHE_COUNT="$(find "$AUDIO_CACHE_DIR" -name '*.pt' 2>/dev/null | wc -l | tr -d ' ')"
echo
echo "Cache files (.pt): $CACHE_COUNT"
echo "Cache directory:   $AUDIO_CACHE_DIR"
