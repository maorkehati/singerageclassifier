#!/usr/bin/env bash
set -euo pipefail

RAD_ROOT=/home/maork/Projects/rad_sandbox
REPO_ROOT=/home/maork/Projects/rad_sandbox/Sandbox/singerclassifier
DATA_ROOT=/home/maork/Projects/rad_sandbox/Sandbox/data/DAMP-S-AG-partial/DAMP-S-AG
ARTIFACT_ROOT=/home/maork/Projects/rad_sandbox/Sandbox/data/singerclassifier
PYTHON=/home/maork/Projects/rad_sandbox/Sandbox/SSL_Tabular/.venv/bin/python
SPLIT_CSV=$ARTIFACT_ROOT/processed/damp_sag_splits.csv
SPLIT_SUMMARY=$ARTIFACT_ROOT/data_inspection/split_summary.json
AUDIO_CACHE_DIR=$ARTIFACT_ROOT/cache/audio_22050_mono
SWEEP_SPEC=$REPO_ROOT/configs/phase6_sweeps.yaml
MANIFEST=$ARTIFACT_ROOT/manifests/phase6_sweep_manifest.csv

CONCURRENCY=1
BUILD_CACHE=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build-cache)
      BUILD_CACHE=true
      shift
      ;;
    *)
      if [[ "$1" =~ ^[0-9]+$ ]]; then
        CONCURRENCY="$1"
      else
        echo "ERROR: Unknown argument: $1" >&2
        echo "Usage: $0 [concurrency] [--build-cache]" >&2
        exit 1
      fi
      shift
      ;;
  esac
done

if ! command -v sbatch >/dev/null 2>&1; then
  echo "ERROR: sbatch not found. This script must run on the SLURM submission node, e.g. mem-ans1." >&2
  echo "From the work Linux machine, use: bash scripts/remote_submit_phase6.sh 1" >&2
  exit 1
fi

if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: Python not found: $PYTHON" >&2
  exit 1
fi

cd "$RAD_ROOT"
export PYTHONPATH="$RAD_ROOT:${PYTHONPATH:-}"

mkdir -p "$ARTIFACT_ROOT/processed" "$ARTIFACT_ROOT/data_inspection" \
  "$ARTIFACT_ROOT/generated_configs/phase6" "$ARTIFACT_ROOT/manifests" \
  "$AUDIO_CACHE_DIR" \
  "$REPO_ROOT/slurm/logs"

echo "Phase 6 sweep preflight"
echo "======================="
echo "Artifact root: $ARTIFACT_ROOT"
echo "Split CSV:     $SPLIT_CSV"
echo "Audio cache:   $AUDIO_CACHE_DIR"
echo "Manifest:      $MANIFEST"
echo "Build cache:   $BUILD_CACHE"

if [[ ! -f "$SPLIT_CSV" ]]; then
  echo "Split CSV missing. Generating splits..."
  "$PYTHON" -m Sandbox.singerclassifier.scripts.prepare_splits \
    --data-root "$DATA_ROOT" \
    --output-csv "$SPLIT_CSV" \
    --summary-json "$SPLIT_SUMMARY" \
    --train-ratio 0.70 \
    --val-ratio 0.15 \
    --test-ratio 0.15 \
    --seed 42
fi

echo "Regenerating sweep configs and manifest..."
"$PYTHON" -m Sandbox.singerclassifier.scripts.generate_sweep_configs \
  --sweep-spec "$SWEEP_SPEC"

if [[ ! -f "$SPLIT_CSV" ]]; then
  echo "ERROR: Split CSV still missing after generation attempt: $SPLIT_CSV" >&2
  exit 1
fi

if [[ ! -f "$MANIFEST" ]]; then
  echo "ERROR: Manifest missing after generation attempt: $MANIFEST" >&2
  exit 1
fi

while IFS= read -r config_path; do
  if [[ -n "$config_path" && ! -f "$config_path" ]]; then
    echo "ERROR: Manifest references missing config: $config_path" >&2
    exit 1
  fi
done < <(tail -n +2 "$MANIFEST" | cut -d, -f6)

if [[ "$BUILD_CACHE" == true ]]; then
  echo "Building audio cache (--build-cache)..."
  "$PYTHON" -m Sandbox.singerclassifier.scripts.precompute_audio_cache \
    --split-csv "$SPLIT_CSV" \
    --cache-dir "$AUDIO_CACHE_DIR" \
    --sample-rate 22050
else
  echo "Validating audio cache (not building)..."
  if [[ ! -d "$AUDIO_CACHE_DIR" ]]; then
    CACHE_COUNT=0
  else
    CACHE_COUNT="$(find "$AUDIO_CACHE_DIR" -name '*.pt' 2>/dev/null | wc -l | tr -d ' ')"
  fi
  SPLIT_COUNT=$(( $(wc -l < "$SPLIT_CSV") - 1 ))

  if [[ "$CACHE_COUNT" -lt "$SPLIT_COUNT" ]]; then
    echo "ERROR: Audio cache is missing or incomplete." >&2
    echo "Expected at least $SPLIT_COUNT .pt files, found $CACHE_COUNT." >&2
    echo "Run this first from the work Linux machine:" >&2
    echo "cd /home/maork/Projects/rad_sandbox/Sandbox/singerclassifier" >&2
    echo "bash scripts/precompute_phase6_cache.sh" >&2
    exit 1
  fi
  echo "Audio cache OK: $CACHE_COUNT .pt files (need >= $SPLIT_COUNT)"
fi

echo "Running cache-based dataloader smoke test..."
"$PYTHON" -m Sandbox.singerclassifier.scripts.smoke_audio_preprocessing \
  --split-csv "$SPLIT_CSV" \
  --batch-size 4 \
  --duration-sec 15 \
  --num-workers 0 \
  --use-audio-cache \
  --audio-cache-dir "$AUDIO_CACHE_DIR" \
  --strict-audio-cache

echo "Running augmentation wiring smoke test..."
"$PYTHON" -m Sandbox.singerclassifier.scripts.smoke_augmentation \
  --split-csv "$SPLIT_CSV"

SAMPLE_AUG_CONFIG="$ARTIFACT_ROOT/generated_configs/phase6/cnn_augmented_lr0p001_auglight.yaml"
if [[ -f "$SAMPLE_AUG_CONFIG" ]]; then
  "$PYTHON" -m Sandbox.singerclassifier.scripts.smoke_augmentation \
    --config "$SAMPLE_AUG_CONFIG"
fi

N=$(($(wc -l < "$MANIFEST") - 1))
if [[ "$N" -le 0 ]]; then
  echo "ERROR: Manifest has no runs: $MANIFEST" >&2
  exit 1
fi

MAX_INDEX=$((N - 1))
echo "Submitting Phase 6 sweep array: 0-${MAX_INDEX} with concurrency %${CONCURRENCY}"
echo "Runs: $N"

PREFLIGHT_AUDIO=$DATA_ROOT/amazing_grace/160395093_196074140.m4a
echo "Running audio decode preflight: $PREFLIGHT_AUDIO"
"$PYTHON" -m Sandbox.singerclassifier.scripts.smoke_audio_preprocessing \
  --audio-path "$PREFLIGHT_AUDIO"

cd "$REPO_ROOT"

SBATCH_OUTPUT="$(sbatch --array="0-${MAX_INDEX}%${CONCURRENCY}" slurm/phase6_sweep_array.sbatch)"
echo "$SBATCH_OUTPUT"

JOB_ID="$(echo "$SBATCH_OUTPUT" | awk '{print $NF}')"
echo "Submitted job id: ${JOB_ID}"
echo "Monitor with: squeue -u \$USER"
echo "From work Linux: bash scripts/remote_status.sh"
