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
CONCURRENCY="${1:-1}"

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

echo "Precomputing/checking audio cache..."
"$PYTHON" -m Sandbox.singerclassifier.scripts.precompute_audio_cache \
  --split-csv "$SPLIT_CSV" \
  --cache-dir "$AUDIO_CACHE_DIR" \
  --sample-rate 22050

echo "Running cache-based dataloader smoke test..."
"$PYTHON" -m Sandbox.singerclassifier.scripts.smoke_audio_preprocessing \
  --split-csv "$SPLIT_CSV" \
  --batch-size 4 \
  --duration-sec 15 \
  --num-workers 0 \
  --use-audio-cache \
  --audio-cache-dir "$AUDIO_CACHE_DIR" \
  --strict-audio-cache

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
