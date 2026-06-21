#!/usr/bin/env bash
# Rerun Phase 6 manifest rows invalidated by augmentation wiring or multi-crop bugs.
set -euo pipefail

RAD_ROOT=/home/maork/Projects/rad_sandbox
REPO_ROOT=/home/maork/Projects/rad_sandbox/Sandbox/singerclassifier
ARTIFACT_ROOT=/home/maork/Projects/rad_sandbox/Sandbox/data/singerclassifier
PYTHON=/home/maork/Projects/rad_sandbox/Sandbox/SSL_Tabular/.venv/bin/python
SPLIT_CSV=$ARTIFACT_ROOT/processed/damp_sag_splits.csv
AUDIO_CACHE_DIR=$ARTIFACT_ROOT/cache/audio_22050_mono
SWEEP_SPEC=$REPO_ROOT/configs/phase6_sweeps.yaml
MANIFEST=$ARTIFACT_ROOT/manifests/phase6_sweep_manifest.csv

# Augmented + multi-crop rows that must be rerun after the wiring fixes.
DEFAULT_ARRAY="9-14"
ARRAY_SPEC="${1:-$DEFAULT_ARRAY}"
CONCURRENCY="${2:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -d "$REPO_ROOT" ]]; then
  :
else
  REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi

if ! command -v sbatch >/dev/null 2>&1; then
  echo "ERROR: sbatch not found. Run via SSH on mem-ans1 or use scripts/remote_rerun_phase6_invalid.sh" >&2
  exit 1
fi

if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: Python not found: $PYTHON" >&2
  exit 1
fi

cd "$RAD_ROOT"
export PYTHONPATH="$RAD_ROOT:${PYTHONPATH:-}"

echo "Phase 6 invalid-run rerun"
echo "========================="
echo "Array indices:  $ARRAY_SPEC"
echo "Concurrency:    %${CONCURRENCY}"
echo "Manifest:       $MANIFEST"

echo "Regenerating sweep configs..."
"$PYTHON" -m Sandbox.singerclassifier.scripts.generate_sweep_configs \
  --sweep-spec "$SWEEP_SPEC"

echo "Running augmentation smoke test..."
"$PYTHON" -m Sandbox.singerclassifier.scripts.smoke_augmentation \
  --split-csv "$SPLIT_CSV"

SAMPLE_CONFIG="$ARTIFACT_ROOT/generated_configs/phase6/cnn_augmented_lr0p001_auglight.yaml"
if [[ -f "$SAMPLE_CONFIG" ]]; then
  "$PYTHON" -m Sandbox.singerclassifier.scripts.smoke_augmentation \
    --config "$SAMPLE_CONFIG"
fi

if [[ ! -d "$AUDIO_CACHE_DIR" ]] || [[ "$(find "$AUDIO_CACHE_DIR" -name '*.pt' 2>/dev/null | wc -l | tr -d ' ')" -lt "$(( $(wc -l < "$SPLIT_CSV") - 1 ))" ]]; then
  echo "ERROR: Audio cache missing or incomplete. Run scripts/precompute_phase6_cache.sh first." >&2
  exit 1
fi

cd "$REPO_ROOT"
mkdir -p slurm/logs

export PHASE6_FORCE_RERUN=1

SBATCH_OUTPUT="$(sbatch --array="${ARRAY_SPEC}%${CONCURRENCY}" slurm/phase6_sweep_array.sbatch)"
echo "$SBATCH_OUTPUT"

JOB_ID="$(echo "$SBATCH_OUTPUT" | awk '{print $NF}')"
echo "Submitted rerun job id: ${JOB_ID}"
echo "Monitor: bash scripts/remote_status.sh"
