#!/usr/bin/env bash
#SBATCH --job-name=phase6_sweep
#SBATCH --partition=base
#SBATCH --nodelist=gpu1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/home/maork/Projects/rad_sandbox/Sandbox/singerclassifier/slurm/logs/phase6_%A_%a.out
#SBATCH --error=/home/maork/Projects/rad_sandbox/Sandbox/singerclassifier/slurm/logs/phase6_%A_%a.err

set -euo pipefail

RAD_ROOT=/home/maork/Projects/rad_sandbox
REPO_ROOT=/home/maork/Projects/rad_sandbox/Sandbox/singerclassifier
PYTHON=/home/maork/Projects/rad_sandbox/Sandbox/SSL_Tabular/.venv/bin/python
MANIFEST=/home/maork/Projects/rad_sandbox/Sandbox/data/singerclassifier/manifests/phase6_sweep_manifest.csv

cd "$RAD_ROOT"
export PYTHONPATH="$RAD_ROOT:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/usr/local/cuda/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"

echo "=== Phase 6 sweep array task ==="
hostname
date
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-unset}"
env -u LD_LIBRARY_PATH nvidia-smi || echo "WARNING: nvidia-smi failed; continuing"
echo "Python: $PYTHON"

if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: Python not found: $PYTHON" >&2
  exit 1
fi

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "ERROR: SLURM_ARRAY_TASK_ID is not set. Submit with --array." >&2
  exit 1
fi

mkdir -p "$REPO_ROOT/slurm/logs"

MANIFEST_ARGS=(--manifest "$MANIFEST" --index "$SLURM_ARRAY_TASK_ID")
if [[ "${PHASE6_FORCE_RERUN:-0}" != "1" ]]; then
  MANIFEST_ARGS+=(--skip-existing)
fi

"$PYTHON" -m Sandbox.singerclassifier.scripts.run_sweep_manifest \
  "${MANIFEST_ARGS[@]}"

echo "=== Array task complete ==="
