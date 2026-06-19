#!/usr/bin/env bash
#SBATCH --job-name=singer_dummy_gpu
#SBATCH --nodelist=gpu1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --partition=base
#SBATCH --output=/home/maork/Projects/rad_sandbox/Sandbox/singerclassifier/slurm/logs/%x_%j.out
#SBATCH --error=/home/maork/Projects/rad_sandbox/Sandbox/singerclassifier/slurm/logs/%x_%j.err

set -euo pipefail

RAD_ROOT=/home/maork/Projects/rad_sandbox
REPO_ROOT=/home/maork/Projects/rad_sandbox/Sandbox/singerclassifier
PYTHON="$REPO_ROOT/.venv/bin/python"
CONFIG="$REPO_ROOT/configs/dummy_gpu.yaml"

cd "$RAD_ROOT"

export PYTHONPATH="$RAD_ROOT:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/usr/local/cuda/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"

echo "=== Job diagnostics ==="
hostname
date
pwd
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
nvidia-smi
echo "Python: $PYTHON"

if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: Virtual environment Python not found or not executable: $PYTHON" >&2
  echo "Run setup first:" >&2
  echo "  /home/maork/Projects/rad_sandbox/Sandbox/singerclassifier/scripts/setup_work_venv.sh" >&2
  exit 1
fi

echo "=== Smoke GPU test ==="
"$PYTHON" -m Sandbox.singerclassifier.scripts.smoke_gpu

echo "=== Dummy GPU training ==="
"$PYTHON" -m Sandbox.singerclassifier.src.dummy_gpu_train --config "$CONFIG"

echo "=== Metrics ==="
cat "$REPO_ROOT/experiments/dummy_gpu/metrics.json"
