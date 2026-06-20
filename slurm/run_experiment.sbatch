#!/usr/bin/env bash
#SBATCH --job-name=singer_experiment
#SBATCH --nodelist=gpu1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --partition=base
#SBATCH --output=/home/maork/Projects/rad_sandbox/Sandbox/singerclassifier/slurm/logs/%x_%j.out
#SBATCH --error=/home/maork/Projects/rad_sandbox/Sandbox/singerclassifier/slurm/logs/%x_%j.err

set -euo pipefail

RAD_ROOT=/home/maork/Projects/rad_sandbox
REPO_ROOT=/home/maork/Projects/rad_sandbox/Sandbox/singerclassifier
PYTHON="$REPO_ROOT/.venv/bin/python"
ARG="${1:?Usage: sbatch slurm/run_experiment.sbatch <config.yaml|phase6>}"

cd "$RAD_ROOT"

export PYTHONPATH="$RAD_ROOT:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/usr/local/cuda/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"

echo "=== Job diagnostics ==="
hostname
date
pwd
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
nvidia-smi || true
echo "Python: $PYTHON"
echo "Argument: $ARG"

if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: Virtual environment Python not found: $PYTHON" >&2
  exit 1
fi

mkdir -p "$REPO_ROOT/slurm/logs"
mkdir -p "$REPO_ROOT/experiments"

if [[ "$ARG" == "phase6" ]]; then
  echo "=== Running Phase 6 experiment set ==="
  "$PYTHON" -m Sandbox.singerclassifier.scripts.run_experiments --experiment-set phase6

  echo "=== Summarizing Phase 6 experiments ==="
  "$PYTHON" -m Sandbox.singerclassifier.scripts.summarize_experiments \
    --experiments-root "$REPO_ROOT/experiments" \
    --run-names majority_baseline cnn_basic cnn_balanced cnn_augmented cnn_augmented_multicrop \
    --output-markdown "$REPO_ROOT/experiments/phase6_summary.md" \
    --output-csv "$REPO_ROOT/experiments/phase6_summary.csv"
else
  CONFIG="$ARG"
  if [[ "$CONFIG" != /* ]]; then
    CONFIG="$REPO_ROOT/$CONFIG"
  fi

  echo "=== Running experiment config: $CONFIG ==="
  "$PYTHON" -m Sandbox.singerclassifier.scripts.run_experiments --configs "$CONFIG"
fi

echo "=== Experiment job complete ==="
