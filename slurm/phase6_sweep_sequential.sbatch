#!/usr/bin/env bash
#SBATCH --job-name=phase6_sweep_seq
#SBATCH --partition=base
#SBATCH --nodelist=gpu1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/home/maork/Projects/rad_sandbox/Sandbox/singerclassifier/slurm/logs/phase6_seq_%j.out
#SBATCH --error=/home/maork/Projects/rad_sandbox/Sandbox/singerclassifier/slurm/logs/phase6_seq_%j.err

set -euo pipefail

RAD_ROOT=/home/maork/Projects/rad_sandbox
REPO_ROOT=/home/maork/Projects/rad_sandbox/Sandbox/singerclassifier
PYTHON=/home/maork/Projects/rad_sandbox/Sandbox/SSL_Tabular/.venv/bin/python
MANIFEST="$REPO_ROOT/experiments/manifests/phase6_sweep_manifest.csv"
SWEEP_SPEC="$REPO_ROOT/configs/phase6_sweeps.yaml"

cd "$RAD_ROOT"
export PYTHONPATH="$RAD_ROOT:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/usr/local/cuda/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"

echo "=== Phase 6 sequential sweep ==="
hostname
date
nvidia-smi || true

mkdir -p "$REPO_ROOT/slurm/logs" "$REPO_ROOT/experiments/manifests"

if [[ ! -f "$MANIFEST" ]]; then
  "$PYTHON" -m Sandbox.singerclassifier.scripts.generate_sweep_configs \
    --sweep-spec "$SWEEP_SPEC"
fi

"$PYTHON" -m Sandbox.singerclassifier.scripts.run_sweep_manifest \
  --manifest "$MANIFEST" \
  --skip-existing \
  --continue-on-error

"$PYTHON" -m Sandbox.singerclassifier.scripts.summarize_sweep \
  --manifest "$MANIFEST"

echo "=== Sequential sweep complete ==="
