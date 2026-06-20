#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=/home/maork/Projects/rad_sandbox/Sandbox/singerclassifier
RAD_ROOT=/home/maork/Projects/rad_sandbox
PYTHON=/home/maork/Projects/rad_sandbox/Sandbox/SSL_Tabular/.venv/bin/python
SWEEP_SPEC="$REPO_ROOT/configs/phase6_sweeps.yaml"
MANIFEST="$REPO_ROOT/experiments/manifests/phase6_sweep_manifest.csv"
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

if [[ ! -f "$MANIFEST" ]]; then
  echo "Manifest not found. Generating sweep configs..."
  "$PYTHON" -m Sandbox.singerclassifier.scripts.generate_sweep_configs \
    --sweep-spec "$SWEEP_SPEC"
fi

N=$(($(wc -l < "$MANIFEST") - 1))
if [[ "$N" -le 0 ]]; then
  echo "ERROR: Manifest has no runs: $MANIFEST" >&2
  exit 1
fi

MAX_INDEX=$((N - 1))
echo "Submitting Phase 6 sweep array: 0-${MAX_INDEX} with concurrency %${CONCURRENCY}"
echo "Manifest: $MANIFEST"
echo "Runs: $N"

cd "$REPO_ROOT"
mkdir -p slurm/logs

SBATCH_OUTPUT="$(sbatch --array="0-${MAX_INDEX}%${CONCURRENCY}" slurm/phase6_sweep_array.sbatch)"
echo "$SBATCH_OUTPUT"

JOB_ID="$(echo "$SBATCH_OUTPUT" | awk '{print $NF}')"
echo "Submitted job id: ${JOB_ID}"
echo "Monitor with: squeue -u \$USER"
echo "From work Linux: bash scripts/remote_status.sh"
