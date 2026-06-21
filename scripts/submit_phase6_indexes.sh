#!/usr/bin/env bash
# Submit selected Phase 6 manifest indexes via SLURM (run on mem-ans1).
set -euo pipefail

RAD_ROOT=/home/maork/Projects/rad_sandbox
REPO_ROOT=/home/maork/Projects/rad_sandbox/Sandbox/singerclassifier
PYTHON=/home/maork/Projects/rad_sandbox/Sandbox/SSL_Tabular/.venv/bin/python
MANIFEST=/home/maork/Projects/rad_sandbox/Sandbox/data/singerclassifier/manifests/phase6_sweep_manifest.csv

usage() {
  echo "Usage: $0 <index_list> [concurrency]" >&2
  echo "Example: $0 9,10,11,12,13,14 1" >&2
  exit 1
}

if [[ $# -lt 1 ]] || [[ -z "${1:-}" ]]; then
  usage
fi

INDEXES="$1"
CONCURRENCY="${2:-1}"

if ! [[ "$INDEXES" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo "ERROR: Invalid index list (expected comma-separated integers): $INDEXES" >&2
  usage
fi

if ! [[ "$CONCURRENCY" =~ ^[0-9]+$ ]] || [[ "$CONCURRENCY" -lt 1 ]]; then
  echo "ERROR: Invalid concurrency (expected positive integer): $CONCURRENCY" >&2
  usage
fi

if ! command -v sbatch >/dev/null 2>&1; then
  echo "ERROR: sbatch not found. Run on the SLURM submission node (mem-ans1)." >&2
  exit 1
fi

if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: Python not found: $PYTHON" >&2
  exit 1
fi

if [[ ! -f "$MANIFEST" ]]; then
  echo "ERROR: Manifest not found: $MANIFEST" >&2
  exit 1
fi

cd "$REPO_ROOT"
mkdir -p slurm/logs

echo "Phase 6 selected-index submission"
echo "=================================="
echo "Selected indexes: $INDEXES"
echo "Concurrency:      %${CONCURRENCY}"
echo "Manifest:         $MANIFEST"
echo "Repo root:        $REPO_ROOT"
echo "Project root:     $RAD_ROOT"
echo "Python:           $PYTHON"
echo "Skip existing:    no (forced rerun)"

export PHASE6_FORCE_RERUN=1

SBATCH_OUTPUT="$(sbatch --array="${INDEXES}%${CONCURRENCY}" slurm/phase6_sweep_array.sbatch)"
echo "$SBATCH_OUTPUT"

JOB_ID="$(echo "$SBATCH_OUTPUT" | awk '{print $NF}')"
echo "Submitted job id: ${JOB_ID}"
echo "Monitor: bash scripts/remote_status.sh"
