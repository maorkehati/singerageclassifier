#!/usr/bin/env bash
set -euo pipefail

RAD_ROOT="/home/maork/Projects/rad_sandbox"
WORK_REPO_ROOT="/home/maork/Projects/rad_sandbox/Sandbox/singerclassifier"
PYTHON="${RAD_ROOT}/Sandbox/SSL_Tabular/.venv/bin/python"
MANIFEST="${WORK_REPO_ROOT}/experiments/manifests/phase6_sweep_manifest.csv"
EXPERIMENTS_ROOT="${WORK_REPO_ROOT}/experiments"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -d "$WORK_REPO_ROOT" ]]; then
  REPO_ROOT="$WORK_REPO_ROOT"
else
  REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
  MANIFEST="${REPO_ROOT}/experiments/manifests/phase6_sweep_manifest.csv"
  EXPERIMENTS_ROOT="${REPO_ROOT}/experiments"
fi

ALL_MD="${EXPERIMENTS_ROOT}/phase6_all_runs_summary.md"
ALL_CSV="${EXPERIMENTS_ROOT}/phase6_all_runs_summary.csv"
BEST_MD="${EXPERIMENTS_ROOT}/phase6_best_by_family.md"
BEST_CSV="${EXPERIMENTS_ROOT}/phase6_best_by_family.csv"

echo "Phase 6 sweep summarizer"
echo "========================"
echo "Python:   ${PYTHON}"
echo "Manifest: ${MANIFEST}"
echo

if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: Python not found: ${PYTHON}" >&2
  exit 1
fi

if [[ ! -f "$MANIFEST" ]]; then
  echo "ERROR: Manifest not found: ${MANIFEST}" >&2
  echo "Generate configs first:" >&2
  echo "  ${PYTHON} -m Sandbox.singerclassifier.scripts.generate_sweep_configs \\" >&2
  echo "    --sweep-spec ${REPO_ROOT}/configs/phase6_sweeps.yaml" >&2
  exit 1
fi

cd "$RAD_ROOT"
export PYTHONPATH="${RAD_ROOT}:${PYTHONPATH:-}"

"$PYTHON" -m Sandbox.singerclassifier.scripts.summarize_sweep \
  --manifest "$MANIFEST"

echo
echo "Summary outputs"
echo "================="
echo "All runs (markdown): ${ALL_MD}"
echo "All runs (csv):      ${ALL_CSV}"
echo "Best by family (md): ${BEST_MD}"
echo "Best by family (csv):${BEST_CSV}"
