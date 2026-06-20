#!/usr/bin/env bash
set -euo pipefail

WORK_REPO_ROOT="/home/maork/Projects/rad_sandbox/Sandbox/singerclassifier"
N="${1:-100}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -d "$WORK_REPO_ROOT" ]]; then
  REPO_ROOT="$WORK_REPO_ROOT"
else
  REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi

LOG_DIR="${REPO_ROOT}/slurm/logs"

echo "Phase 6 SLURM logs (last ${N} lines)"
echo "Log directory: ${LOG_DIR}"
echo

shopt -s nullglob
OUT_FILES=( "${LOG_DIR}"/phase6_*.out )
ERR_FILES=( "${LOG_DIR}"/phase6_*.err )

if [[ ${#OUT_FILES[@]} -eq 0 && ${#ERR_FILES[@]} -eq 0 ]]; then
  echo "No Phase 6 logs found yet."
  echo "Logs appear after a sweep is submitted and array tasks start running."
  exit 0
fi

if [[ ${#OUT_FILES[@]} -gt 0 ]]; then
  echo "=== stdout (${#OUT_FILES[@]} file(s)) ==="
  tail -n "$N" "${OUT_FILES[@]}"
  echo
fi

if [[ ${#ERR_FILES[@]} -gt 0 ]]; then
  echo "=== stderr (${#ERR_FILES[@]} file(s)) ==="
  tail -n "$N" "${ERR_FILES[@]}"
fi
