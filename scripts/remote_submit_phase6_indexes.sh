#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="mem-ans1.transchip.com"
REPO_ROOT="/home/maork/Projects/rad_sandbox/Sandbox/singerclassifier"
MANIFEST="/home/maork/Projects/rad_sandbox/Sandbox/data/singerclassifier/manifests/phase6_sweep_manifest.csv"

usage() {
  echo "Usage: $0 <index_list> [concurrency]" >&2
  echo "Example: $0 9,10,11,12,13,14 1" >&2
  echo >&2
  echo "Delete invalid run folders before rerunning, e.g.:" >&2
  echo "  rm -rf ${REPO_ROOT}/experiments/cnn_augmented_*" >&2
  echo "  rm -rf ${REPO_ROOT}/experiments/cnn_augmented_multicrop_*" >&2
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

if ! command -v ssh >/dev/null 2>&1; then
  echo "ERROR: ssh not found. Install OpenSSH client and retry." >&2
  exit 1
fi

REMOTE_CMD="cd ${REPO_ROOT} && bash scripts/submit_phase6_indexes.sh ${INDEXES} ${CONCURRENCY}"

echo "Phase 6 selected-index remote submission"
echo "========================================"
echo "Remote host:      ${REMOTE_HOST}"
echo "Selected indexes: ${INDEXES}"
echo "Concurrency:      ${CONCURRENCY}"
echo "Manifest:         ${MANIFEST}"
echo "Repo root:        ${REPO_ROOT}"
echo
echo "Remote command:"
echo "  ssh -o BatchMode=yes -o ConnectTimeout=20 ${REMOTE_HOST} \"${REMOTE_CMD}\""
echo

set +e
REMOTE_OUTPUT="$(ssh -o BatchMode=yes -o ConnectTimeout=20 "${REMOTE_HOST}" "${REMOTE_CMD}" 2>&1)"
REMOTE_STATUS=$?
set -e

echo "$REMOTE_OUTPUT"

if [[ "$REMOTE_STATUS" -ne 0 ]]; then
  echo "ERROR: Remote submission failed (exit ${REMOTE_STATUS})." >&2
  exit "$REMOTE_STATUS"
fi

JOB_ID=""
if echo "$REMOTE_OUTPUT" | grep -q "Submitted job id:"; then
  JOB_ID="$(echo "$REMOTE_OUTPUT" | grep "Submitted job id:" | tail -1 | awk '{print $NF}')"
elif echo "$REMOTE_OUTPUT" | grep -q "Submitted batch job"; then
  JOB_ID="$(echo "$REMOTE_OUTPUT" | grep "Submitted batch job" | tail -1 | awk '{print $NF}')"
fi

echo
echo "Next steps"
echo "=========="
if [[ -n "$JOB_ID" ]]; then
  echo "Submitted job id: ${JOB_ID}"
  echo "Cancel:           bash scripts/remote_cancel_phase6.sh ${JOB_ID}"
fi
echo "Check queue:      bash scripts/remote_status.sh"
echo "Tail logs:        bash scripts/remote_tail_logs.sh 100"
