#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="mem-ans1.transchip.com"
WORK_REPO_ROOT="/home/maork/Projects/rad_sandbox/Sandbox/singerclassifier"
ARRAY_SPEC="${1:-9-14}"
CONCURRENCY="${2:-1}"

REMOTE_CMD="cd ${WORK_REPO_ROOT} && bash scripts/rerun_phase6_invalid.sh ${ARRAY_SPEC} ${CONCURRENCY}"

echo "Phase 6 invalid-run remote rerun"
echo "================================="
echo "Remote host:   ${REMOTE_HOST}"
echo "Array spec:    ${ARRAY_SPEC}"
echo "Concurrency:   ${CONCURRENCY}"
echo

set +e
REMOTE_OUTPUT="$(ssh -o BatchMode=yes -o ConnectTimeout=20 "${REMOTE_HOST}" "${REMOTE_CMD}" 2>&1)"
REMOTE_STATUS=$?
set -e

echo "$REMOTE_OUTPUT"

if [[ "$REMOTE_STATUS" -ne 0 ]]; then
  echo "ERROR: Remote rerun submission failed (exit ${REMOTE_STATUS})." >&2
  exit "$REMOTE_STATUS"
fi
