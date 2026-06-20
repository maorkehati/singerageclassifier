#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="mem-ans1.transchip.com"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <job_id>" >&2
  echo "Example: bash scripts/remote_cancel_phase6.sh 12345" >&2
  exit 1
fi

JOB_ID="$1"

if ! [[ "$JOB_ID" =~ ^[0-9]+$ ]]; then
  echo "ERROR: job_id must be a numeric SLURM job id, got: ${JOB_ID}" >&2
  exit 1
fi

echo "Cancelling SLURM job ${JOB_ID} on ${REMOTE_HOST}"

if ! command -v ssh >/dev/null 2>&1; then
  echo "ERROR: ssh not found. Install OpenSSH client and retry." >&2
  exit 1
fi

set +e
REMOTE_OUTPUT="$(ssh "${REMOTE_HOST}" "scancel ${JOB_ID}" 2>&1)"
REMOTE_STATUS=$?
set -e

if [[ -n "$REMOTE_OUTPUT" ]]; then
  echo "$REMOTE_OUTPUT"
fi

if [[ "$REMOTE_STATUS" -ne 0 ]]; then
  echo "ERROR: scancel failed (exit ${REMOTE_STATUS})." >&2
  exit "$REMOTE_STATUS"
fi

echo "Cancel request sent for job ${JOB_ID}."
