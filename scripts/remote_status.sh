#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="mem-ans1.transchip.com"
SLURM_USER="maork"

echo "SLURM queue for ${SLURM_USER} on ${REMOTE_HOST}"
echo "================================================"

if ! command -v ssh >/dev/null 2>&1; then
  echo "ERROR: ssh not found. Install OpenSSH client and retry." >&2
  exit 1
fi

set +e
REMOTE_OUTPUT="$(ssh "${REMOTE_HOST}" "squeue -u ${SLURM_USER}" 2>&1)"
REMOTE_STATUS=$?
set -e

echo "$REMOTE_OUTPUT"

if [[ "$REMOTE_STATUS" -ne 0 ]]; then
  echo
  echo "ERROR: Remote queue check failed (exit ${REMOTE_STATUS})." >&2
  echo "Check SSH access to ${REMOTE_HOST}." >&2
  exit "$REMOTE_STATUS"
fi
