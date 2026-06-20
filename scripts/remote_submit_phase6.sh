#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="mem-ans1.transchip.com"
WORK_REPO_ROOT="/home/maork/Projects/rad_sandbox/Sandbox/singerclassifier"
ARTIFACT_ROOT="/home/maork/Projects/rad_sandbox/Sandbox/data/singerclassifier"
CONCURRENCY="${1:-1}"
MANIFEST="${ARTIFACT_ROOT}/manifests/phase6_sweep_manifest.csv"
SPLIT_CSV="${ARTIFACT_ROOT}/processed/damp_sag_splits.csv"
AUDIO_CACHE_DIR="${ARTIFACT_ROOT}/cache/audio_22050_mono"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -d "$WORK_REPO_ROOT" ]]; then
  REPO_ROOT="$WORK_REPO_ROOT"
else
  REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi

mkdir -p "${REPO_ROOT}/slurm/logs"

REMOTE_CMD="cd ${WORK_REPO_ROOT} && bash slurm/submit_phase6_sweep.sh ${CONCURRENCY}"

echo "Phase 6 remote SLURM submission"
echo "================================="
echo "Local repo path:  ${REPO_ROOT}"
echo "Artifact root:    ${ARTIFACT_ROOT}"
echo "Remote host:      ${REMOTE_HOST}"
echo "Concurrency:      ${CONCURRENCY}"
echo "Split CSV:        ${SPLIT_CSV}"
echo "Audio cache:      ${AUDIO_CACHE_DIR}"
echo "Manifest:         ${MANIFEST}"
echo
echo "Note: submission validates the audio cache but does not build it."
echo "Precompute cache first (one-time or after split changes):"
echo "  bash scripts/precompute_phase6_cache.sh"
echo
echo "Remote command:   ssh -o BatchMode=yes -o ConnectTimeout=20 ${REMOTE_HOST} \"${REMOTE_CMD}\""
echo

if ! command -v ssh >/dev/null 2>&1; then
  echo "ERROR: ssh not found. Install OpenSSH client and retry." >&2
  exit 1
fi

set +e
REMOTE_OUTPUT="$(ssh -o BatchMode=yes -o ConnectTimeout=20 "${REMOTE_HOST}" "${REMOTE_CMD}" 2>&1)"
REMOTE_STATUS=$?
set -e

echo "$REMOTE_OUTPUT"

if [[ "$REMOTE_STATUS" -ne 0 ]]; then
  echo
  echo "ERROR: Remote submission failed (exit ${REMOTE_STATUS})." >&2
  if echo "$REMOTE_OUTPUT" | grep -qi "Audio cache is missing or incomplete"; then
    echo "Build the audio cache first:" >&2
    echo "  cd ${REPO_ROOT}" >&2
    echo "  bash scripts/precompute_phase6_cache.sh" >&2
  elif echo "$REMOTE_OUTPUT" | grep -qi "sbatch not found"; then
    echo "sbatch must run on the SLURM submission node (${REMOTE_HOST})." >&2
    echo "Use this script from the work Linux machine; do not run slurm/submit_phase6_sweep.sh locally." >&2
  else
    echo "Check SSH access to ${REMOTE_HOST} and that ${WORK_REPO_ROOT} exists on the shared filesystem." >&2
  fi
  exit "$REMOTE_STATUS"
fi

JOB_ID=""
if echo "$REMOTE_OUTPUT" | grep -q "Submitted batch job"; then
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
echo "Summarize:        bash scripts/remote_summarize_phase6.sh (after runs complete)"
echo "Log directory:    ${REPO_ROOT}/slurm/logs/"
