#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=/home/maork/Projects/rad_sandbox/Sandbox/singerclassifier

cd "$REPO_ROOT"

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi

# shellcheck source=/dev/null
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python -m py_compile src/dummy_gpu_train.py scripts/smoke_gpu.py
python -c "import torch, yaml, tqdm, numpy; print('imports ok')"

echo "Virtual environment ready at: $REPO_ROOT/.venv"
