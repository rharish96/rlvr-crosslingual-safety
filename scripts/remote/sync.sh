#!/usr/bin/env bash
# Local -> pod code sync (run from the Mac). Excludes venvs, caches, raw data and git internals.
# Usage: scripts/remote/sync.sh [host]      (default host alias: runpod)
set -euo pipefail
HOST=${1:-runpod}
HERE=$(cd "$(dirname "$0")/../.." && pwd)
ssh "$HOST" 'mkdir -p /workspace/rlvr-crosslingual-safety'
rsync -rlptz --no-o --no-g --delete \
  --exclude '.venv' --exclude '.git' --exclude '__pycache__' --exclude '.pytest_cache' --exclude '.ruff_cache' \
  --exclude 'data/raw' --exclude 'outputs' --exclude 'checkpoints' --exclude 'adapters' --exclude 'generations' \
  --exclude 'trackio' --exclude '*.db' \
  "$HERE/" "$HOST:/workspace/rlvr-crosslingual-safety/"
echo "synced to $HOST:/workspace/rlvr-crosslingual-safety"
