#!/usr/bin/env bash
# Pod -> Mac results sync (run from the Mac). Pulls /workspace/outputs into outputs/ and the training
# log histories into outputs/adapters_meta/. Never pushes outputs upward; code goes up via sync.sh.
# Usage: scripts/remote/pull.sh [host]      (default host alias: runpod)
set -euo pipefail
HOST=${1:-runpod}
HERE=$(cd "$(dirname "$0")/../.." && pwd)
mkdir -p "$HERE/outputs" "$HERE/outputs/adapters_meta"
rsync -rltz --no-o --no-g \
  --exclude '*.safetensors' --exclude '*.bin' --exclude '*.pt' \
  "$HOST:/workspace/outputs/" "$HERE/outputs/"
# small per-run metadata: log_history.json, run_meta.json, trainer_state.json of each checkpoint
rsync -rltz --no-o --no-g --prune-empty-dirs \
  --include '*/' --include 'log_history.json' --include 'run_meta.json' --include 'trainer_state.json' --exclude '*' \
  "$HOST:/workspace/adapters/" "$HERE/outputs/adapters_meta/"
echo "pulled $HOST:/workspace/outputs -> $HERE/outputs (and adapter metadata -> outputs/adapters_meta)"
