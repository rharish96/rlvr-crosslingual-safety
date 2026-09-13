#!/usr/bin/env bash
# One-shot (idempotent) setup of a fresh Runpod container against the persistent /workspace volume.
# Run as: bash /workspace/rlvr-crosslingual-safety/scripts/remote/bootstrap.sh
set -euo pipefail

WS=/workspace
PROJECT=$WS/rlvr-crosslingual-safety
ENVSH=$PROJECT/scripts/remote/env.sh

# 1. shell env for every future login
grep -q 'scripts/remote/env.sh' ~/.bashrc 2>/dev/null || echo "source $ENVSH" >> ~/.bashrc
# shellcheck disable=SC1090
source "$ENVSH"

# 2. small system tools (container disk; cheap to redo after a restart)
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq >/dev/null 2>&1 || true
apt-get install -y -qq tmux rsync htop >/dev/null 2>&1 || true

# 3. uv on the volume
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$UV_INSTALL_DIR" UV_NO_MODIFY_PATH=1 sh >/dev/null
fi
uv --version

# 4. project env: Python 3.12 + locked deps incl. gpu extra (vllm, trl)
cd "$PROJECT"
uv python install 3.12 >/dev/null
uv sync --frozen --extra gpu --group dev

# 5. sanity
uv run python - <<'EOF'
import torch, vllm, trl, transformers, peft, math_verify
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0); print("gpu", p.name, f"{p.total_memory/2**30:.0f} GiB")
print("vllm", vllm.__version__, "| trl", trl.__version__, "| transformers", transformers.__version__, "| peft", peft.__version__)
EOF
echo "bootstrap OK"
