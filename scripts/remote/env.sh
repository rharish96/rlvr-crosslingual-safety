# Source this on the GPU box. Everything lives on the network volume (/workspace) so it survives pod stops.
export WS=/workspace
export PROJECT=$WS/rlvr-crosslingual-safety

# uv + Python + package cache on the volume (container disk is wiped on stop)
export UV_INSTALL_DIR=$WS/bin
export UV_PYTHON_INSTALL_DIR=$WS/uv/python
export UV_CACHE_DIR=$WS/uv/cache
export PATH=$WS/bin:$PATH

# Hugging Face cache and token (token file copied with scp; never echo it)
export HF_HOME=$WS/hf
export HF_HUB_ENABLE_HF_TRANSFER=0
if [ -f "$WS/.hf_token" ]; then
  export HF_TOKEN="$(cat "$WS/.hf_token")"
fi

# vLLM / torch niceties
export VLLM_LOGGING_LEVEL=WARNING
export TOKENIZERS_PARALLELISM=false

cd "$PROJECT" 2>/dev/null || true
