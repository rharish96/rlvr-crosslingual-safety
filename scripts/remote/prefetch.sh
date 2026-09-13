#!/usr/bin/env bash
# Pre-download models and datasets into $HF_HOME on the volume so training runs never wait on downloads.
set -euo pipefail
source /workspace/rlvr-crosslingual-safety/scripts/remote/env.sh
cd "$PROJECT"

for m in Qwen/Qwen2.5-3B-Instruct Qwen/Qwen2.5-7B-Instruct google/gemma-2b qylu4156/strongreject-15k-v1; do
  echo "== $m"
  uv run hf download "$m" --exclude "*.gguf" --exclude "original/*" --quiet >/dev/null
done

# datasets: pinned JSONL + AceReason (for English reconstruction) + StrongREJECT CSV, via our loaders
uv run python - <<'EOF'
from rlvr_crosslingual_safety.data import load_macereason
from rlvr_crosslingual_safety.safety_eval import load_strongreject
es = load_macereason("es"); en = load_macereason("en"); sr = load_strongreject()
print("es train/test", len(es["train"]), len(es["test"]), "| en", len(en["train"]), len(en["test"]), "| strongreject", len(sr))
EOF
du -sh "$HF_HOME" "$PROJECT/data/raw" 2>/dev/null
echo "prefetch OK"
