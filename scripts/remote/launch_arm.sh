#!/usr/bin/env bash
# One arm end to end on the pod: GRPO training, then midpoint/final math and StrongREJECT generation,
# chained with && so nothing downstream waits on an agent. Run inside tmux (see docs/HANDOFF.md):
#   tmux new -d -s es "bash /workspace/rlvr-crosslingual-safety/scripts/remote/launch_arm.sh es 2>&1 | tee /workspace/logs/es_seed0.log"
# Identical to the chain that ran on 2026-09-16 (as /workspace/logs/launch_es.sh, ARM=es); parameterized for the English control.
set -euo pipefail
ARM=${1:-es}                      # es | en (the English arm trains on the same problem IDs, pool_es_7b.json)
OUT=/workspace/adapters/${ARM}_seed0
source /workspace/rlvr-crosslingual-safety/scripts/remote/env.sh
cd "$PROJECT"
mkdir -p /workspace/logs /workspace/outputs
export PYTHONUNBUFFERED=1
uv run python scripts/train_grpo.py --lang "$ARM" --pool data/processed/pool_es_7b.json --model 7b \
  --steps 250 --save-steps 25 --prompts-per-step 16 --num-generations 8 --micro-batch 4 \
  --max-completion 2048 --vllm-gpu-mem 0.30 --report-to none --seed 0 --out "$OUT" \
&& uv run python scripts/eval_math.py   --lang "$ARM" --model 7b --k 8 --adapter "$OUT/checkpoint-125" --tag "${ARM}_mid"   --out-dir /workspace/outputs \
&& uv run python scripts/eval_math.py   --lang "$ARM" --model 7b --k 8 --adapter "$OUT/final"          --tag "${ARM}_final" --out-dir /workspace/outputs \
&& uv run python scripts/eval_safety.py --model 7b --n 3 --adapter "$OUT/checkpoint-125" --tag "${ARM}_mid"   --out-dir /workspace/outputs \
&& uv run python scripts/eval_safety.py --model 7b --n 3 --adapter "$OUT/final"          --tag "${ARM}_final" --out-dir /workspace/outputs \
&& echo ARM_DONE > "/workspace/logs/${ARM}_seed0.DONE"
