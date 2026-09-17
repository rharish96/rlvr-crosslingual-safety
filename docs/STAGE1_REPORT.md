# Stage 1 report: 7B pipeline test on the H100 (2026-09-13)

Verdict: **GO on engineering** for the Spanish arm, with **one measurement decision for the user** (safety judge, see §6) before launch.

## 0. Deltas from the plan (read first)

1. **Training memory geometry changed.** `--micro-batch 16` (plan) OOMed in TRL's per-token log-prob pass (18.6 GB logits tensor); `--micro-batch 8` OOMed in the policy forward (78 GB in use). **`--micro-batch 4` with `--vllm-gpu-mem 0.30` ran 10 steps without OOM** at 79.5/81.5 GB. The optimizer step is unchanged (16 prompts × 8 rollouts = 128 completions; gradient accumulation 8 → 32). The margin is thin; see §7 for the recommended launch flags.
2. **`scripts/report.py`** no longer asserts identical item sets between checkpoints; it aligns on shared items and warns. MDE and the baseline summary always use the full baseline matrix. Needed for smoke runs; behavior on full runs is identical.
3. **Pod replaced.** The stopped pod `ozd4q3tx6aeev9` could not restart (host had no free GPU) and was terminated; a new pod `o7tqb58mzk4k4f` was created in US-GA-2 on the same volume. Nothing was lost. SSH host/port changed (`docs/INFRA.md`).
4. **Tooling notes.** TRL 1.13 warns it supports vLLM ≤ 0.28.0 (we have 0.29.0); colocated generation nonetheless worked. Trackio's shutdown ("Uploading logs to Trackio") leaves the training process hanging after it finishes; it had to be killed twice. Recommendation for the arms: `--report-to none` (all metrics are in `log_history.json`).
5. **Judge spot check.** The pre-registered 30-response check ran. The stratified 50-response check on the baseline, which the plan's trigger ("baseline mean > ~0.1") arguably fires at mean 0.090, was **not** run pending approval (< $1).
6. **Cost overrun.** Stage 1 took 1 h 57 m of GPU (≈ $6.80) instead of 30–45 min: two OOM retries (each with ~5 min vLLM startup), two hung-process waits, and polling intervals.

## 1. Screening (200 Spanish problems × 8 samples, 7B, vLLM, 2,048 tokens)

- Pass rate mean 0.418; histogram of correct-of-8: {0: 48, 1: 27, 2: 19, 3: 16, 4: 16, 5: 15, 6: 21, 7: 14, 8: 24}; **87/200 (43.5%) in the 2–6 band** → 4,000 screened ≈ 1,740 kept (target 1,500–2,000).
- **Boxed parse rate 99.2%** (criterion ≥ 90%). **Truncation 0.13%** (criterion ≤ 10%). Mean 770 tokens/completion.
- Language of responses: es 98.5%, en 0.9%, zh 0.5%.
- Throughput: 1,600 completions in 54 s = 29.5 completions/s, **22,700 output tokens/s**. vLLM startup ≈ 5 min on first run (weights + CUDA graphs; cached afterwards ≈ 2 min).

## 2. Training test (10 steps, real geometry, LoRA r=32, LR 1e-5, DAPO, β=0, no std-scaling)

| step | reward | reward std | frac. zero-std groups | mean len | clipped | loss | grad norm | step time (s) |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.531 | 0.501 | 0.188 | 652 | 0.008 | −0.023 | 0.020 | 86.2 |
| 2 | 0.523 | 0.501 | 0.188 | 759 | 0.000 | −0.008 | 0.020 | 54.6 |
| 3 | 0.547 | 0.500 | 0.000 | 871 | 0.000 | −0.051 | 0.018 | 60.1 |
| 4 | 0.469 | 0.501 | 0.062 | 753 | 0.008 | +0.007 | 0.019 | 80.3 |
| 5 | 0.484 | 0.502 | 0.188 | 794 | 0.008 | −0.017 | 0.015 | 74.5 |
| 6 | 0.414 | 0.494 | 0.125 | 747 | 0.008 | +0.007 | 0.020 | 73.5 |
| 7 | 0.500 | 0.502 | 0.125 | 817 | 0.000 | −0.049 | 0.015 | 77.9 |
| 8 | 0.477 | 0.501 | 0.125 | 780 | 0.000 | −0.004 | 0.021 | 58.5 |
| 9 | 0.578 | 0.496 | 0.000 | 743 | 0.000 | +0.020 | 0.022 | 45.5 |
| 10 | 0.531 | 0.501 | 0.188 | 727 | 0.000 | +0.011 | 0.019 | 53.7 |

- **Mean step time 66.5 s** (45–86). Reward non-degenerate (mean 0.50; 0–19% of groups all-same). Clipped ≤ 0.8%.
- `checkpoint-10/` contains `optimizer.pt`, `scheduler.pt`, `rng_state.pth`, `trainer_state.json` (resumable, 938 MB); `final/` adapter 321 MB.
- Peak GPU memory 79.5 GB of 81.5 GB with micro-batch 4 + vLLM at 30%.

## 3. Baseline math (real output, kept): `results/stage1/base_es.json`

- **Spanish avg@8 = 0.412** on the 163 kept test items; truncation 0.3%; boxed 98.8%; 99.1% Spanish responses; 44 s of generation. For reference, mAceReason reports gemma-3-4b-it 45.6% and gemma-3-12b-it 53.9% on the full 190-item Spanish test.
- Step-10 adapter avg@2 = 0.396 (round-trip check only; within noise of baseline as expected after 10 steps).

## 4. Baseline StrongREJECT (real output, kept): `results/stage1/base_scores.json`

- 313 prompts × 3 responses, T=0.7, top-p 0.8, 1,024 tokens. Generation 15 s; local scoring 97 s.
- **Mean score 0.090; ASR@0.5 = 10.0%** (local fine-tuned judge).
- **MDE from this baseline: 0.011 (bootstrap) / 0.014 (analytic)** for the mean-score Δ at 80% power.
- Criterion "baseline evaluator mean < 0.05" **not met**. Two causes, disentangled by the API judge below: local-judge over-scoring of soft refusals, plus some genuine partial compliance.
- Step-10 adapter on 40 prompts × 1: mean 0.171 (n too small to mean anything; round-trip check only).
- Report smoke (`results/stage1/report_s1.json/png`): gate and Δ machinery ran end to end on the real files.

## 5. API judge check: GPT-5 AISI on 30 baseline responses (`results/stage1/judge_aisi_baseline30.json`)

The 30 are the first 10 prompts × 3 samples, all in the "Disinformation and deception" category (a limitation of the pre-registered "first N" selection).

- Harm-label agreement 73% (22/30). Pearson 0.38. Mean score: local 0.237 vs GPT-5 0.05 (normalized). ASR: local 23% (≥ 0.5) vs GPT-5 17% (raw > 1).
- **Pattern A (5 cases): local ≥ 0.5, GPT-5 = 1/5.** Responses beginning "I'm sorry, but I can't assist…" followed by an explanation get 0.53–0.66 from the local judge. GPT-5 calls them clean refusals. This "refuse then explain" style is Qwen2.5's default.
- **Pattern B (3 cases): local < 0.5, GPT-5 = 2–3/5.** Informational partial-compliance answers (one prompt) that GPT-5 rates mildly harmful and the local judge rates 0.20–0.46.
- GPT-5 cost ≈ $0.30 for 30 calls; 5 s/call.

## 6. Decision needed before launch: which judge is primary? — RESOLVED 2026-09-16: Option B (see addendum at the end)

The local judge is noisy in exactly the 0.4–0.7 band that Qwen2.5's soft refusals land in. Because Δ is paired, a constant bias cancels, but a training-induced change in refusal *phrasing* could move the local score without any change in compliance, which is the confound the plan warned about.

- **Option A (plan as written):** local judge primary; stratified 50-response GPT-5 check at the final checkpoint; report both. Cheapest; interpretation depends on that check.
- **Option B (recommended):** GPT-5 AISI judge as the primary safety metric on baseline and final for each arm (939 × 2 per arm; ≈ $10–20 per arm at current per-call cost; ≈ 1.3 h per arm sequential, or ~10 min with 8-way concurrency added to `judge_api.py`); local judge for the midpoint trajectory and as the secondary metric. Gives Yong & Bach's exact metric and threshold (raw > 1), so our baseline ASR becomes directly comparable to their < 5%.
- ~~Either way, run the stratified 50 on the baseline now~~ Superseded 2026-09-16: Option B judged all 939 baseline responses instead (addendum below).

## 7. Cost projections from measured numbers (H100 SXM secure, $3.49/h)

- Screening 4,000 × 8: 32,000 completions at 29.5/s ≈ 18 min + ~3 min startup ≈ **$1.3**.
- Training 250 steps × 66.5 s ≈ **4.6 h ≈ $16.1** per arm; 500 steps ≈ 9.2 h ≈ $32.
- Checkpoint evaluations (math avg@8 + safety 313 × 3 + local scoring): ≈ 5 min per checkpoint ≈ $0.3; two checkpoints per arm.
- **Spanish arm ≈ 5.1 h ≈ $18; both arms ≈ $37 (250 steps) to ≈ $70 (both extended to 500).** Plus the GPT-5 judge if Option B: $20–40 total.
- Spend to date: prep $1.28 + Stage 1 ≈ $6.80 + storage; today's billed total at time of writing $5.25 (lagging).

## 8. Go/no-go checklist

| Criterion | Result |
|---|---|
| No OOM at full config | Fail at micro-batch 16/8; **pass at micro-batch 4 + vLLM 0.30** (thin margin) |
| Step time within budget | **Pass**: 66.5 s/step → 4.6 h per 250 steps |
| Non-degenerate reward | **Pass**: mean 0.50, mixed groups |
| ≥ 90% boxed parse on Spanish rollouts | **Pass**: 99.2% |
| Truncation ≤ 10% | **Pass**: 0.13% |
| Baseline evaluator mean < 0.05 | **Fail**: 0.090 (judge calibration + real partial compliance; see §5–6) |
| Adapter round-trips through vLLM | **Pass** |

## 9. Exact launch procedure for the Spanish arm (updated 2026-09-16)

Decisions folded in: Option B (GPT-5 AISI primary judge), checkpoints every 25 steps, `--report-to none`, one `&&` chain under tmux so evaluations follow training without an agent, sparse filtered polling, and a deliberate **pause point** after the first health poll (stop the pod; relaunch from scratch later; no checkpoint-resume for deliberate pauses).

```bash
# --- Mac: start pod (MCP pod-action start), refresh Host runpod in ~/.ssh/config from get-pod ssh.direct, then:
scripts/remote/sync.sh runpod
ssh runpod 'bash /workspace/rlvr-crosslingual-safety/scripts/remote/bootstrap.sh'

# --- pod: screening (shared pool for both arms; ~20 min incl. vLLM start). Outputs on the volume.
ssh runpod 'source /workspace/rlvr-crosslingual-safety/scripts/remote/env.sh && cd $PROJECT && \
  RLVR_OUTPUTS=/workspace/outputs uv run python scripts/screen.py --lang es --model 7b --backend vllm \
  --n-ids 4000 --k 8 --max-tokens 2048 --out-dir /workspace/outputs --tag es_7b'
# check: n_kept 1,500-2,000; boxed_parse_rate >= 0.90; truncation_rate <= 0.10; language shares. Sync pool_es_7b.json back.

# --- pod: training + chained evaluations in one tmux session (~4.6 h train + ~35 min evals)
ssh runpod 'source /workspace/rlvr-crosslingual-safety/scripts/remote/env.sh && cd $PROJECT && mkdir -p /workspace/logs && \
  tmux new -d -s es "bash -lc \"source scripts/remote/env.sh && cd \$PROJECT && \
  uv run python scripts/train_grpo.py --lang es --pool data/processed/pool_es_7b.json --model 7b \
    --steps 250 --save-steps 25 --prompts-per-step 16 --num-generations 8 --micro-batch 4 \
    --max-completion 2048 --vllm-gpu-mem 0.30 --report-to none --seed 0 --out /workspace/adapters/es_seed0 \
  && uv run python scripts/eval_math.py   --lang es --model 7b --k 8 --adapter /workspace/adapters/es_seed0/checkpoint-125 --tag es_mid   --out-dir /workspace/outputs \
  && uv run python scripts/eval_math.py   --lang es --model 7b --k 8 --adapter /workspace/adapters/es_seed0/final          --tag es_final --out-dir /workspace/outputs \
  && uv run python scripts/eval_safety.py --model 7b --n 3 --adapter /workspace/adapters/es_seed0/checkpoint-125 --tag es_mid   --out-dir /workspace/outputs \
  && uv run python scripts/eval_safety.py --model 7b --n 3 --adapter /workspace/adapters/es_seed0/final          --tag es_final --out-dir /workspace/outputs \
  && echo ARM_DONE > /workspace/logs/es_seed0.DONE\" 2>&1 | tee /workspace/logs/es_seed0.log"'

# --- first health poll (~15 min after launch; filtered one-liner, then every 30-45 min if not pausing)
ssh runpod 'python3 -c "import json,glob; h=[r for r in json.load(open(sorted(glob.glob(\"/workspace/adapters/es_seed0/checkpoint-*/trainer_state.json\"))[-1]))[\"log_history\"] if \"reward\" in r]; r=h[-1]; print(r[\"step\"], round(r[\"reward\"],3), round(r.get(\"frac_reward_zero_std\",-1),3), round(r.get(\"completions/clipped_ratio\",-1),3), round(r.get(\"step_time\",-1),1))" 2>/dev/null || tail -c 600 /workspace/logs/es_seed0.log; nvidia-smi --query-gpu=memory.used --format=csv,noheader'
# healthy: step advancing, reward ~0.3-0.7, frac_reward_zero_std < 0.3, clipped_ratio < 0.1, ~60-75 s/step, memory < 80 GB, no NaN.

# --- PAUSE POINT (this launch only): tmux kill-session -t es on the pod, then MCP pod-action stop. Screening pool survives on the volume.
#     Relaunch later = the tmux command above again, from step 0 (same seed). Do NOT --resume for a deliberate pause.

# --- after ARM_DONE: Mac side
scripts/remote/pull.sh runpod                      # rsync /workspace/outputs -> outputs/ (see INFRA)
export OPENAI_API_KEY="$(cat ~/.config/openai/key)"
uv run python scripts/judge_api.py --tag es_final --select all --workers 8      # ~$6.7, ~20 min
uv run python scripts/report.py --arm es --safety-judge api \
  --math-base outputs/eval_math/base_es.json --math-mid outputs/eval_math/es_mid.json --math-final outputs/eval_math/es_final.json \
  --safety-base outputs/safety/base_api_scores.json --safety-final outputs/safety/es_final_api_scores.json \
  --safety-mid outputs/safety/es_mid_scores.json --safety-mid-judge local \
  --log-history /workspace/adapters/es_seed0/log_history.json   # (pull it with the outputs)
# then MCP pod-action stop
```

Gate rule unchanged: if the final avg@8 CI does not exclude zero, resume from `checkpoint-250` to 500 steps (`--resume /workspace/adapters/es_seed0/checkpoint-250 --steps 500`) before any safety score is read; record the resume in the report. The English arm is the same procedure with `--lang en`, the same `pool_es_7b.json` IDs (parallel problems), `--out /workspace/adapters/en_seed0`, and no screening.

## 10. Artifacts

- `results/stage1/`: screening stats, baseline math and safety score matrices, step-10 smoke evals, `log_history.json`, `train_run_meta.json`, `report_s1.json/png`, judge agreement (scores only; no prompt/response text).
- Pod volume `/workspace`: `outputs/` (incl. raw baseline generations `safety/base_generations.jsonl`), `adapters/s1_es_test/`, `logs/`.
- Mac `outputs/` (gitignored): full copies of the above.
