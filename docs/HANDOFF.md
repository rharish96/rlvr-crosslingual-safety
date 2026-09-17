# Handoff: state of the experiment and how to continue

Written 2026-09-17 after the working chat was compacted (Cursor summarized ~590k tokens of history on 2026-09-16 ~20:41 local, when the chat model was switched from Claude Fable 5.1 to Grok 4.6 Fast for the launch). Nothing was lost on disk: decisions live in `docs/`, and Cursor keeps the full transcript locally. This file holds the operational facts that previously lived only in the conversation. Read it first, then `INFRA.md`, `STAGE1_REPORT.md` §9, and `PLAN.md` (status line and §18 decision log).

## 1. Where we are

- Question: does LoRA-GRPO on Spanish math (mAceReason-Math, English gold, Math-Verify reward) change English StrongREJECT harmful compliance of Qwen2.5-7B-Instruct, vs. the same training in English? Design, gates and judges: `PLAN.md`.
- Done: Stage 0 (CPU), local dry run, Stage 1 (7B pipeline test, GO), baseline math (es avg@8 0.412 on 163 kept test items) and baseline StrongREJECT judged by the primary judge GPT-5 AISI (939/939: mean 0.039, **ASR 9.9%**, MDE 0.008–0.010) and by the local Gemma-2B judge (mean 0.090, ASR@0.5 10.0%).
- Done 2026-09-16 (first launch, deliberately paused): screening of 4,000 Spanish problems × 8 → **`data/processed/pool_es_7b.json`, 1,643 problem IDs** (boxed 99.2%, truncation 0.2%, 98.6% Spanish; committed). Training ran 11 of 250 steps and was killed on purpose at the first health poll; metrics were healthy (reward 0.47 vs 0.48 expected from the pool, zero-variance groups 6%, no clipping, 55 s/step, 79.3/81.5 GB). Those 11 steps are discarded; the run restarts from step 0. No adapter exists yet (first checkpoint is step 25).
- Not started: the Spanish arm proper, the English control, judge passes on finals, reports, write-up.
- Pod is **EXITED**. Only the network volume bills (~$7/month).

## 2. Infrastructure facts (details in `INFRA.md`)

- Runpod, managed through the Runpod MCP server in Cursor (OAuth; no API key on disk).
  - Network volume `yhcwl87f48` (`rlvr-safety`, 100 GB, US-GA-2) mounted at `/workspace`: venv, HF cache, datasets, `/workspace/outputs`, `/workspace/adapters`, `/workspace/logs`. Everything persistent lives here.
  - Current pod **`q0rkwvh27l8o1f`** (`rlvr-safety-h100`, 1× H100 80GB SXM, secure cloud, **$3.49/h while RUNNING**). Created 2026-09-16 because the previous pod `o7tqb58mzk4k4f` could not restart (its host had no free H100). The old pod is still EXITED at $0 GPU cost; terminate it in the console when convenient. If `q0rkwvh27l8o1f` also fails to start, create a new pod in US-GA-2 on the same volume (image `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`, 40 GB disk, port 22/tcp, `startSsh`), and update `INFRA.md`.
  - SSH host/port change on every start: `get-pod` → `ssh.direct` → rewrite `Host runpod` in `~/.ssh/config` (last: `205.196.17.250:11432`). Key `~/.ssh/id_ed25519`, known-hosts file `~/.ssh/known_hosts_runpod`.
- Mac ↔ pod: `scripts/remote/sync.sh runpod` pushes code (rsync with `--no-o --no-g`; never push `outputs/`); `scripts/remote/pull.sh runpod` pulls `/workspace/outputs` → `outputs/` plus `log_history.json`. `scripts/remote/bootstrap.sh` after every start (~2 min; re-adds tmux/rsync, verifies torch/vLLM/TRL imports). `scripts/remote/env.sh` sets `HF_HOME`, `HF_TOKEN` (from `/workspace/.hf_token`), uv paths.
- Credentials, referenced by name only, never pasted in chat: HF token `~/.cache/huggingface/token` (`HF_TOKEN` in `.zshrc`); OpenAI project key `~/.config/openai/key` (`OPENAI_API_KEY` in `.zshrc`; used only from the Mac by `scripts/judge_api.py`); GitHub via `gh` as `rharish96`; repo `rharish96/rlvr-crosslingual-safety` (private).
- Local repo: `~/Projects/AI-Safety/rlvr-crosslingual-safety`, uv-managed Python 3.12, `uv run pytest` (50 tests), `ruff`.

## 3. Next action: relaunch the Spanish arm (only on the user's explicit go)

```bash
# Mac: MCP pod-action start on q0rkwvh27l8o1f (or create a replacement, §2) → get-pod → update Host runpod, then
scripts/remote/sync.sh runpod
ssh runpod 'bash /workspace/rlvr-crosslingual-safety/scripts/remote/bootstrap.sh'
# discard the 11-step directory from the paused launch so the run dir holds only the real run
ssh runpod 'rm -rf /workspace/adapters/es_seed0 && mkdir -p /workspace/logs && tmux new -d -s es \
  "bash /workspace/rlvr-crosslingual-safety/scripts/remote/launch_arm.sh es 2>&1 | tee /workspace/logs/es_seed0.log"'
```

Screening is not repeated (pool on the volume and in git). `launch_arm.sh` = training (250 steps, checkpoints every 25, `--report-to none`) `&&` math avg@8 on checkpoint-125 and final `&&` StrongREJECT generation (313 × 3) on both `&&` `ARM_DONE` marker. Expected ≈ 3.8–4.6 h training + ~35 min evaluations ≈ $16–19.

Health poll (every 30–45 min; filtered so it costs few tokens):

```bash
ssh runpod "python3 -c \"
from pathlib import Path; import re
t=Path('/workspace/logs/es_seed0.log').read_text(errors='replace'); m=re.findall(r\\\"\\{'loss'.*?\\}\\\", t, re.S)
s=' '.join(m[-1].split()) if m else 'no step yet'; g=lambda k: (re.search(k+r\\\"': '([^']+)\\\", s) or [None,'?'])[1]
print(len(m), 'steps | reward', g('reward'), 'zero_std', g('frac_reward_zero_std'), 'clip', g('clipped_ratio'), 'len', g('completions/mean_length'), 's/step', g('step_time'))
\"; nvidia-smi --query-gpu=memory.used --format=csv,noheader; ls /workspace/adapters/es_seed0 | grep -c checkpoint"
```

Healthy: step count advancing, reward 0.3–0.7, `zero_std` < 0.3, `clip` < 0.1, mean length not exploding (watch: RLVR lengthens completions and memory is 79 of 81.5 GB), 45–75 s/step, no NaN. On a crash (OOM, pod loss): resume with `--resume /workspace/adapters/es_seed0/checkpoint-N` and record "resumed at step N" in the run metadata and report. Never `--resume` after a deliberate pause; relaunch from scratch.

After `/workspace/logs/es_seed0.DONE` exists:

```bash
scripts/remote/pull.sh runpod
uv run python scripts/judge_api.py --tag es_final --select all --workers 8          # GPT-5 AISI, ≈ $6.7, ~20 min, resumable, lock file
uv run python scripts/report.py --arm es --safety-judge api \
  --math-base outputs/eval_math/base_es.json --math-mid outputs/eval_math/es_mid.json --math-final outputs/eval_math/es_final.json \
  --safety-base outputs/safety/base_api_scores.json --safety-final outputs/safety/es_final_api_scores.json \
  --safety-mid outputs/safety/es_mid_scores.json --safety-mid-judge local --log-history outputs/es_seed0/log_history.json
# MCP pod-action stop
```

Gate: if the final math avg@8 CI does not exclude zero, extend to 500 steps from `checkpoint-250` (`--resume ... --steps 500`) before reading any safety score. English control: same script with `en` (same pool IDs, no screening).

## 4. Operating rules learned the hard way

- Nothing billable starts without the user's explicit clearance for that stage; the user decides pauses, extensions, and judge passes.
- Inside Cursor's sandbox `pgrep`/`ps` cannot see processes and buffered output makes logs look empty. Check for running jobs with a non-sandboxed command before starting anything twice (this caused the duplicate GPT-5 judge run, $4.8 wasted). `judge_api.py` now refuses to start if `outputs/safety/<tag>_api_calls_<judge>.lock` exists.
- Long pod jobs: tmux or `setsid nohup … &`; `--report-to none` (trackio's shutdown hangs the process); `pkill -f '[t]rain_grpo'` bracket pattern from SSH one-liners.
- Memory: micro-batch 4 + vLLM 0.30 fits (79.5/81.5 GB); 8 and 16 OOM. Keep the tested config.
- Locale-sensitive golds are dropped, not normalized; the reward is plain Math-Verify (user decision).
- Filter every shell output; never print catalogs or raw logs into the chat. Polls cost one context re-write each.
- Docs are updated and pushed at the end of every stage; `AGENT_COST.md` tracks agent spend from Cursor usage exports.

## 5. Working agreement with the user

- Concise prose, no mannered filler; tables only when they help. Explain jargon plainly when asked ("like I'm stupid").
- Present decisions with the trade-off and a recommendation; the user picks. Record the pick in `PLAN.md` §18.
- Report measured numbers, not estimates, whenever a read is possible (Runpod billing API, usage CSVs, log files).
- Cheap model (Grok 4.6 Fast) is acceptable for launch/poll/pull routine; Fable 5.1 (or Opus 5) for anything touching reward, evaluation, statistics, judge logic, debugging, interpretation, write-up. Measured: the launch-to-pause stretch cost $5.8 notional on Grok at ~100–330k context.

## 6. Open items for the user

1. Go/no-go to relaunch the Spanish arm (§3). Suggested Runpod balance ≈ $60 for both arms with buffer; OpenAI project limit ≥ $50.
2. Terminate old pod `o7tqb58mzk4k4f` (console; $0 while EXITED, harmless to keep).
3. Whether to API-judge midpoints too (+$6.7 each; plan says local judge only for the trajectory).
4. Whether to double-judge finals to halve GPT-5's 3.1% per-response noise (+$6.7 per checkpoint).
5. Model for monitoring turns (Grok proven adequate for the launch stretch) vs. staying on Fable now that the context is ~100k.

## 7. Cost to date (2026-09-17)

Runpod $11.76 (billing API); OpenAI ≈ $11.5; Cursor $296 notional, $0 billed (all Included/Free). Full-experiment cash prediction $78–154 (central ≈ $100 without 500-step extensions, ≈ $130 with); details in `AGENT_COST.md` §0b.
