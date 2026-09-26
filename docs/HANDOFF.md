# Handoff: state of the experiment and how to continue

Written 2026-09-17 after the working chat was compacted (Cursor summarized ~590k tokens of history on 2026-09-16 ~20:41 local, when the chat model was switched from Claude Fable 5.1 to Grok 4.6 Fast for the launch). Nothing was lost on disk: decisions live in `docs/`, and Cursor keeps the full transcript locally. This file holds the operational facts that previously lived only in the conversation. Read it first, then `INFRA.md`, `STAGE1_REPORT.md` §9, and `PLAN.md` (status line and §18 decision log).

## 1. Where we are

- Question: does LoRA-GRPO on Spanish math (mAceReason-Math, English gold, Math-Verify reward) change English StrongREJECT harmful compliance of Qwen2.5-7B-Instruct, vs. the same training in English? Design, gates and judges: `PLAN.md`.
- Done: Stage 0 (CPU), local dry run, Stage 1 (7B pipeline test, GO), baseline math (es avg@8 0.412 on 163 kept test items) and baseline StrongREJECT judged by the primary judge GPT-5 AISI (939/939: mean 0.039, **ASR 9.9%**, MDE 0.008–0.010) and by the local Gemma-2B judge (mean 0.090, ASR@0.5 10.0%).
- Done 2026-09-16 (first launch, deliberately paused): screening of 4,000 Spanish problems × 8 → **`data/processed/pool_es_7b.json`, 1,643 problem IDs** (boxed 99.2%, truncation 0.2%, 98.6% Spanish; committed). Training ran 11 of 250 steps and was killed on purpose at the first health poll; metrics were healthy (reward 0.47 vs 0.48 expected from the pool, zero-variance groups 6%, no clipping, 55 s/step, 79.3/81.5 GB). Those 11 steps are discarded; the run restarts from step 0. No adapter exists yet (first checkpoint is step 25).
- **Spanish arm complete (2026-09-18 22:39Z → 09-19 ~04:00Z, pod `117rcbsl79ii1k`, uninterrupted, $18.98).** Math gate passed: final avg@8 0.462 vs 0.412, +5.0 pp, CI [+1.8, +8.2]; no extension. Local-judge safety Δ +0.019 [+0.003, +0.034] (secondary). 13% of responses to Spanish prompts are now English (0.5% at baseline). Full details: `docs/ES_ARM_REPORT.md`.
- **Primary safety result (GPT-5 AISI, 2026-09-19): Δ = +0.012, 95% CI [+0.005, +0.019]; ASR 9.9% → 12.4%; $6.05.** Two of 939 responses were rejected by OpenAI's biological-risk filter (refusals by the local judge; filled with the prompt's other generations). `ES_ARM_REPORT.md` §4.
- **English control, 250 steps, complete (2026-09-19 20:48Z → 09-20 02:25Z, pod `117rcbsl79ii1k`, uninterrupted, 5.72 h ≈ $20).** `base_en` math baseline run afterwards (avg@8 0.499). **Math gate NOT passed at 250: final 0.521, Δ +2.1 pp, CI [−0.9, +5.2]** (midpoint +2.5 pp [−0.1, +5.1]). Training reward 0.63 → 0.68; 24% of groups had zero reward variance (Spanish arm: 11%) because the pool was screened on Spanish pass rates and the model solves more of these problems in English, so the English arm received fewer effective updates. Pod EXITED. User chose option B (2026-09-19): judge the 250-step final for a steps-matched comparison, then decide on extending **both** arms to 500.
- **English final judged (GPT-5 AISI, $5.74, 938/939): Δ = +0.002, CI [−0.005, +0.009], null; ASR 9.9% → 10.3%.** **Spanish − English finals: +0.010, CI [+0.003, +0.018].** Full report and comparison: `docs/EN_ARM_REPORT.md`, `results/compare_arms.{json,png}`.
- Decision pending: extend both arms to 500 steps (resume each from `checkpoint-250`, ≈ $17 GPU + $6.7 judge per arm, ~5.5 h each, sequential on one pod) to test whether the English arm passes the gate and whether the Spanish − English gap persists at matched capability. Needs a resume variant of `launch_arm.sh` (`--resume <ckpt> --steps 500`, midpoint eval = checkpoint-250 already done, final eval on the 500-step adapter). Otherwise: write-up.
- Running `judge_api.py` from the agent: Cursor's auto-review blocks the `export OPENAI_API_KEY="$(cat ~/.config/openai/key)"` step until approved; macOS has no `setsid`, so run the judge as a plain foreground command (the tool backgrounds it itself) rather than `setsid nohup … &`.

## 2. Infrastructure facts (details in `INFRA.md`)

- Runpod, managed through the Runpod MCP server in Cursor (OAuth; no API key on disk).
  - Network volume `yhcwl87f48` (`rlvr-safety`, 100 GB, US-GA-2) mounted at `/workspace`: venv, HF cache, datasets, `/workspace/outputs`, `/workspace/adapters`, `/workspace/logs`. Everything persistent lives here.
  - Current pod **`117rcbsl79ii1k`** (`rlvr-safety-h100`, 1× H100 80GB SXM, secure cloud, **$3.49/h while RUNNING**), created 2026-09-18 for the Spanish arm. Predecessors `q0rkwvh27l8o1f` (09-16) and `o7tqb58mzk4k4f` (09-13) each failed to restart because their host had no free H100; both sit EXITED at $0 GPU cost and can be terminated in the console. Expect this at every session start: if `start` returns "not enough free GPUs", create a new pod in US-GA-2 on the same volume (image `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`, 40 GB disk, port 22/tcp, `startSsh`) and update `INFRA.md`.
  - SSH host/port change on every start: `get-pod` → `ssh.direct` → rewrite `Host runpod` in `~/.ssh/config` as its own step (last: `205.196.19.19:8967`). Key `~/.ssh/id_ed25519`, known-hosts file `~/.ssh/known_hosts_runpod`.
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
- Agent model (user decision 2026-09-18): **Claude Fable 5.1, extra-high thinking, 1M context, in the existing chat**. No fresh chats, no switching to cheaper models for monitoring, and the agent must not change chat settings. Cost per cold turn grows with the context (~$1.5–2 at 150k, ~$12.5 at 1M); accepted. Polls every ~45 min. (For the record: the 09-16 launch-to-pause stretch cost $5.8 notional on Grok 4.6 Fast at ~100–330k context.)
- Cursor's auto-review prompts for approval on some routine steps: a shell call that edits `~/.ssh/config` together with ssh commands, and the MCP pod stop. Edit the SSH config as its own step (ssh-only commands are on the user's allowlist); expect one click for the stop. Neither is a failure.

## 6. Open items for the user

1. Go/no-go to relaunch the Spanish arm (§3). Estimate (2026-09-18): GPU 4.5–5.4 h ≈ $16–19 nominal, $22–25 with a crash/idle buffer, +$15–17 if extended to 500 steps; GPT-5 judge on the final $6.7 (+$6.7 if extended); Cursor ≈ $30–60 notional on Fable in this chat. Suggested Runpod balance ≥ $30 for this arm, ≥ $50 to cover the extension without a mid-run top-up (both arms with extensions ≈ $90–100); OpenAI project limit with ≥ $25 headroom; Cursor on-demand cap $50 per arm if one must be set.
2. Terminate old pod `o7tqb58mzk4k4f` (console; $0 while EXITED, harmless to keep).
3. Whether to API-judge midpoints too (+$6.7 each; plan says local judge only for the trajectory).
4. Whether to double-judge finals to halve GPT-5's 3.1% per-response noise (+$6.7 per checkpoint).
5. ~~Model for monitoring turns~~ Resolved 2026-09-18: Fable 5.1 in this chat throughout (§5).

The launch-to-pause procedure in §3 has been executed twice (2026-09-16 and 2026-09-17, the second time from this document alone) with matching first-11-step metrics; see `INFRA.md` session log. The 500-step extension is decided by the pre-registered math gate only (`PLAN.md` §7): if the final avg@8 CI over the 163 test items includes zero, resume from `checkpoint-250` to 500 and re-test once before any GPT-5 judge pass; the English arm then matches the step count.

## 7. Cost to date (2026-09-17)

Runpod $11.76 (billing API); OpenAI ≈ $11.5; Cursor $296 notional, $0 billed (all Included/Free). Full-experiment cash prediction $78–154 (central ≈ $100 without 500-step extensions, ≈ $130 with); details in `AGENT_COST.md` §0b.
