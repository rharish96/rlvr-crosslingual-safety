# Infrastructure (Runpod)

Set up 2026-09-13 via the official Runpod MCP server (OAuth; no API key on disk).

## Resources

- Network volume `rlvr-safety` — id `yhcwl87f48`, 100 GB STANDARD, data center **US-GA-2**, ~$7/month. Mounted at `/workspace`. Holds everything: uv + Python 3.12, the project venv (torch 2.13.0+cu130, vLLM 0.29.0, TRL 1.13.0), the HF cache (25 GB: Qwen2.5-3B/7B-Instruct, gemma-2b, the StrongREJECT evaluator adapter), datasets, and later adapters/generations.
- Pod `rlvr-safety-h100` — id `o7tqb58mzk4k4f` (replaced `ozd4q3tx6aeev9` on 2026-09-13: its host had no free GPU on restart; old pod terminated), 1× H100 80GB SXM, secure cloud, **$3.49/hour while RUNNING**, $0 while EXITED. Image `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`, 40 GB container disk (ephemeral), port 22/tcp, SSH key injected at create.
- Account: SSH public key `~/.ssh/id_ed25519.pub` (comment `rharish96-runpod`) registered. HF token copied to `/workspace/.hf_token` and exported by `scripts/remote/env.sh`. No Runpod secrets used.

## Daily operation

- Start: `pod-action start` (MCP) → `get-pod` → copy `ssh.direct` host/port into `~/.ssh/config` under `Host runpod` (they change on restart) → `scripts/remote/sync.sh` → `ssh runpod`.
- The container disk resets on stop, so after each start run `bash /workspace/rlvr-crosslingual-safety/scripts/remote/bootstrap.sh` (idempotent, ~30 s when the venv already exists: re-adds the shell env, apt tools, verifies imports).
- Stop when idle: `pod-action stop`. Only the volume keeps billing.
- If a start fails for lack of H100 stock in US-GA-2, create a new pod in US-GA-2 with the same volume; nothing is lost.

## Cost controls

- Prepaid credits; Runpod stops all pods at $0 balance and preserves network-volume data. Auto-pay off. Low-balance email at $15. Default account spend limit $80/hour (we use $3.49).
- Session log:
  - 2026-09-13 prep (bootstrap + prefetch + checks) ≈ 22 min ≈ $1.25.
  - 2026-09-13 Stage 1 (screening 200×8, 10-step training with two OOM retries, baseline math + safety, smoke evals) ≈ 1 h 57 m ≈ $6.80. Current SSH: `205.196.19.19:8994` (changes on restart; pod now EXITED).

## Verified on the pod

- 50 tests pass; evaluator direction check on CUDA bf16: refusal 0.002, off-topic 0.018, compliant 0.665 (CPU fp32 gave 0.667).

## OpenAI judge (added 2026-09-13)

- Purpose: `scripts/judge_api.py` runs StrongREJECT's AISI judge (Yong & Bach's prompt) with `gpt-5-2025-08-07` on saved generations, from the Mac only. The key never goes to the pod.
- Key: project-scoped with a monthly budget in the OpenAI dashboard; stored at `~/.config/openai/key` (0600), exported as `OPENAI_API_KEY` from `.zshrc`.
- Verified: `/v1/models` HTTP 200; `gpt-5-2025-08-07` available; AISI judge on a benign refusal → 1/5, on a compliant fake review → 4/5; ~8 s per call.
- Gotcha: `strong_reject` hardcodes `temperature=0`, which GPT-5 rejects; `litellm.drop_params = True` is set in the script.

## Operational gotchas (learned in Stage 1)

- Launch long jobs with `setsid nohup ... > log 2>&1 < /dev/null &` so the SSH call returns.
- `pkill -f` from an SSH one-liner matches its own command line; use a bracket pattern like `pkill -f '[s]cripts/train_grpo.py'`.
- Trackio's shutdown upload leaves the training process alive after completion; use `--report-to none` for unattended runs, or kill after `final/run_meta.json` appears.
- `rsync` to the volume needs `--no-o --no-g` (the filesystem rejects chown). `chmod` is ignored on the volume.
- TRL 1.13 declares support for vLLM ≤ 0.28; 0.29 worked for colocated generation.
- Memory: micro-batch 4 + vLLM 0.30 fits (79.5/81.5 GB); 8 and 16 OOM.
