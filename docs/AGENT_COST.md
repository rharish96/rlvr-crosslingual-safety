# Agent (Cursor) token usage and cost

## 0. Actuals from the Cursor usage export (2026-09-17) — supersedes the estimates below

Source: `usage-events-2026-09-17.csv` (58 events, 2026-09-10 21:05 → 2026-09-13 15:09 local, all model `claude-fable-5-1-thinking-xhigh`, all `Kind = Included`, i.e. covered by the plan allowance; $0 billed on-demand). Events are per agent run/user message, not per tool call.

| | Cache write | Uncached input | Cache read | Output | Total | Notional at list rates* |
|---|---|---|---|---|---|---|
| Whole conversation (3 days) | 14.37M | 638 | 83.75M | 370k | 98.5M | **$219** (write $180, read $21, output $19) |
| Sun 09-13 (Stage 1 + multitask workers + cost analysis) | 11.37M | | 70.43M | 192k | | $169 |
| Stage 1 worker window (12:00–14:15) | 3.14M | | 28.07M | 52k | | **$49** (est. was $15) |
| Largest single event (the Stage 1 fork, 13:10) | 1.58M | | 18.22M | 39k | 19.8M | ~$25 |

\* Fable 5.1: $12.5/M cache write, $0.25/M cache read, $50/M output; uncached input assumed $10/M.

What the actuals change:
- **Cache writes are 82% of the cost**, not output. The context is now ~500k tokens (events of 490–560k written with zero cache read = a full re-write; 22 of 58 events were full re-writes, ~$6 each). The Section 2 estimates assumed a 240k context and undercounted writes ~3×.
- The Stage 1 worker cost ~$49, not $15, for the same reason: a self-fork re-reads the whole coordinator history on every tool call (18.2M cache-read tokens in one run).
- Output tokens are a minor term (370k ≈ $19). Thinking is included in output and is not what drives cost.
- Ratios and the recommendation are unchanged; the absolute savings from context hygiene are larger than estimated: a fresh 40k-token chat re-writes for ~$0.5 instead of ~$6, and a brief-based 25k worker costs ~$3–5 per Stage-1-sized run instead of ~$49.

Projection for the remaining work (Spanish + English arms, evals, judge passes, reports, write-up; 150–325 agent turns), all on Fable 5.1 (no model downgrade):
- Status quo (500k coordinator context, self-forked workers): **$150–400 notional**.
- Fresh coordinator chat (~40k) + brief-based workers (~25k): **$30–100 notional**.
Whether any of this bills on-demand depends on the remaining "Included" allowance in the Cursor dashboard; to date it has been $0.

Incident log: on 2026-09-16 the baseline GPT-5 judge pass was accidentally started twice in parallel (a sandbox quirk hid the first process). 679 duplicate calls ≈ $4.8 wasted on the OpenAI side; results unaffected (cache keyed per response). Guard added: the run command now aborts if a judge process is already running.

---


Scope: tokens consumed by the coding agent itself (coordinator chat + subagents), not GPU training tokens. Sections 1–6 are the 2026-09-13 *estimates* from transcripts; Section 0 above has the measured actuals.

## 1. Method and caveats

- Cursor's local transcripts record user text, assistant text and tool-call arguments. They do **not** record tool outputs, model thinking, or usage/cost fields. Everything below is therefore an **estimate**:
  - tokens = characters / 4;
  - tool outputs estimated per tool type (shell tails ~1.2k chars, web searches ~7k, the RunPod GPU catalog ~42k, etc.);
  - thinking tokens assumed at 1,000/turn for the coordinator and 700/turn for workers (plausible range 0.5×–2×);
  - system prompt + tool schemas assumed at 15k tokens per call;
  - a fixed per-call system prompt and Anthropic-style prompt caching: every prior token is a cache *read* on each call, new tokens are a cache *write* once, and the whole context is re-written after an idle gap longer than the cache TTL. Cursor does not publish its TTL, so both 5-minute and 60-minute cases are shown. Cursor may also compact long contexts, which would lower the read figures.
- Prices are Cursor's published per-model API rates as of 2026-09-13 (cursor.com/docs/models-and-pricing). Individual plans draw third-party models from the "Other Models" included allowance and bill on-demand beyond it at these same rates; Composer and Grok draw from the separate, larger "Cursor Models" pool. The $0.25/M Cursor Token Rate applies only to Teams/Enterprise.
- Uncertainty on any single dollar figure is roughly ±40%; the *ratios* between scenarios are much more robust than the absolute numbers.

## 2. Measured usage so far

| Thread | API calls (turns) | Tool calls | Own content (tokens) | Cache reads (M) | Cache writes (M), 5-min / 60-min TTL | Output incl. est. thinking (k) |
|---|---|---|---|---|---|---|
| Coordinator (whole chat, 3 days) | 174 | 219 | 231k | 26.1 | 4.2 / 1.1 | 398 |
| Stage 1 worker (self-fork) | 57 | 77 | 43k own + 242k inherited | 14.8 | 0.57 | 81 |
| Plan-doc worker (self-fork) | 7 | 12 | 6k own + 242k inherited | 1.7 | 0.25 | 11 |
| **Whole conversation** | **238** | **308** | | **42.6** | **5.0 / 1.9** | **491** |

Model used throughout: Claude Fable 5.1 (thinking, xhigh). Stage 1 worker tool mix: 48 Shell, 17 AwaitShell, 6 RunPod MCP calls, 4 StrReplace, 1 Read, 1 Write; wall-clock 12:03–14:12.

### Estimated cost of what has been spent

| Model (rates $/M: cache write / cache read / output) | Stage 1 worker | Plan worker | Coordinator (5-min TTL) | Coordinator (60-min TTL) | Whole conversation |
|---|---|---|---|---|---|
| **Claude Fable 5.1** (12.5 / 0.25 / 50) — actual | **$15** | **$4** | **$79** | **$40** | **$59–98** |
| Claude Opus 5 (6.25 / 0.5 / 25) | $13 | $3 | $49 | $30 | $45–65 |
| GPT-5.6 Sol (5 / 0.4 / 20) | $10 | $2 | $39 | $24 | $36–52 |
| Grok 4.6 Fast (4 / 1 / 12) | $18 | $3 | $48 | $35 | $56–68 |
| Composer 2.5 Fast (3 / 0.5 / 15) | $10 | $2 | $32 | $22 | $34–44 |
| Composer 2.5 non-fast (0.5 / 0.2 / 2.5), main-chat only | $3.5 | $0.5 | $8 | $7 | $11–12 |

Two things stand out. Fable 5.1's unusually cheap cache reads ($0.25/M) make it only 1.5× Opus 5 in practice, and *cheaper* than Grok 4.6 Fast for the Stage 1 pattern. And the coordinator thread, not Stage 1, is where most of the money went.

## 3. Where Stage 1's tokens actually went

- **93% of the worker's input volume was the inherited coordinator history.** The self-fork carried ~242k tokens of context into each of its 57 calls (13.8M of 14.8M cache-read tokens). The worker's own work added 43k tokens in total.
- Of its own 43k: tool-call arguments 13k (shell commands, file edits), assistant prose 8k, estimated tool outputs 21k. Output side: ~21k text + args plus an estimated ~40k thinking.
- Retries were cheap in tokens but expensive in wall-clock: the two OOM retries and two hung processes cost ~1 h of GPU ($3.50), but only ~10 agent turns.
- Idle waits are cache-hostile: a poll after a >5-minute wait re-writes the whole context. With a 250k context on Fable 5.1 that is ~$3.2 per poll; with a 25k context it is ~$0.38; on Composer 2.5 Fast with 25k it is ~$0.10.

Conclusion: model choice is the second lever. Context size per call is the first.

## 4. Projection for the rest of the experiment

Remaining agent work (turns, low–high): Spanish arm launch + 4.6 h monitoring + mid/final evals 35–50; its report/judge/write-up 15–20; English arm 30–45 and 15–20; 500-step extensions 0–60; option-B judge and add-ons 10–25; final synthesis 15–25; coordinator Q&A and routing 30–80. Total **150–325 turns** (Stage 1 was 57 turns of mostly debugging; monitoring turns are cheaper).

Two workflows:

- **A. Status quo**: workers self-forked from the coordinator (~250k inherited context each), coordinator thread continuing at ~260k and growing.
- **B. Fresh briefs**: workers launched as new subagents with a written brief (`docs/INFRA.md`, `STAGE1_REPORT.md §9`, the plan), ~25k context; coordinator moved to a new chat seeded with the docs (~40k).

| Model for all remaining work | A: status quo | B: fresh briefs |
|---|---|---|
| Claude Fable 5.1 (current) | $123–337 | $31–96 |
| Claude Opus 5 | $76–203 | $18–55 |
| GPT-5.6 Sol | $61–162 | $14–44 |
| Grok 4.6 Fast | $75–189 | $14–45 |
| Composer 2.5 Fast | $48–122 | $11–32 |

**Mixed, recommended** (workflow B; monitoring-heavy runs on a cheap model, everything else on Fable 5.1): **$24–73** with Composer 2.5 Fast, $25–78 with Grok 4.6 Fast, $25–77 with GPT-5.6 Sol. The coordinator alone, in a new chat on Fable 5.1, is $12–47 of that.

Reading the table: switching to workflow B saves 3–4× regardless of model; switching model on top saves a further ~25% on the runs that are moved. Composer and Grok also come out of the separate Cursor Models pool, so on Pro/Pro+/Ultra their usage may not touch the dollar allowance at all.

## 5. Recommendation

Keep on the strongest model (Fable 5.1, or Opus 5 as the half-price fallback):
- anything that changes the reward, evaluation, statistics or judge logic;
- OOM and training-instability debugging (the Stage 1 memory decisions);
- the go/no-go and measurement decisions (judge A vs B, extension to 500 steps), the report interpretation, the final write-up;
- the coordinator thread itself, but restarted in a new chat so each turn is ~40k context, not ~260k.

Safe to move to Composer 2.5 Fast or Grok 4.6 Fast (available as subagent models), or GPT-5.6 Sol:
- pod start/stop, SSH config refresh, `sync.sh`, `bootstrap.sh`;
- launching pre-written commands from `STAGE1_REPORT.md §9` under tmux and polling `log_history.json`;
- running `eval_math.py` / `eval_safety.py` / `report.py` / `judge_api.py` with known arguments and copying results back;
- doc updates that transcribe numbers already computed.

Rule for the cheap worker: it reports facts and stops at the first anomaly (OOM, NaN loss, reward stuck at 0 or 1, evaluator mean far from baseline, any exception); a Fable turn then decides. Do not let the cheap model change code under `src/` or `scripts/`.

Token-saving practices for the monitoring phase:
- Launch workers as fresh subagents with a written brief instead of self-forks. This is the single largest saving.
- `--report-to none` on training runs; poll `log_history.json` via a filtered one-liner (last reward, step, s/step) rather than tailing raw logs.
- Poll rarely and with long waits (every 30–45 min for a 4.6 h run); each poll costs one context re-write, so fewer polls beat frequent cheap ones.
- Have scripts write a one-paragraph `SUMMARY.txt` next to their outputs and have the agent read only that.
- Filter every shell output (`tail`, `grep`, `jq`); never print catalogs or full logs into context (the RunPod GPU catalog alone was ~10k tokens).
- Start a new coordinator chat for the execution phase; the planning history lives in `docs/`.

## 6. Cursor billing notes (as of 2026-09-13)

- Pro $20 / Pro+ $60 / Ultra $200 per month. Each includes an "Other Models" allowance billed at API rates (historically $20 / $70 / $400; Cursor now lists it only as "Included" with 3× and 20× Agent limits) plus a separate, larger "Cursor Models" pool for Composer 2.5 and Grok 4.5/4.6. On-demand usage beyond the allowance is billed in arrears at the same API rates.
- Subagent slugs available here and their pool: `claude-fable-5-1-thinking-xhigh` and `claude-opus-5-thinking-high` (Other Models); `gpt-5.6-sol-medium` (Other Models; promotional price through 2026-11-21); `composer-2.5-fast` and `cursor-grok-4.6-high-fast` (Cursor Models pool). Non-fast Composer 2.5 ($0.5/$0.2/$2.5) is only selectable as the main chat model, not as a subagent model.
- Thinking tokens bill as output. Cache writes are not listed for Composer/Grok; assumed equal to input above.
