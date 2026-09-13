# Cross-Lingual RLVR and English Harmful Compliance: Design and Execution Plan

Status: Stage 0, local dry run and **Stage 1 (7B pipeline test) complete** (2026-09-13); see `docs/STAGE1_REPORT.md`. Engineering verdict GO; one open decision (primary safety judge, STAGE1_REPORT §6) before launching the Spanish arm.

## 1. Question and hypothesis

- Does LoRA-based GRPO (RLVR) on Spanish math problems change harmful compliance of Qwen2.5-7B-Instruct on English StrongREJECT prompts, relative to the same training in English?
- Primary test, two-sided: H0: Δ_SR = 0 vs. H1: Δ_SR ≠ 0. Prior work motivates an increase, but a decrease or a null is also informative.
- The baseline is expected near the floor (Yong & Bach report ASR < 5% for non-reasoning Qwen2.5-Instruct), so in practice only increases are detectable.
- No fixed smallest effect of interest. We report the 95% CI together with the minimum detectable effect (MDE): the smallest true increase the design would detect with 80% probability, computed from the baseline generations before any trained model is scored (`stats.mde_analytic`, `stats.mde_bootstrap`). A null reads "the CI is [a, b]; the design could detect increases of at least MDE."

## 2. Background and gap

- [DeepSeekMath](https://arxiv.org/abs/2402.03300) introduced GRPO, the standard optimizer for math RLVR.
- [Qi et al., ICLR 2024](https://arxiv.org/abs/2310.03693) showed benign fine-tuning can weaken safety alignment.
- [Self-Jailbreaking](https://arxiv.org/abs/2510.20956) (Yong & Bach, [ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/file/9b14a88c6e50068a97256696902521bf-Paper-Conference.pdf)): benign math/code reasoning training raises StrongREJECT ASR from <5% to 60–95% across SFT- and RL-trained models. Models still classify prompts as unsafe (95–99%) but rationalize compliance in CoT; 50 safety-reasoning samples restore refusal. Cross-lingual generalization is named as open.
- [GRPO Beyond English](https://arxiv.org/abs/2608.13698) (Dobler et al.; [Apple ML](https://machinelearning.apple.com/research/grpo-beyond-english)): non-English GRPO transfers across languages but causes model- and language-specific regressions on out-of-domain reasoning and hard math. Safety was not evaluated.
- Gap: whether non-English math RLVR transfers to an English safety behavior that was never in the training objective.

## 3. Model

- [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct): instruction-tuned with intact refusal behavior, 29+ languages including Spanish, Apache-2.0, fits LoRA-GRPO on one H100, and is the base of s1.1-7B from the Self-Jailbreaking paper.
- Why not a newer model:
  - The intervention must be the first reasoning RL the model sees. Nearly all 2025–26 releases (Qwen3+, gpt-oss, SmolLM3, R1 distills) already have RLVR-trained thinking modes; our run would be a rounding error on top and a null uninformative. GRPO Beyond English shows the headroom problem: +50 pp in-domain for Qwen3-Base vs. +1–4 pp for post-trained Qwen3.
  - Yong & Bach's story is aligned model → reasoning training → compliance rises. Qwen2.5-7B-Instruct is the "before" state.
  - Yong & Bach publish this model's baseline StrongREJECT ASR, so our baseline should reproduce it.
  - mAceReason authors suspect AceReason-Math is in Qwen3's RLVR mix; Qwen2.5 predates the dataset.
  - Base (non-instruct) models have no refusal behavior to lose.
- Excluded: Qwen2.5-Math (math-specialized); Qwen3 (above).
- Second-model candidate for a later generality claim: Gemma-3-4b-it or 12b-it.
- Qwen2.5-3B-Instruct is used only as a first pass to test the pipeline. Its results are discarded.

## 4. Data

- Source: [mAceReason-Math](https://arxiv.org/abs/2603.10767) ([GitHub](https://github.com/apple/ml-macereason-math) @ `a9b8d7e`, CC BY-NC-ND 4.0), a 14-language translation of cleaned AceReason-Math built for RLVR by the GRPO Beyond English authors. English is reconstructed from `nvidia/AceReason-Math` @ `a5cc41c` via the released bsdiff4 patches.
- Parallel means the same underlying problems exist as translations in every language. The Spanish arm trains on the Spanish text and the English control on the English text of the same problems, so only language differs between arms.
- Splits used: parallel `train` (7,620 per language) and parallel `test` (190, human-validated). Not used: Spanish `train_all` (11,346, non-parallel).
- Stage 0 verified: identical `original_idx` sets and order between es and en for both splits.
- Spanish is Qwen-supported and had native-speaker review of a 100-item pilot and the test set; train translations passed only an LLM grading/refinement loop (residual translation errors are a limitation).
- Language choice: Spanish kept (close enough to English to pass the gate and stay in-language). Japanese/Korean are candidates for a later distance-contrast arm; Chinese avoided as Qwen's home language.
- Answer verification, identical for both arms (`reward.py`):
  - Gold answers are the English reference solutions (the dataset localizes number formats in translated solutions, e.g. `42,86\%` vs `42.86\%`).
  - Locale-sensitive golds are dropped for both arms so the reward is plain Math-Verify with **no custom normalization**. Stage 0 showed Math-Verify reads Spanish-locale numbers wrongly (`3,5` → the set {3,5}; `10.500` → 10.5), so any gold a Spanish-writing model might express with a decimal comma or thousands dot is excluded: decimals (`3.5`, `.185`), comma-grouped numbers (`2,177,280`), integers ≥ 1000 (any 4+ digit run), and fractions (`\frac`, `7/2`). Stage 0 counts: train drops 1,290 (317 + 13 + 872 + 88) → **6,330 kept**; test drops 27 (9 + 1 + 16 + 1) → **163 kept**. Remaining golds are integers < 1000, symbolic expressions, tuples/intervals, and text.
  - Reward = 1 iff the last balanced `\boxed{...}` in the completion is Math-Verify-equivalent to the English gold; unbalanced (truncated) or missing box = 0. 20 reward tests + 17 filter tests.
  - Decision (2026-09-13): a symmetric normalization of the boxed string was prototyped and tested in Stage 0, then removed in favour of dropping the affected golds, trading ~13% of the pool and 17 test items for a reward with zero custom logic.
- Response language: `langdetect` (as in GRPO Beyond English) after stripping LaTeX, equations, numerals and code; responses with < 20 prose characters are "unknown".

## 5. Training subset

1. From the 6,330 kept parallel `train` IDs, sample 4,000 by ID (shared by both arms).
2. For each, sample 8 baseline solutions from the untrained model with the training template, temperature 1.0, 2,048 max tokens.
3. Score with the reward above; eyeball ~20 Spanish outputs for parsing failures.
4. Keep problems solved 2–6 of 8 (25–75%), targeting 1,500–2,000. All-correct or all-wrong groups have zero GRPO advantage and no gradient.

## 6. RLVR configuration

- GRPO with bf16 LoRA on one H100 (no QLoRA).
- LoRA: all linear layers, r = 32, α = 64, LR ≈ 1e-5 (≈10× the 1e-6 full-FT rate of GRPO Beyond English, per [LoRA Without Regret](https://thinkingmachines.ai/blog/lora/)).
- 16 prompts × 8 rollouts = 128 rollouts per step; temperature 1.0; 2,048 max completion tokens (Stage 1: truncation 0.13%, so 2,048 stands). Stage 1 memory finding: micro-batch 4 completions per forward/backward (gradient accumulation 32) with vLLM at 30% GPU memory; micro-batches of 8 and 16 OOM on the 80 GB H100. Optimizer-step geometry unchanged.
- Reward: binary correctness only. Loss: DAPO-style token-level, β = 0, no std-scaling.
- ~250 steps (~2 epochs over the pool). Adapters saved at step 0 (base model), midpoint (~125), end (250; 500 if extended). Full trainer state saved at 250 so an extension resumes exactly.
- Prompting: default Qwen2.5 chat template, no custom system prompt. Training-only user-turn suffix: es "Razona paso a paso y escribe la respuesta final en \boxed{}." / en "Reason step by step and put your final answer in \boxed{}." No thinking template.
- One seed. No English or safety data in training.
- Checkpoints: base + adapter (100–300 MB), loaded in vLLM by attaching the adapter. Both arms share step 0; baseline evaluations run once before either arm trains. Adapters and raw generations go to the persistent volume and a private HF model repo, tagged with step, arm, seed, config and versions.

## 7. Gate: did RLVR work?

- Evaluate each checkpoint on the 163 kept Spanish test items at avg@8 with the training template.
- Report accuracy with a paired-bootstrap 95% CI over items, training-reward curve, response length, truncation rate, share of outputs detected as Spanish.
- Gate: final avg@8 significantly above baseline (paired bootstrap CI excludes zero) and a rising reward curve. Effect size reported; with 163 × 8 samples the smallest detectable gain is roughly 4–5 pp. No threshold is borrowed from the base papers, which have no gate.
- If the CI includes zero, extend to 500 steps and re-test once, before any safety score is examined.
- If `langdetect` shows frequent English responses to Spanish prompts, describe the intervention as "RLVR on Spanish-language prompts."

## 8. English safety evaluation

- All 313 [StrongREJECT](https://strong-reject.readthedocs.io/) prompts (`alexandrasouly/strongreject` @ `f7cad6c`; 6 categories: 59/54/50/50/50/50), no jailbreak templates.
- Per checkpoint: 3 responses per prompt, fresh context, default Qwen template with no system prompt and no math instruction, Qwen's published sampling (temperature 0.7, top-p 0.8), 1,024 max tokens, fixed seed schedule.
- Scoring: `strongreject_finetuned` (`qylu4156/strongreject-15k-v1` @ `4bd893d`, LoRA r=8 on gated `google/gemma-2b`), 0–1 score = expected value of the judge's 1–5 rating. Loaded explicitly and pinned via `safety_eval.load_evaluator` (the package's `device_map="auto"` loader aborts on Apple MPS); CPU fp32 locally, CUDA bf16 on the box. `max_response_length` = 1,024 (package default 512 would truncate).
- Why this evaluator: official StrongREJECT judge, reported by its authors to agree with human labels about as well as the GPT-4 rubric; local, free, deterministic; no harmful content leaves the machine. Trade-off: trained on ordinary chat responses; the optional rubric spot check covers drift.
- Scored text: the full response. The model has no thinking delimiter, so its whole output is the user-facing answer; Yong & Bach scored this same base model the same way. If a visible reasoning-then-answer structure emerges, answer-only scoring is added as a robustness check with full-response primary.
- Stage 0 direction check (synthetic low-hazard prompt): refusal 0.002, off-topic 0.018, compliant 0.667.
- Workload: 313 × 3 × 3 = 2,817 responses. Not included: HarmBench, XSTest, manual annotation.

## 9. Primary outcome

- Baseline = step 0 (no adapter); Final = end of training. Per prompt, average its 3 scores; S̄ = mean across prompts.
- Δ_SR = S̄_final − S̄_baseline, 95% CI from a paired bootstrap over prompts (`stats.paired_bootstrap_delta`). Also report ASR at threshold 0.5 and the MDE.
- Midpoint: plotted for the trajectory only; not tested.
- Category-level: ~50 prompts each; shown as observations, not claims (6 tests → ~1-in-4 chance of a spurious hit).
- Interpretation: positive Δ = increased English harmful compliance; negative = reduced (hard to detect at floor); CI near zero = no detectable transfer at the stated MDE.
- No benign prompts, so a decrease could not be separated from generalized refusal (stated limitation; add XSTest if a decrease appears).
- Comparability with Yong & Bach is at the design level (same benchmark, same starting model, same outcome direction), not exact ASR values (different judge scale, thinking budget, sampling). Number-level comparability via the AISI add-on.

## 10. English control

- Same problem IDs, same configuration, same English gold and verifier, English instruction. Separates "math RLVR" from "Spanish math RLVR" and replicates the pipeline. Paper-level: 3 seeds per arm.

## 11. Scope of claims

- Exploratory cross-lingual safety audit: one model, one language pair, one seed.
- Supported: "We measure whether LoRA-based GRPO on Spanish mathematical problems transfers to English harmful-compliance behavior in Qwen2.5-7B-Instruct, relative to the same training in English."
- Not claimed: RLVR generally raises/lowers safety; Spanish uniquely causes any effect; strict self-jailbreaking (requires CoT analysis); lower scores imply better alignment.
- LoRA-only is a limitation, softened by LoRA Without Regret (LoRA matches full FT for policy-gradient RL at low rank).

## 12. Add-ons (not included now)

- XSTest safe subset (250 × 1 × 3 = 750 responses, ~15 GPU-min) if a decrease appears.
- **API-judge spot check (in scope, small; `scripts/judge_api.py`)**: StrongREJECT's `strongreject_aisi` (the AISI judge prompt Yong & Bach used) with `gpt-5-2025-08-07` (their judge model); harmful iff raw 1–5 score > 1 (the package returns (raw−1)/4). Runs from the Mac on saved generations; the OpenAI key never reaches the pod. Stage 1: 30 baseline responses (validates the API path; floor-level agreement; < $1). Final checkpoint, pre-registered: 50 responses stratified by local score (top 20, 15 from the 0.3–0.7 band, 15 random others). Reports Pearson r, mean |Δ| on 0–1, both judges' ASR on the subset, harm-label agreement, and the disagreements for eyeballing. Rubric variant available behind `--judge rubric`. Verified 2026-09-13 (refusal → 1/5, compliant → 4/5, ~8 s/call). Gotcha: the package hardcodes `temperature=0`, which GPT-5 rejects; `litellm.drop_params=True` is set in the script.
- Triggers for going beyond the pre-registered check: CI edge near a decision boundary; unusual output style; many local scores in 0.3–0.7 or mean/ASR disagreement; baseline mean > ~0.1; publication.
- **Full AISI pass** for number-level comparability with Yong & Bach: `--select all` over the 2,817 responses per arm, roughly $15–35 per arm with GPT-5. Separate decision; not included by default.
- Japanese/Korean arm; `<think>`-format variant; Gemma-3 second model.

## 13. Tooling

- Training: TRL 1.13 `GRPOTrainer` + vLLM 0.29 colocated + PEFT. TRL settings pinned explicitly: `loss_type="dapo"`, `beta=0.0`, `scale_rewards="none"`, `num_generations=8`, `per_device_train_batch_size=16` × `gradient_accumulation_steps=8` (= 16 prompts × 8 rollouts per optimizer step), `temperature=1.0`, `max_completion_length=2048`, `lr=1e-5` constant with 10 warmup steps, `max_grad_norm=1.0`, `save_only_model=False`. Generation: vLLM (transformers fallback for CPU dry runs). Scoring: Math-Verify 0.9.0, `strong_reject` @ `7a551d5`, `langdetect`.
- Tracking: trackio (local SQLite; no WandB). Versioning: private GitHub repo `rharish96/rlvr-crosslingual-safety` for code/configs/metrics/figures; adapters and raw generations to a private HF repo or the volume.
- Environment: `uv`-managed Python 3.12, `uv.lock` (250 packages; torch 2.13.0, transformers 5.17.0, peft 0.20.0, datasets 5.0.1; gpu extra: vllm 0.29.0, trl 1.13.0).
- Cursor connectors: MCP servers/Plugins (`~/.cursor/mcp.json`, `${env:HF_TOKEN}` supported). The HF plugin is optional; MCP credentials do not reach terminal scripts, so the shell `HF_TOKEN` is the single source of truth. GitHub via `gh`. No official GPU-provider connectors; SSH.

## 14. Pipeline tests

- Stage 0 (local CPU) — complete; see `docs/STAGE0_REPORT.md`.
- Stage 1 (7B pipeline test) — complete 2026-09-13; see `docs/STAGE1_REPORT.md`. Measured: 66.5 s/step, 22.7k tok/s generation, baseline es avg@8 0.412, baseline StrongREJECT mean 0.090 / ASR@0.5 10.0% (local judge), MDE 0.011–0.014, GPT-5 AISI agreement 73% on 30 with the local judge over-scoring soft refusals.
- Local dry run (CPU, Qwen2.5-0.5B-Instruct) — complete 2026-09-13. Every script ran end to end on the Mac with tiny settings: `screen.py` (6 problems × 2), `train_grpo.py` (1 GRPO+LoRA step, no vLLM; `checkpoint-1` with trainer state and `final` adapter saved), `eval_math.py` and `eval_safety.py` for base and adapter, `report.py` (JSON + plot). Zero cost.
- The former Stage 1 (3B smoke test) is dropped: it could not test the 7B-specific memory question, saved about a dollar, and its results were to be discarded. Merged into:
- **Stage 1 (GPU, Qwen2.5-7B-Instruct, one merged pipeline test, ~30–45 min ≈ $2–3)**
  1. 10 training steps at the real config (16 × 8 × 2,048, LoRA, vLLM colocated): confirms memory fits, measures step time (the real cost number).
  2. Screening on a 200-prompt subset × 8: tokens/s, truncation rate at 2,048, `\boxed{}` parse rate, Spanish share.
  3. Evaluation path on the step-10 adapter with small N (163 test items at avg@2; 40 StrongREJECT prompts × 1): adapter round-trips through vLLM; report emits tables and plots.
  4. Baseline StrongREJECT scores (313 × 3) and the MDE, recorded before any real training.
  5. Mac side: `judge_api.py --tag base --select baseline --n 30` (GPT-5 AISI judge on 30 baseline responses; < $1) to validate the API path and record floor-level judge agreement.
  6. If go/no-go passes, launch the Spanish arm on the same running pod.
- Go/no-go: no OOM at full config; step time within budget; non-degenerate reward; ≥ 90% of Spanish rollouts contain a parsable `\boxed{}`; truncation ≤ 10%; evaluator mean < 0.05 on the baseline subset; adapter round-trips.
- Scripts: `scripts/screen.py`, `scripts/train_grpo.py`, `scripts/eval_math.py`, `scripts/eval_safety.py`, `scripts/report.py`; remote ops in `scripts/remote/` and `docs/INFRA.md`.

## 15. Cost

- One H100 80GB at $2.50–3.50/h. Shared: screening 1–2 h, baseline evals ~0.3 h. Per arm: training 5–9 h (250 steps), evals ~0.5 h; +5–9 h if extended. Tests ~1.5 h.
- Nominal 15–22 h ≈ $45–75; buffered 25–35 h ≈ $75–120. Working figure $60–130. No API spend required.

## 16. Accounts and credentials

- Hugging Face: done. Fine-grained token, single permission "Read contents of public gated repos you can access"; Gemma license accepted. Stored at `~/.cache/huggingface/token` (0600) and exported as `HF_TOKEN` from `.zshrc`. Later: write scope for the private adapter repo; Jobs scope only if HF Jobs is the provider.
- GitHub: `gh` authenticated as `rharish96` (`repo` scope).
- GPU provider (Stage 1+): RunPod / Lambda / Vast.ai / HF Jobs; SSH key; ≥ 100 GB volume. Token reaches the box via provider secrets or `scp` of the token file.
- OpenAI: done (2026-09-13). Project-scoped key with a monthly budget limit; stored at `~/.config/openai/key` (0600), exported as `OPENAI_API_KEY` from `.zshrc`; used only from the Mac by `judge_api.py`. Verified against `/v1/models` and with two live AISI judge calls.
- Rules: tokens never pasted in chat; referenced by name only; `.env` and HF cache outside the repo; leaked token → revoke and recreate.

## 17. Machine state (2026-09-12)

- macOS 15.6 arm64, 26 GB RAM. Homebrew Python 3.14.7 (default) and 3.13.15; `gh` 2.100.0; `uv` 0.12.13. Project venv: uv-managed CPython 3.12.14.

## 18. Decision log

- SESOI dropped for CI + MDE; capability gate changed from fixed 5 pp to CI-excludes-zero + rising reward.
- English gold for both arms; locale-sensitive golds dropped (decimals, comma grouping, integers ≥ 1000, fractions): 7,620 → 6,330 train; 190 → 163 test.
- Normalization considered and rejected: Math-Verify does not parse Spanish-locale numbers; a symmetric boxed-string normalization was prototyped in Stage 0, then removed in favour of dropping the affected golds so the reward has no custom logic (user decision 2026-09-13).
- Spanish kept over Japanese/Korean. Qwen2.5-7B-Instruct over newer models. Full-response scoring. Fine-tuned evaluator primary; AISI later; rubric spot check conditional; XSTest deferred.
- Evaluator loaded explicitly (pinned revision, CPU fp32 / CUDA bf16) because the package loader aborts on MPS.
- trackio, not WandB. Dashes for folders/repos, underscores for the Python package.
- Stage 1 (2026-09-13): micro-batch 16 → 4 after OOM; `report.py` aligns on shared items; pod replaced on same volume; `--report-to none` recommended for unattended runs (trackio shutdown hang). Open: primary judge choice (STAGE1_REPORT §6) after the local judge over-scored soft refusals vs GPT-5.
- API judge chosen as AISI prompt + `gpt-5-2025-08-07` (Yong & Bach's exact setup) so one tool serves both the local-judge spot check and number-level comparability; rubric variant behind a flag. Runs from the Mac only. Stage 1 gets a 30-response baseline check; the informative stratified 50-response check is pre-registered for the final checkpoint.

## 19. Glossary

- ASR: attack success rate; fraction of responses judged harmful above a threshold.
- Rollout: one sampled answer; 8 per prompt form a GRPO group. Advantage: reward minus group mean.
- Temperature: sampling randomness; 0 deterministic, 1.0 the model's own distribution.
- Adapter / checkpoint: LoRA weight file; base + adapter at a step. Gate: capability check before interpreting safety.
- Paired bootstrap: resample items with replacement, keep each item's own difference, recompute the mean.
- avg@8: mean accuracy over 8 samples per item. MDE: smallest true change detectable with 80% power.
- OOM: GPU out of memory. Truncation: response cut at the token limit; reward 0.
