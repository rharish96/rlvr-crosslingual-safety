# Stage 0 report (local CPU, 2026-09-12)

All checks passed. No GPU used. Nothing here depends on a trained model.

## Environment

- `uv`-managed CPython 3.12.14; `uv.lock` resolves 250 packages for macOS (CPU) and Linux (CUDA, `--extra gpu`).
- torch 2.13.0, transformers 5.17.0, peft 0.20.0, datasets 5.0.1, math-verify 0.9.0, trackio 0.37.1, `strong_reject` @ `7a551d5` (git; not on PyPI), gpu extra: vllm 0.29.0, trl 1.13.0.
- Private repo: `github.com/rharish96/rlvr-crosslingual-safety`.

## Data (`scripts/stage0_data.py` → `results/stage0_data_summary.json`, `data/processed/parallel_ids.json`)

- mAceReason-Math @ `a9b8d7e`; English reconstructed from `nvidia/AceReason-Math` @ `a5cc41c`.
- es/en parallel `train`: 7,620 each, identical `original_idx` set and order. `test`: 190 each, identical.
- 345 train solutions differ as strings between es and en; all are localization (`42,86\%` vs `42.86\%`, `2.177.280` vs `2,177,280`, `120°` vs `120^\circ`, `\$70` vs `$70`), which is why the English gold is used.
- Locale-sensitive gold filter (English gold with a decimal, comma grouping, an integer ≥ 1000, or a fraction): train drops 1,290 (317 + 13 + 872 + 88) → **6,330 kept**; test drops 27 (9 + 1 + 16 + 1) → **163 kept**.
- Before the ≥ 1000 and fraction rules were added, gold types (train) were 6,160 integers (85%), 825 expressions/text, 227 other, 94 fractions, with 872 integers ≥ 1000.
- Only 45 Spanish problems (0.6%) contain a `d.ddd` thousands pattern in the prompt text, so the prompts rarely prime Spanish grouping.

## Reward (`reward.py`, `tests/test_reward.py`, `scripts/stage0_reward_check.py`)

- Math-Verify handles: `42`≡`42.0`, `\frac{7}{2}`≡`3.5`≡`7/2`, `5\sqrt{2}`≡`\sqrt{50}`, `x^2+1`≡`1+x^2`, `(1,2)`≡`(1, 2)`, units/currency/degree stripped, `\text{42}`, unicode minus, `10,500`≡`10500`.
- Math-Verify does **not** handle Spanish-locale numbers: `3,5` parses as the set {3, 5}; `10.500` as 10.5; `2.177.280` fails.
- A symmetric, pattern-based normalization of the boxed string was prototyped and passed 36 tests, then **removed by decision (2026-09-13)** in favour of dropping every gold a Spanish-writing model might format with a comma or thousands dot. The reward is now plain Math-Verify. `results/stage0_reward_check.json` records the locale cases scoring 0, which is the evidence for the filter.
- Truncated completions (unbalanced `\boxed{`) score 0. Last box wins.
- 20 reward tests and 17 filter tests pass.

## Language ID (`langid.py`, `tests/test_langid.py`)

- LaTeX/inline math/`\boxed{}`/code/numerals stripped; `langdetect` seeded for determinism; < 20 prose chars → unknown. Spanish and English math solutions detected correctly; math-only text → unknown.

## StrongREJECT and evaluator (`safety_eval.py`, `scripts/stage0_safety_check.py` → `results/stage0_safety_check.json`)

- 313 prompts @ `f7cad6c`; categories 59/54/50/50/50/50 as expected.
- Evaluator `qylu4156/strongreject-15k-v1` @ `4bd893d` (LoRA r=8 on gated `google/gemma-2b`) downloaded with `HF_TOKEN`; Gemma gate access confirmed.
- The package's own loader (`device_map="auto"`, bf16) aborts with SIGABRT on Apple MPS. `load_evaluator` now loads base + adapter explicitly (CPU fp32 here, CUDA bf16 on the box), merges, and registers it in the package cache, so the official scoring code runs unchanged.
- Direction check on a synthetic low-hazard prompt (fake review): refusal **0.002**, off-topic **0.018**, compliant **0.667**. ~7 s per response on CPU; the 2,817 real responses will be scored on the GPU.

## Statistics (`stats.py`, `tests/test_stats.py`)

- Paired bootstrap over items for Δ with 95% CI; ASR at a threshold; MDE by an analytic formula and by within-item resampling under H0.
- Synthetic floor baseline (313 × 3, mean 0.023): MDE ≈ 0.003–0.004. Synthetic gate (180 × 8 in the test; 163 × 8 in practice, p ≈ 0.4): MDE ≈ 0.05; a +5–9 pp shift is detected with the CI excluding zero.
- 6 stats tests pass (null coverage, shift detection, MDE sanity).

## Plan changes recorded

- Test set for the gate is the 163 locale-safe items (was 190); training pool is 6,330 (was 7,620).
- Reward stays plain Math-Verify; locale handling moved entirely into the data filter.
- Evaluator loading pinned and explicit.

## Next: Stage 1 (GPU, Qwen2.5-3B-Instruct)

Needs a GPU provider account and SSH access. Scripts to write on the box: screening, GRPO training (TRL 1.13 API to be checked against docs), checkpoint evaluation, StrongREJECT generation, and the end-to-end report.

## Addendum: local dry run (2026-09-13, CPU, Qwen2.5-0.5B-Instruct)

Purpose: catch API and config mistakes before any GPU minute is billed. TRL 1.13 API inspected directly (`GRPOConfig` fields and defaults) rather than from memory.

- `screen.py --lang es --model 0.5b --backend hf --n-ids 6 --k 2 --max-tokens 96`: ran; all completions truncated (expected at 96 tokens); language shares es 1.0; pool file written.
- `train_grpo.py ... --steps 1 --prompts-per-step 2 --num-generations 2 --max-completion 48 --no-vllm --cpu`: one step in 12 s; `checkpoint-1` (trainer state) and `final` adapter saved; `log_history.json` exported. Rewards all 0 → zero advantage → loss 0 (expected degenerate group).
- `eval_math.py` and `eval_safety.py` for base and adapter (k=2 / n=1, tiny N): ran; evaluator scored on CPU; identical base/adapter outputs under identical seeds confirm the generation path is deterministic.
- `report.py`: JSON report and 3-panel plot produced.
- Artifacts deleted afterwards (nothing from the dry run is kept in `results/`).
