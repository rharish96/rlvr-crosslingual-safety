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
- Locale-sensitive gold filter (English gold with a decimal or comma thousands): train drops 330 (317 + 13) → **7,290 kept**; test drops 10 (9 + 1) → **180 kept**.
- Kept gold types (train): 6,160 integers (85%), 825 expressions/text, 227 other, 94 fractions. 872 kept golds are integers ≥ 1000.
- Only 45 Spanish problems (0.6%) contain a `d.ddd` thousands pattern in the prompt text, so the prompts rarely prime Spanish grouping.

## Reward (`reward.py`, `tests/test_reward.py`, `scripts/stage0_reward_check.py`)

- Math-Verify handles: `42`≡`42.0`, `\frac{7}{2}`≡`3.5`≡`7/2`, `5\sqrt{2}`≡`\sqrt{50}`, `x^2+1`≡`1+x^2`, `(1,2)`≡`(1, 2)`, units/currency/degree stripped, `\text{42}`, unicode minus, `10,500`≡`10500`.
- Math-Verify does **not** handle Spanish-locale numbers: `3,5` parses as the set {3, 5}; `10.500` as 10.5; `2.177.280` fails.
- Added a symmetric, pattern-based normalization of the boxed string (thousands dot/space/thin-space → integer; lone decimal comma → point when the gold has no comma). After it, every locale case scores correctly; the only remaining 0 in the recorded table is `\sqrt{2}` vs `1.414`, which is correct behaviour.
- Truncated completions (unbalanced `\boxed{`) score 0. Last box wins.
- 36 reward tests pass.

## Language ID (`langid.py`, `tests/test_langid.py`)

- LaTeX/inline math/`\boxed{}`/code/numerals stripped; `langdetect` seeded for determinism; < 20 prose chars → unknown. Spanish and English math solutions detected correctly; math-only text → unknown.

## StrongREJECT and evaluator (`safety_eval.py`, `scripts/stage0_safety_check.py` → `results/stage0_safety_check.json`)

- 313 prompts @ `f7cad6c`; categories 59/54/50/50/50/50 as expected.
- Evaluator `qylu4156/strongreject-15k-v1` @ `4bd893d` (LoRA r=8 on gated `google/gemma-2b`) downloaded with `HF_TOKEN`; Gemma gate access confirmed.
- The package's own loader (`device_map="auto"`, bf16) aborts with SIGABRT on Apple MPS. `load_evaluator` now loads base + adapter explicitly (CPU fp32 here, CUDA bf16 on the box), merges, and registers it in the package cache, so the official scoring code runs unchanged.
- Direction check on a synthetic low-hazard prompt (fake review): refusal **0.002**, off-topic **0.018**, compliant **0.667**. ~7 s per response on CPU; the 2,817 real responses will be scored on the GPU.

## Statistics (`stats.py`, `tests/test_stats.py`)

- Paired bootstrap over items for Δ with 95% CI; ASR at a threshold; MDE by an analytic formula and by within-item resampling under H0.
- Synthetic floor baseline (313 × 3, mean 0.023): MDE ≈ 0.003–0.004. Synthetic gate (180 × 8, p ≈ 0.4): MDE ≈ 0.05; a +5–9 pp shift is detected with the CI excluding zero.
- 6 stats tests pass (null coverage, shift detection, MDE sanity).

## Plan changes recorded

- Test set for the gate is the 180 decimal-free items (was 190).
- Reward normalization added (was "none"), with the evidence above.
- Evaluator loading pinned and explicit.

## Next: Stage 1 (GPU, Qwen2.5-3B-Instruct)

Needs a GPU provider account and SSH access. Scripts to write on the box: screening, GRPO training (TRL 1.13 API to be checked against docs), checkpoint evaluation, StrongREJECT generation, and the end-to-end report.
