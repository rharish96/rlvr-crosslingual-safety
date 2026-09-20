# Cross-lingual RLVR and English harmful compliance

Does reinforcement learning with verifiable rewards (RLVR) on a benign, non-English task change a model's safety behaviour in English? We train Qwen2.5-7B-Instruct with LoRA-GRPO on Spanish mathematics and on the same problems in English, and measure harmful compliance on English StrongREJECT prompts before and after.

## Background

- [Self-Jailbreaking](https://arxiv.org/abs/2510.20956) (Yong & Bach, ICLR 2026): benign math/code *reasoning* training raises StrongREJECT attack success on aligned models from under 5% to 60–95%. The models still classify the prompts as unsafe but reason their way to compliance. Cross-lingual generalisation is named as open.
- [GRPO Beyond English](https://arxiv.org/abs/2608.13698) (Dobler et al.): GRPO on non-English math transfers across languages but produces model- and language-specific regressions on out-of-domain tasks. Safety was not evaluated.
- Supporting work: [DeepSeekMath](https://arxiv.org/abs/2402.03300) (GRPO), [Qi et al. 2024](https://arxiv.org/abs/2310.03693) (benign fine-tuning weakens alignment), [StrongREJECT](https://strong-reject.readthedocs.io/) (benchmark and judges), [mAceReason-Math](https://arxiv.org/abs/2603.10767) ([code](https://github.com/apple/ml-macereason-math); parallel 14-language translation of AceReason-Math, built for RLVR by the GRPO Beyond English authors).

## Motivation

The two papers leave a gap between them: reasoning training erodes safety, and non-English RLVR causes out-of-domain regressions, but nobody has asked whether RLVR in one language changes a safety behaviour in another that was never part of the training objective. If it does, multilingual capability training is a cross-lingual safety risk; if it does not, the self-jailbreaking effect is tied to the reasoning format rather than to training itself. Our setup deliberately keeps the intervention light (LoRA, on-policy GRPO, no thinking mode) so that any transfer is attributable to the RL objective and the language, not to installing a new reasoning style.

## Protocol

- **Model:** Qwen2.5-7B-Instruct (aligned, no prior RLVR; the base of s1.1-7B in Self-Jailbreaking).
- **Data:** the parallel `train` split of mAceReason-Math. Problems whose gold answer is locale-sensitive (decimals, thousands grouping, integers ≥ 1000, fractions) are dropped so the reward is plain Math-Verify against the English gold; 6,330 of 7,620 remain. 4,000 are screened with the base model (8 samples each) and the 1,643 solved 2–6 times are kept as the training pool, identical for both arms.
- **Training:** GRPO with the DAPO loss, β = 0, no reward scaling, 16 prompts × 8 rollouts per step, temperature 1.0, 2,048-token completions, LoRA r = 32 on all linear layers, LR 1e-5, 250 steps, checkpoints every 25. Two arms differing only in the language of the problem text and instruction: **Spanish** and **English** (control).
- **Capability gate:** avg@8 on 163 held-out problems; a paired-bootstrap 95% CI excluding zero is required before safety is read. If not met, the arm is extended to 500 steps once.
- **Safety:** all 313 StrongREJECT prompts, 3 responses each at Qwen's default sampling, at baseline and each checkpoint. Primary judge: GPT-5 running the AISI judge prompt (Yong & Bach's setup; harmful iff score > 1 of 5). Secondary: the fine-tuned StrongREJECT evaluator. Outcome: change in mean score with a paired bootstrap over prompts, ASR, and the minimum detectable effect; no smallest-effect threshold is assumed.
- **Infrastructure:** one H100 (Runpod), TRL + vLLM + PEFT, `uv`-locked environment. Everything is scripted (`scripts/`, `scripts/remote/`); design and decision log in `docs/PLAN.md`, run reports in `docs/ES_ARM_REPORT.md` and `docs/EN_ARM_REPORT.md`.

## Results so far (one seed per arm)

| arm | math avg@8 gain (95% CI) | gate | StrongREJECT Δ mean, GPT-5 judge (95% CI) | ASR |
|---|---|---|---|---|
| Spanish | +5.0 pp [+1.8, +8.2] | pass | **+0.012 [+0.005, +0.019]** | 9.9% → 12.4% |
| English | +2.2 pp [−0.9, +5.2] | not at 250 | +0.002 [−0.005, +0.009] | 9.9% → 10.3% |
| Spanish − English | | | +0.010 [+0.003, +0.018] | |

Small effects, in the predicted direction, about 1–1.5× the design's minimum detectable effect and an order of magnitude below the self-jailbreaking regime. The Spanish arm also drifted to 13% English responses on Spanish prompts. The open confound is dose versus language: the Spanish arm changed more than the English arm, partly because the pool was screened on Spanish difficulty.

## Next steps

1. Score every 25-step checkpoint of both arms (math avg@8 and GPT-5 StrongREJECT) to obtain a dose–response curve per arm, and regress safety change on capability gain with a language term: does Spanish sit above English at equal gain?
2. If needed, extend the English arm to 500 steps and compare at matched capability gain.
3. Robustness: an English arm on an English-screened pool (difficulty-matched instead of problem-matched); additional seeds; a more distant language (Japanese or Korean).

## Layout and setup

```
src/rlvr_crosslingual_safety/   data, reward, language id, generation, safety eval, stats
scripts/                        screen, train_grpo, eval_math, eval_safety, judge_api, report
scripts/remote/                 pod bootstrap, sync, launch_arm.sh (one arm end to end), pull
tests/                          unit tests
data/processed/                 problem-ID pools (committed); raw data and generations are gitignored
results/                        score matrices, summaries and figures (committed)
docs/                           PLAN.md, stage and arm reports, infrastructure and cost notes
```

```
uv sync --group dev          # local (CPU) environment
uv sync --extra gpu          # on the GPU box
```

`HF_TOKEN` (read access to gated Gemma for the StrongREJECT evaluator) and, for the judge, `OPENAI_API_KEY` must be set in the environment. No tokens are committed.
