# rlvr-crosslingual-safety

Does LoRA-based GRPO (RLVR) on Spanish math problems change harmful compliance of
Qwen2.5-7B-Instruct on English StrongREJECT prompts, relative to the same training in English?

See `docs/PLAN.md` for the full design and execution plan.

## Layout

```
src/rlvr_crosslingual_safety/   library code (data, reward, language id, stats, safety eval)
scripts/                        stage scripts (stage0_*.py run on CPU; later stages on the GPU box)
tests/                          unit tests
data/raw/                       downloaded datasets (gitignored)
results/                        small JSON/CSV summaries and figures (committed)
docs/                           plan and reports
```

## Setup

```
uv sync --group dev          # local (CPU) environment
uv sync --extra gpu          # on the Linux GPU box
```

`HF_TOKEN` must be set in the environment (read access to gated Gemma for the StrongREJECT
evaluator). Never commit tokens.
