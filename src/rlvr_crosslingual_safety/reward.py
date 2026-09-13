"""Binary correctness reward: 1 iff the last \\boxed{...} in the completion matches the English gold.

Plain Math-Verify (huggingface/Math-Verify) equivalence, no custom normalization. Locale-sensitive
golds (decimals, comma-grouped numbers, integers >= 1000, fractions) are excluded from the pool
upstream (see data.DROP_RULES) so Spanish number formatting cannot produce false negatives.
Identical for both arms.
"""

from __future__ import annotations

from functools import cache

from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify

_EXTRACTION = [LatexExtractionConfig(), ExprExtractionConfig()]


def extract_last_boxed(text: str) -> str | None:
    """Return the content of the last \\boxed{...} (brace-balanced), or None if absent."""
    start = text.rfind("\\boxed{")
    if start == -1:
        return None
    i = start + len("\\boxed{")
    depth = 1
    j = i
    while j < len(text) and depth > 0:
        c = text[j]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        j += 1
    if depth != 0:
        return None  # unbalanced (e.g. truncated completion)
    return text[i : j - 1].strip()


def _wrap(s: str) -> str:
    s = s.strip()
    if s.startswith(("$", "\\(", "\\[")):
        return s
    return f"${s}$"


@cache
def parse_gold(gold: str):
    return parse(_wrap(gold), extraction_config=_EXTRACTION)


def is_correct(completion: str, gold: str) -> bool:
    boxed = extract_last_boxed(completion)
    if boxed is None or boxed == "":
        return False
    gold_parsed = parse_gold(gold)
    if not gold_parsed:
        return False
    ans_parsed = parse(_wrap(boxed), extraction_config=_EXTRACTION)
    if not ans_parsed:
        return False
    try:
        return bool(verify(gold_parsed, ans_parsed))
    except Exception:  # noqa: BLE001 - sympy/timeout errors must never crash a training step
        return False


def reward(completion: str, gold: str) -> float:
    """1.0 if correct else 0.0. Truncated completions have no balanced \\boxed{} and score 0."""
    return 1.0 if is_correct(completion, gold) else 0.0


def batch_reward(completions: list[str], golds: list[str]) -> list[float]:
    return [reward(c, g) for c, g in zip(completions, golds)]
