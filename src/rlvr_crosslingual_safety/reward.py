"""Binary correctness reward: 1 iff the last \\boxed{...} in the completion matches the English gold.

Uses Math-Verify (huggingface/Math-Verify) for symbolic/numeric equivalence. Identical for both arms.
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


import re

# Pure-number outputs written with locale grouping/decimal marks. Applied to the boxed string only,
# identically in both arms. Decimal-point and comma-thousands golds are excluded from the pool, so
# these rewrites can only turn a locale-formatted correct integer/decimal into a parseable one.
_THOUSANDS_DOT_RE = re.compile(r"^-?\d{1,3}(?:\.\d{3})+$")  # 10.500  2.177.280
_THOUSANDS_COMMA_RE = re.compile(r"^-?\d{1,3}(?:,\d{3})+$")  # 10,500  (en grouping; Math-Verify native)
_THOUSANDS_THIN_RE = re.compile(r"^-?\d{1,3}(?:(?:\\,|\\;|\\ |\s|\u202f|\u00a0)\d{3})+$")  # 10\,500  10 500
_DECIMAL_COMMA_RE = re.compile(r"^-?\d+(?:\{,\}|,)\d+$")  # 3,5  3{,}5  0,25


def normalize_boxed(boxed: str, gold: str) -> str:
    s = boxed.strip()
    if _THOUSANDS_DOT_RE.match(s):
        return s.replace(".", "")
    if _THOUSANDS_THIN_RE.match(s):
        return re.sub(r"\\[,; ]|\s|\u202f|\u00a0", "", s)
    if _THOUSANDS_COMMA_RE.match(s):
        # "10,500": ambiguous between en grouping (10500) and es decimal (10.5). Decimal golds are
        # excluded from the pool, so only the integer reading can be correct; leave it to Math-Verify.
        return s
    if _DECIMAL_COMMA_RE.match(s) and "," not in gold:
        # gold has no comma, so it is not a tuple/list; read the comma as a decimal mark
        return s.replace("{,}", ".").replace(",", ".")
    return s


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
    ans_parsed = parse(_wrap(normalize_boxed(boxed, gold)), extraction_config=_EXTRACTION)
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
