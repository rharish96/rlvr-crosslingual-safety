"""Response-language detection after stripping math (langdetect, as in GRPO Beyond English)."""

from __future__ import annotations

import re

from langdetect import DetectorFactory, LangDetectException, detect_langs

DetectorFactory.seed = 0  # deterministic

_MATH_PATTERNS = [
    r"\$\$.*?\$\$",  # display math
    r"\$[^$]*\$",  # inline math
    r"\\\[.*?\\\]",  # \[ ... \]
    r"\\\(.*?\\\)",  # \( ... \)
    r"\\boxed\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}",  # \boxed{...} (one nesting level)
    r"\\[a-zA-Z]+(\{[^{}]*\})*",  # \command{...}
    r"```.*?```",  # code blocks
]
_MATH_RE = re.compile("|".join(_MATH_PATTERNS), flags=re.DOTALL)
_NUMERIC_RE = re.compile(r"[\d.,:%/*+\-=<>^_()\[\]{}|\\~`'\"°·×÷±≈≠≤≥∞√∑∏∫]+")
_WS_RE = re.compile(r"\s+")

MIN_CHARS = 20


def strip_math(text: str) -> str:
    text = _MATH_RE.sub(" ", text)
    text = _NUMERIC_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def detect_language(text: str) -> tuple[str | None, float]:
    """Return (lang_code, prob) of the dominant language of the prose, or (None, 0.0) if too short."""
    prose = strip_math(text)
    if len(prose) < MIN_CHARS:
        return None, 0.0
    try:
        best = detect_langs(prose)[0]
        return best.lang, float(best.prob)
    except LangDetectException:
        return None, 0.0


def language_shares(texts: list[str]) -> dict[str, float]:
    """Fraction of texts detected as each language ('unknown' for undetectable)."""
    counts: dict[str, int] = {}
    for t in texts:
        lang, _ = detect_language(t)
        key = lang or "unknown"
        counts[key] = counts.get(key, 0) + 1
    n = max(len(texts), 1)
    return {k: v / n for k, v in sorted(counts.items(), key=lambda kv: -kv[1])}
