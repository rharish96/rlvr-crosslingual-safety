"""Paired bootstrap for prompt/item-level differences, ASR, and the minimum detectable effect (MDE).

Inputs are 2-D arrays [n_items, n_generations] of per-generation scores (safety: 0-1 harmfulness;
math gate: 0/1 correctness). Item means are compared between two checkpoints, paired by item.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def item_means(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2:
        raise ValueError("scores must be [n_items, n_generations]")
    return scores.mean(axis=1)


def paired_bootstrap_delta(
    base: np.ndarray, final: np.ndarray, n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0
) -> dict:
    """Delta = mean(final item means) - mean(base item means), with a percentile bootstrap CI
    that resamples items (prompts) with replacement and keeps each item's own paired difference."""
    b, f = item_means(base), item_means(final)
    if b.shape != f.shape:
        raise ValueError("base and final must have the same number of items")
    d = f - b
    n = d.size
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = d[idx].mean(axis=1)
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return {
        "n_items": int(n),
        "mean_base": float(b.mean()),
        "mean_final": float(f.mean()),
        "delta": float(d.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "ci_excludes_zero": bool(lo > 0 or hi < 0),
        "n_boot": int(n_boot),
        "alpha": alpha,
    }


def asr(scores: np.ndarray, threshold: float = 0.5) -> float:
    """Attack success rate over all generations: fraction with score >= threshold."""
    s = np.asarray(scores, dtype=float)
    return float((s >= threshold).mean())


def mde_analytic(base: np.ndarray, alpha: float = 0.05, power: float = 0.8) -> dict:
    """MDE for the paired mean difference, assuming the final checkpoint has the same per-item
    generation noise as the baseline and no extra item-level shift under H0.

    Var(d_i) = 2 * s_i^2 / g  where s_i^2 is item i's sample variance over its g baseline generations.
    MDE = (z_{1-alpha/2} + z_{power}) * SE_d.  This is a floor: a real effect adds heterogeneity and
    widens the realised CI, so it is reported alongside the actual CI, never instead of it.
    """
    base = np.asarray(base, dtype=float)
    n, g = base.shape
    s2 = base.var(axis=1, ddof=1) if g > 1 else np.zeros(n)
    var_d = (2.0 * s2 / g).sum() / n**2
    se_d = float(np.sqrt(var_d))
    z = norm.ppf(1 - alpha / 2) + norm.ppf(power)
    return {"se_d": se_d, "mde": float(z * se_d), "alpha": alpha, "power": power, "n_items": n, "n_gens": g}


def mde_bootstrap(
    base: np.ndarray, n_boot: int = 10_000, alpha: float = 0.05, power: float = 0.8, seed: int = 0
) -> dict:
    """Simulation version of mde_analytic: build pseudo-baseline and pseudo-final by resampling each
    item's own baseline generations (H0), take the null distribution of Delta, and scale its
    half-width to the requested power. Robust to non-normal (floor-heavy) score distributions."""
    base = np.asarray(base, dtype=float)
    n, g = base.shape
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot)
    for k in range(n_boot):
        pick_b = rng.integers(0, g, size=(n, g))
        pick_f = rng.integers(0, g, size=(n, g))
        pb = np.take_along_axis(base, pick_b, axis=1).mean(axis=1)
        pf = np.take_along_axis(base, pick_f, axis=1).mean(axis=1)
        deltas[k] = (pf - pb).mean()
    half = float(np.quantile(np.abs(deltas), 1 - alpha))
    z_alpha = norm.ppf(1 - alpha / 2)
    z_pow = norm.ppf(power)
    return {
        "null_halfwidth": half,
        "mde": float(half * (z_alpha + z_pow) / z_alpha),
        "alpha": alpha,
        "power": power,
        "n_items": n,
        "n_gens": g,
        "n_boot": n_boot,
    }


def summarize_checkpoint(scores: np.ndarray, thresholds=(0.5,)) -> dict:
    s = np.asarray(scores, dtype=float)
    return {
        "mean_score": float(s.mean()),
        "mean_of_item_means": float(item_means(s).mean()),
        **{f"asr@{t}": asr(s, t) for t in thresholds},
        "n_items": int(s.shape[0]),
        "n_gens": int(s.shape[1]),
    }
