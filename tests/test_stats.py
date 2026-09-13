import numpy as np

from rlvr_crosslingual_safety.stats import (
    asr,
    item_means,
    mde_analytic,
    mde_bootstrap,
    paired_bootstrap_delta,
)


def _floor_baseline(rng, n=313, g=3):
    # 95% of prompts near 0, 5% occasionally leaky: mimics an aligned instruct model
    leaky = rng.random(n) < 0.05
    p = np.where(leaky, 0.3, 0.01)
    return rng.beta(p[:, None] * 20 + 1e-3, (1 - p[:, None]) * 20, size=(n, g)).clip(0, 1)


def test_item_means_shape():
    x = np.arange(12).reshape(4, 3)
    assert item_means(x).tolist() == [1.0, 4.0, 7.0, 10.0]


def test_asr():
    assert asr(np.array([[0.1, 0.6], [0.9, 0.0]])) == 0.5


def test_bootstrap_null_covers_zero():
    rng = np.random.default_rng(1)
    base = _floor_baseline(rng)
    final = _floor_baseline(rng)  # same distribution, independent draw
    r = paired_bootstrap_delta(base, final, n_boot=2000, seed=1)
    assert r["ci_low"] <= 0.0 <= r["ci_high"]


def test_bootstrap_detects_shift():
    rng = np.random.default_rng(2)
    base = _floor_baseline(rng)
    final = base.copy()
    hit = rng.random(base.shape[0]) < 0.15  # 15% of prompts become compliant
    final[hit] = rng.uniform(0.5, 1.0, size=(hit.sum(), base.shape[1]))
    r = paired_bootstrap_delta(base, final, n_boot=2000, seed=2)
    true_delta = (item_means(final) - item_means(base)).mean()
    assert r["ci_excludes_zero"]
    assert r["ci_low"] <= true_delta <= r["ci_high"]
    assert r["delta"] > 0.05


def test_mde_is_small_at_floor_and_consistent():
    rng = np.random.default_rng(3)
    base = _floor_baseline(rng)
    a = mde_analytic(base)
    b = mde_bootstrap(base, n_boot=1000, seed=3)
    assert 0 < a["mde"] < 0.05
    assert 0 < b["mde"] < 0.05
    # the two estimators should agree within a factor of 2 on this smooth-ish distribution
    assert 0.5 < a["mde"] / b["mde"] < 2.0


def test_mde_zero_generation_variance():
    base = np.zeros((10, 3))
    assert mde_analytic(base)["mde"] == 0.0
