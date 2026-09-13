"""Assemble the per-arm report: math gate (paired bootstrap), safety Δ with CI, ASR, MDE, trajectory plot.

Example
  uv run python scripts/report.py --arm es \
     --math-base outputs/eval_math/base.json --math-mid outputs/eval_math/es_mid.json --math-final outputs/eval_math/es_final.json \
     --safety-base outputs/safety/base_scores.json --safety-mid outputs/safety/es_mid_scores.json --safety-final outputs/safety/es_final_scores.json \
     --log-history /workspace/adapters/es_seed0/log_history.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rlvr_crosslingual_safety.data import REPO_ROOT, save_json
from rlvr_crosslingual_safety.stats import (
    mde_analytic,
    mde_bootstrap,
    paired_bootstrap_delta,
    summarize_checkpoint,
)


def _load(p):
    return json.loads(Path(p).read_text()) if p else None


def _cat_breakdown(base, final):
    cats = sorted(set(base["category"]))
    out = {}
    b, f = np.array(base["matrix"]), np.array(final["matrix"])
    for c in cats:
        m = np.array([x == c for x in base["category"]])
        r = paired_bootstrap_delta(b[m], f[m], n_boot=4000)
        out[c] = {"n": int(m.sum()), "delta": r["delta"], "ci": [r["ci_low"], r["ci_high"]]}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--math-base", required=True); ap.add_argument("--math-mid"); ap.add_argument("--math-final", required=True)
    ap.add_argument("--safety-base", required=True); ap.add_argument("--safety-mid"); ap.add_argument("--safety-final", required=True)
    ap.add_argument("--log-history", default=None)
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "results"))
    args = ap.parse_args()

    mb, mm, mf = _load(args.math_base), _load(args.math_mid), _load(args.math_final)
    sb, sm, sf = _load(args.safety_base), _load(args.safety_mid), _load(args.safety_final)
    assert mb["ids"] == mf["ids"] and sb["idx"] == sf["idx"], "checkpoints must be evaluated on identical items"

    # --- math gate ---
    gate = paired_bootstrap_delta(np.array(mb["matrix"]), np.array(mf["matrix"]))
    reward_curve = []
    if args.log_history:
        for rec in _load(args.log_history):
            if "reward" in rec and "step" in rec:
                reward_curve.append({"step": rec["step"], "reward": rec["reward"]})
    curve_rose = None
    if len(reward_curve) >= 10:
        r = np.array([x["reward"] for x in reward_curve]); q = len(r) // 5
        curve_rose = bool(r[-q:].mean() > r[:q].mean())
    gate_pass = bool(gate["ci_excludes_zero"] and gate["delta"] > 0 and (curve_rose is not False))

    # --- safety ---
    b, f = np.array(sb["matrix"]), np.array(sf["matrix"])
    delta = paired_bootstrap_delta(b, f)
    mde_a, mde_b = mde_analytic(b), mde_bootstrap(b, n_boot=3000)
    ckpts = {"baseline": summarize_checkpoint(b), "final": summarize_checkpoint(f)}
    if sm:
        ckpts["midpoint"] = summarize_checkpoint(np.array(sm["matrix"]))

    report = {
        "arm": args.arm,
        "math_gate": {**gate, "avg_at_k_base": mb["avg_at_k"], "avg_at_k_final": mf["avg_at_k"],
                      "avg_at_k_mid": mm["avg_at_k"] if mm else None, "k": mf["k"], "n_items": mf["n_items"],
                      "reward_curve_rose": curve_rose, "pass": gate_pass,
                      "language_shares_final": mf["language_shares"], "truncation_rate_final": mf["truncation_rate"]},
        "safety": {"delta": delta, "mde_analytic": mde_a, "mde_bootstrap": mde_b, "checkpoints": ckpts,
                   "by_category_exploratory": _cat_breakdown(sb, sf)},
        "inputs": {k: v for k, v in vars(args).items() if k not in ("out_dir",)},
    }
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    save_json(report, out_dir / f"report_{args.arm}.json")

    # --- plots ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
        if reward_curve:
            axes[0].plot([x["step"] for x in reward_curve], [x["reward"] for x in reward_curve], lw=1)
        axes[0].set_title("training reward"); axes[0].set_xlabel("step")
        xs = ["baseline"] + (["midpoint"] if mm else []) + ["final"]
        axes[1].plot(xs, [mb["avg_at_k"]] + ([mm["avg_at_k"]] if mm else []) + [mf["avg_at_k"]], "o-")
        axes[1].set_title(f"{args.arm} test avg@{mf['k']} (n={mf['n_items']})")
        axes[2].plot(xs, [ckpts["baseline"]["mean_score"]] + ([ckpts["midpoint"]["mean_score"]] if sm else []) + [ckpts["final"]["mean_score"]], "o-", color="C3")
        axes[2].set_title("StrongREJECT mean score"); axes[2].set_ylim(bottom=0)
        fig.tight_layout(); fig.savefig(out_dir / f"report_{args.arm}.png", dpi=130)
    except Exception as e:  # noqa: BLE001
        print("plot skipped:", e)

    print(json.dumps({"math_gate": {k: report["math_gate"][k] for k in ("avg_at_k_base", "avg_at_k_final", "delta", "ci_low", "ci_high", "pass")},
                      "safety": {"delta": delta["delta"], "ci": [delta["ci_low"], delta["ci_high"]],
                                 "mean_base": ckpts["baseline"]["mean_score"], "mean_final": ckpts["final"]["mean_score"],
                                 "asr_base": ckpts["baseline"]["asr@0.5"], "asr_final": ckpts["final"]["asr@0.5"],
                                 "mde_analytic": mde_a["mde"], "mde_bootstrap": mde_b["mde"]}}, indent=1))
    print(f"-> {out_dir / f'report_{args.arm}.json'}")


if __name__ == "__main__":
    main()
