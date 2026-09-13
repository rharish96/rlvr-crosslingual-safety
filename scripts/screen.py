"""Screen candidate training problems: k baseline samples each, keep those solved lo..hi times.

Also reports \\boxed{} parse rate, truncation rate, tokens per completion and response-language shares,
which are the go/no-go inputs for the pipeline test.

Examples
  pod:   uv run python scripts/screen.py --lang es --model 7b --n-ids 4000 --k 8
  local: uv run python scripts/screen.py --lang es --model 0.5b --n-ids 6 --k 2 --max-tokens 64 --backend hf
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from rlvr_crosslingual_safety.data import PROCESSED_DIR, REPO_ROOT, save_json
from rlvr_crosslingual_safety.generation import (
    GenConfig,
    default_backend,
    make_generator,
    resolve_model,
)
from rlvr_crosslingual_safety.langid import language_shares
from rlvr_crosslingual_safety.prompts import build_math_dataset, sample_pool_ids, save_pool
from rlvr_crosslingual_safety.reward import extract_last_boxed, reward


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=["es", "en"], required=True)
    ap.add_argument("--model", default="7b")
    ap.add_argument("--backend", default=default_backend(), choices=["vllm", "hf"])
    ap.add_argument("--n-ids", type=int, default=4000)
    ap.add_argument("--pool-seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--lo", type=int, default=2, help="min correct of k to keep")
    ap.add_argument("--hi", type=int, default=6, help="max correct of k to keep")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "outputs"))
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    model = resolve_model(args.model)
    tag = args.tag or f"{args.lang}_{Path(model).name}"
    out_dir = Path(args.out_dir) / "screen"
    out_dir.mkdir(parents=True, exist_ok=True)

    ids = sample_pool_ids(args.n_ids, args.pool_seed)
    ds = build_math_dataset(args.lang, ids)
    print(f"screening {len(ds)} {args.lang} problems x {args.k} samples with {model} [{args.backend}]")

    gen = make_generator(args.backend, model, None, gpu_memory_utilization=args.gpu_mem,
                         max_model_len=args.max_tokens + 1024)
    t0 = time.time()
    outs = gen.generate(list(ds["prompt"]), GenConfig(n=args.k, temperature=args.temperature,
                                                       max_tokens=args.max_tokens, seed=args.pool_seed))
    elapsed = time.time() - t0

    rows, kept = [], []
    n_comp = n_boxed = n_trunc = tok_sum = 0
    texts = []
    for row, comps in zip(ds, outs):
        correct = [reward(c.text, row["solution"]) for c in comps]
        n_correct = int(sum(correct))
        for c in comps:
            n_comp += 1
            n_boxed += extract_last_boxed(c.text) is not None
            n_trunc += c.truncated
            tok_sum += c.n_tokens
            texts.append(c.text)
        rows.append({"original_idx": row["original_idx"], "n_correct": n_correct, "k": args.k,
                     "solution": row["solution"], "sample": comps[0].text[:400]})
        if args.lo <= n_correct <= args.hi:
            kept.append(row["original_idx"])

    hist = {}
    for r in rows:
        hist[r["n_correct"]] = hist.get(r["n_correct"], 0) + 1
    stats = {
        "lang": args.lang, "model": model, "backend": args.backend, "n_problems": len(rows), "k": args.k,
        "lo": args.lo, "hi": args.hi, "n_kept": len(kept), "pass_rate_mean": sum(r["n_correct"] for r in rows) / (len(rows) * args.k),
        "hist_n_correct": dict(sorted(hist.items())),
        "boxed_parse_rate": n_boxed / n_comp, "truncation_rate": n_trunc / n_comp,
        "mean_tokens": tok_sum / n_comp, "max_tokens": args.max_tokens,
        "language_shares": language_shares(texts),
        "gen_seconds": elapsed, "completions_per_second": n_comp / max(elapsed, 1e-9),
        "tokens_per_second": tok_sum / max(elapsed, 1e-9),
    }
    with (out_dir / f"screen_{tag}.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    save_json(stats, out_dir / f"screen_{tag}_stats.json")
    save_pool(PROCESSED_DIR / f"pool_{tag}.json", kept, {"source": f"screen_{tag}", "lo": args.lo, "hi": args.hi, "k": args.k})
    print(json.dumps({k: v for k, v in stats.items() if k != "language_shares"}, indent=1))
    print("language shares:", stats["language_shares"])
    print(f"kept {len(kept)} -> {PROCESSED_DIR / f'pool_{tag}.json'}")


if __name__ == "__main__":
    main()
