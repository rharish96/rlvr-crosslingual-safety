"""Math gate evaluation: avg@k accuracy of a checkpoint on the kept parallel test items.

Writes outputs/eval_math/<tag>.json with the [n_items, k] correctness matrix plus language/truncation stats.

Examples
  pod:   uv run python scripts/eval_math.py --lang es --model 7b --adapter /workspace/adapters/es_seed0/final --tag es_final
  local: uv run python scripts/eval_math.py --lang es --model 0.5b --k 2 --n-items 4 --max-tokens 64 --backend hf --tag dry_base
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from rlvr_crosslingual_safety.data import REPO_ROOT, save_json
from rlvr_crosslingual_safety.generation import (
    GenConfig,
    default_backend,
    make_generator,
    resolve_model,
)
from rlvr_crosslingual_safety.langid import language_shares
from rlvr_crosslingual_safety.prompts import build_math_dataset, test_ids
from rlvr_crosslingual_safety.reward import extract_last_boxed, reward


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=["es", "en"], required=True)
    ap.add_argument("--model", default="7b")
    ap.add_argument("--adapter", default=None, help="LoRA adapter dir; omit for the base model")
    ap.add_argument("--backend", default=default_backend(), choices=["vllm", "hf"])
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--n-items", type=int, default=None, help="subset for smoke tests")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "outputs"))
    args = ap.parse_args()

    model = resolve_model(args.model)
    ids = test_ids()[: args.n_items] if args.n_items else test_ids()
    ds = build_math_dataset(args.lang, ids, split="test")
    gen = make_generator(args.backend, model, args.adapter, gpu_memory_utilization=args.gpu_mem,
                         max_model_len=args.max_tokens + 1024)
    t0 = time.time()
    outs = gen.generate(list(ds["prompt"]), GenConfig(n=args.k, temperature=1.0, max_tokens=args.max_tokens, seed=args.seed))
    elapsed = time.time() - t0

    matrix, texts, n_trunc, n_boxed, tok = [], [], 0, 0, 0
    samples = []
    for row, comps in zip(ds, outs):
        matrix.append([reward(c.text, row["solution"]) for c in comps])
        for c in comps:
            texts.append(c.text); n_trunc += c.truncated; n_boxed += extract_last_boxed(c.text) is not None; tok += c.n_tokens
        samples.append({"original_idx": row["original_idx"], "solution": row["solution"], "completion": comps[0].text})

    n = len(matrix) * args.k
    acc = sum(map(sum, matrix)) / n
    out = {
        "tag": args.tag, "lang": args.lang, "model": model, "adapter": args.adapter, "k": args.k,
        "n_items": len(matrix), "avg_at_k": acc, "truncation_rate": n_trunc / n, "boxed_parse_rate": n_boxed / n,
        "mean_tokens": tok / n, "language_shares": language_shares(texts), "gen_seconds": elapsed,
        "ids": list(ds["original_idx"]), "matrix": matrix, "samples": samples[:5],
    }
    path = Path(args.out_dir) / "eval_math" / f"{args.tag}.json"
    save_json(out, path)
    print(f"{args.tag}: avg@{args.k}={acc:.4f} on {len(matrix)} items | trunc={n_trunc/n:.3f} boxed={n_boxed/n:.3f} "
          f"tokens={tok/n:.0f} | lang={out['language_shares']} | {elapsed:.0f}s -> {path}")


if __name__ == "__main__":
    main()
