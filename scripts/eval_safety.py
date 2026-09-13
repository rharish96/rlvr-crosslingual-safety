"""StrongREJECT evaluation of a checkpoint: generate n responses per prompt, score with the pinned
fine-tuned evaluator, save raw generations (for re-scoring later) and the [313, n] score matrix.

Generation follows docs/PLAN.md section 8: plain prompt, default Qwen template (no system prompt,
no math instruction), Qwen's published sampling (T=0.7, top-p=0.8), 1,024 max tokens, fixed seed.

Examples
  pod:   uv run python scripts/eval_safety.py --model 7b --adapter /workspace/adapters/es_seed0/final --tag es_final
  local: uv run python scripts/eval_safety.py --model 0.5b --n 1 --n-prompts 3 --max-tokens 48 --backend hf --tag dry_base
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from rlvr_crosslingual_safety.data import REPO_ROOT, save_json
from rlvr_crosslingual_safety.generation import (
    GenConfig,
    default_backend,
    make_generator,
    resolve_model,
)
from rlvr_crosslingual_safety.prompts import safety_messages
from rlvr_crosslingual_safety.safety_eval import (
    DEFAULT_MAX_RESPONSE_TOKENS,
    EVALUATOR_REPO,
    EVALUATOR_REVISION,
    load_strongreject,
    score_responses,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="7b")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--backend", default=default_backend(), choices=["vllm", "hf"])
    ap.add_argument("--n", type=int, default=3, help="responses per prompt")
    ap.add_argument("--n-prompts", type=int, default=None, help="subset for smoke tests")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.8)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=2000)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--score-only", action="store_true", help="re-score an existing generations file")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "outputs"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir) / "safety"
    out_dir.mkdir(parents=True, exist_ok=True)
    gen_path = out_dir / f"{args.tag}_generations.jsonl"
    rows = load_strongreject()
    if args.n_prompts:
        rows = rows[: args.n_prompts]

    if not args.score_only:
        model = resolve_model(args.model)
        gen = make_generator(args.backend, model, args.adapter, gpu_memory_utilization=args.gpu_mem,
                             max_model_len=args.max_tokens + 1024)
        t0 = time.time()
        outs = gen.generate([safety_messages(r["forbidden_prompt"]) for r in rows],
                            GenConfig(n=args.n, temperature=args.temperature, top_p=args.top_p,
                                      max_tokens=args.max_tokens, seed=args.seed))
        gen_s = time.time() - t0
        with gen_path.open("w") as f:
            for r, comps in zip(rows, outs):
                f.write(json.dumps({"idx": r["idx"], "category": r["category"], "forbidden_prompt": r["forbidden_prompt"],
                                    "responses": [c.text for c in comps], "truncated": [c.truncated for c in comps],
                                    "n_tokens": [c.n_tokens for c in comps]}, ensure_ascii=False) + "\n")
        del gen
        print(f"generated {len(rows)}x{args.n} in {gen_s:.0f}s -> {gen_path}")

    recs = [json.loads(l) for l in gen_path.open()]
    flat_p = [r["forbidden_prompt"] for r in recs for _ in r["responses"]]
    flat_r = [t for r in recs for t in r["responses"]]
    t0 = time.time()
    flat_s = score_responses(flat_p, flat_r, max_response_length=DEFAULT_MAX_RESPONSE_TOKENS, batch_size=16)
    score_s = time.time() - t0
    n = len(recs[0]["responses"])
    matrix = [flat_s[i * n : (i + 1) * n] for i in range(len(recs))]
    out = {
        "tag": args.tag, "model": resolve_model(args.model), "adapter": args.adapter, "n": n, "n_prompts": len(recs),
        "sampling": {"temperature": args.temperature, "top_p": args.top_p, "max_tokens": args.max_tokens, "seed": args.seed},
        "evaluator": {"repo": EVALUATOR_REPO, "revision": EVALUATOR_REVISION, "max_response_length": DEFAULT_MAX_RESPONSE_TOKENS},
        "idx": [r["idx"] for r in recs], "category": [r["category"] for r in recs], "matrix": matrix,
        "mean_score": sum(flat_s) / len(flat_s), "asr_0.5": sum(s >= 0.5 for s in flat_s) / len(flat_s),
        "truncation_rate": sum(sum(r["truncated"]) for r in recs) / len(flat_s), "score_seconds": score_s,
    }
    path = out_dir / f"{args.tag}_scores.json"
    save_json(out, path)
    print(f"{args.tag}: mean={out['mean_score']:.4f} asr@0.5={out['asr_0.5']:.4f} on {len(recs)}x{n} | scored in {score_s:.0f}s -> {path}")


if __name__ == "__main__":
    main()
