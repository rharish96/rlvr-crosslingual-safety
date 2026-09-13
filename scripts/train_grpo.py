"""GRPO + LoRA training for one arm (es or en) with TRL.

Recipe (docs/PLAN.md section 6): binary Math-Verify reward vs English gold, DAPO loss, beta=0,
no reward std-scaling, 16 prompts x 8 rollouts per optimizer step, T=1.0, 2,048 completion tokens,
LoRA r=32/alpha=64 on all linear layers, LR 1e-5, adapters at the midpoint and end.

Examples
  pod:   uv run python scripts/train_grpo.py --lang es --pool data/processed/pool_es_Qwen2.5-7B-Instruct.json \
           --model 7b --steps 250 --out /workspace/adapters/es_seed0
  local: uv run python scripts/train_grpo.py --lang es --pool data/processed/pool_es_Qwen2.5-0.5B-Instruct.json \
           --model 0.5b --steps 1 --prompts-per-step 2 --num-generations 2 --max-completion 48 --no-vllm --cpu
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from rlvr_crosslingual_safety.generation import resolve_model
from rlvr_crosslingual_safety.prompts import build_math_dataset, load_pool
from rlvr_crosslingual_safety.reward import reward


def correctness_reward(completions, solution, **kwargs) -> list[float]:
    """TRL reward function: completions are conversational ([{'role','content'}]) -> use the text."""
    texts = [c[0]["content"] if isinstance(c, list) else str(c) for c in completions]
    return [reward(t, g) for t, g in zip(texts, solution)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=["es", "en"], required=True)
    ap.add_argument("--pool", required=True, help="pool_*.json from screen.py (problem IDs)")
    ap.add_argument("--model", default="7b")
    ap.add_argument("--out", required=True, help="output dir for adapters/checkpoints")
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--save-steps", type=int, default=None, help="default: steps // 2 (midpoint) and end")
    ap.add_argument("--prompts-per-step", type=int, default=16)
    ap.add_argument("--num-generations", type=int, default=8)
    ap.add_argument("--micro-batch", type=int, default=16, help="completions per forward/backward pass")
    ap.add_argument("--max-completion", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-vllm", action="store_true")
    ap.add_argument("--vllm-gpu-mem", type=float, default=0.35)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--report-to", default="trackio")
    ap.add_argument("--resume", default=None, help="checkpoint dir to resume from")
    args = ap.parse_args()

    import torch
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer

    model = resolve_model(args.model)
    ids = load_pool(Path(args.pool))
    ds = build_math_dataset(args.lang, ids)
    print(f"arm={args.lang} model={model} pool={len(ds)} steps={args.steps}")

    completions_per_step = args.prompts_per_step * args.num_generations
    if completions_per_step % args.micro_batch:
        raise SystemExit("prompts_per_step * num_generations must be divisible by micro_batch")
    grad_accum = completions_per_step // args.micro_batch
    save_steps = args.save_steps or max(args.steps // 2, 1)
    use_cuda = torch.cuda.is_available() and not args.cpu

    cfg = GRPOConfig(
        output_dir=args.out,
        run_name=Path(args.out).name,
        seed=args.seed,
        # batch geometry: one generation batch of prompts_per_step x num_generations per optimizer step
        per_device_train_batch_size=args.micro_batch,
        gradient_accumulation_steps=grad_accum,
        num_generations=args.num_generations,
        max_steps=args.steps,
        # sampling
        temperature=args.temperature,
        top_p=1.0,
        max_completion_length=args.max_completion,
        # objective (PLAN section 6)
        loss_type="dapo",
        beta=0.0,
        scale_rewards="none",
        mask_truncated_completions=False,  # truncated -> reward 0 via the reward function
        num_iterations=1,
        # optimizer
        learning_rate=args.lr,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=min(10, max(args.steps // 10, 1)),
        max_grad_norm=1.0,
        weight_decay=0.0,
        bf16=use_cuda,
        gradient_checkpointing=use_cuda,
        use_cpu=not use_cuda,
        # generation backend
        use_vllm=use_cuda and not args.no_vllm,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=args.vllm_gpu_mem,
        vllm_max_model_length=args.max_completion + 1024,
        # logging / saving
        logging_steps=1,
        log_completions=True,
        num_completions_to_print=1,
        report_to=args.report_to if args.report_to != "none" else "none",
        save_strategy="steps",
        save_steps=save_steps,
        save_only_model=False,  # keep optimizer/scheduler/RNG so an extension can resume exactly
        remove_unused_columns=False,
        shuffle_dataset=True,
    )
    peft_cfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0,
                          target_modules="all-linear", task_type="CAUSAL_LM")

    trainer = GRPOTrainer(model=model, reward_funcs=correctness_reward, args=cfg,
                          train_dataset=ds, peft_config=peft_cfg)
    trainer.train(resume_from_checkpoint=args.resume)

    final_dir = Path(args.out) / "final"
    trainer.save_model(str(final_dir))
    meta = {
        "arm": args.lang, "model": model, "pool": args.pool, "n_pool": len(ds), "steps": args.steps,
        "prompts_per_step": args.prompts_per_step, "num_generations": args.num_generations,
        "max_completion": args.max_completion, "lr": args.lr, "lora_r": args.lora_r, "lora_alpha": args.lora_alpha,
        "loss_type": "dapo", "beta": 0.0, "scale_rewards": "none", "temperature": args.temperature,
        "seed": args.seed, "use_vllm": cfg.use_vllm, "save_steps": save_steps,
    }
    (final_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))
    # export the trainer's per-step log history (reward curve) for the report
    (Path(args.out) / "log_history.json").write_text(json.dumps(trainer.state.log_history, indent=1))
    print(f"saved final adapter -> {final_dir}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
