"""Batched chat generation with n samples per prompt, on vLLM (GPU) or transformers (CPU dry runs).

Both backends return, per prompt, a list of n completions plus a `truncated` flag per completion.
Adapters (LoRA) are attached by directory path; None means the base model.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class GenConfig:
    n: int = 1
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int = 2048
    seed: int = 0
    max_model_len: int = 4096


@dataclass
class Completion:
    text: str
    truncated: bool
    n_tokens: int


class Generator:
    def generate(self, conversations: list[list[dict]], cfg: GenConfig) -> list[list[Completion]]:
        raise NotImplementedError


class VLLMGenerator(Generator):
    def __init__(self, model: str, adapter: str | None = None, gpu_memory_utilization: float = 0.85,
                 max_model_len: int = 4096, max_lora_rank: int = 32):
        from vllm import LLM

        self.adapter = adapter
        self.llm = LLM(
            model=model,
            dtype="bfloat16",
            enable_lora=adapter is not None,
            max_lora_rank=max_lora_rank if adapter is not None else 16,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            enable_prefix_caching=True,
            seed=0,
        )
        self._lora = None
        if adapter is not None:
            from vllm.lora.request import LoRARequest

            self._lora = LoRARequest("adapter", 1, adapter)

    def generate(self, conversations, cfg: GenConfig):
        from vllm import SamplingParams

        sp = SamplingParams(n=cfg.n, temperature=cfg.temperature, top_p=cfg.top_p,
                            max_tokens=cfg.max_tokens, seed=cfg.seed)
        outs = self.llm.chat(conversations, sampling_params=sp, lora_request=self._lora, use_tqdm=True)
        result = []
        for o in outs:
            comps = []
            for c in o.outputs:
                comps.append(Completion(text=c.text, truncated=(c.finish_reason == "length"),
                                        n_tokens=len(c.token_ids)))
            result.append(comps)
        return result


class HFGenerator(Generator):
    """Slow transformers backend for CPU dry runs and tiny models. Not for real experiments."""

    def __init__(self, model: str, adapter: str | None = None, device: str | None = None, batch_size: int = 4):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.tok = AutoTokenizer.from_pretrained(model, padding_side="left")
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        m = AutoModelForCausalLM.from_pretrained(model, dtype=dtype)
        if adapter is not None:
            from peft import PeftModel

            m = PeftModel.from_pretrained(m, adapter).merge_and_unload()
        self.model = m.to(self.device).eval()
        self.batch_size = batch_size

    def generate(self, conversations, cfg: GenConfig):
        import torch

        torch.manual_seed(cfg.seed)
        texts = [self.tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True) for c in conversations]
        result: list[list[Completion]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            enc = self.tok(batch, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                out = self.model.generate(
                    **enc, do_sample=cfg.temperature > 0, temperature=max(cfg.temperature, 1e-5),
                    top_p=cfg.top_p, max_new_tokens=cfg.max_tokens, num_return_sequences=cfg.n,
                    pad_token_id=self.tok.pad_token_id,
                )
            gen = out[:, enc["input_ids"].shape[1]:]
            for b in range(len(batch)):
                comps = []
                for j in range(cfg.n):
                    ids = gen[b * cfg.n + j]
                    n_tok = int((ids != self.tok.pad_token_id).sum())
                    text = self.tok.decode(ids, skip_special_tokens=True)
                    comps.append(Completion(text=text, truncated=(n_tok >= cfg.max_tokens), n_tokens=n_tok))
                result.append(comps)
        return result


def make_generator(backend: str, model: str, adapter: str | None = None, **kw) -> Generator:
    if backend == "vllm":
        return VLLMGenerator(model, adapter, **kw)
    if backend == "hf":
        return HFGenerator(model, adapter, **{k: v for k, v in kw.items() if k in ("device", "batch_size")})
    raise ValueError(f"unknown backend {backend!r}")


def default_backend() -> str:
    try:
        import vllm  # noqa: F401

        return "vllm"
    except ImportError:
        return "hf"


def resolve_model(name: str) -> str:
    """Allow short names; keep HF ids otherwise."""
    aliases = {
        "7b": "Qwen/Qwen2.5-7B-Instruct",
        "3b": "Qwen/Qwen2.5-3B-Instruct",
        "0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    }
    return aliases.get(name.lower(), name)


def outputs_dir() -> str:
    return os.environ.get("RLVR_OUTPUTS", "outputs")
