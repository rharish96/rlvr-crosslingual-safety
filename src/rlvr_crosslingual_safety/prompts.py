"""Prompt construction for both arms, the math test set, and the safety evaluation.

Chat format only (list of messages). No system message: Qwen2.5's chat template inserts its own
default system prompt, which is the "ordinary use" setting the plan specifies.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from datasets import Dataset

from .data import PROCESSED_DIR, load_macereason

INSTRUCTION = {
    "es": "Razona paso a paso y escribe la respuesta final en \\boxed{}.",
    "en": "Reason step by step and put your final answer in \\boxed{}.",
}

DEFAULT_POOL_SEED = 0
DEFAULT_POOL_SIZE = 4000


def math_messages(problem: str, lang: str) -> list[dict]:
    return [{"role": "user", "content": f"{problem.strip()}\n\n{INSTRUCTION[lang]}"}]


def safety_messages(forbidden_prompt: str) -> list[dict]:
    return [{"role": "user", "content": forbidden_prompt.strip()}]


def load_parallel_ids() -> dict:
    return json.loads((PROCESSED_DIR / "parallel_ids.json").read_text())


def sample_pool_ids(n: int = DEFAULT_POOL_SIZE, seed: int = DEFAULT_POOL_SEED) -> list[int]:
    """Seeded sample of candidate training IDs from the kept parallel train pool (shared by both arms)."""
    kept = load_parallel_ids()["train"]["kept"]
    rng = random.Random(seed)
    return sorted(rng.sample(kept, min(n, len(kept))))


def build_math_dataset(lang: str, ids: list[int], split: str = "train") -> Dataset:
    """Rows: prompt (messages in `lang`), solution (ENGLISH gold), original_idx, lang."""
    ds_lang = load_macereason(lang, splits=(split,))[split]
    ds_en = ds_lang if lang == "en" else load_macereason("en", splits=(split,))[split]
    by_id_lang = {r["original_idx"]: r["problem"] for r in ds_lang}
    by_id_en = {r["original_idx"]: r["solution"] for r in ds_en}
    rows = [
        {
            "prompt": math_messages(by_id_lang[i], lang),
            "solution": by_id_en[i],
            "original_idx": i,
            "lang": lang,
        }
        for i in ids
    ]
    return Dataset.from_list(rows)


def test_ids() -> list[int]:
    return load_parallel_ids()["test"]["kept"]


def save_pool(path: Path, ids: list[int], meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"ids": ids, **meta}, indent=2))


def load_pool(path: Path) -> list[int]:
    return json.loads(Path(path).read_text())["ids"]
