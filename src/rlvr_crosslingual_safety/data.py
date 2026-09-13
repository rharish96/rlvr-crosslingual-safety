"""mAceReason-Math loading (pinned), English reconstruction, parallel alignment, gold filtering.

Dataset: Dobler et al., "mAceReason-Math" (arXiv:2603.10767), CC BY-NC-ND 4.0.
https://github.com/apple/ml-macereason-math
"""

from __future__ import annotations

import base64
import json
import re
import urllib.request
from collections.abc import Iterable
from pathlib import Path

import bsdiff4
from datasets import Dataset, load_dataset

MACEREASON_COMMIT = "a9b8d7e9d3ed30e2e5f4a6b2792211b566c8b676"
MACEREASON_RAW = f"https://raw.githubusercontent.com/apple/ml-macereason-math/{MACEREASON_COMMIT}"
ACEREASON_REVISION = "a5cc41c5ecfc1d4a6571f98ade92d7fec100b2a8"

LANGS = ("bn", "de", "en", "es", "fr", "it", "ja", "ko", "pt", "ru", "sw", "te", "th", "zh")
PARALLEL_SPLITS = ("train", "test")
EXPECTED_SIZES = {"train": 7620, "test": 190}

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"


def _download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        urllib.request.urlretrieve(url, dest)
    return dest


def _jsonl_path(lang_dir: str, split: str) -> Path:
    return RAW_DIR / "macereason" / MACEREASON_COMMIT / lang_dir / f"{split}.jsonl"


def load_macereason(lang: str, splits: Iterable[str] = PARALLEL_SPLITS) -> dict[str, Dataset]:
    """Load mAceReason-Math for one language at the pinned commit.

    For lang == "en", reconstruct the cleaned English problems/solutions by applying the
    released bsdiff4 patches to nvidia/AceReason-Math at the pinned revision.
    Returns {split: Dataset} with columns original_idx, problem, solution, english_has_been_cleaned.
    """
    if lang not in LANGS:
        raise ValueError(f"unknown lang {lang!r}; choose from {LANGS}")
    splits = tuple(splits)
    lang_dir = "en_modifications" if lang == "en" else lang
    files = {s: str(_download(f"{MACEREASON_RAW}/{lang_dir}/{s}.jsonl", _jsonl_path(lang_dir, s))) for s in splits}
    ds = load_dataset("json", data_files=files)

    if lang != "en":
        return {s: ds[s] for s in splits}

    original = load_dataset("nvidia/AceReason-Math", split="train", revision=ACEREASON_REVISION)

    def reconstruct(mod):
        orig = original[mod["original_idx"]]
        problem = orig["problem"]
        if mod["english_problem_modification"]:
            problem = bsdiff4.patch(
                problem.encode(), base64.b64decode(mod["english_problem_modification"])
            ).decode()
        solution = orig["answer"]
        if mod["english_solution_modification"]:
            solution = bsdiff4.patch(
                solution.encode(), base64.b64decode(mod["english_solution_modification"])
            ).decode()
        return {
            "original_idx": mod["original_idx"],
            "problem": problem,
            "solution": solution,
            "english_has_been_cleaned": mod["english_has_been_cleaned"],
        }

    out = ds.map(
        reconstruct,
        remove_columns=["english_problem_modification", "english_solution_modification"],
        desc="reconstruct en",
    )
    return {s: out[s] for s in splits}


def align_parallel(a: Dataset, b: Dataset) -> dict:
    """Check two parallel splits share exactly the same original_idx set. Returns a summary."""
    ids_a = list(a["original_idx"])
    ids_b = list(b["original_idx"])
    set_a, set_b = set(ids_a), set(ids_b)
    summary = {
        "n_a": len(ids_a),
        "n_b": len(ids_b),
        "unique_a": len(set_a),
        "unique_b": len(set_b),
        "only_in_a": sorted(set_a - set_b)[:10],
        "only_in_b": sorted(set_b - set_a)[:10],
        "same_id_set": set_a == set_b,
        "same_order": ids_a == ids_b,
    }
    if set_a == set_b:
        sol_a = dict(zip(ids_a, a["solution"]))
        sol_b = dict(zip(ids_b, b["solution"]))
        diff = [i for i in ids_a if sol_a[i].strip() != sol_b[i].strip()]
        summary["n_solution_string_mismatch"] = len(diff)
        summary["solution_mismatch_examples"] = [
            {"original_idx": i, "a": sol_a[i], "b": sol_b[i]} for i in diff[:8]
        ]
    return summary


# Locale-sensitive gold answers are excluded from the pool for BOTH arms so the reward can be plain
# Math-Verify against the English gold with no custom normalization. Stage 0 showed Math-Verify
# reads Spanish-locale numbers wrongly ("3,5" -> the set {3,5}; "10.500" -> 10.5), so any gold a
# Spanish-writing model might express with a decimal comma or thousands dot is removed.
_DECIMAL_RE = re.compile(r"\d\.\d|(?<!\d)\.\d")  # 3.5 or .185
_COMMA_THOUSANDS_RE = re.compile(r"\d,\d{3}(?!\d)")  # 2,177,280
_INT_GE_1000_RE = re.compile(r"(?<![\d.,])\d{4,}(?![\d.,])")  # 10500 (model might write 10.500)
_FRACTION_RE = re.compile(r"\\[dt]?frac|(?<![\w^])-?\d+/\d+(?![\w])")  # \frac{7}{2}, 7/2 (model might write 3,5)

DROP_RULES = (
    ("decimal", _DECIMAL_RE),
    ("comma_thousands", _COMMA_THOUSANDS_RE),
    ("integer_ge_1000", _INT_GE_1000_RE),
    ("fraction", _FRACTION_RE),
)


def drop_reason(solution: str) -> str | None:
    """Why a problem is excluded from the pool (locale-sensitive gold), or None to keep it."""
    for name, rx in DROP_RULES:
        if rx.search(solution):
            return name
    return None


def filter_ids(en_split: Dataset) -> dict:
    """Split parallel IDs into kept vs. dropped (locale-sensitive gold), with counts per rule."""
    ids = list(en_split["original_idx"])
    sols = list(en_split["solution"])
    reasons = {i: drop_reason(s) for i, s in zip(ids, sols)}
    dropped = [i for i in ids if reasons[i]]
    kept = [i for i in ids if not reasons[i]]
    sol_by_id = dict(zip(ids, sols))
    per_rule = {name: sum(1 for i in dropped if reasons[i] == name) for name, _ in DROP_RULES}
    return {
        "n_total": len(ids),
        "n_dropped": len(dropped),
        "n_dropped_by_rule": per_rule,
        "n_kept": len(kept),
        "dropped_examples": {name: [sol_by_id[i] for i in dropped if reasons[i] == name][:5] for name, _ in DROP_RULES},
        "kept_ids": kept,
        "dropped_ids": dropped,
    }


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
