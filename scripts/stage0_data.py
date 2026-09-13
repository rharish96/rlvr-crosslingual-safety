"""Stage 0: load mAceReason es + en (parallel train/test), verify alignment, filter decimal-gold problems.

Writes:
  results/stage0_data_summary.json        counts and alignment checks (committed)
  data/processed/parallel_ids.json        kept / dropped original_idx lists (committed)
"""

from rlvr_crosslingual_safety.data import (
    EXPECTED_SIZES,
    PROCESSED_DIR,
    REPO_ROOT,
    align_parallel,
    filter_ids,
    load_macereason,
    save_json,
)


def main() -> None:
    es = load_macereason("es")
    en = load_macereason("en")

    summary = {"sizes": {}, "alignment": {}, "filter": {}}
    for split in ("train", "test"):
        summary["sizes"][split] = {"es": len(es[split]), "en": len(en[split]), "expected": EXPECTED_SIZES[split]}
        assert len(es[split]) == len(en[split]) == EXPECTED_SIZES[split], summary["sizes"][split]
        al = align_parallel(es[split], en[split])
        assert al["same_id_set"], al
        summary["alignment"][split] = al

    ids = {}
    for split in ("train", "test"):
        f = filter_ids(en[split])
        ids[split] = {"kept": f.pop("kept_ids"), "dropped": f.pop("dropped_ids")}
        summary["filter"][split] = f

    # a few side-by-side examples for the report
    summary["examples"] = []
    for k in range(3):
        row_es, row_en = es["train"][k], en["train"][k]
        assert row_es["original_idx"] == row_en["original_idx"]
        summary["examples"].append(
            {
                "original_idx": row_es["original_idx"],
                "es_problem": row_es["problem"][:300],
                "en_problem": row_en["problem"][:300],
                "es_solution": row_es["solution"],
                "en_solution": row_en["solution"],
            }
        )

    save_json(summary, REPO_ROOT / "results" / "stage0_data_summary.json")
    save_json(ids, PROCESSED_DIR / "parallel_ids.json")

    for split in ("train", "test"):
        f = summary["filter"][split]
        al = summary["alignment"][split]
        print(
            f"{split}: n={f['n_total']} kept={f['n_kept']} dropped={f['n_dropped']} "
            f"(decimal={f['n_dropped_decimal_gold']}, comma_thousands={f['n_dropped_comma_thousands_gold']}) "
            f"kept_large_int={f['n_kept_with_large_integer_gold']} | id_set_equal={al['same_id_set']} "
            f"same_order={al['same_order']} sol_str_mismatch={al.get('n_solution_string_mismatch')}"
        )
    print("dropped examples (train):", summary["filter"]["train"]["dropped_examples"][:5])
    print("solution mismatches (train):", summary["alignment"]["train"].get("solution_mismatch_examples", [])[:3])


if __name__ == "__main__":
    main()
