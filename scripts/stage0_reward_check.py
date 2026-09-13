"""Stage 0: document Math-Verify behaviour on locale-sensitive model outputs (recorded, not asserted).

Writes results/stage0_reward_check.json.
"""

from rlvr_crosslingual_safety.data import REPO_ROOT, save_json
from rlvr_crosslingual_safety.reward import reward

# (gold [English], model boxed answer, note)
CASES = [
    # Spanish decimal comma in the model output (gold is a fraction, since decimal golds are dropped)
    (r"\frac{7}{2}", "3,5", "es decimal comma vs fraction gold"),
    (r"\frac{7}{2}", "3{,}5", "es LaTeX decimal comma"),
    (r"\frac{1}{4}", "0,25", "es decimal comma"),
    # Spanish thousands separator in the model output (872 kept train golds are integers >= 1000)
    ("10500", "10.500", "es thousands dot"),
    ("10500", "10\\,500", "LaTeX thin-space thousands"),
    ("10500", "10,500", "en thousands comma"),
    ("10500", "10 500", "space thousands"),
    ("2177280", "2.177.280", "es thousands dots, 7 digits"),
    # degrees / units / currency
    (r"120^\circ", "120", "degree symbol dropped"),
    (r"120^\circ", "120°", "unicode degree"),
    (r"\$70", "70", "currency stripped"),
    ("70", r"\$70", "currency added"),
    ("5", "5 cm", "unit added"),
    # radicals / numeric approximations
    (r"\sqrt{2}", "1.414", "decimal approximation of radical"),
    (r"\sqrt{2}", r"\sqrt{2}", "exact radical"),
    # wrapping / formatting
    ("42", r"\text{42}", "text wrapper"),
    ("42", "42.", "trailing period"),
    ("42", "$42$", "inner dollars"),
    ("-3", "−3", "unicode minus"),
    ("(1,2)", "(1, 2)", "tuple spacing"),
    (r"\frac{1}{2}", "1/2", "slash fraction"),
]


def main() -> None:
    rows = []
    for gold, boxed, note in CASES:
        r = reward(f"... \\boxed{{{boxed}}}", gold)
        rows.append({"gold": gold, "boxed": boxed, "reward": r, "note": note})
        print(f"{r:.0f}  gold={gold!r:16} boxed={boxed!r:16} {note}")
    save_json(rows, REPO_ROOT / "results" / "stage0_reward_check.json")


if __name__ == "__main__":
    main()
