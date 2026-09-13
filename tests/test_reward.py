import pytest

from rlvr_crosslingual_safety.reward import extract_last_boxed, reward


def test_extract_last_boxed_nested_and_last():
    assert extract_last_boxed(r"so \boxed{1} no wait \boxed{\frac{1}{2}}") == r"\frac{1}{2}"
    assert extract_last_boxed("no box here") is None
    assert extract_last_boxed(r"truncated \boxed{\frac{1}{2") is None  # unbalanced -> None


@pytest.mark.parametrize(
    "gold, boxed, expected",
    [
        # integers
        ("42", "42", 1.0),
        ("42", "42.0", 1.0),
        ("42", "43", 0.0),
        ("-3", "-3", 1.0),
        ("10500", "10500", 1.0),
        # fractions / decimals (gold fractions are kept; decimal golds are dropped from the pool)
        (r"\frac{7}{2}", "3.5", 1.0),
        (r"\frac{7}{2}", "7/2", 1.0),
        (r"\frac{7}{2}", r"\dfrac{7}{2}", 1.0),
        (r"\frac{7}{2}", "3.4", 0.0),
        (r"\frac{1}{2}", "0.5", 1.0),
        # radicals / expressions
        (r"5\sqrt{2}", r"5\sqrt2", 1.0),
        (r"5\sqrt{2}", r"\sqrt{50}", 1.0),
        ("x^2+1", "1+x^2", 1.0),
        ("x^2+1", "x^2-1", 0.0),
        # tuples / intervals
        ("(1,2)", "(1, 2)", 1.0),
        ("(1,2)", "(2,1)", 0.0),
        # text-ish
        (r"120^\circ", r"120^\circ", 1.0),
    ],
)
def test_reward_core(gold, boxed, expected):
    completion = f"Razonamiento... La respuesta final es \\boxed{{{boxed}}}."
    assert reward(completion, gold) == expected


@pytest.mark.parametrize(
    "gold, boxed",
    [
        ("10500", "10.500"),  # es thousands dot -> Math-Verify reads 10.5
        (r"\frac{7}{2}", "3,5"),  # es decimal comma -> Math-Verify reads the set {3,5}
        ("2177280", "2.177.280"),  # es thousands dots
    ],
)
def test_locale_formats_are_not_understood(gold, boxed):
    """Documents WHY such golds are excluded from the pool (data.DROP_RULES): no normalization here."""
    assert reward(f"\\boxed{{{boxed}}}", gold) == 0.0


def test_no_boxed_is_zero():
    assert reward("La respuesta es 42.", "42") == 0.0


def test_uses_last_boxed():
    assert reward(r"\boxed{41} ... corrijo: \boxed{42}", "42") == 1.0
    assert reward(r"\boxed{42} ... corrijo: \boxed{41}", "42") == 0.0
