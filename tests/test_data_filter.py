import pytest

from rlvr_crosslingual_safety.data import drop_reason


@pytest.mark.parametrize(
    "gold, reason",
    [
        ("42", None),
        ("-7", None),
        ("999", None),
        (r"5\sqrt{2}", None),
        ("x^2+1", None),
        ("(1,2)", None),
        (r"120^\circ", None),
        ("2^{10}", None),
        ("3.5", "decimal"),
        (".185", "decimal"),
        (r"42.86\%", "decimal"),
        ("2,177,280", "comma_thousands"),
        ("10500", "integer_ge_1000"),
        ("2^{1000}", "integer_ge_1000"),  # conservative: any 4+ digit run
        (r"\frac{7}{2}", "fraction"),
        (r"\dfrac{1}{3}", "fraction"),
        ("7/2", "fraction"),
    ],
)
def test_drop_reason(gold, reason):
    assert drop_reason(gold) == reason
