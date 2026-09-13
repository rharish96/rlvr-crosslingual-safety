from rlvr_crosslingual_safety.langid import detect_language, language_shares, strip_math

ES = (
    "Primero, observamos que el paralelogramo tiene tres vértices dados. Calculamos el punto medio "
    r"de la diagonal: $M = \left(\frac{1+4}{2}, \frac{2+1}{2}\right)$. Por lo tanto, la suma de las "
    r"posibles coordenadas es \boxed{8}."
)
EN = (
    "First, note that the parallelogram has three given vertices. We compute the midpoint of the "
    r"diagonal: $M = \left(\frac{1+4}{2}, \frac{2+1}{2}\right)$. Therefore the sum of the possible "
    r"coordinates is \boxed{8}."
)
MATH_ONLY = r"$x^2 + 2x + 1 = 0$ \boxed{-1}"


def test_strip_math_removes_latex_and_numbers():
    s = strip_math(ES)
    assert "\\frac" not in s and "$" not in s and "boxed" not in s
    assert "paralelogramo" in s


def test_detects_spanish_and_english():
    assert detect_language(ES)[0] == "es"
    assert detect_language(EN)[0] == "en"


def test_math_only_is_unknown():
    assert detect_language(MATH_ONLY) == (None, 0.0)


def test_language_shares():
    shares = language_shares([ES, ES, EN, MATH_ONLY])
    assert shares["es"] == 0.5 and shares["en"] == 0.25 and shares["unknown"] == 0.25
