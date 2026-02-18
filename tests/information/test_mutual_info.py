import numpy as np

from tneuro.information.entropy import entropy_discrete
from tneuro.information.mutual_info import mutual_info_discrete


def test_mutual_info_independent_is_small() -> None:
    rng = np.random.default_rng(123)
    x = rng.integers(0, 4, size=5000)
    y = rng.integers(0, 4, size=5000)
    mi = mutual_info_discrete(x, y, base=2.0)
    assert mi < 0.05


def test_mutual_info_perfect_coupling_equals_entropy_object() -> None:
    x = np.array(list("abacabbccabacaba"), dtype=object)
    y = x.copy()
    mi = mutual_info_discrete(x, y, base=2.0)
    h = entropy_discrete(x, base=2.0)
    assert abs(mi - h) < 1e-12
