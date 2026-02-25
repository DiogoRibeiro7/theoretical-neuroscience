import numpy as np
import pytest

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


def test_mutual_info_rejects_non_1d_inputs() -> None:
    x = np.array([[0, 1], [1, 0]])
    y = np.array([[0, 1], [1, 0]])
    with pytest.raises(ValueError, match="1D"):
        mutual_info_discrete(x, y, base=2.0)

    with pytest.raises(ValueError, match="same shape"):
        mutual_info_discrete(np.array([0, 1, 2]), np.array([0, 1]), base=2.0)
