from __future__ import annotations

import numpy as np

from tneuro.information.entropy import entropy_discrete


def test_entropy_bits() -> None:
    # two equiprobable outcomes -> 1 bit
    x = np.array([0, 1] * 100)
    h = entropy_discrete(x, base=2.0)
    assert np.isclose(h, 1.0, atol=1e-6)

def test_entropy_empty() -> None:
    assert entropy_discrete(np.array([]), base=2.0) == 0.0
