from __future__ import annotations

from typing import Optional

import numpy as np


def make_rng(seed: Optional[int] = None) -> np.random.Generator:
    """Create a NumPy Generator with an optional seed."""
    if seed is None:
        return np.random.default_rng()
    return np.random.default_rng(int(seed))
