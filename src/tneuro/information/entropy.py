from __future__ import annotations

from typing import Any, Optional

import numpy as np


def entropy_discrete(x: Any, *, base: float = 2.0) -> float:
    """Shannon entropy of a 1D discrete variable.

    Parameters
    ----------
    x:
        Observations (any hashable values). Converted to a NumPy array.
    base:
        Log base. Use 2.0 for bits, np.e for nats.

    Returns
    -------
    float
        Entropy H(X).

    Notes
    -----
    This is the plug-in estimator (maximum likelihood). It is biased for small samples.
    Bias-corrected estimators can be added later (Miller-Madow, NSB, etc.).
    """
    if base <= 0.0 or not np.isfinite(base):
        raise ValueError("base must be finite and > 0")

    arr = np.asarray(x)
    if arr.ndim != 1:
        raise ValueError("x must be 1D")

    if arr.size == 0:
        return 0.0

    # unique counts (works for numeric + object arrays)
    _, counts = np.unique(arr, return_counts=True)
    p = counts.astype(float) / float(arr.size)

    # avoid log(0); p is strictly positive from counts
    h = -np.sum(p * (np.log(p) / np.log(base)))
    return float(h)
