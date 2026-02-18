from __future__ import annotations

from typing import Any

import numpy as np


def entropy_discrete(x: Any, *, base: float = 2.0, method: str = "plugin") -> float:
    """Shannon entropy of a 1D discrete variable.

    Parameters
    ----------
    x:
        Observations (any hashable values). Converted to a NumPy array.
    base:
        Log base. Use 2.0 for bits, np.e for nats.
    method:
        Estimator to use: ``"plugin"`` (maximum likelihood) or ``"miller_madow"``.

    Returns
    -------
    float
        Entropy H(X).

    Notes
    -----
    The plug-in estimator is biased for small samples. Miller-Madow adds a
    first-order correction: (k - 1) / (2 n ln(base)), where k is the number
    of observed categories and n is the sample size. This correction is only
    approximate and can still be biased when sample sizes are very small or
    when many categories are unobserved.
    """
    if base <= 0.0 or not np.isfinite(base):
        raise ValueError("base must be finite and > 0")

    if method not in {"plugin", "miller_madow"}:
        raise ValueError("method must be 'plugin' or 'miller_madow'")

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
    if method == "plugin":
        return float(h)

    k = float(counts.size)
    n = float(arr.size)
    correction = (k - 1.0) / (2.0 * n * np.log(base))
    return float(h + correction)
