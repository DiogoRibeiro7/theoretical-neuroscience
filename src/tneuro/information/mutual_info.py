from __future__ import annotations

from typing import Any

import numpy as np

from .entropy import entropy_discrete


def mutual_info_discrete(x: Any, y: Any, *, base: float = 2.0) -> float:
    """Mutual information between two 1D discrete variables (plug-in estimator).

    Parameters
    ----------
    x, y:
        Observations (any hashable values). Converted to NumPy arrays.
    base:
        Log base. Use 2.0 for bits, np.e for nats.

    Returns
    -------
    float
        Mutual information I(X; Y).
    """
    x_arr = np.asarray(x)
    y_arr = np.asarray(y)
    if x_arr.ndim != 1 or y_arr.ndim != 1:
        raise ValueError("x and y must be 1D")
    if x_arr.shape != y_arr.shape:
        raise ValueError("x and y must have the same shape")
    if x_arr.size == 0:
        return 0.0

    pairs = np.empty(x_arr.size, dtype=object)
    pairs[:] = list(zip(x_arr, y_arr, strict=False))
    h_x = entropy_discrete(x_arr, base=base, method="plugin")
    h_y = entropy_discrete(y_arr, base=base, method="plugin")
    h_xy = entropy_discrete(pairs, base=base, method="plugin")
    return float(h_x + h_y - h_xy)


__all__ = ["mutual_info_discrete"]
