from __future__ import annotations

from typing import Any

import numpy as np


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
    if base <= 0.0 or not np.isfinite(base):
        raise ValueError("base must be finite and > 0")

    x_arr = np.asarray(x)
    y_arr = np.asarray(y)
    if x_arr.ndim != 1 or y_arr.ndim != 1:
        raise ValueError("x and y must be 1D")
    if x_arr.shape != y_arr.shape:
        raise ValueError("x and y must have the same shape")
    if x_arr.size == 0:
        return 0.0

    _, x_counts = np.unique(x_arr, return_counts=True)
    _, y_counts = np.unique(y_arr, return_counts=True)
    p_x = x_counts.astype(float) / float(x_arr.size)
    p_y = y_counts.astype(float) / float(y_arr.size)

    pairs = np.empty(x_arr.size, dtype=object)
    pairs[:] = list(zip(x_arr, y_arr))
    _, xy_counts = np.unique(pairs, return_counts=True)
    p_xy = xy_counts.astype(float) / float(x_arr.size)

    h_x = -np.sum(p_x * (np.log(p_x) / np.log(base)))
    h_y = -np.sum(p_y * (np.log(p_y) / np.log(base)))
    h_xy = -np.sum(p_xy * (np.log(p_xy) / np.log(base)))
    return float(h_x + h_y - h_xy)


__all__ = ["mutual_info_discrete"]
