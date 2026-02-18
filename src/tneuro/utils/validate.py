from __future__ import annotations

from typing import Any

import numpy as np


def require_1d_float_array(x: Any, *, name: str) -> np.ndarray:
    """Convert input to a 1D float NumPy array.

    Raises:
        TypeError: If the array cannot be converted to float.
        ValueError: If the array is not 1D.
    """
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape={arr.shape}")
    return arr


def require_positive_scalar(x: Any, *, name: str) -> float:
    """Validate a positive scalar."""
    val = float(x)
    if not np.isfinite(val) or val <= 0.0:
        raise ValueError(f"{name} must be a finite positive scalar, got {x!r}")
    return val


def require_non_negative_scalar(x: Any, *, name: str) -> float:
    """Validate a non-negative scalar."""
    val = float(x)
    if not np.isfinite(val) or val < 0.0:
        raise ValueError(f"{name} must be a finite non-negative scalar, got {x!r}")
    return val
