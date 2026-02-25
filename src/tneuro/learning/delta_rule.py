from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tneuro.typing import ArrayF

from tneuro.utils.validate import require_non_negative_scalar, require_positive_scalar


@dataclass(frozen=True, slots=True)
class DeltaRuleResult:
    """Result of delta rule training."""

    weights: ArrayF
    losses: ArrayF


def _validate_supervised_xy(
    x: ArrayF | np.ndarray,
    y: ArrayF | np.ndarray,
) -> tuple[ArrayF, ArrayF]:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if x_arr.ndim != 2:
        raise ValueError("x must be 2D with shape (n_samples, n_features).")
    if y_arr.ndim != 1:
        raise ValueError("y must be 1D with shape (n_samples,).")
    if x_arr.shape[0] != y_arr.shape[0]:
        raise ValueError("x and y must have the same number of samples.")
    if not np.all(np.isfinite(x_arr)) or not np.all(np.isfinite(y_arr)):
        raise ValueError("x and y must be finite.")
    return x_arr, y_arr


def delta_rule_fit(
    x: ArrayF | np.ndarray,
    y: ArrayF | np.ndarray,
    *,
    lr: float,
    n_epochs: int,
    w0: ArrayF | np.ndarray | None = None,
    shuffle: bool = True,
    l2: float = 0.0,
    seed: int | None = None,
) -> DeltaRuleResult:
    """Fit a linear model with the delta rule (stochastic gradient descent).

    Parameters
    ----------
    x:
        Design matrix with shape (n_samples, n_features).
    y:
        Target vector with shape (n_samples,).
    lr:
        Learning rate (>0).
    n_epochs:
        Number of passes over the data (>0).
    w0:
        Optional initial weights, shape (n_features,).
    shuffle:
        Whether to shuffle samples each epoch.
    l2:
        L2 regularization strength (>=0).
    seed:
        Optional RNG seed for shuffling.
    """
    # TODO: Add mini-batch option for SGD. LABELS:learning,enhancement ASSIGNEE:diogoribeiro7
    # TODO: Add early stopping based on loss plateau. LABELS:learning,analysis ASSIGNEE:diogoribeiro7
    x_arr, y_arr = _validate_supervised_xy(x, y)
    lr_val = require_positive_scalar(lr, name="lr")
    n_epochs_val = int(require_positive_scalar(n_epochs, name="n_epochs"))
    l2_val = require_non_negative_scalar(l2, name="l2")

    n_samples, n_features = x_arr.shape
    if w0 is None:
        w: ArrayF = np.zeros(n_features, dtype=float)
    else:
        w = np.asarray(w0, dtype=float).copy()
        if w.shape != (n_features,):
            raise ValueError("w0 must have shape (n_features,).")
        if not np.all(np.isfinite(w)):
            raise ValueError("w0 must be finite.")

    rng = np.random.default_rng(seed)
    losses: ArrayF = np.empty(n_epochs_val, dtype=float)

    for epoch in range(n_epochs_val):
        idx = rng.permutation(n_samples) if shuffle else np.arange(n_samples)

        for i in idx:
            xi = x_arr[i]
            err = y_arr[i] - float(xi @ w)
            w += lr_val * (err * xi - l2_val * w)

        preds = x_arr @ w
        losses[epoch] = float(0.5 * np.mean((y_arr - preds) ** 2))

    return DeltaRuleResult(weights=w.astype(float), losses=losses)


def delta_rule_predict(
    x: ArrayF | np.ndarray,
    weights: ArrayF | np.ndarray,
) -> ArrayF:
    """Predict with linear weights learned by the delta rule."""
    x_arr = np.asarray(x, dtype=float)
    w = np.asarray(weights, dtype=float)
    if x_arr.ndim != 2:
        raise ValueError("x must be 2D with shape (n_samples, n_features).")
    if w.ndim != 1 or w.shape[0] != x_arr.shape[1]:
        raise ValueError("weights must be 1D with shape (n_features,).")
    return np.asarray(x_arr @ w, dtype=float)


__all__ = [
    "DeltaRuleResult",
    "delta_rule_fit",
    "delta_rule_predict",
]
