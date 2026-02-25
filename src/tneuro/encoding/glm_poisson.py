from __future__ import annotations

from dataclasses import dataclass
from math import lgamma

import numpy as np

from tneuro.typing import ArrayF, ArrayI
from tneuro.utils.validate import require_1d_float_array, require_non_negative_scalar


@dataclass(frozen=True, slots=True)
class PoissonGLMResult:
    """Result of Poisson GLM fitting."""

    coef: ArrayF
    se: ArrayF
    log_likelihood: float
    n_iter: int
    converged: bool


def build_design_matrix(
    stim: ArrayF | np.ndarray,
    lags: ArrayI | np.ndarray,
    *,
    add_intercept: bool = True,
) -> tuple[ArrayF, ArrayI]:
    """Construct a design matrix from a 1D stimulus and integer lags.

    Parameters
    ----------
    stim:
        1D stimulus array.
    lags:
        1D integer lags in samples (negative values look back in time).
    add_intercept:
        Whether to add a column of ones.

    Returns
    -------
    X:
        Design matrix with shape (n_samples_valid, n_lags + intercept).
    valid_idx:
        Indices into the original stimulus where all lags are valid.
    """
    x = require_1d_float_array(stim, name="stim")
    lags_arr = np.asarray(lags, dtype=int)
    if lags_arr.ndim != 1 or lags_arr.size == 0:
        raise ValueError("lags must be a non-empty 1D array of integers.")

    min_lag = int(np.min(lags_arr))
    max_lag = int(np.max(lags_arr))
    start = max(0, -min_lag)
    stop = x.size - max_lag
    if stop <= start:
        raise ValueError("lags are incompatible with stim length.")

    valid_idx = np.arange(start, stop)
    n_rows = valid_idx.size
    n_lags = lags_arr.size

    x_mat = np.empty((n_rows, n_lags + (1 if add_intercept else 0)), dtype=float)
    col = 0
    if add_intercept:
        x_mat[:, 0] = 1.0
        col = 1

    for j, lag in enumerate(lags_arr):
        x_mat[:, col + j] = x[valid_idx + lag]

    return x_mat, valid_idx.astype(np.int64)


def predict_rate(x_mat: ArrayF | np.ndarray, coef: ArrayF | np.ndarray) -> ArrayF:
    """Predict Poisson rate using a log link."""
    eta = x_mat @ coef
    return np.asarray(np.exp(eta), dtype=float)


def log_likelihood_poisson(y: np.ndarray, rate: np.ndarray) -> float:
    """Poisson log-likelihood for observed counts."""
    y_arr = np.asarray(y, dtype=float)
    rate_arr = np.asarray(rate, dtype=float)
    if y_arr.shape != rate_arr.shape:
        raise ValueError("y and rate must have the same shape.")
    if np.any(rate_arr <= 0.0) or not np.all(np.isfinite(rate_arr)):
        raise ValueError("rate must be finite and positive.")
    return float(np.sum(y_arr * np.log(rate_arr) - rate_arr - np.vectorize(lgamma)(y_arr + 1.0)))


def fit_poisson_glm(
    stim: ArrayF | np.ndarray,
    spikes: ArrayF | np.ndarray,
    lags: ArrayI | np.ndarray,
    *,
    add_intercept: bool = True,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> PoissonGLMResult:
    """Fit a Poisson GLM using IRLS (log link).

    Parameters
    ----------
    stim:
        1D stimulus array.
    spikes:
        1D spike counts aligned to ``stim``.
    lags:
        1D integer lags in samples.
    add_intercept:
        Whether to include an intercept column in the design matrix.
    max_iter:
        Maximum IRLS iterations.
    tol:
        Convergence tolerance on coefficient change (L2 norm).
    """
    # TODO: Add stronger regularization options (ridge). LABELS:encoding,enhancement ASSIGNEE:diogoribeiro7
    # TODO: Add a warm-start option for coefficients. LABELS:encoding,enhancement ASSIGNEE:diogoribeiro7
    # TODO: Add convergence diagnostics (deviance or log-likelihood trace). LABELS:encoding,analysis ASSIGNEE:diogoribeiro7
    # TODO: Add fit diagnostics output (AIC/BIC). LABELS:encoding,analysis ASSIGNEE:diogoribeiro7
    y = require_1d_float_array(spikes, name="spikes")
    x_mat, valid_idx = build_design_matrix(stim, lags, add_intercept=add_intercept)
    y_valid = y[valid_idx]
    if y_valid.shape[0] != x_mat.shape[0]:
        raise ValueError("spikes must align with stim after lagging.")
    if np.any(y_valid < 0.0) or not np.all(np.isfinite(y_valid)):
        raise ValueError("spikes must be finite and non-negative.")

    require_non_negative_scalar(tol, name="tol")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive.")

    n_features = x_mat.shape[1]
    coef = np.zeros(n_features, dtype=float)
    converged = False

    for i in range(max_iter):
        eta = x_mat @ coef
        mu = np.exp(eta)
        w = mu
        z = eta + (y_valid - mu) / mu

        xtw = x_mat.T * w
        fisher = xtw @ x_mat
        rhs = xtw @ z
        try:
            coef_new = np.linalg.solve(fisher, rhs)
        except np.linalg.LinAlgError:
            coef_new = np.linalg.lstsq(fisher, rhs, rcond=None)[0]

        if np.linalg.norm(coef_new - coef) <= tol:
            coef = coef_new
            converged = True
            n_iter = i + 1
            break
        coef = coef_new
    else:
        n_iter = max_iter

    eta = x_mat @ coef
    mu = np.exp(eta)
    xtw = x_mat.T * mu
    fisher = xtw @ x_mat
    try:
        cov = np.linalg.inv(fisher)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(fisher)
    se = np.sqrt(np.diag(cov))
    ll = log_likelihood_poisson(y_valid, mu)

    return PoissonGLMResult(coef=coef, se=se, log_likelihood=ll, n_iter=n_iter, converged=converged)


__all__ = [
    "PoissonGLMResult",
    "build_design_matrix",
    "fit_poisson_glm",
    "predict_rate",
    "log_likelihood_poisson",
]
