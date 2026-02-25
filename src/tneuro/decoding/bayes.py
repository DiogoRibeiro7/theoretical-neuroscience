from __future__ import annotations

import numpy as np

from tneuro.typing import ArrayF, ArrayI

from tneuro.utils.validate import require_1d_float_array, require_positive_scalar


def _validate_rate_and_prior(
    rate_hz: ArrayF | np.ndarray,
    prior: ArrayF | np.ndarray | None,
) -> tuple[ArrayF, ArrayF]:
    rates = np.asarray(rate_hz, dtype=float)
    if rates.ndim != 2:
        raise ValueError("rate_hz must be 2D with shape (n_states, n_neurons).")
    if not np.all(np.isfinite(rates)) or np.any(rates < 0.0):
        raise ValueError("rate_hz must be finite and non-negative.")

    if prior is None:
        prior_arr = np.full(rates.shape[0], 1.0 / rates.shape[0], dtype=float)
    else:
        prior_arr = require_1d_float_array(prior, name="prior")
        if prior_arr.size != rates.shape[0]:
            raise ValueError("prior must match the number of states.")
        if np.any(prior_arr < 0.0) or not np.all(np.isfinite(prior_arr)):
            raise ValueError("prior must be finite and non-negative.")
        if prior_arr.sum() <= 0.0:
            raise ValueError("prior must have positive total mass.")
        prior_arr = prior_arr / prior_arr.sum()

    return rates, prior_arr


def decode_bayes_poisson(
    rate_hz: ArrayF | np.ndarray,
    spike_counts: ArrayF | np.ndarray,
    *,
    dt_s: float,
    prior: ArrayF | np.ndarray | None = None,
) -> tuple[ArrayF, int]:
    """Bayesian decoding with independent Poisson spike counts.

    Parameters
    ----------
    rate_hz:
        Tuning curves with shape (n_states, n_neurons), in Hz.
    spike_counts:
        Observed spike counts for each neuron, shape (n_neurons,).
    dt_s:
        Bin width in seconds.
    prior:
        Optional prior over states, shape (n_states,). If None, a uniform
        prior is used.

    Returns
    -------
    posterior:
        Posterior over states, shape (n_states,). Sums to 1.
    map_index:
        Index of the maximum a posteriori (MAP) state.
    """
    # TODO: Add numerical-stability tests for extreme rates. LABELS:decoding,tests ASSIGNEE:diogoribeiro7
    # TODO: Add optional log-rate caching for repeated calls. LABELS:decoding,performance ASSIGNEE:diogoribeiro7
    # TODO: Add posterior diagnostics (entropy/credible intervals). LABELS:decoding,analysis ASSIGNEE:diogoribeiro7
    rates, prior_arr = _validate_rate_and_prior(rate_hz, prior)

    counts: ArrayF = require_1d_float_array(spike_counts, name="spike_counts")
    if counts.size != rates.shape[1]:
        raise ValueError("spike_counts must match the number of neurons.")
    if np.any(counts < 0.0) or not np.all(np.isfinite(counts)):
        raise ValueError("spike_counts must be finite and non-negative.")

    dt = require_positive_scalar(dt_s, name="dt_s")

    log_rate = np.where(rates > 0.0, np.log(rates), 0.0)
    log_like = counts @ log_rate.T - dt * np.sum(rates, axis=1)

    impossible = (rates == 0.0) & (counts[None, :] > 0.0)
    if np.any(impossible):
        log_like = np.where(np.any(impossible, axis=1), -np.inf, log_like)

    log_prior = np.where(prior_arr > 0.0, np.log(prior_arr), -np.inf)
    log_post = log_like + log_prior

    max_log = np.max(log_post)
    if not np.isfinite(max_log):
        raise ValueError("All states have zero posterior mass.")
    weights = np.exp(log_post - max_log)
    total = float(np.sum(weights))
    if total <= 0.0 or not np.isfinite(total):
        raise ValueError("Failed to normalize posterior.")
    posterior = weights / total
    map_index = int(np.argmax(posterior))
    return posterior.astype(float), map_index


def decode_bayes_poisson_batch(
    rate_hz: ArrayF | np.ndarray,
    spike_counts: ArrayF | np.ndarray,
    *,
    dt_s: float,
    prior: ArrayF | np.ndarray | None = None,
) -> tuple[ArrayF, ArrayI]:
    """Vectorized Bayesian decoding for multiple time bins.

    Parameters
    ----------
    rate_hz:
        Tuning curves with shape (n_states, n_neurons), in Hz.
    spike_counts:
        Observed spike counts for each neuron, shape (n_samples, n_neurons).
    dt_s:
        Bin width in seconds.
    prior:
        Optional prior over states, shape (n_states,). If None, a uniform
        prior is used.

    Returns
    -------
    posterior:
        Posterior over states, shape (n_samples, n_states). Rows sum to 1.
    map_index:
        Index of the MAP state for each sample, shape (n_samples,).
    """
    rates, prior_arr = _validate_rate_and_prior(rate_hz, prior)
    counts = np.asarray(spike_counts, dtype=float)
    if counts.ndim != 2:
        raise ValueError("spike_counts must be 2D with shape (n_samples, n_neurons).")
    if counts.shape[1] != rates.shape[1]:
        raise ValueError("spike_counts must match the number of neurons.")
    if np.any(counts < 0.0) or not np.all(np.isfinite(counts)):
        raise ValueError("spike_counts must be finite and non-negative.")

    dt = require_positive_scalar(dt_s, name="dt_s")

    log_rate = np.where(rates > 0.0, np.log(rates), 0.0)
    log_like = counts @ log_rate.T - dt * np.sum(rates, axis=1)

    impossible = (rates[None, :, :] == 0.0) & (counts[:, None, :] > 0.0)
    if np.any(impossible):
        log_like = np.where(np.any(impossible, axis=2), -np.inf, log_like)

    log_prior = np.where(prior_arr > 0.0, np.log(prior_arr), -np.inf)
    log_post = log_like + log_prior[None, :]

    max_log = np.max(log_post, axis=1)
    if not np.all(np.isfinite(max_log)):
        raise ValueError("All states have zero posterior mass for at least one sample.")
    weights = np.exp(log_post - max_log[:, None])
    totals = np.sum(weights, axis=1)
    if np.any(totals <= 0.0) or not np.all(np.isfinite(totals)):
        raise ValueError("Failed to normalize posterior.")
    posterior = weights / totals[:, None]
    map_index = np.argmax(posterior, axis=1).astype(np.int64)
    return posterior.astype(float), map_index


__all__ = ["decode_bayes_poisson", "decode_bayes_poisson_batch"]
