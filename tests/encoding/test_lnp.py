import numpy as np

from tneuro.encoding.lnp import build_lag_matrix, fit_lnp_poisson_irls


def test_fit_lnp_recovers_filter_shape() -> None:
    rng = np.random.default_rng(123)
    fs_hz = 200.0
    dt_s = 1.0 / fs_hz
    n = 12000
    stim = rng.standard_normal(n)

    window_s = 0.05
    n_lags = int(window_s * fs_hz) + 1
    lags = np.arange(-n_lags + 1, 1, dtype=int)
    lags_s = lags * dt_s
    true_filter = np.exp(lags_s / 0.02)
    true_filter /= np.linalg.norm(true_filter)

    x_mat, valid_idx = build_lag_matrix(stim, lags, add_intercept=True)
    lin_drive = x_mat @ np.concatenate(([0.0], true_filter))
    rate_hz = np.exp(-0.6 + 1.2 * lin_drive)
    rate_hz = np.clip(rate_hz, 1e-3, 150.0)

    spikes_valid = rng.poisson(rate_hz * dt_s)
    spikes = np.zeros_like(stim, dtype=float)
    spikes[valid_idx] = spikes_valid

    result = fit_lnp_poisson_irls(stim, spikes, lags, alpha=0.0)
    coef = result.coef[1:]
    coef_norm = coef / np.linalg.norm(coef)
    corr = float(np.corrcoef(true_filter, coef_norm)[0, 1])
    assert corr > 0.7
    assert result.converged
