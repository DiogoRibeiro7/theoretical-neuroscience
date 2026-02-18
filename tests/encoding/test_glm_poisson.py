import numpy as np

from tneuro.encoding.glm_poisson import (
    build_design_matrix,
    fit_poisson_glm,
    predict_rate,
)


def test_poisson_glm_recovers_params() -> None:
    rng = np.random.default_rng(7)
    n = 6000
    stim = rng.normal(0.0, 1.0, size=n)
    lags = np.arange(-8, 1)
    X, valid_idx = build_design_matrix(stim, lags, add_intercept=True)

    true_coef = np.linspace(-0.5, 0.8, X.shape[1])
    rate = predict_rate(X, true_coef)
    spikes = np.zeros(n, dtype=float)
    spikes[valid_idx] = rng.poisson(rate)

    res = fit_poisson_glm(stim, spikes, lags, add_intercept=True, max_iter=100, tol=1e-6)
    assert res.converged
    assert res.n_iter < 100
    assert res.coef.shape == true_coef.shape
    assert np.all(res.se > 0.0)

    corr = float(np.corrcoef(res.coef, true_coef)[0, 1])
    assert corr > 0.95


def test_poisson_glm_intercept_only() -> None:
    rng = np.random.default_rng(11)
    n = 2000
    stim = rng.normal(0.0, 1.0, size=n)
    lags = np.array([0])

    X, valid_idx = build_design_matrix(stim, lags, add_intercept=True)
    true_coef = np.array([-0.2, 0.0])
    rate = predict_rate(X, true_coef)
    spikes = np.zeros(n, dtype=float)
    spikes[valid_idx] = rng.poisson(rate)

    res = fit_poisson_glm(stim, spikes, lags, add_intercept=True)
    assert res.converged
    assert res.coef.shape == true_coef.shape
    assert abs(res.coef[0] - true_coef[0]) < 0.2
