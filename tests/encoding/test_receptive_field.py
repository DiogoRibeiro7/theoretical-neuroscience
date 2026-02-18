import numpy as np

from tneuro.encoding.receptive_field import fit_linear_rf


def test_fit_linear_rf_recovers_filter() -> None:
    rng = np.random.default_rng(42)
    n = 4000
    stim = rng.normal(0.0, 1.0, size=n)
    lags = np.arange(-15, 1)
    true_rf = np.exp(-np.linspace(0.0, 2.0, lags.size))
    true_rf /= np.linalg.norm(true_rf)

    X = np.stack([np.roll(stim, -lag) for lag in lags], axis=1)
    spikes = X @ true_rf + rng.normal(0.0, 0.2, size=n)

    rf, _ = fit_linear_rf(stim, spikes, lags, alpha=1e-2)
    rf /= np.linalg.norm(rf)
    corr = float(np.dot(rf, true_rf))
    assert corr > 0.9


def test_fit_linear_rf_shape_mismatch() -> None:
    stim = np.zeros(100, dtype=float)
    spikes = np.zeros(99, dtype=float)
    lags = np.array([-1, 0])

    try:
        fit_linear_rf(stim, spikes, lags, alpha=0.0)
    except ValueError as exc:
        assert "same shape" in str(exc)
    else:
        raise AssertionError("Expected ValueError for mismatched shapes.")
