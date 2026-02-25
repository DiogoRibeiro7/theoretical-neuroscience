from __future__ import annotations

import numpy as np

from tneuro.learning.delta_rule import delta_rule_fit, delta_rule_predict


def test_delta_rule_recovers_linear_weights() -> None:
    rng = np.random.default_rng(0)
    n = 2000
    n_features = 3
    x = rng.normal(0.0, 1.0, size=(n, n_features))
    true_w = np.array([0.4, -0.2, 0.8])
    y = x @ true_w

    res = delta_rule_fit(x, y, lr=0.05, n_epochs=10, seed=1)
    assert res.weights.shape == true_w.shape
    assert np.allclose(res.weights, true_w, atol=0.05)

    preds = delta_rule_predict(x, res.weights)
    mse = float(np.mean((preds - y) ** 2))
    assert mse < 1e-3


def test_delta_rule_monte_carlo_similarity() -> None:
    rng = np.random.default_rng(42)
    n_trials = 20
    sims = []
    for seed in range(n_trials):
        n = 800
        x = rng.normal(0.0, 1.0, size=(n, 2))
        true_w = np.array([1.0, -0.5])
        y = x @ true_w + rng.normal(0.0, 0.2, size=n)
        res = delta_rule_fit(x, y, lr=0.03, n_epochs=8, seed=seed)
        sim = float(np.dot(res.weights, true_w) / (np.linalg.norm(res.weights) * np.linalg.norm(true_w)))
        sims.append(sim)

    assert float(np.mean(sims)) > 0.9
