import numpy as np

from tneuro.information.entropy import entropy_discrete


def test_miller_madow_reduces_bias_on_average() -> None:
    rng = np.random.default_rng(123)
    p = np.array([0.1, 0.2, 0.3, 0.4], dtype=float)
    true_h = -np.sum(p * np.log2(p))

    n = 20
    n_trials = 200
    plugin_err = []
    mm_err = []
    for _ in range(n_trials):
        x = rng.choice(len(p), size=n, p=p)
        h_plugin = entropy_discrete(x, base=2.0, method="plugin")
        h_mm = entropy_discrete(x, base=2.0, method="miller_madow")
        plugin_err.append(abs(h_plugin - true_h))
        mm_err.append(abs(h_mm - true_h))

    assert float(np.mean(mm_err)) < float(np.mean(plugin_err))
