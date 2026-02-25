from __future__ import annotations

import builtins

import numpy as np
import pytest

from tneuro.utils import plot as plot_utils


def _block_matplotlib_import(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def fake_import(name: str, *args, **kwargs):
        if name.startswith("matplotlib"):
            raise ImportError("blocked for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)


def test_plot_functions_require_matplotlib(monkeypatch: pytest.MonkeyPatch) -> None:
    _block_matplotlib_import(monkeypatch)

    with pytest.raises(ImportError, match="matplotlib is required for plotting"):
        plot_utils.plot_voltage_trace(np.array([0.0, 1.0]), np.array([0.0, 1.0]))

    with pytest.raises(ImportError, match="matplotlib is required for plotting"):
        plot_utils.plot_spike_raster(np.array([0.1, 0.2]))


def test_plot_functions_smoke_when_matplotlib_available() -> None:
    pytest.importorskip("matplotlib")

    t = np.linspace(0.0, 1.0, 10)
    v = np.sin(2.0 * np.pi * t)
    ax_v = plot_utils.plot_voltage_trace(t, v)
    assert ax_v is not None

    spikes = np.array([0.1, 0.3, 0.7])
    ax_r = plot_utils.plot_spike_raster(spikes)
    assert ax_r is not None
