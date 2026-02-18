from __future__ import annotations

import tneuro


def test_imports() -> None:
    assert hasattr(tneuro, "__version__")
    assert hasattr(tneuro, "SpikeTrain")
    assert hasattr(tneuro, "simulate_lif")
