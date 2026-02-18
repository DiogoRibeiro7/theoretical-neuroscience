"""tneuro: theoretical neuroscience building blocks.

This package is designed as a small, typed core library that can be extended
chapter-by-chapter with notebooks under examples/.
"""

from __future__ import annotations

from ._version import __version__
from .neurons.lif import LIFParams, simulate_lif
from .spiketrain.core import SpikeTrain

__all__ = [
    "__version__",
    "SpikeTrain",
    "LIFParams",
    "simulate_lif",
]
