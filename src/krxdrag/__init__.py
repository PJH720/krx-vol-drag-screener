"""KRX variance-drag screener.

Ranks Korean listed equities by the Itô correction term 0.5*sigma^2 -- the wedge
between arithmetic expected return mu and realised compound growth g.
"""

from .config import ScreenConfig
from .diagnostics import Diagnostics, compute_diagnostics
from .jumps import JumpMetrics, decompose_jumps, simulate_jump_diffusion
from .metrics import DragMetrics, compute_drag, log_returns, simulate_gbm
from .rolling import cross_sectional_drag, drag_trend, rolling_drag, rolling_panel
from .screener import screen, summarise

__version__ = "0.1.0"

__all__ = [
    "ScreenConfig",
    "Diagnostics",
    "compute_diagnostics",
    "JumpMetrics",
    "decompose_jumps",
    "simulate_jump_diffusion",
    "DragMetrics",
    "compute_drag",
    "log_returns",
    "simulate_gbm",
    "cross_sectional_drag",
    "drag_trend",
    "rolling_drag",
    "rolling_panel",
    "screen",
    "summarise",
]
