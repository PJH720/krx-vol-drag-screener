"""Variance drag through time.

compute_drag() collapses a whole history into one number, which hides the thing
a practitioner most wants to know: whether a name's drag is where it usually
sits, or whether volatility has just doubled. Estimating the same Ito
decomposition over a moving window puts the wedge on a time axis.

The estimators here are deliberately the *same* formulas as metrics.py --
g = A*mean(r), sigma^2 = A*var(r, ddof=1), drag = sigma^2/2, mu = g + drag --
computed with pandas rolling rather than a Python loop. A regression test pins
the full-length window to compute_drag() so the two paths cannot drift apart.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import TRADING_DAYS, log_returns

DEFAULT_WINDOW: int = 126  # ~6 months of trading days


def _returns_series(prices) -> pd.Series:
    """Log returns as a Series, keeping the price index where there is one."""
    if isinstance(prices, pd.Series):
        clean = prices[np.isfinite(prices) & (prices > 0)]
        if clean.size < 2:
            return pd.Series(dtype=float)
        return pd.Series(
            np.diff(np.log(clean.to_numpy(dtype=float))), index=clean.index[1:]
        )
    return pd.Series(log_returns(prices))


def rolling_drag(
    prices,
    window: int = DEFAULT_WINDOW,
    periods_per_year: int = TRADING_DAYS,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """Annualised Ito decomposition over a moving window.

    One row per window end, with columns sigma, sigma_sq, g, mu, drag. Windows
    that do not have `min_periods` observations are dropped, so a series shorter
    than the window yields an empty frame rather than raising.
    """
    if window < 2:
        raise ValueError(f"window must be at least 2, got {window}")

    r = _returns_series(prices)
    min_periods = window if min_periods is None else min_periods
    if r.empty or r.size < min_periods:
        return pd.DataFrame(columns=["sigma", "sigma_sq", "g", "mu", "drag"])

    a = float(periods_per_year)
    roll = r.rolling(window=window, min_periods=min_periods)

    sigma_sq = a * roll.var(ddof=1)
    g = a * roll.mean()
    drag = 0.5 * sigma_sq

    out = pd.DataFrame(
        {
            "sigma": np.sqrt(sigma_sq),
            "sigma_sq": sigma_sq,
            "g": g,
            "mu": g + drag,
            "drag": drag,
        }
    )
    return out.dropna(how="all")


def rolling_panel(
    wide: pd.DataFrame,
    window: int = DEFAULT_WINDOW,
    periods_per_year: int = TRADING_DAYS,
    field: str = "drag",
) -> pd.DataFrame:
    """One rolling field for every column of a date x symbol price matrix.

    Returns a date x symbol frame of that field -- `drag` by default, which is
    what the cross-sectional time series in the report is built from.
    """
    if field not in {"sigma", "sigma_sq", "g", "mu", "drag"}:
        raise ValueError(f"unknown field {field!r}")

    a = float(periods_per_year)
    returns = np.log(wide.where(wide > 0)).diff()
    roll = returns.rolling(window=window, min_periods=window)

    if field in {"g", "mu"}:
        g = a * roll.mean()
        if field == "g":
            return g.dropna(how="all")
        return (g + 0.5 * a * roll.var(ddof=1)).dropna(how="all")

    sigma_sq = a * roll.var(ddof=1)
    if field == "sigma_sq":
        return sigma_sq.dropna(how="all")
    if field == "sigma":
        return np.sqrt(sigma_sq).dropna(how="all")
    return (0.5 * sigma_sq).dropna(how="all")


def drag_trend(
    prices,
    window: int = DEFAULT_WINDOW,
    periods_per_year: int = TRADING_DAYS,
) -> float:
    """Most recent window's drag minus the immediately preceding window's.

    Positive means the drag is building. NaN when there is not enough history
    for two non-overlapping windows.
    """
    r = _returns_series(prices)
    if r.size < 2 * window:
        return float("nan")

    a = float(periods_per_year)
    recent = r.iloc[-window:]
    prior = r.iloc[-2 * window : -window]
    return float(0.5 * a * (recent.var(ddof=1) - prior.var(ddof=1)))


def cross_sectional_drag(
    wide: pd.DataFrame,
    window: int = DEFAULT_WINDOW,
    periods_per_year: int = TRADING_DAYS,
    min_names: int = 5,
) -> pd.DataFrame:
    """Median and quartile drag across all names, per date.

    This is the market-level view: when the median line lifts, the whole
    cross-section is paying more drag, not just one volatile name.
    """
    panel = rolling_panel(wide, window=window, periods_per_year=periods_per_year)
    if panel.empty:
        return pd.DataFrame(columns=["n_names", "q1", "median", "q3"])

    counts = panel.notna().sum(axis=1)
    out = pd.DataFrame(
        {
            "n_names": counts,
            "q1": panel.quantile(0.25, axis=1),
            "median": panel.median(axis=1),
            "q3": panel.quantile(0.75, axis=1),
        }
    )
    return out[out["n_names"] >= min_names]
