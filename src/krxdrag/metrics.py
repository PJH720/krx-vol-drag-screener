"""Itô decomposition of realized returns.

Under geometric Brownian motion  dS = mu*S*dt + sigma*S*dB, Itô's lemma gives

    d(ln S) = (mu - 0.5*sigma^2) dt + sigma dB

so the *log* drift observed in data is not mu but g = mu - 0.5*sigma^2.
The wedge D = mu - g = 0.5*sigma^2 is the "variance drag": the amount of
arithmetic expected return that never shows up in compounded wealth.

Every quantity here is estimated from daily log returns and annualised.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
from scipy import stats

TRADING_DAYS: int = 252


@dataclass(frozen=True)
class DragMetrics:
    """Annualised Itô decomposition for a single price series."""

    n_obs: int
    sigma: float          # annualised volatility
    sigma_sq: float       # annualised variance
    g: float              # geometric (log) drift  = mu - 0.5*sigma^2
    mu: float             # arithmetic drift       = g + 0.5*sigma^2
    drag: float           # 0.5*sigma^2
    drag_ratio: float     # drag / mu, NaN when mu <= 0
    se_g: float
    se_drag: float
    sigma_sq_lo: float    # chi-square CI for sigma^2
    sigma_sq_hi: float
    realized_qv: float    # (A/n) * sum(r^2), the quadratic-variation estimate
    simple_mean: float    # annualised mean of simple returns, cross-check on mu
    total_log_return: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def log_returns(prices: np.ndarray) -> np.ndarray:
    """Daily log returns from a price level series, dropping non-finite rows."""
    p = np.asarray(prices, dtype=float)
    p = p[np.isfinite(p) & (p > 0)]
    if p.size < 2:
        return np.empty(0, dtype=float)
    return np.diff(np.log(p))


def compute_drag(
    prices: np.ndarray,
    periods_per_year: int = TRADING_DAYS,
    confidence: float = 0.95,
) -> DragMetrics | None:
    """Estimate the Itô decomposition for one price series.

    Returns None when there is not enough data to form a sample variance.
    """
    r = log_returns(prices)
    n = r.size
    if n < 20:
        return None

    a = float(periods_per_year)

    # --- core estimators -------------------------------------------------
    g = a * float(np.mean(r))
    s2_daily = float(np.var(r, ddof=1))
    sigma_sq = a * s2_daily
    sigma = float(np.sqrt(sigma_sq))
    drag = 0.5 * sigma_sq
    mu = g + drag  # Itô correction; by construction mu - g == drag exactly

    # --- uncertainty -----------------------------------------------------
    # SE(g) = a * s/sqrt(n) = sigma * sqrt(a/n)
    se_g = sigma * np.sqrt(a / n)
    # Under normality Var(s^2) = 2*sigma^4/(n-1)  =>  SE(sigma^2)=sigma^2*sqrt(2/(n-1))
    se_sigma_sq = sigma_sq * np.sqrt(2.0 / (n - 1))
    se_drag = 0.5 * se_sigma_sq

    alpha = 1.0 - confidence
    chi_hi = stats.chi2.ppf(1 - alpha / 2, df=n - 1)
    chi_lo = stats.chi2.ppf(alpha / 2, df=n - 1)
    sigma_sq_lo = (n - 1) * s2_daily * a / chi_hi
    sigma_sq_hi = (n - 1) * s2_daily * a / chi_lo

    # --- cross-checks ----------------------------------------------------
    # Itô: (dB)^2 = dt, so realized quadratic variation should track sigma^2.
    realized_qv = a * float(np.mean(r**2))
    simple = np.expm1(r)
    simple_mean = a * float(np.mean(simple))

    drag_ratio = drag / mu if mu > 0 else float("nan")

    return DragMetrics(
        n_obs=n,
        sigma=sigma,
        sigma_sq=sigma_sq,
        g=g,
        mu=mu,
        drag=drag,
        drag_ratio=drag_ratio,
        se_g=float(se_g),
        se_drag=float(se_drag),
        sigma_sq_lo=float(sigma_sq_lo),
        sigma_sq_hi=float(sigma_sq_hi),
        realized_qv=realized_qv,
        simple_mean=simple_mean,
        total_log_return=float(np.sum(r)),
    )


def simulate_gbm(
    mu: float,
    sigma: float,
    n_days: int,
    s0: float = 100.0,
    periods_per_year: int = TRADING_DAYS,
    seed: int | None = None,
) -> np.ndarray:
    """Exact GBM sample path, used by the test-suite to validate estimators."""
    rng = np.random.default_rng(seed)
    dt = 1.0 / periods_per_year
    shocks = rng.normal(
        loc=(mu - 0.5 * sigma**2) * dt,
        scale=sigma * np.sqrt(dt),
        size=n_days,
    )
    return s0 * np.exp(np.concatenate([[0.0], np.cumsum(shocks)]))
