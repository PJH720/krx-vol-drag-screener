"""Why leveraged ETFs decay: drag is quadratic in the multiple.

A daily-rebalanced Lx fund tracks L times the underlying's *simple* daily
return, so under GBM its value follows dX/X = L*dS/S and its log drift is

    g_L = L*mu - 0.5*L^2*sigma^2

Substituting mu = g + 0.5*sigma^2 rewrites that in terms of the drag D:

    g_L = L*g - (L^2 - L)*D

The fund does not lose L times the underlying's drag; it loses L times the
underlying's *growth* minus a penalty of (L^2 - L)*D. That penalty is zero at
L=1, but 2D at L=2, 6D at L=3, and 6D at L=-2 -- an inverse 2x fund pays the
same penalty as a 3x long one. This is the entire reason such products decay in
choppy markets while tracking their index perfectly day to day.

Two numbers fall out and both are directly usable:

    critical volatility   sigma* = sqrt(2*mu/L)   above which g_L turns negative
    optimal leverage      L*     = mu/sigma^2     the log-growth-maximising Kelly multiple

The KRX product table below is a convenience, not an authority. Rather than
trust its declared multiples, estimate_realized_leverage() regresses the fund's
daily simple returns on the underlying's and reports the slope, so a mislabelled
or misremembered entry is flagged by the data instead of silently believed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from .metrics import TRADING_DAYS, DragMetrics

# Declared multiple mismatch beyond this flags the row for review.
LEVERAGE_TOLERANCE: float = 0.15


@dataclass(frozen=True)
class LeveragedETF:
    """A KRX-listed geared product and the index it tracks."""

    code: str
    name: str
    leverage: float
    underlying: str       # KRX code of the 1x reference product
    market: str = "KOSPI"

    @property
    def symbol(self) -> str:
        return f"{self.code}.KS" if self.market == "KOSPI" else f"{self.code}.KQ"


# Seed table of KRX geared ETFs. Multiples are as marketed; every one of them is
# re-estimated from prices before use, and a row whose realised slope disagrees
# with its declared multiple is reported rather than trusted. Verify codes
# against KRX before relying on this list.
KRX_LEVERAGED_ETFS: tuple[LeveragedETF, ...] = (
    LeveragedETF("069500", "KODEX 200", 1.0, "069500"),
    LeveragedETF("122630", "KODEX 레버리지", 2.0, "069500"),
    LeveragedETF("252670", "KODEX 200선물인버스2X", -2.0, "069500"),
    LeveragedETF("114800", "KODEX 인버스", -1.0, "069500"),
    LeveragedETF("123320", "TIGER 레버리지", 2.0, "069500"),
    LeveragedETF("252710", "TIGER 200선물인버스2X", -2.0, "069500"),
    LeveragedETF("229200", "KODEX 코스닥150", 1.0, "229200"),
    LeveragedETF("233740", "KODEX 코스닥150레버리지", 2.0, "229200"),
    LeveragedETF("251340", "KODEX 코스닥150선물인버스", -1.0, "229200"),
)


@dataclass(frozen=True)
class LeverageMetrics:
    """What geared exposure does to one underlying's Ito decomposition."""

    leverage: float
    mu: float                  # underlying arithmetic drift
    sigma: float               # underlying volatility
    drag: float                # underlying 0.5*sigma^2
    levered_sigma: float       # |L| * sigma
    levered_drag: float        # 0.5 * L^2 * sigma^2
    levered_mu: float          # L * mu
    levered_g: float           # L*mu - 0.5*L^2*sigma^2, net of costs
    naive_g: float             # L * g, what a holder might expect
    drag_penalty: float        # (L^2 - L) * drag, the shortfall against naive_g
    cost: float                # annual fee + financing applied to levered_g
    critical_sigma: float      # volatility at which levered_g crosses zero
    optimal_leverage: float    # mu / sigma^2

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def critical_volatility(mu: float, leverage: float) -> float:
    """Volatility above which an Lx fund's log growth turns negative.

    From g_L = L*mu - 0.5*L^2*sigma^2 = 0. Only defined when L*mu > 0: if the
    geared position points against a drifting underlying it loses at every
    volatility, so there is no crossing point.
    """
    if leverage == 0.0 or leverage * mu <= 0.0:
        return float("nan")
    return float(np.sqrt(2.0 * mu / leverage))


def optimal_leverage(mu: float, sigma_sq: float) -> float:
    """The multiple maximising log growth (Kelly): L* = mu / sigma^2."""
    if sigma_sq <= 0.0:
        return float("nan")
    return float(mu / sigma_sq)


def leveraged_metrics(
    m: DragMetrics,
    leverage: float,
    fee: float = 0.0,
    financing: float = 0.0,
) -> LeverageMetrics:
    """Apply a constant daily-rebalanced multiple to an estimated decomposition.

    `fee` is the annual expense ratio. `financing` is the annual borrowing rate
    charged on the geared portion, applied to |L| - 1 of exposure.
    """
    sigma_sq = m.sigma_sq
    levered_drag = 0.5 * leverage**2 * sigma_sq
    levered_mu = leverage * m.mu
    cost = fee + financing * max(abs(leverage) - 1.0, 0.0)

    return LeverageMetrics(
        leverage=float(leverage),
        mu=m.mu,
        sigma=m.sigma,
        drag=m.drag,
        levered_sigma=float(abs(leverage) * m.sigma),
        levered_drag=float(levered_drag),
        levered_mu=float(levered_mu),
        levered_g=float(levered_mu - levered_drag - cost),
        naive_g=float(leverage * m.g),
        drag_penalty=float((leverage**2 - leverage) * m.drag),
        cost=float(cost),
        critical_sigma=critical_volatility(m.mu, leverage),
        optimal_leverage=optimal_leverage(m.mu, sigma_sq),
    )


def leverage_curve(
    mu: float,
    sigma: float,
    leverages: np.ndarray | None = None,
) -> pd.DataFrame:
    """Log growth as a function of the multiple: an inverted parabola in L.

    The peak sits at the Kelly multiple and the right-hand zero crossing is
    where more leverage stops helping and starts destroying capital.
    """
    if leverages is None:
        leverages = np.linspace(-3.0, 5.0, 161)
    grid = np.asarray(leverages, dtype=float)

    sigma_sq = sigma**2
    g = mu * grid - 0.5 * sigma_sq * grid**2
    return pd.DataFrame(
        {
            "leverage": grid,
            "levered_g": g,
            "levered_sigma": np.abs(grid) * sigma,
            "levered_drag": 0.5 * sigma_sq * grid**2,
        }
    )


def simulate_leveraged_path(
    underlying: np.ndarray,
    leverage: float,
    fee_per_day: float = 0.0,
) -> np.ndarray:
    """Exact daily-rebalanced Lx price path from an underlying price series.

    The fund earns L times each day's simple return. A day where L*R <= -1 wipes
    it out; real products halt before that, so the level is floored at a token
    fraction rather than allowed to go negative.
    """
    p = np.asarray(underlying, dtype=float)
    simple = p[1:] / p[:-1] - 1.0
    factors = 1.0 + leverage * simple - fee_per_day

    level = np.empty(p.size, dtype=float)
    level[0] = 100.0
    floor = 1e-8
    for i, f in enumerate(factors, start=1):
        level[i] = max(level[i - 1] * f, level[i - 1] * floor)
    return level


def estimate_realized_leverage(
    fund_prices: np.ndarray,
    underlying_prices: np.ndarray,
) -> tuple[float, float]:
    """Regress the fund's daily simple returns on the underlying's.

    Returns (slope, r_squared). The slope is the multiple the fund actually
    delivered, which is what the product table is checked against.
    """
    f = np.asarray(fund_prices, dtype=float)
    u = np.asarray(underlying_prices, dtype=float)
    n = min(f.size, u.size)
    if n < 20:
        return float("nan"), float("nan")

    fr = f[-n:][1:] / f[-n:][:-1] - 1.0
    ur = u[-n:][1:] / u[-n:][:-1] - 1.0
    keep = np.isfinite(fr) & np.isfinite(ur)
    fr, ur = fr[keep], ur[keep]
    if fr.size < 20 or np.var(ur) == 0.0:
        return float("nan"), float("nan")

    slope = float(np.cov(fr, ur, ddof=1)[0, 1] / np.var(ur, ddof=1))
    resid = fr - slope * ur
    ss_tot = float(np.sum((fr - fr.mean()) ** 2))
    r2 = float(1.0 - np.sum(resid**2) / ss_tot) if ss_tot > 0 else float("nan")
    return slope, r2


def audit_leveraged_etfs(
    wide: pd.DataFrame,
    products: tuple[LeveragedETF, ...] = KRX_LEVERAGED_ETFS,
    periods_per_year: int = TRADING_DAYS,
    tolerance: float = LEVERAGE_TOLERANCE,
) -> pd.DataFrame:
    """Check each product's declared multiple against the one it delivered.

    `wide` is a date x symbol close matrix. Products whose prices are absent are
    skipped, so this degrades to an empty frame offline rather than raising.
    """
    from .metrics import compute_drag

    rows: list[dict] = []
    for etf in products:
        under_symbol = f"{etf.underlying}.KS" if etf.market == "KOSPI" else f"{etf.underlying}.KQ"
        if etf.symbol not in wide.columns or under_symbol not in wide.columns:
            continue

        pair = wide[[etf.symbol, under_symbol]].dropna()
        if len(pair) < 60:
            continue

        fund = pair[etf.symbol].to_numpy(dtype=float)
        under = pair[under_symbol].to_numpy(dtype=float)

        under_m = compute_drag(under, periods_per_year=periods_per_year)
        fund_m = compute_drag(fund, periods_per_year=periods_per_year)
        if under_m is None or fund_m is None:
            continue

        slope, r2 = estimate_realized_leverage(fund, under)
        theory = leveraged_metrics(under_m, etf.leverage)

        rows.append(
            {
                "code": etf.code,
                "name": etf.name,
                "declared_leverage": etf.leverage,
                "realized_leverage": slope,
                "r_squared": r2,
                "leverage_mismatch": bool(
                    np.isfinite(slope) and abs(slope - etf.leverage) > tolerance
                ),
                "underlying_sigma": under_m.sigma,
                "underlying_drag": under_m.drag,
                "theoretical_g": theory.levered_g,
                "actual_g": fund_m.g,
                "drag_penalty": theory.drag_penalty,
                "critical_sigma": theory.critical_sigma,
            }
        )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)
