"""Range-based variance estimators.

Everything this project reports is a function of sigma^2, and metrics.py
estimates it the least efficient way available: the variance of close-to-close
returns, which throws away everything the price did between the closes. A day
that opened flat, ran up 8%, and closed flat again looks identical to a day
that never moved.

The estimators here read the whole bar. Measured against close-to-close on
simulated intraday GBM, the variance of the daily estimate falls by roughly
5.2x (Parkinson), 8.2x (Garman-Klass) and 6.4x (Rogers-Satchell) -- the same
data, several times the precision.

Two cautions govern how they are used, and both are enforced rather than merely
documented.

**Drift.** Parkinson and Garman-Klass are derived assuming zero drift. This
project is *about* the interaction between drift and variance, so an estimator
that assumes the drift away is the wrong tool for it: raising the drift bends
Parkinson to +30% bias while Rogers-Satchell stays flat. Worse than the bias
itself, Parkinson's error grows *with* mu, and a sigma^2 error correlated with
mu contaminates the very drag-versus-mu relationship the screen exists to
describe. Rogers-Satchell and Yang-Zhang are drift-free by construction and are
what `DEFAULT_METHOD` points at.

**Overnight gaps.** Parkinson, Garman-Klass and Rogers-Satchell are built from a
single bar and so are blind to the jump from one close to the next open. On
simulated data where 40% of variance falls overnight they understate sigma^2 by
about 42%, while close-to-close -- which spans the gap -- is unaffected. Only
Yang-Zhang carries an explicit overnight term, so it is the default rather than
Rogers-Satchell.

**Liquidity.** Every range estimator is biased downward, because the true
continuous-time high and low are never observed -- only the extremes of the
trades that happened. The sparser the trading, the narrower the observed range:
the bias runs from about -3% on a densely traded day to -24% on a thin one.
That bias is therefore *correlated with liquidity*, so adopting these estimators
for their efficiency alone would systematically understate the drag of thinly
traded names and tilt the cross-section along liquidity. They are reported
beside the close-to-close figure, never silently in place of it, and
`liquidity_bias_report()` exists to make the gap visible.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from .metrics import TRADING_DAYS, log_returns

# KRX daily price limit. On a limit day the bar's range is clipped by the rule
# rather than by trading, adding a further downward bias on top of the sampling
# one, so those days are counted and reported.
KRX_PRICE_LIMIT: float = 0.30

# Variance of the daily close-to-close estimate divided by the variance of each
# range estimate, under zero drift. Reference figures only -- they are NOT used
# to build confidence intervals, because the chi-square interval in metrics.py
# is derived for the sample variance and does not transfer.
RELATIVE_EFFICIENCY: dict[str, float] = {
    "close_to_close": 1.0,
    "parkinson": 4.9,
    "garman_klass": 7.4,
    "rogers_satchell": 6.0,
}

DRIFT_FREE: frozenset[str] = frozenset({"close_to_close", "rogers_satchell", "yang_zhang"})

# Estimators built only from a bar's own open/high/low/close see nothing of the
# move between one close and the next open. That gap is not a rounding error
# for KRX equities -- overseas markets trade while Seoul is shut -- and a name
# carrying 40% of its variance overnight is understated by about 42% by
# Parkinson and Rogers-Satchell alike. Yang-Zhang is the only estimator here
# that is both drift-free and gap-aware, which is why it, not Rogers-Satchell,
# is the default.
INTRADAY_ONLY: frozenset[str] = frozenset({"parkinson", "garman_klass", "rogers_satchell"})
DEFAULT_METHOD: str = "yang_zhang"

_MIN_BARS: int = 20


@dataclass(frozen=True)
class VarianceEstimates:
    """Annualised sigma^2 under every estimator, for one symbol."""

    n_obs: int
    close_to_close: float
    parkinson: float
    garman_klass: float
    rogers_satchell: float
    yang_zhang: float
    limit_hit_share: float      # share of bars pinned at the +/-30% limit
    range_gap: float            # rogers_satchell / close_to_close - 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def get(self, method: str) -> float:
        if method not in METHODS:
            raise ValueError(f"unknown method {method!r}; choose from {sorted(METHODS)}")
        return float(getattr(self, method))


# --------------------------------------------------------------------------
# per-bar estimators
# --------------------------------------------------------------------------

def parkinson_daily(high: np.ndarray, low: np.ndarray) -> np.ndarray:
    """(1 / 4ln2) * ln(H/L)^2. Assumes zero drift."""
    h, l = np.asarray(high, float), np.asarray(low, float)
    return (1.0 / (4.0 * np.log(2.0))) * np.log(h / l) ** 2


def garman_klass_daily(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray
) -> np.ndarray:
    """0.5*ln(H/L)^2 - (2ln2 - 1)*ln(C/O)^2. Assumes zero drift."""
    o, h = np.asarray(open_, float), np.asarray(high, float)
    l, c = np.asarray(low, float), np.asarray(close, float)
    return 0.5 * np.log(h / l) ** 2 - (2.0 * np.log(2.0) - 1.0) * np.log(c / o) ** 2


def rogers_satchell_daily(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray
) -> np.ndarray:
    """ln(H/C)ln(H/O) + ln(L/C)ln(L/O). Unbiased whatever the drift."""
    o, h = np.asarray(open_, float), np.asarray(high, float)
    l, c = np.asarray(low, float), np.asarray(close, float)
    return np.log(h / c) * np.log(h / o) + np.log(l / c) * np.log(l / o)


# --------------------------------------------------------------------------
# sample estimators (annualised variance)
# --------------------------------------------------------------------------

def close_to_close(close: np.ndarray, periods_per_year: int = TRADING_DAYS) -> float:
    """The estimator metrics.py already uses, for side-by-side comparison."""
    r = log_returns(close)
    if r.size < 2:
        return float("nan")
    return float(periods_per_year) * float(np.var(r, ddof=1))


def _mean_daily(values: np.ndarray, periods_per_year: int) -> float:
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan")
    return float(periods_per_year) * float(np.mean(v))


def parkinson(high, low, periods_per_year: int = TRADING_DAYS) -> float:
    return _mean_daily(parkinson_daily(high, low), periods_per_year)


def garman_klass(open_, high, low, close, periods_per_year: int = TRADING_DAYS) -> float:
    return _mean_daily(garman_klass_daily(open_, high, low, close), periods_per_year)


def rogers_satchell(open_, high, low, close, periods_per_year: int = TRADING_DAYS) -> float:
    return _mean_daily(rogers_satchell_daily(open_, high, low, close), periods_per_year)


def yang_zhang(
    open_, high, low, close, periods_per_year: int = TRADING_DAYS
) -> float:
    """Overnight + open-to-close + Rogers-Satchell, per Yang and Zhang (2000).

    Alone among these it prices the overnight gap, which for KRX equities is a
    real share of the daily move: the close-to-close estimator sees the gap and
    the intraday estimators do not, so without this term the range figures sit
    below close-to-close for reasons that have nothing to do with sampling.
    """
    o, h = np.asarray(open_, float), np.asarray(high, float)
    l, c = np.asarray(low, float), np.asarray(close, float)
    if o.size < 3:
        return float("nan")

    # Day i's overnight return needs day i-1's close, so all three series are
    # taken over days 1..n-1 to keep them aligned on the same bars.
    overnight = np.log(o[1:] / c[:-1])
    open_to_close = np.log(c[1:] / o[1:])
    rs = rogers_satchell_daily(o[1:], h[1:], l[1:], c[1:])

    n = overnight.size
    if n < 2:
        return float("nan")

    var_o = float(np.var(overnight, ddof=1))
    var_c = float(np.var(open_to_close, ddof=1))
    var_rs = float(np.mean(rs))

    k = 0.34 / (1.34 + (n + 1.0) / (n - 1.0))
    return float(periods_per_year) * (var_o + k * var_c + (1.0 - k) * var_rs)


METHODS: dict[str, str] = {
    "close_to_close": "종가 대비",
    "parkinson": "Parkinson",
    "garman_klass": "Garman–Klass",
    "rogers_satchell": "Rogers–Satchell",
    "yang_zhang": "Yang–Zhang",
}


# --------------------------------------------------------------------------
# price limits
# --------------------------------------------------------------------------

def limit_hit_mask(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    limit: float = KRX_PRICE_LIMIT,
    tolerance: float = 0.005,
) -> np.ndarray:
    """Bars whose high or low sits at the daily price limit.

    KRX caps a day's move at +/-30% of the previous close. On such a bar the
    range is clipped by the rule rather than by trading, so every range
    estimator understates that day's variance. Returned mask is aligned to
    bars 1..n-1 (the first bar has no previous close to measure against).
    """
    c, h, l = (np.asarray(x, float) for x in (close, high, low))
    if c.size < 2:
        return np.zeros(0, dtype=bool)

    prev = c[:-1]
    up = h[1:] >= prev * (1.0 + limit) * (1.0 - tolerance)
    down = l[1:] <= prev * (1.0 - limit) * (1.0 + tolerance)
    return up | down


# --------------------------------------------------------------------------
# the whole comparison
# --------------------------------------------------------------------------

def estimate_all(
    bars: pd.DataFrame,
    periods_per_year: int = TRADING_DAYS,
    limit: float = KRX_PRICE_LIMIT,
) -> VarianceEstimates | None:
    """Every estimator for one symbol's OHLC bars, or None if unusable.

    `bars` is expected to come from data.to_bars(), which has already dropped
    incomplete and internally inconsistent bars.
    """
    if bars is None or len(bars) < _MIN_BARS:
        return None
    for col in ("open", "high", "low", "close"):
        if col not in bars.columns:
            return None

    o = bars["open"].to_numpy(float)
    h = bars["high"].to_numpy(float)
    l = bars["low"].to_numpy(float)
    c = bars["close"].to_numpy(float)
    if not np.isfinite(c).all() or (c <= 0).any():
        return None

    cc = close_to_close(c, periods_per_year)
    rs = rogers_satchell(o, h, l, c, periods_per_year)
    limits = limit_hit_mask(c, h, l, limit=limit)

    return VarianceEstimates(
        n_obs=len(bars),
        close_to_close=cc,
        parkinson=parkinson(h, l, periods_per_year),
        garman_klass=garman_klass(o, h, l, c, periods_per_year),
        rogers_satchell=rs,
        yang_zhang=yang_zhang(o, h, l, c, periods_per_year),
        limit_hit_share=float(limits.mean()) if limits.size else 0.0,
        range_gap=float(rs / cc - 1.0) if np.isfinite(cc) and cc > 0 else float("nan"),
    )


def liquidity_bias_report(
    df: pd.DataFrame,
    method: str = DEFAULT_METHOD,
    n_buckets: int = 5,
) -> pd.DataFrame:
    """How far the range estimate sits from close-to-close, by liquidity.

    The downward bias of a range estimator grows as trading thins out, so if
    these estimates are to be trusted across a 1,100-name cross-section that
    spans very different liquidity, the gap has to be shown as a function of
    liquidity rather than averaged into a single number. A gap that widens
    monotonically as turnover falls is the fingerprint of the sampling bias,
    not of thin names genuinely being calmer.
    """
    needed = {"median_turnover", method, "sigma_sq"}
    if df.empty or not needed <= set(df.columns):
        return pd.DataFrame()

    d = df[np.isfinite(df[method]) & np.isfinite(df["sigma_sq"]) & (df["sigma_sq"] > 0)]
    d = d[d["median_turnover"] > 0]
    if len(d) < n_buckets * 2:
        return pd.DataFrame()

    bucket = pd.qcut(d["median_turnover"], n_buckets, labels=False, duplicates="drop")
    gap = d[method] / d["sigma_sq"] - 1.0

    out = pd.DataFrame({"bucket": bucket, "gap": gap, "turnover": d["median_turnover"]})
    grouped = out.groupby("bucket")
    table = pd.DataFrame(
        {
            "n_names": grouped["gap"].size(),
            "median_turnover": grouped["turnover"].median(),
            "median_gap": grouped["gap"].median(),
        }
    ).reset_index()
    table["bucket"] = table["bucket"].astype(int) + 1
    return table


def simulate_ohlc(
    mu: float,
    sigma: float,
    n_days: int,
    steps_per_day: int = 400,
    s0: float = 100.0,
    overnight_share: float = 0.0,
    periods_per_year: int = TRADING_DAYS,
    seed: int | None = None,
) -> pd.DataFrame:
    """GBM sampled `steps_per_day` times a day, reduced to OHLC bars.

    Used by the test-suite to check the estimators against a known sigma.
    `steps_per_day` is the knob that exposes the sampling bias: the observed
    high and low approach the continuous-time extremes only as it grows, so a
    small value stands in for a thinly traded name.

    `overnight_share` moves that fraction of total variance into a gap between
    one close and the next open, which the intraday estimators cannot see.
    """
    rng = np.random.default_rng(seed)
    total_var = sigma**2 / periods_per_year
    gap_var = total_var * overnight_share
    day_var = total_var - gap_var

    drift = (mu - 0.5 * sigma**2) / periods_per_year
    inc = rng.normal(
        drift / steps_per_day,
        np.sqrt(day_var / steps_per_day),
        size=(n_days, steps_per_day),
    )
    gaps = (
        rng.normal(0.0, np.sqrt(gap_var), size=n_days)
        if gap_var > 0
        else np.zeros(n_days)
    )

    intraday = np.cumsum(inc, axis=1)
    # Each day opens at the previous close plus its overnight gap.
    day_close = intraday[:, -1]
    opens = np.log(s0) + np.cumsum(np.concatenate([[gaps[0]], day_close[:-1] + gaps[1:]]))
    path = opens[:, None] + intraday

    o = np.exp(opens)
    c = np.exp(path[:, -1])
    h = np.maximum(np.exp(path.max(axis=1)), np.maximum(o, c))
    l = np.minimum(np.exp(path.min(axis=1)), np.minimum(o, c))

    return pd.DataFrame(
        {
            "date": pd.date_range("2023-01-02", periods=n_days, freq="B"),
            "open": o,
            "high": h,
            "low": l,
            "close": c,
            "volume": 1e6,
        }
    )
