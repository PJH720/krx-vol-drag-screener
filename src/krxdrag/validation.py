"""Does the estimate hold up out of sample?

Since its first commit this project has asserted, in README and ROADMAP alike:

    "mu's standard error is very large. sigma^2 is estimated comparatively
     accurately, so the drag itself is more trustworthy than mu."

The standard errors say so and the literature agrees, but nothing in this
repository has ever demonstrated it, and no report has carried evidence for it.
A tool that goes to the trouble of reporting how badly its own GBM assumptions
fail should not leave its founding premise unmeasured. This module measures it.

The design is a horse race between quantities, not a backtest of returns. The
history is cut into consecutive non-overlapping windows; each quantity is
estimated on window t and compared against what actually materialised in window
t+1. Two numbers per quantity:

- **Spearman rho** -- does the *ranking* survive into the next window? This is
  the one that matters for a leaderboard, which is what this project produces.
- **Out-of-sample R2** -- is last window's estimate a better forecast than
  simply guessing the cross-sectional average? It goes negative when the
  estimate is worse than that, which is a stronger indictment than a low
  correlation.

There is a ceiling on both, and it is knowable in advance. If an estimate is
truth plus independent noise, then across a cross-section

    corr(est_t, est_t+1) = Var(signal) / (Var(signal) + Var(noise))

so pure estimation noise caps persistence even when the underlying quantity
never changes. `reliability()` computes that ceiling, and the report shows it
beside the measurement: a measured persistence near the ceiling means the
quantity is as stable as it could be, while one far below it means the
underlying parameter is genuinely moving.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from .metrics import TRADING_DAYS

QUANTITIES: dict[str, str] = {
    "sigma_sq": "σ² (분산)",
    "drag": "드래그 ½σ²",
    "g": "g (기하 성장률)",
    "mu": "μ (산술 드리프트)",
}

DEFAULT_WINDOW: int = 126
MIN_NAMES: int = 10


@dataclass(frozen=True)
class PersistenceResult:
    """How well one quantity estimated on window t survives into window t+1."""

    quantity: str
    n_pairs: int          # consecutive window pairs measured
    n_names: int          # median names per pair
    spearman: float       # rank persistence, the leaderboard-relevant number
    pearson: float
    r2_oos: float         # negative => worse than guessing the cross-sectional mean

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def reliability(signal_var: float, noise_var: float) -> float:
    """Var(signal) / (Var(signal) + Var(noise)).

    The persistence an estimate can reach when the underlying quantity is
    perfectly stable and only estimation noise degrades it.
    """
    total = signal_var + noise_var
    if not np.isfinite(total) or total <= 0:
        return float("nan")
    return float(signal_var / total)


def noise_variance(
    quantity: str,
    sigma: np.ndarray,
    n: int,
    periods_per_year: int = TRADING_DAYS,
) -> float:
    """Average sampling variance of one quantity's estimator, from metrics.py.

    Var(sigma^2 hat) = 2 sigma^4 / (n - 1);  Var(g hat) = A sigma^2 / n.
    mu = g + sigma^2/2, and g's noise dominates it by two orders of magnitude
    at any realistic window length, which is the whole reason drag is a
    steadier quantity than mu.
    """
    s = np.asarray(sigma, float)
    s = s[np.isfinite(s) & (s > 0)]
    if s.size == 0 or n < 2:
        return float("nan")

    var_sigma_sq = float(np.mean(2.0 * s**4 / (n - 1)))
    var_g = float(np.mean(periods_per_year * s**2 / n))

    return {
        "sigma_sq": var_sigma_sq,
        "drag": 0.25 * var_sigma_sq,
        "g": var_g,
        "mu": var_g + 0.25 * var_sigma_sq,
    }[quantity]


# --------------------------------------------------------------------------
# window estimates
# --------------------------------------------------------------------------

def window_estimates(
    wide: pd.DataFrame,
    window: int = DEFAULT_WINDOW,
    periods_per_year: int = TRADING_DAYS,
    min_periods: int | None = None,
) -> dict[str, pd.DataFrame]:
    """Estimate every quantity on each consecutive non-overlapping window.

    Returns quantity -> (window index x symbol) frame. Windows are disjoint so
    that consecutive pairs share no data; overlapping windows would manufacture
    persistence out of the shared observations.
    """
    if window < 2:
        raise ValueError(f"window must be at least 2, got {window}")

    returns = np.log(wide.where(wide > 0)).diff().iloc[1:]
    n_blocks = len(returns) // window
    if n_blocks < 2:
        return {}

    if min_periods is None:
        min_periods = max(2, int(round(0.8 * window)))

    a = float(periods_per_year)
    g_rows, var_rows = [], []
    for b in range(n_blocks):
        block = returns.iloc[b * window : (b + 1) * window]
        enough = block.notna().sum() >= min_periods
        g_rows.append((a * block.mean()).where(enough))
        var_rows.append((a * block.var(ddof=1)).where(enough))

    g = pd.DataFrame(g_rows).reset_index(drop=True)
    sigma_sq = pd.DataFrame(var_rows).reset_index(drop=True)

    return {
        "sigma_sq": sigma_sq,
        "drag": 0.5 * sigma_sq,
        "g": g,
        "mu": g + 0.5 * sigma_sq,
    }


def _pair_scores(prior: pd.Series, later: pd.Series) -> tuple[float, float, float]:
    """Spearman, Pearson and out-of-sample R2 for one window pair."""
    both = pd.concat([prior, later], axis=1).dropna()
    if len(both) < 3:
        return (float("nan"),) * 3

    x = both.iloc[:, 0].to_numpy(float)
    y = both.iloc[:, 1].to_numpy(float)
    if np.std(x) == 0 or np.std(y) == 0:
        return (float("nan"),) * 3

    rho = float(stats.spearmanr(x, y).statistic)
    r = float(np.corrcoef(x, y)[0, 1])

    # Using the prior window's estimate as the forecast, against the naive
    # forecast of the realised cross-sectional mean.
    ss_res = float(np.sum((y - x) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return rho, r, r2


def persistence(
    wide: pd.DataFrame,
    window: int = DEFAULT_WINDOW,
    periods_per_year: int = TRADING_DAYS,
    min_names: int = MIN_NAMES,
) -> pd.DataFrame:
    """Measure how well each quantity's window-t estimate predicts window t+1.

    One row per quantity, ordered by rank persistence -- the ordering is the
    answer the project has been assuming.
    """
    estimates = window_estimates(wide, window=window, periods_per_year=periods_per_year)
    if not estimates:
        return pd.DataFrame()

    rows: list[dict] = []
    for quantity, frame in estimates.items():
        scores, counts = [], []
        for t in range(len(frame) - 1):
            prior, later = frame.iloc[t], frame.iloc[t + 1]
            usable = int((prior.notna() & later.notna()).sum())
            if usable < min_names:
                continue
            rho, r, r2 = _pair_scores(prior, later)
            if not np.isfinite(rho):
                continue
            scores.append((rho, r, r2))
            counts.append(usable)

        if not scores:
            continue
        arr = np.array(scores, dtype=float)
        rows.append(
            PersistenceResult(
                quantity=quantity,
                n_pairs=len(scores),
                n_names=int(np.median(counts)),
                spearman=float(np.nanmean(arr[:, 0])),
                pearson=float(np.nanmean(arr[:, 1])),
                r2_oos=float(np.nanmean(arr[:, 2])),
            ).to_dict()
        )

    if not rows:
        return pd.DataFrame()
    return (
        pd.DataFrame(rows)
        .sort_values("spearman", ascending=False)
        .reset_index(drop=True)
    )


def persistence_with_ceiling(
    wide: pd.DataFrame,
    window: int = DEFAULT_WINDOW,
    periods_per_year: int = TRADING_DAYS,
    min_names: int = MIN_NAMES,
) -> pd.DataFrame:
    """persistence(), plus the ceiling estimation noise alone would impose.

    `ceiling` is what the rank correlation would be if every name's true
    parameter were frozen and only sampling error moved the estimate. Measured
    persistence close to it means the quantity is about as stable as it can be
    measured; far below means the parameter itself is moving.
    """
    table = persistence(wide, window=window, periods_per_year=periods_per_year,
                        min_names=min_names)
    if table.empty:
        return table

    estimates = window_estimates(wide, window=window, periods_per_year=periods_per_year)
    sigma = np.sqrt(estimates["sigma_sq"].to_numpy(float).ravel())

    ceilings, signals = [], []
    for quantity in table["quantity"]:
        observed = estimates[quantity].to_numpy(float).ravel()
        observed = observed[np.isfinite(observed)]
        noise = noise_variance(quantity, sigma, window, periods_per_year)
        # Var(observed) = Var(signal) + Var(noise), so back out the signal.
        signal = max(float(np.var(observed)) - noise, 0.0)
        signals.append(signal)
        ceilings.append(reliability(signal, noise))

    out = table.copy()
    out["signal_var"] = signals
    out["ceiling"] = ceilings
    return out
