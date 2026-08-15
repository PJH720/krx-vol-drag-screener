"""Tests of the GBM assumptions that the drag estimate rests on.

compute_drag() is only as good as its model. GBM assumes daily log returns are
i.i.d. normal; real equities are neither. These diagnostics report *how badly*
that assumption fails for each name, so a leaderboard rank can be discounted
accordingly rather than taken at face value.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class Diagnostics:
    skew: float
    excess_kurtosis: float
    jarque_bera: float
    jb_pvalue: float
    lb_sq_stat: float          # Ljung-Box on squared returns
    lb_sq_pvalue: float
    normal_ok: bool            # JB fails to reject at 5%
    no_arch_ok: bool           # Ljung-Box fails to reject at 5%
    gbm_score: float           # 0..1, 1 = assumptions look fine

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def ljung_box(x: np.ndarray, lags: int = 10) -> tuple[float, float]:
    """Ljung-Box Q statistic and p-value for autocorrelation up to `lags`."""
    x = np.asarray(x, dtype=float)
    n = x.size
    if n <= lags + 1:
        return float("nan"), float("nan")
    xc = x - x.mean()
    denom = float(np.sum(xc**2))
    if denom == 0.0:
        return float("nan"), float("nan")

    q = 0.0
    for k in range(1, lags + 1):
        rho_k = float(np.sum(xc[k:] * xc[:-k])) / denom
        q += (rho_k**2) / (n - k)
    q *= n * (n + 2)
    p = float(stats.chi2.sf(q, df=lags))
    return float(q), p


def compute_diagnostics(returns: np.ndarray, lags: int = 10) -> Diagnostics | None:
    """Normality and volatility-clustering diagnostics for a return series."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    if n < 30:
        return None

    skew = float(stats.skew(r, bias=False))
    exk = float(stats.kurtosis(r, fisher=True, bias=False))

    jb, jb_p = stats.jarque_bera(r)
    jb, jb_p = float(jb), float(jb_p)

    lb_stat, lb_p = ljung_box(r**2, lags=lags)

    normal_ok = bool(jb_p > 0.05)
    no_arch_ok = bool(np.isfinite(lb_p) and lb_p > 0.05)

    # Heuristic 0..1 score. Heavy tails and volatility clustering both push it
    # down; it is a triage aid for ranking trust, not a formal test.
    tail_penalty = min(abs(exk) / 10.0, 1.0)
    skew_penalty = min(abs(skew) / 3.0, 1.0)
    arch_penalty = 0.0 if no_arch_ok else 1.0
    gbm_score = float(
        max(0.0, 1.0 - (0.45 * tail_penalty + 0.2 * skew_penalty + 0.35 * arch_penalty))
    )

    return Diagnostics(
        skew=skew,
        excess_kurtosis=exk,
        jarque_bera=jb,
        jb_pvalue=jb_p,
        lb_sq_stat=lb_stat,
        lb_sq_pvalue=lb_p,
        normal_ok=normal_ok,
        no_arch_ok=no_arch_ok,
        gbm_score=gbm_score,
    )
