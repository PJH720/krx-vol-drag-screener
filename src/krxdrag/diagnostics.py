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
import pandas as pd
from scipy import stats

# Default false-discovery rate for the cross-sectional correction below.
FDR_Q: float = 0.05

# The per-name tests this screen runs, and what a *rejection* of each means.
# Keyed by the p-value column the screen carries.
DIAGNOSTIC_TESTS: dict[str, str] = {
    "jb_pvalue": "정규성 기각 (Jarque–Bera)",
    "lb_sq_pvalue": "변동성 군집 있음 (Ljung–Box)",
    "bns_pvalue": "점프 유의 (BNS)",
}


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


# --------------------------------------------------------------------------
# multiple testing across the cross-section
# --------------------------------------------------------------------------
#
# Each diagnostic above is a hypothesis test run once per name. Run at the 5%
# level across 1,100 names, a test rejects roughly 55 of them by chance alone
# even when every null is true -- so "55 names show significant jumps" may be
# no evidence of anything. Benjamini-Hochberg controls the expected share of
# false positives *among the rejections*, which is the quantity a reader of a
# leaderboard actually cares about.
#
# Two of the three tests reject so overwhelmingly on real KRX data (Jarque-Bera
# rejects normality for ~99.9% of names) that the correction changes little;
# it matters for the jump flag, where the rejection rate is moderate and the
# chance floor is a real share of it. The correction is applied to all three
# regardless, and both counts are reported, so the reader can see which is
# which instead of being told.


def benjamini_hochberg(pvalues, q: float = FDR_Q) -> np.ndarray:
    """Step-up procedure: which hypotheses to reject at false-discovery rate q.

    Non-finite p-values take no part in the procedure and are never rejected;
    m counts only the tests that actually ran.
    """
    p = np.asarray(pvalues, dtype=float)
    rejected = np.zeros(p.shape, dtype=bool)

    finite = np.isfinite(p)
    m = int(finite.sum())
    if m == 0:
        return rejected

    idx = np.flatnonzero(finite)
    order = idx[np.argsort(p[idx], kind="stable")]
    ranked = p[order]

    below = np.flatnonzero(ranked <= q * np.arange(1, m + 1) / m)
    if below.size:
        # Reject everything up to the largest passing rank, not just the ranks
        # that individually pass -- that step-up is what makes it BH.
        rejected[order[: below[-1] + 1]] = True
    return rejected


def bh_adjusted(pvalues) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (q-values).

    The smallest FDR at which each hypothesis would be rejected, made monotone
    by the usual running minimum from the largest p-value down.
    """
    p = np.asarray(pvalues, dtype=float)
    out = np.full(p.shape, np.nan, dtype=float)

    finite = np.isfinite(p)
    m = int(finite.sum())
    if m == 0:
        return out

    idx = np.flatnonzero(finite)
    order = idx[np.argsort(p[idx], kind="stable")]
    scaled = p[order] * m / np.arange(1, m + 1)
    out[order] = np.minimum.accumulate(scaled[::-1])[::-1].clip(max=1.0)
    return out


def fdr_report(
    df: pd.DataFrame,
    tests: dict[str, str] | None = None,
    q: float = FDR_Q,
) -> pd.DataFrame:
    """Raw versus FDR-controlled rejection counts, one row per test.

    `expected_by_chance` is q * m -- how many rejections a test would produce
    on this many names if every null were true. A raw count near it is
    indistinguishable from noise; one far above it is not.
    """
    tests = tests or DIAGNOSTIC_TESTS
    rows: list[dict] = []

    for column, label in tests.items():
        if column not in df.columns:
            continue
        p = df[column].to_numpy(dtype=float)
        m = int(np.isfinite(p).sum())
        if m == 0:
            continue

        raw = int((p < q).sum())
        controlled = int(benjamini_hochberg(p, q=q).sum())
        rows.append(
            {
                "test": label,
                "n_tested": m,
                "raw_rejections": raw,
                "fdr_rejections": controlled,
                "expected_by_chance": q * m,
                "raw_share": raw / m,
                "fdr_share": controlled / m,
            }
        )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)
