"""Splitting the variance drag into diffusive and jump components.

compute_drag() reports a single number, D = 0.5*sigma^2, and treats every unit
of variance the same. But realized variance mixes two very different things: the
continuous diffusion GBM actually assumes, and discrete jumps it does not. A
name whose drag is mostly jump risk is failing the model in a different way than
one that is merely volatile, and the leaderboard should be able to say which.

The split follows Barndorff-Nielsen and Shephard. Realized variance

    RV = sum r_i^2

is inflated by jumps, because a jump contributes its whole squared size. Bipower
variation

    BPV = mu1^-2 * (n/(n-1)) * sum |r_i| |r_{i-1}|,    mu1 = E|Z| = sqrt(2/pi)

pairs each return with its neighbour. A jump lands in only two of those products
and is multiplied by an ordinary-sized return each time, so as the sampling grid
fills in BPV converges to the *continuous* part of the variance alone. The gap

    JV = max(RV - BPV, 0)

is therefore the jump variance, and the drag decomposes exactly:

    D = 0.5*sigma^2 = 0.5*sigma^2_cont + 0.5*sigma^2_jump

Daily data is a coarse grid for this -- the asymptotics assume intraday
sampling -- so read the split as an indication of where a name's variance comes
from, not as a precise measurement.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy import special, stats

from .metrics import TRADING_DAYS


def _abs_moment(p: float) -> float:
    """E|Z|^p for standard normal Z:  2^(p/2) * Gamma((p+1)/2) / sqrt(pi)."""
    return float(2.0 ** (p / 2.0) * special.gamma((p + 1.0) / 2.0) / np.sqrt(np.pi))


# mu1 = E|Z|; BPV's scale constant is mu1^-2 = pi/2.
_MU1 = _abs_moment(1.0)
_BPV_SCALE = 1.0 / _MU1**2

# mu_{4/3} = E|Z|^{4/3}, the constant tripower quarticity is built from.
_MU_43 = _abs_moment(4.0 / 3.0)

# Asymptotic variance constant of the BNS ratio statistic: pi^2/4 + pi - 5.
_THETA = np.pi**2 / 4.0 + np.pi - 5.0


@dataclass(frozen=True)
class JumpMetrics:
    """Continuous / jump split of one series' realized variance."""

    n_obs: int
    rv: float                 # realized variance, daily units
    bpv: float                # bipower variation, daily units
    jump_variance: float      # max(rv - bpv, 0), daily units
    jump_ratio: float         # jump_variance / rv, in [0, 1]
    sigma_sq_cont: float      # annualised continuous variance
    sigma_sq_jump: float      # annualised jump variance
    drag_cont: float          # 0.5 * sigma_sq_cont
    drag_jump: float          # 0.5 * sigma_sq_jump
    bns_stat: float           # BNS ratio test statistic (standard normal)
    bns_pvalue: float         # one-sided; small => jumps present
    has_jumps: bool           # bns_pvalue < 0.05

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def realized_variance(returns: np.ndarray) -> float:
    """Sum of squared returns."""
    r = np.asarray(returns, dtype=float)
    return float(np.sum(r**2))


def bipower_variation(returns: np.ndarray) -> float:
    """Jump-robust estimate of the continuous part of realized variance."""
    r = np.abs(np.asarray(returns, dtype=float))
    n = r.size
    if n < 2:
        return float("nan")
    products = float(np.sum(r[1:] * r[:-1]))
    return _BPV_SCALE * (n / (n - 1.0)) * products


def tripower_quarticity(returns: np.ndarray) -> float:
    """Jump-robust estimate of integrated quarticity, the BNS test's denominator."""
    r = np.abs(np.asarray(returns, dtype=float))
    n = r.size
    if n < 3:
        return float("nan")
    products = float(np.sum(r[2:] ** (4 / 3) * r[1:-1] ** (4 / 3) * r[:-2] ** (4 / 3)))
    return n * (_MU_43**-3) * (n / (n - 2.0)) * products


def decompose_jumps(
    returns: np.ndarray,
    periods_per_year: int = TRADING_DAYS,
) -> JumpMetrics | None:
    """Split realized variance into continuous and jump parts.

    Returns None when the series is too short or degenerate to support the
    neighbour-product estimators.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    if n < 30:
        return None

    rv = realized_variance(r)
    if rv <= 0.0:
        return None

    bpv = bipower_variation(r)
    if not np.isfinite(bpv):
        return None

    # BPV can exceed RV by sampling noise on a jump-free series; the jump part
    # is a variance and cannot be negative.
    jump_variance = max(rv - bpv, 0.0)
    jump_ratio = jump_variance / rv

    a = float(periods_per_year)
    # RV/n is the mean daily variance; annualise the same way metrics.py does.
    sigma_sq_cont = a * min(bpv, rv) / n
    sigma_sq_jump = a * jump_variance / n

    # BNS ratio (relative jump) test. The max(1, TQ/BPV^2) floor is the standard
    # guard against a vanishing denominator on very quiet series.
    tq = tripower_quarticity(r)
    denom_ratio = 1.0
    if np.isfinite(tq) and bpv > 0.0:
        denom_ratio = max(1.0, tq / bpv**2)
    variance = _THETA * denom_ratio / n
    if variance > 0.0:
        bns_stat = (rv - bpv) / rv / np.sqrt(variance)
    else:
        bns_stat = float("nan")
    bns_pvalue = float(stats.norm.sf(bns_stat)) if np.isfinite(bns_stat) else float("nan")

    return JumpMetrics(
        n_obs=n,
        rv=rv,
        bpv=float(bpv),
        jump_variance=float(jump_variance),
        jump_ratio=float(jump_ratio),
        sigma_sq_cont=float(sigma_sq_cont),
        sigma_sq_jump=float(sigma_sq_jump),
        drag_cont=float(0.5 * sigma_sq_cont),
        drag_jump=float(0.5 * sigma_sq_jump),
        bns_stat=float(bns_stat),
        bns_pvalue=bns_pvalue,
        has_jumps=bool(np.isfinite(bns_pvalue) and bns_pvalue < 0.05),
    )


def simulate_jump_diffusion(
    mu: float,
    sigma: float,
    n_days: int,
    jump_intensity: float,
    jump_mean: float,
    jump_std: float,
    s0: float = 100.0,
    periods_per_year: int = TRADING_DAYS,
    seed: int | None = None,
) -> np.ndarray:
    """Merton jump-diffusion path, used by the test-suite to validate the split.

    jump_intensity is the annual expected number of jumps; jump_mean and
    jump_std describe the log jump size.
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / periods_per_year

    diffusive = rng.normal(
        loc=(mu - 0.5 * sigma**2) * dt,
        scale=sigma * np.sqrt(dt),
        size=n_days,
    )
    counts = rng.poisson(jump_intensity * dt, size=n_days)
    jumps = np.array(
        [
            rng.normal(jump_mean, jump_std, size=k).sum() if k else 0.0
            for k in counts
        ]
    )
    increments = diffusive + jumps
    return s0 * np.exp(np.concatenate([[0.0], np.cumsum(increments)]))
