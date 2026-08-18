"""Bipower variation must separate diffusion from jumps.

Validated the same way the rest of the estimators are: simulate a process whose
jump content is known, then check the decomposition recovers it.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import special

from krxdrag.jumps import (
    _MU1,
    _MU_43,
    bipower_variation,
    decompose_jumps,
    realized_variance,
    simulate_jump_diffusion,
    tripower_quarticity,
)
from krxdrag.metrics import TRADING_DAYS, compute_drag, log_returns, simulate_gbm


# --- constants ----------------------------------------------------------------

@pytest.mark.parametrize("p, constant", [(1.0, _MU1), (4 / 3, _MU_43)])
def test_absolute_moment_constants_match_the_normal_distribution(p, constant):
    """E|Z|^p = 2^(p/2) Gamma((p+1)/2)/sqrt(pi), checked against a large sample."""
    z = np.random.default_rng(0).normal(size=2_000_000)
    assert constant == pytest.approx(float(np.mean(np.abs(z) ** p)), rel=2e-3)
    assert constant == pytest.approx(
        2 ** (p / 2) * special.gamma((p + 1) / 2) / np.sqrt(np.pi), rel=1e-12
    )


def test_bipower_variation_is_unbiased_for_iid_normal():
    """With no jumps BPV estimates n*s^2, the same target as RV."""
    s = 0.013
    r = np.random.default_rng(1).normal(0.0, s, size=500_000)
    assert bipower_variation(r) == pytest.approx(r.size * s**2, rel=0.01)


# --- identities ----------------------------------------------------------------

def test_continuous_and_jump_variance_sum_to_realized_variance():
    path = simulate_jump_diffusion(
        mu=0.10, sigma=0.30, n_days=2000,
        jump_intensity=12.0, jump_mean=-0.02, jump_std=0.05, seed=4,
    )
    r = log_returns(path)
    j = decompose_jumps(r)
    m = compute_drag(path)
    assert j is not None and m is not None

    # The split is exact: annualised cont + jump == annualised RV.
    rv_annualised = TRADING_DAYS * realized_variance(r) / r.size
    assert j.sigma_sq_cont + j.sigma_sq_jump == pytest.approx(rv_annualised, rel=1e-12)
    # and metrics.py's quadratic variation is that same quantity
    assert rv_annualised == pytest.approx(m.realized_qv, rel=1e-12)
    # so the drag splits too
    assert j.drag_cont + j.drag_jump == pytest.approx(0.5 * rv_annualised, rel=1e-12)


def test_jump_ratio_is_bounded():
    for seed in range(5):
        path = simulate_jump_diffusion(
            mu=0.05, sigma=0.4, n_days=800,
            jump_intensity=20.0, jump_mean=0.0, jump_std=0.08, seed=seed,
        )
        j = decompose_jumps(log_returns(path))
        assert j is not None
        assert 0.0 <= j.jump_ratio <= 1.0


# --- behaviour on jump-free data ----------------------------------------------

def test_pure_gbm_shows_almost_no_jump_content():
    path = simulate_gbm(mu=0.08, sigma=0.30, n_days=4000, seed=21)
    j = decompose_jumps(log_returns(path))
    assert j is not None
    assert j.bpv == pytest.approx(j.rv, rel=0.10)
    assert j.jump_ratio < 0.10


def test_bns_test_holds_its_size_on_jump_free_paths():
    """A 5% test must not fire much more than 5% of the time without jumps."""
    rejections = sum(
        decompose_jumps(
            log_returns(simulate_gbm(mu=0.06, sigma=0.25, n_days=1500, seed=s))
        ).has_jumps
        for s in range(40)
    )
    assert rejections <= 6  # 15% of 40, loose enough not to flake


# --- behaviour on jumpy data ---------------------------------------------------

def test_injected_jumps_are_detected():
    path = simulate_jump_diffusion(
        mu=0.05, sigma=0.20, n_days=3000,
        jump_intensity=25.0, jump_mean=0.0, jump_std=0.12, seed=8,
    )
    j = decompose_jumps(log_returns(path))
    assert j is not None
    assert j.has_jumps
    assert j.bns_pvalue < 0.01
    assert j.jump_ratio > 0.15


def test_recovered_jump_variance_tracks_its_generator():
    """Annualised jump variance should approach lambda*(m^2 + s^2)."""
    lam, jm, js = 30.0, 0.0, 0.10
    expected = lam * (jm**2 + js**2)

    got = [
        decompose_jumps(
            log_returns(
                simulate_jump_diffusion(
                    mu=0.04, sigma=0.20, n_days=4000,
                    jump_intensity=lam, jump_mean=jm, jump_std=js, seed=s,
                )
            )
        ).sigma_sq_jump
        for s in range(12)
    ]
    # Daily sampling is a coarse grid for this asymptotic, so allow a wide band;
    # the point is that the magnitude is right, not that it is precise.
    assert expected * 0.4 < float(np.mean(got)) < expected * 1.6


def test_more_jumps_means_more_jump_variance():
    def jump_var(intensity: float) -> float:
        return float(np.mean([
            decompose_jumps(
                log_returns(
                    simulate_jump_diffusion(
                        mu=0.05, sigma=0.25, n_days=2000,
                        jump_intensity=intensity, jump_mean=0.0, jump_std=0.10, seed=s,
                    )
                )
            ).sigma_sq_jump
            for s in range(8)
        ]))

    assert jump_var(2.0) < jump_var(20.0) < jump_var(60.0)


# --- degenerate input ----------------------------------------------------------

def test_short_or_flat_series_returns_none():
    assert decompose_jumps(np.zeros(10)) is None      # too short
    assert decompose_jumps(np.zeros(200)) is None     # zero realized variance


def test_estimators_are_nan_safe_on_tiny_input():
    assert np.isnan(bipower_variation(np.array([0.01])))
    assert np.isnan(tripower_quarticity(np.array([0.01, 0.02])))


def test_nonfinite_returns_are_dropped():
    r = np.concatenate([np.random.default_rng(2).normal(0, 0.01, 200), [np.nan, np.inf]])
    j = decompose_jumps(r)
    assert j is not None
    assert j.n_obs == 200
