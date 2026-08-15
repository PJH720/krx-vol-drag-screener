"""The estimators must recover known GBM parameters. This is the load-bearing test."""

from __future__ import annotations

import numpy as np
import pytest

from krxdrag.metrics import TRADING_DAYS, compute_drag, log_returns, simulate_gbm


def test_ito_identity_is_exact():
    """mu - g must equal 0.5*sigma^2 to floating point, by construction."""
    path = simulate_gbm(mu=0.12, sigma=0.30, n_days=1500, seed=7)
    m = compute_drag(path)
    assert m is not None
    assert m.mu - m.g == pytest.approx(m.drag, abs=1e-12)
    assert m.drag == pytest.approx(0.5 * m.sigma_sq, abs=1e-12)


def test_zero_volatility_means_zero_drag():
    """A deterministic path has no drag: g and mu coincide."""
    a = TRADING_DAYS
    mu = 0.08
    path = 100.0 * np.exp(mu * np.arange(0, 800) / a)
    m = compute_drag(path)
    assert m is not None
    assert m.sigma == pytest.approx(0.0, abs=1e-9)
    assert m.drag == pytest.approx(0.0, abs=1e-9)
    assert m.g == pytest.approx(mu, rel=1e-6)
    assert m.mu == pytest.approx(m.g, abs=1e-9)


@pytest.mark.parametrize("sigma", [0.15, 0.30, 0.55])
def test_sigma_recovered_within_chi2_interval(sigma):
    """The chi-square CI for sigma^2 must cover the true value."""
    path = simulate_gbm(mu=0.10, sigma=sigma, n_days=3000, seed=42)
    m = compute_drag(path)
    assert m is not None
    assert m.sigma_sq_lo <= sigma**2 <= m.sigma_sq_hi


def test_geometric_drift_recovered_within_two_standard_errors():
    """g estimate should sit near the true mu - 0.5*sigma^2."""
    mu, sigma = 0.10, 0.25
    true_g = mu - 0.5 * sigma**2
    path = simulate_gbm(mu=mu, sigma=sigma, n_days=6000, seed=11)
    m = compute_drag(path)
    assert m is not None
    assert abs(m.g - true_g) < 2.5 * m.se_g


def test_quadratic_variation_tracks_sigma_squared():
    """Ito's (dB)^2 = dt: realized QV should match the variance estimate."""
    path = simulate_gbm(mu=0.05, sigma=0.35, n_days=4000, seed=3)
    m = compute_drag(path)
    assert m is not None
    assert m.realized_qv == pytest.approx(m.sigma_sq, rel=0.05)


def test_mu_cross_checks_against_simple_returns():
    """The Ito-corrected mu should track the mean of simple returns."""
    path = simulate_gbm(mu=0.10, sigma=0.30, n_days=8000, seed=5)
    m = compute_drag(path)
    assert m is not None
    assert m.mu == pytest.approx(m.simple_mean, abs=0.02)


def test_higher_volatility_produces_larger_drag():
    """Drag is monotone increasing in sigma."""
    drags = [
        compute_drag(simulate_gbm(mu=0.08, sigma=s, n_days=4000, seed=99)).drag
        for s in (0.10, 0.25, 0.50)
    ]
    assert drags == sorted(drags)


def test_drag_ratio_is_nan_for_nonpositive_mu():
    path = simulate_gbm(mu=-0.30, sigma=0.10, n_days=1500, seed=17)
    m = compute_drag(path)
    assert m is not None
    assert m.mu < 0
    assert np.isnan(m.drag_ratio)


def test_short_or_degenerate_series_returns_none():
    assert compute_drag(np.array([100.0, 101.0, 102.0])) is None
    assert compute_drag(np.array([])) is None


def test_log_returns_drop_nonpositive_and_nan():
    r = log_returns(np.array([100.0, np.nan, 110.0, -5.0, 121.0]))
    assert r.size == 2
    assert np.all(np.isfinite(r))
