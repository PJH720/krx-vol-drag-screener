"""Diagnostics must flag GBM violations and stay quiet when there are none."""

from __future__ import annotations

import numpy as np

from krxdrag.diagnostics import compute_diagnostics, ljung_box


def test_clean_normal_returns_pass_normality():
    rng = np.random.default_rng(0)
    r = rng.normal(0.0003, 0.012, size=4000)
    d = compute_diagnostics(r)
    assert d is not None
    assert d.normal_ok
    assert abs(d.excess_kurtosis) < 0.4
    assert d.gbm_score > 0.7


def test_fat_tails_are_detected():
    rng = np.random.default_rng(1)
    r = rng.standard_t(df=3, size=4000) * 0.01
    d = compute_diagnostics(r)
    assert d is not None
    assert not d.normal_ok
    assert d.excess_kurtosis > 1.0


def test_volatility_clustering_is_detected():
    """A GARCH-like series must trip the Ljung-Box test on squared returns."""
    rng = np.random.default_rng(2)
    n = 4000
    r = np.zeros(n)
    var = 1e-4
    for i in range(n):
        var = 1e-6 + 0.12 * (r[i - 1] ** 2 if i else 0.0) + 0.85 * var
        r[i] = rng.normal(0.0, np.sqrt(var))
    d = compute_diagnostics(r)
    assert d is not None
    assert not d.no_arch_ok


def test_ljung_box_quiet_on_white_noise():
    rng = np.random.default_rng(3)
    _, p = ljung_box(rng.normal(size=3000), lags=10)
    assert p > 0.01


def test_short_series_returns_none():
    assert compute_diagnostics(np.zeros(5)) is None
