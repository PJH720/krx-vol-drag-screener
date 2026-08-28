"""The claim this project has always asserted, finally measured.

README and ROADMAP have said since the first commit that sigma^2 is estimated
far more accurately than mu, so the drag is the trustworthy number. These tests
check that on panels whose true parameters are known and *constant through
time*, so every point of lost persistence is estimation noise and nothing else
-- which makes the answer predictable in advance from the standard errors in
metrics.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from krxdrag.metrics import TRADING_DAYS
from krxdrag.validation import (
    QUANTITIES,
    noise_variance,
    persistence,
    persistence_with_ceiling,
    reliability,
    window_estimates,
)

WINDOW = 126


def make_panel(sigma, mu, n_names=300, window=WINDOW, blocks=7, seed=11) -> pd.DataFrame:
    """GBM panel with heterogeneous but time-constant (mu_i, sigma_i)."""
    rng = np.random.default_rng(seed)
    sigma = np.broadcast_to(np.asarray(sigma, float), (n_names,))
    mu = np.broadcast_to(np.asarray(mu, float), (n_names,))

    n_days = window * blocks
    dt = 1.0 / TRADING_DAYS
    inc = rng.normal(
        (mu - 0.5 * sigma**2) * dt, sigma * np.sqrt(dt), size=(n_days, n_names)
    )
    prices = 100.0 * np.exp(np.vstack([np.zeros(n_names), np.cumsum(inc, axis=0)]))
    return pd.DataFrame(
        prices,
        index=pd.date_range("2020-01-02", periods=n_days + 1, freq="B"),
        columns=[f"N{i}" for i in range(n_names)],
    )


@pytest.fixture(scope="module")
def spread_panel() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    return make_panel(rng.uniform(0.20, 0.90, 300), rng.uniform(-0.20, 0.60, 300))


@pytest.fixture(scope="module")
def measured(spread_panel) -> pd.DataFrame:
    return persistence_with_ceiling(spread_panel, window=WINDOW).set_index("quantity")


# --- the headline result ------------------------------------------------------

def test_variance_ranking_survives_into_the_next_window(measured):
    assert measured.loc["sigma_sq", "spearman"] > 0.85


def test_drift_ranking_does_not(measured):
    assert measured.loc["mu", "spearman"] < 0.35
    assert measured.loc["g", "spearman"] < 0.35


def test_the_gap_between_them_is_the_projects_central_claim(measured):
    """sigma^2 must out-persist mu by a wide margin, not merely edge it."""
    assert measured.loc["sigma_sq", "spearman"] - measured.loc["mu", "spearman"] > 0.5


def test_drift_forecasts_are_worse_than_guessing_the_average(measured):
    """Negative out-of-sample R2: last window's mu is worse than the mean."""
    assert measured.loc["mu", "r2_oos"] < -0.3
    assert measured.loc["sigma_sq", "r2_oos"] > 0.5


def test_quantities_are_ordered_by_persistence(spread_panel):
    table = persistence(spread_panel, window=WINDOW)
    assert table["spearman"].is_monotonic_decreasing
    assert table.iloc[0]["quantity"] in {"sigma_sq", "drag"}
    assert table.iloc[-1]["quantity"] in {"mu", "g"}


# --- it matches what the standard errors predict --------------------------------

def test_measured_persistence_matches_the_analytic_reliability(spread_panel, measured):
    """Var(signal)/(Var(signal)+Var(noise)) predicts the answer in advance.

    True parameters are frozen, so persistence is capped by sampling error
    alone, and metrics.py's standard errors say exactly what that cap is.
    """
    lo, hi = 0.20, 0.90
    e_s2 = (hi**3 - lo**3) / (3 * (hi - lo))
    e_s4 = (hi**5 - lo**5) / (5 * (hi - lo))

    predicted_sigma_sq = reliability(e_s4 - e_s2**2, 2 * e_s4 / (WINDOW - 1))
    predicted_mu = reliability(0.8**2 / 12, TRADING_DAYS * e_s2 / WINDOW)

    assert predicted_sigma_sq == pytest.approx(0.95, abs=0.02)
    assert predicted_mu == pytest.approx(0.07, abs=0.03)

    assert measured.loc["sigma_sq", "pearson"] == pytest.approx(predicted_sigma_sq, abs=0.06)
    assert measured.loc["mu", "pearson"] == pytest.approx(predicted_mu, abs=0.10)


def test_reported_ceiling_brackets_the_measurement(measured):
    """The module's own ceiling must land near what it measures."""
    for quantity in ("sigma_sq", "drag"):
        ceiling = measured.loc[quantity, "ceiling"]
        assert ceiling == pytest.approx(measured.loc[quantity, "pearson"], abs=0.10)


# --- the control that proves the metric measures signal, not ease of estimation ---

def test_persistence_collapses_when_every_name_shares_one_sigma():
    """With no cross-sectional spread there is nothing to rank, so a high score
    for sigma^2 would mean the metric is measuring the estimator, not the signal."""
    rng = np.random.default_rng(5)
    flat = persistence(
        make_panel(0.45, rng.uniform(-0.20, 0.60, 300), seed=3), window=WINDOW
    ).set_index("quantity")

    assert flat.loc["sigma_sq", "spearman"] < 0.20
    assert flat.loc["sigma_sq", "r2_oos"] < 0.0


def test_wider_sigma_spread_raises_variance_persistence():
    rng = np.random.default_rng(7)
    mu = rng.uniform(-0.20, 0.60, 300)
    narrow = persistence(make_panel(rng.uniform(0.40, 0.50, 300), mu, seed=8), window=WINDOW)
    wide = persistence(make_panel(rng.uniform(0.15, 1.00, 300), mu, seed=8), window=WINDOW)

    narrow_rho = narrow.set_index("quantity").loc["sigma_sq", "spearman"]
    wide_rho = wide.set_index("quantity").loc["sigma_sq", "spearman"]
    assert wide_rho > narrow_rho + 0.3


# --- invariants ------------------------------------------------------------------

def test_drag_persistence_is_variance_persistence(measured):
    """drag = sigma^2 / 2, a monotone transform, so the rank score is identical.

    Worth pinning: it means "the drag ranking is stable" and "the variance
    ranking is stable" are the same statement, not two pieces of evidence.
    """
    assert measured.loc["drag", "spearman"] == pytest.approx(
        measured.loc["sigma_sq", "spearman"], abs=1e-12
    )


def test_windows_do_not_overlap(spread_panel):
    """Overlapping windows would manufacture persistence from shared data."""
    est = window_estimates(spread_panel, window=WINDOW)
    n_returns = len(spread_panel) - 1
    assert len(est["sigma_sq"]) == n_returns // WINDOW


def test_every_quantity_is_reported(spread_panel):
    table = persistence(spread_panel, window=WINDOW)
    assert set(table["quantity"]) == set(QUANTITIES)


def test_mu_equals_g_plus_half_variance_in_every_window(spread_panel):
    est = window_estimates(spread_panel, window=WINDOW)
    assert np.allclose(est["mu"], est["g"] + est["drag"], equal_nan=True)


# --- the noise model ---------------------------------------------------------------

def test_noise_variance_matches_the_standard_errors_in_metrics():
    sigma = np.array([0.3, 0.5])
    n = 126
    assert noise_variance("sigma_sq", sigma, n) == pytest.approx(
        float(np.mean(2 * sigma**4 / (n - 1)))
    )
    assert noise_variance("g", sigma, n) == pytest.approx(
        float(np.mean(TRADING_DAYS * sigma**2 / n))
    )
    # drag = sigma^2 / 2 scales its variance by a quarter
    assert noise_variance("drag", sigma, n) == pytest.approx(
        0.25 * noise_variance("sigma_sq", sigma, n)
    )


def test_drift_noise_dwarfs_variance_noise():
    """The arithmetic reason the drag is the steadier quantity."""
    sigma = np.full(50, 0.5)
    ratio = noise_variance("mu", sigma, 126) / noise_variance("sigma_sq", sigma, 126)
    assert ratio > 100


@pytest.mark.parametrize(
    "signal, noise, expected", [(1.0, 0.0, 1.0), (0.0, 1.0, 0.0), (1.0, 1.0, 0.5)]
)
def test_reliability_formula(signal, noise, expected):
    assert reliability(signal, noise) == pytest.approx(expected)


def test_reliability_is_nan_without_variance():
    assert np.isnan(reliability(0.0, 0.0))


# --- degenerate input ----------------------------------------------------------------

def test_too_little_history_yields_nothing(spread_panel):
    assert window_estimates(spread_panel.head(200), window=WINDOW) == {}
    assert persistence(spread_panel.head(200), window=WINDOW).empty


def test_too_few_names_yields_nothing(spread_panel):
    assert persistence(spread_panel.iloc[:, :3], window=WINDOW, min_names=10).empty


def test_window_below_two_is_rejected(spread_panel):
    with pytest.raises(ValueError, match="at least 2"):
        window_estimates(spread_panel, window=1)


def test_gaps_do_not_break_the_estimate(spread_panel):
    holed = spread_panel.copy()
    holed.iloc[300:310, 0] = np.nan
    table = persistence(holed, window=WINDOW)
    assert not table.empty
    assert np.isfinite(table["spearman"]).all()


def test_ceiling_on_an_empty_frame():
    assert persistence_with_ceiling(pd.DataFrame()).empty
