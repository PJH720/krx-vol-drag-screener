"""Geared exposure: the theory must match paths actually built by rebalancing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from krxdrag.leverage import (
    KRX_LEVERAGED_ETFS,
    LeveragedETF,
    audit_leveraged_etfs,
    critical_volatility,
    estimate_realized_leverage,
    leverage_curve,
    leveraged_metrics,
    optimal_leverage,
    simulate_leveraged_path,
)
from krxdrag.metrics import compute_drag, simulate_gbm


# --- the load-bearing test: theory vs a real rebalanced path --------------------

@pytest.mark.parametrize("leverage", [1.0, 2.0, 3.0, -1.0, -2.0])
def test_simulated_geared_path_matches_the_closed_form(leverage):
    """g_L = L*mu - 0.5*L^2*sigma^2 must hold on a path built day by day."""
    underlying = simulate_gbm(mu=0.10, sigma=0.30, n_days=20_000, seed=3)
    m = compute_drag(underlying)

    geared = simulate_leveraged_path(underlying, leverage)
    realized_g = compute_drag(geared).g
    predicted_g = leveraged_metrics(m, leverage).levered_g

    # Discrete daily rebalancing differs from the continuous limit by a
    # higher-order term that grows with |L|, so scale the tolerance with it.
    assert realized_g == pytest.approx(predicted_g, abs=0.002 * leverage**2)


@pytest.mark.parametrize("leverage", [2.0, 3.0, -2.0])
def test_the_penalty_against_naive_expectations_is_l_squared_minus_l(leverage):
    """A holder expecting L*g is short by exactly (L^2 - L)*D."""
    m = compute_drag(simulate_gbm(mu=0.12, sigma=0.40, n_days=4000, seed=6))
    lm = leveraged_metrics(m, leverage)

    assert lm.naive_g - lm.levered_g == pytest.approx(lm.drag_penalty, abs=1e-12)
    assert lm.drag_penalty == pytest.approx((leverage**2 - leverage) * m.drag, abs=1e-12)


def test_an_inverse_2x_pays_the_same_penalty_as_a_3x_long():
    """(L^2 - L) is 6 for both L=-2 and L=3 -- worth stating explicitly."""
    m = compute_drag(simulate_gbm(mu=0.08, sigma=0.35, n_days=3000, seed=11))
    assert leveraged_metrics(m, -2.0).drag_penalty == pytest.approx(
        leveraged_metrics(m, 3.0).drag_penalty, abs=1e-12
    )


def test_unit_leverage_is_the_identity():
    m = compute_drag(simulate_gbm(mu=0.09, sigma=0.28, n_days=2500, seed=4))
    lm = leveraged_metrics(m, 1.0)

    assert lm.levered_g == pytest.approx(m.g, abs=1e-12)
    assert lm.levered_mu == pytest.approx(m.mu, abs=1e-12)
    assert lm.levered_sigma == pytest.approx(m.sigma, abs=1e-12)
    assert lm.levered_drag == pytest.approx(m.drag, abs=1e-12)
    assert lm.drag_penalty == pytest.approx(0.0, abs=1e-12)


# --- critical volatility --------------------------------------------------------

@pytest.mark.parametrize("mu, leverage", [(0.10, 2.0), (0.20, 3.0), (0.05, 1.0)])
def test_growth_vanishes_exactly_at_the_critical_volatility(mu, leverage):
    sigma_star = critical_volatility(mu, leverage)
    g_at_star = leverage * mu - 0.5 * leverage**2 * sigma_star**2
    assert g_at_star == pytest.approx(0.0, abs=1e-12)


def test_critical_volatility_has_the_documented_value():
    """mu=10%, L=2 -> sigma* = sqrt(2*0.10/2) = 31.6%."""
    assert critical_volatility(0.10, 2.0) == pytest.approx(np.sqrt(0.10), rel=1e-12)


def test_critical_volatility_is_undefined_against_the_drift():
    assert np.isnan(critical_volatility(0.10, -2.0))   # inverse on a rising asset
    assert np.isnan(critical_volatility(-0.10, 2.0))   # long a falling asset
    assert np.isnan(critical_volatility(0.10, 0.0))


def test_growth_is_negative_above_the_critical_volatility():
    mu, leverage = 0.10, 2.0
    star = critical_volatility(mu, leverage)
    for sigma in (star * 1.1, star * 1.5):
        assert leverage * mu - 0.5 * leverage**2 * sigma**2 < 0


# --- optimal leverage -------------------------------------------------------------

def test_optimal_leverage_maximises_the_curve():
    mu, sigma = 0.12, 0.30
    star = optimal_leverage(mu, sigma**2)
    assert star == pytest.approx(mu / sigma**2)

    curve = leverage_curve(mu, sigma, np.linspace(-1, 4, 2001))
    assert curve.loc[curve["levered_g"].idxmax(), "leverage"] == pytest.approx(
        star, abs=0.01
    )


def test_optimal_leverage_is_nan_without_variance():
    assert np.isnan(optimal_leverage(0.10, 0.0))


# --- the curve ----------------------------------------------------------------------

def test_curve_is_an_inverted_parabola_through_the_origin():
    curve = leverage_curve(0.10, 0.30)
    at_zero = curve.loc[curve["leverage"].abs().idxmin()]
    assert at_zero["levered_g"] == pytest.approx(0.0, abs=1e-9)
    assert at_zero["levered_drag"] == pytest.approx(0.0, abs=1e-9)
    # drag is symmetric in L, growth is not
    left = curve[curve["leverage"] == -2.0]["levered_drag"].iloc[0]
    right = curve[curve["leverage"] == 2.0]["levered_drag"].iloc[0]
    assert left == pytest.approx(right)


def test_curve_drag_is_quadratic_in_leverage():
    curve = leverage_curve(0.10, 0.40, np.array([1.0, 2.0, 3.0]))
    drags = curve["levered_drag"].to_numpy()
    assert drags[1] == pytest.approx(4 * drags[0])
    assert drags[2] == pytest.approx(9 * drags[0])


# --- realised leverage regression ------------------------------------------------------

@pytest.mark.parametrize("leverage", [2.0, 3.0, -1.0, -2.0])
def test_regression_recovers_the_multiple_that_built_the_path(leverage):
    underlying = simulate_gbm(mu=0.08, sigma=0.25, n_days=1500, seed=15)
    geared = simulate_leveraged_path(underlying, leverage)

    slope, r2 = estimate_realized_leverage(geared, underlying)
    assert slope == pytest.approx(leverage, abs=1e-6)
    assert r2 == pytest.approx(1.0, abs=1e-6)


def test_regression_is_nan_on_insufficient_or_flat_data():
    assert np.isnan(estimate_realized_leverage(np.ones(5), np.ones(5))[0])
    assert np.isnan(estimate_realized_leverage(np.ones(100), np.ones(100))[0])


# --- fees and wipeout ---------------------------------------------------------------------

def test_costs_reduce_levered_growth():
    m = compute_drag(simulate_gbm(mu=0.10, sigma=0.30, n_days=2000, seed=8))
    free = leveraged_metrics(m, 2.0)
    charged = leveraged_metrics(m, 2.0, fee=0.008, financing=0.03)

    # fee applies always, financing on the |L|-1 geared portion
    assert charged.cost == pytest.approx(0.008 + 0.03 * 1.0)
    assert charged.levered_g == pytest.approx(free.levered_g - charged.cost)


def test_no_financing_charge_below_unit_exposure():
    m = compute_drag(simulate_gbm(mu=0.10, sigma=0.30, n_days=2000, seed=8))
    assert leveraged_metrics(m, 0.5, fee=0.01, financing=0.03).cost == pytest.approx(0.01)


def test_a_wipeout_day_does_not_produce_negative_prices():
    """A -60% day at 2x would take the fund below zero; it must floor instead."""
    underlying = np.array([100.0, 40.0, 45.0, 50.0] + [50.0] * 30)
    geared = simulate_leveraged_path(underlying, 2.0)
    assert np.all(geared > 0)
    assert np.all(np.isfinite(geared))


# --- the product table ------------------------------------------------------------------------

def test_product_table_entries_are_well_formed():
    for etf in KRX_LEVERAGED_ETFS:
        assert len(etf.code) == 6 and etf.code.isdigit()
        assert etf.leverage != 0.0
        assert etf.symbol.endswith((".KS", ".KQ"))
        assert len(etf.underlying) == 6 and etf.underlying.isdigit()


def test_audit_flags_a_mislabelled_product():
    """The table is not trusted: a wrong declared multiple must be caught."""
    underlying = simulate_gbm(mu=0.09, sigma=0.30, n_days=800, seed=31)
    truly_2x = simulate_leveraged_path(underlying, 2.0)

    wide = pd.DataFrame({"069500.KS": underlying, "999999.KS": truly_2x})
    products = (
        LeveragedETF("999999", "Mislabelled 3x", 3.0, "069500"),
        LeveragedETF("999999", "Correct 2x", 2.0, "069500"),
    )
    out = audit_leveraged_etfs(wide, products=products)

    assert len(out) == 2
    assert bool(out.iloc[0]["leverage_mismatch"]) is True
    assert bool(out.iloc[1]["leverage_mismatch"]) is False
    assert out.iloc[0]["realized_leverage"] == pytest.approx(2.0, abs=1e-6)


def test_audit_theoretical_and_actual_growth_agree_for_a_real_product():
    underlying = simulate_gbm(mu=0.09, sigma=0.30, n_days=6000, seed=32)
    wide = pd.DataFrame(
        {"069500.KS": underlying, "122630.KS": simulate_leveraged_path(underlying, 2.0)}
    )
    products = (LeveragedETF("122630", "KODEX 레버리지", 2.0, "069500"),)
    row = audit_leveraged_etfs(wide, products=products).iloc[0]

    assert row["actual_g"] == pytest.approx(row["theoretical_g"], abs=0.01)
    assert not row["leverage_mismatch"]


def test_audit_skips_products_without_prices():
    """Only rows whose fund *and* underlying are both priced may appear."""
    wide = pd.DataFrame({"069500.KS": simulate_gbm(0.05, 0.2, 300, seed=1)})
    out = audit_leveraged_etfs(wide)

    # 069500 is the KOSPI200 baseline and is its own underlying, so it audits;
    # every geared product referencing it is skipped for want of prices.
    assert set(out["code"]) == {"069500"}
    assert out.iloc[0]["declared_leverage"] == 1.0

    assert audit_leveraged_etfs(pd.DataFrame()).empty


def test_audit_degrades_to_empty_when_nothing_is_priced():
    """Offline, with no ETF prices at all, the section simply disappears."""
    unrelated = pd.DataFrame({"005930.KS": simulate_gbm(0.05, 0.2, 300, seed=1)})
    assert audit_leveraged_etfs(unrelated).empty


# --- self-referential 1x products (regression) ---------------------------------

def test_a_one_x_product_that_is_its_own_underlying_reports_the_underlying():
    """069500 and 229200 ship with code == underlying.

    Selecting wide[[sym, sym]] yields duplicate columns, so pair[sym] is a
    2-D frame and log_returns flattens it into an interleaved [0, r1, 0, r2, ...]
    series -- every metric on the row comes out wrong.
    """
    underlying = simulate_gbm(mu=0.08, sigma=0.30, n_days=300, seed=5)
    truth = compute_drag(underlying)
    wide = pd.DataFrame({"069500.KS": underlying})

    row = audit_leveraged_etfs(
        wide, products=(LeveragedETF("069500", "KODEX 200", 1.0, "069500"),)
    ).iloc[0]

    assert row["underlying_sigma"] == pytest.approx(truth.sigma, rel=1e-9)
    assert row["actual_g"] == pytest.approx(truth.g, rel=1e-9)
    assert row["realized_leverage"] == pytest.approx(1.0, abs=1e-9)
    assert not row["leverage_mismatch"]


def test_every_shipped_self_referential_product_is_clean():
    """Whatever the table holds, no 1x self-reference may distort its own row."""
    selfref = [e for e in KRX_LEVERAGED_ETFS if e.code == e.underlying]
    assert selfref, "the table is expected to carry 1x baselines"

    underlying = simulate_gbm(mu=0.06, sigma=0.25, n_days=400, seed=6)
    truth = compute_drag(underlying)
    for etf in selfref:
        wide = pd.DataFrame({etf.symbol: underlying})
        row = audit_leveraged_etfs(wide, products=(etf,)).iloc[0]
        assert row["underlying_sigma"] == pytest.approx(truth.sigma, rel=1e-9), etf.code


def test_underlying_symbol_property_matches_symbol_construction():
    for etf in KRX_LEVERAGED_ETFS:
        twin = LeveragedETF(etf.underlying, "x", 1.0, etf.underlying, etf.market)
        assert etf.underlying_symbol == twin.symbol
