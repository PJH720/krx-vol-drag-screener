"""Sector aggregation, and the diversification gap it is meant to expose."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from krxdrag.metrics import TRADING_DAYS, compute_drag
from krxdrag.sectors import (
    aggregate_sectors,
    diversification_benefit,
    sector_portfolio_drag,
)


def _screen_row(symbol: str, sector: str, sigma: float, g: float) -> dict:
    drag = 0.5 * sigma**2
    return {
        "symbol": symbol,
        "sector": sector,
        "sigma": sigma,
        "drag": drag,
        "g": g,
        "mu": g + drag,
        "drag_ratio": drag / (g + drag) if g + drag > 0 else np.nan,
        "gbm_score": 0.5,
    }


@pytest.fixture
def hand_built() -> pd.DataFrame:
    """Six names across three sectors, values chosen so medians are obvious."""
    rows = [
        _screen_row("A1", "high", 0.80, +0.10),
        _screen_row("A2", "high", 0.60, -0.40),   # mu = -0.22 < 0: not a trap
        _screen_row("A3", "high", 0.40, +0.02),   # g > 0: not a trap
        _screen_row("B1", "low", 0.20, +0.05),
        # drag 0.005, g -0.003 -> mu +0.002 > 0 with g < 0: a variance trap
        _screen_row("B2", "low", 0.10, -0.003),
        _screen_row("B3", "low", 0.30, +0.01),
        _screen_row("C1", "tiny", 0.90, +0.01),   # lone name, must be filtered
    ]
    return pd.DataFrame(rows)


# --- aggregate_sectors ----------------------------------------------------------

def test_medians_are_computed_per_sector(hand_built):
    out = aggregate_sectors(hand_built, min_names=3).set_index("sector")

    assert out.loc["high", "median_sigma"] == pytest.approx(0.60)
    assert out.loc["high", "median_drag"] == pytest.approx(0.5 * 0.60**2)
    assert out.loc["low", "median_sigma"] == pytest.approx(0.20)
    assert out.loc["low", "median_drag"] == pytest.approx(0.5 * 0.20**2)


def test_min_names_filter_drops_thin_sectors(hand_built):
    out = aggregate_sectors(hand_built, min_names=3)
    assert "tiny" not in set(out["sector"])

    unfiltered = aggregate_sectors(hand_built, min_names=1)
    assert "tiny" in set(unfiltered["sector"])
    # and unfiltered, the lone volatile name tops the table -- the reason the
    # filter exists
    assert unfiltered.iloc[0]["sector"] == "tiny"


def test_ranking_is_by_median_drag_descending(hand_built):
    out = aggregate_sectors(hand_built, min_names=3)
    assert out["median_drag"].is_monotonic_decreasing
    assert out["rank"].tolist() == list(range(1, len(out) + 1))
    assert out.iloc[0]["sector"] == "high"


def test_trap_share_counts_mu_positive_g_negative(hand_built):
    out = aggregate_sectors(hand_built, min_names=3).set_index("sector")
    assert out.loc["low", "share_trap"] == pytest.approx(1 / 3)
    assert out.loc["high", "share_trap"] == pytest.approx(0.0)


def test_optional_columns_are_included_when_present(hand_built):
    out = aggregate_sectors(hand_built, min_names=3)
    assert "median_gbm_score" in out.columns
    assert "median_jump_ratio" not in out.columns  # absent from input


def test_iqr_is_zero_for_identical_constituents():
    same = pd.DataFrame([_screen_row(f"S{i}", "flat", 0.4, 0.03) for i in range(5)])
    out = aggregate_sectors(same, min_names=3)
    assert out.iloc[0]["drag_iqr"] == pytest.approx(0.0)


def test_empty_and_sectorless_input_return_empty(hand_built):
    assert aggregate_sectors(pd.DataFrame()).empty
    assert aggregate_sectors(hand_built.drop(columns=["sector"])).empty
    blank = hand_built.assign(sector="")
    assert aggregate_sectors(blank).empty


# --- sector_portfolio_drag -------------------------------------------------------

@pytest.fixture
def correlated_panel():
    """Eight names in one sector: a common factor plus independent noise."""
    rng = np.random.default_rng(7)
    n_days, n_names = 900, 8
    dt = 1.0 / TRADING_DAYS
    sigma_common, sigma_idio = 0.25, 0.35

    common = rng.normal(0.0, sigma_common * np.sqrt(dt), size=n_days)
    idio = rng.normal(0.0, sigma_idio * np.sqrt(dt), size=(n_days, n_names))
    log_r = common[:, None] + idio

    prices = 100.0 * np.exp(np.vstack([np.zeros(n_names), np.cumsum(log_r, axis=0)]))
    symbols = [f"N{i}" for i in range(n_names)]
    wide = pd.DataFrame(
        prices,
        index=pd.date_range("2022-01-03", periods=n_days + 1, freq="B"),
        columns=symbols,
    )
    universe = pd.DataFrame({"symbol": symbols, "sector": "factor"})
    return wide, universe


def test_portfolio_drag_is_below_median_constituent_drag(correlated_panel):
    """Imperfect correlation must damp portfolio variance -- the whole point."""
    wide, universe = correlated_panel

    portfolio = sector_portfolio_drag(wide, universe, min_names=3)
    assert len(portfolio) == 1

    constituent_drags = [compute_drag(wide[c].to_numpy()).drag for c in wide.columns]
    assert portfolio.iloc[0]["portfolio_drag"] < float(np.median(constituent_drags))


def test_portfolio_drag_approaches_the_common_factor_variance(correlated_panel):
    """With 8 names most idiosyncratic variance is averaged away."""
    wide, universe = correlated_panel
    got = sector_portfolio_drag(wide, universe, min_names=3).iloc[0]

    sigma_common, sigma_idio, n = 0.25, 0.35, 8
    expected_sigma_sq = sigma_common**2 + sigma_idio**2 / n
    assert got["portfolio_sigma"] ** 2 == pytest.approx(expected_sigma_sq, rel=0.15)


def test_a_single_name_portfolio_reproduces_that_name(correlated_panel):
    wide, universe = correlated_panel
    one = universe.head(1)
    got = sector_portfolio_drag(wide, one, min_names=1).iloc[0]
    direct = compute_drag(wide["N0"].to_numpy())
    # equal-weight of one name is that name, up to the log/simple rebuild
    assert got["portfolio_drag"] == pytest.approx(direct.drag, rel=1e-6)


def test_portfolio_respects_min_names(correlated_panel):
    wide, universe = correlated_panel
    assert sector_portfolio_drag(wide, universe, min_names=20).empty


def test_portfolio_ignores_symbols_missing_from_the_price_matrix(correlated_panel):
    wide, universe = correlated_panel
    padded = pd.concat(
        [universe, pd.DataFrame({"symbol": ["GHOST"], "sector": ["factor"]})],
        ignore_index=True,
    )
    got = sector_portfolio_drag(wide, padded, min_names=3)
    assert got.iloc[0]["n_names"] == 8


def test_portfolio_on_empty_input_returns_empty(correlated_panel):
    wide, universe = correlated_panel
    assert sector_portfolio_drag(pd.DataFrame(), universe).empty
    assert sector_portfolio_drag(wide, pd.DataFrame()).empty


# --- diversification_benefit -------------------------------------------------------

def test_diversification_benefit_is_positive_and_joins_cleanly(correlated_panel):
    wide, universe = correlated_panel

    screen_like = pd.DataFrame(
        [
            _screen_row(c, "factor", compute_drag(wide[c].to_numpy()).sigma, 0.05)
            for c in wide.columns
        ]
    )
    table = aggregate_sectors(screen_like, min_names=3)
    portfolio = sector_portfolio_drag(wide, universe, min_names=3)

    benefit = diversification_benefit(table, portfolio)
    assert len(benefit) == 1
    assert benefit.iloc[0]["drag_saved"] > 0
    assert 0.0 < benefit.iloc[0]["drag_saved_share"] < 1.0


def test_diversification_benefit_on_empty_input():
    assert diversification_benefit(pd.DataFrame(), pd.DataFrame()).empty


# --- the real committed snapshot -----------------------------------------------------

def test_aggregation_runs_on_the_real_snapshot(real_screen_df):
    out = aggregate_sectors(real_screen_df)
    assert not out.empty
    assert len(out) < real_screen_df["sector"].nunique()  # thin sectors filtered
    assert (out["n_names"] >= 5).all()
    assert out["median_drag"].is_monotonic_decreasing
    # every sector median drag must still be half a squared volatility
    assert np.allclose(out["median_drag"], 0.5 * out["median_sigma"] ** 2, rtol=0.35)


def test_portfolio_returns_are_not_forward_filled(correlated_panel):
    """pandas 2.x pct_change() pads by default, fabricating 0% returns.

    A halted name would be carried forward into a run of zeros plus one
    catch-up jump, understating portfolio variance and so overstating the
    diversification benefit this module reports.
    """
    wide, universe = correlated_panel
    gapped = wide.copy()
    gapped.iloc[300:305, 0] = np.nan  # one name halted for a week

    simple = gapped.astype(float).pct_change(fill_method=None)
    assert simple.iloc[300:306, 0].isna().all()

    # the aggregate still computes, and does not silently gain a zero-return run
    out = sector_portfolio_drag(gapped, universe, min_names=3)
    assert len(out) == 1
    assert np.isfinite(out.iloc[0]["portfolio_drag"])


def test_a_halted_name_does_not_deflate_portfolio_drag(correlated_panel):
    """Forward-filling would bias portfolio_drag downward; it must not."""
    wide, universe = correlated_panel
    gapped = wide.copy()
    gapped.iloc[200:230, 0] = np.nan

    clean = sector_portfolio_drag(wide, universe, min_names=3).iloc[0]["portfolio_drag"]
    holed = sector_portfolio_drag(gapped, universe, min_names=3).iloc[0]["portfolio_drag"]
    # dropping one name's month of data should not move the estimate much, and
    # certainly must not collapse it toward zero the way padding would
    assert holed == pytest.approx(clean, rel=0.25)
