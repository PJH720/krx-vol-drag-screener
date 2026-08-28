"""The screen must recover the parameters that generated its input.

These run entirely on synthetic GBM paths (see conftest), so the data layer is
exercised without touching the network.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from krxdrag.config import ScreenConfig
from krxdrag.data import to_wide
from krxdrag.screener import screen, summarise

from .conftest import SYNTHETIC_SPEC


def _cfg(**kw) -> ScreenConfig:
    base = dict(lookback_days=600, min_obs=250, min_median_turnover=1e8, batch_size=8)
    base.update(kw)
    return ScreenConfig(**base)


def test_screen_recovers_known_sigma(offline_screen):
    """Every surviving name's sigma^2 must land within 4 SE of the truth.

    Deliberately *not* asserting the 95% chi-square interval covers the truth
    for each name: with seven names that fails ~30% of the time by design.
    Interval coverage is checked statistically in test_metrics.py; here we want
    a consistency bound that never flakes.
    """
    df = screen(_cfg(), use_cache=False)
    assert not df.empty

    for _, row in df.iterrows():
        true_sq = SYNTHETIC_SPEC[row["symbol"]][4] ** 2
        se = true_sq * np.sqrt(2.0 / (row["n_obs"] - 1))
        assert abs(row["sigma_sq"] - true_sq) < 4.0 * se, row["symbol"]


def test_screen_preserves_ito_identity(offline_screen):
    df = screen(_cfg(), use_cache=False)
    residual = np.abs((df["mu"] - df["g"]) - df["drag"]).max()
    assert residual < 1e-12


def test_turnover_filter_drops_illiquid_name(offline_screen):
    """000008.KQ trades 1 share a day and must not survive the liquidity floor."""
    df = screen(_cfg(min_median_turnover=1e8), use_cache=False)
    assert "000008.KQ" not in set(df["symbol"])

    loose = screen(_cfg(min_median_turnover=0.0), use_cache=False)
    assert "000008.KQ" in set(loose["symbol"])


def test_min_obs_filter_rejects_everything_when_impossible(offline_screen):
    df = screen(_cfg(min_obs=10_000), use_cache=False)
    assert df.empty


def test_results_are_ranked_by_drag_descending(offline_screen):
    df = screen(_cfg(), use_cache=False)
    assert df["drag"].is_monotonic_decreasing
    assert df["rank"].tolist() == list(range(1, len(df) + 1))


def test_markets_argument_is_honoured(offline_screen):
    df = screen(_cfg(markets=("KOSPI",)), use_cache=False)
    assert set(df["market"]) == {"KOSPI"}


def test_empty_universe_returns_empty_frame(monkeypatch, offline_screen):
    from krxdrag import screener

    monkeypatch.setattr(
        screener, "load_prices", lambda *a, **k: pd.DataFrame()
    )
    assert screen(_cfg(), use_cache=False).empty


def test_summarise_reports_expected_keys(offline_screen):
    s = summarise(screen(_cfg(), use_cache=False))
    for key in (
        "n_names",
        "median_sigma",
        "median_drag",
        "share_g_negative_mu_positive",
        "median_gbm_score",
    ):
        assert key in s
    assert s["median_drag"] == pytest.approx(0.5 * s["median_sigma"] ** 2, rel=0.35)


def test_summarise_of_empty_frame_is_empty():
    assert summarise(pd.DataFrame()) == {}


def test_to_wide_tolerates_duplicate_rows(synthetic_prices):
    """A duplicated batch must not sink the run."""
    doubled = pd.concat([synthetic_prices, synthetic_prices.head(50)], ignore_index=True)
    wide = to_wide(doubled)
    assert wide.index.is_unique
    assert set(wide.columns) == set(synthetic_prices["symbol"].unique())


def test_jump_columns_are_present_and_consistent(offline_screen):
    """The jump split must reconcile with the drag it decomposes."""
    df = screen(_cfg(), use_cache=False)
    for col in ("drag_cont", "drag_jump", "jump_ratio", "bns_pvalue", "has_jumps"):
        assert col in df.columns

    total = df["drag_cont"] + df["drag_jump"]
    # cont + jump reconstructs the quadratic-variation drag (not 0.5*sample
    # variance, which differs by the mean-return term)
    assert np.allclose(total, 0.5 * df["realized_qv"], rtol=1e-10)
    assert (df["jump_ratio"].between(0.0, 1.0)).all()


def test_jump_decomposition_can_be_switched_off(offline_screen):
    df = screen(_cfg(decompose_jumps=False), use_cache=False)
    assert "drag_cont" not in df.columns
    assert not df.empty


def test_drag_trend_column_is_present(offline_screen):
    df = screen(_cfg(rolling_window=126), use_cache=False)
    assert "drag_trend" in df.columns
    assert np.isfinite(df["drag_trend"]).all()


def test_rolling_window_zero_disables_the_trend_column(offline_screen):
    df = screen(_cfg(rolling_window=0), use_cache=False)
    assert "drag_trend" not in df.columns


def test_range_volatility_columns_appear_beside_close_to_close(offline_screen):
    """Range estimates are added, never substituted for the headline sigma."""
    df = screen(_cfg(), use_cache=False)

    assert "sigma_sq" in df.columns          # close-to-close, still the primary
    for name in ("parkinson", "garman_klass", "rogers_satchell", "yang_zhang"):
        assert f"sigma_sq_{name}" in df.columns
        assert f"drag_{name}" in df.columns
        assert np.allclose(df[f"drag_{name}"], 0.5 * df[f"sigma_sq_{name}"])

    assert "limit_hit_share" in df.columns
    assert "range_gap" in df.columns


def test_headline_drag_still_comes_from_close_to_close(offline_screen):
    """The chi-square interval only holds for the sample variance, so the
    reported drag must keep coming from it even when range estimates exist."""
    df = screen(_cfg(), use_cache=False)
    assert np.allclose(df["drag"], 0.5 * df["sigma_sq"])
    assert not np.allclose(df["drag"], df["drag_yang_zhang"])


def test_range_volatility_can_be_switched_off(offline_screen):
    df = screen(_cfg(range_volatility=False), use_cache=False)
    assert "sigma_sq_yang_zhang" not in df.columns
    assert "sigma_sq" in df.columns


def test_screen_survives_prices_without_ohlc(monkeypatch, offline_screen, synthetic_prices):
    """A feed that returns only closes must degrade, not crash."""
    from krxdrag import screener

    closes_only = synthetic_prices.drop(columns=["open", "high", "low"])
    monkeypatch.setattr(screener, "load_prices", lambda *a, **k: closes_only)

    df = screen(_cfg(), use_cache=False)
    assert not df.empty
    assert "sigma_sq" in df.columns
    assert "sigma_sq_yang_zhang" not in df.columns


def test_fdr_columns_are_added_cross_sectionally(offline_screen):
    """The correction is a property of the whole screen, not of one name."""
    df = screen(_cfg(), use_cache=False)

    for base in ("jb", "lb_sq", "bns"):
        assert f"{base}_qvalue" in df.columns
        assert f"{base}_rejected_fdr" in df.columns

    # a q-value can only inflate its p-value, never shrink it
    finite = np.isfinite(df["jb_qvalue"])
    assert (df.loc[finite, "jb_qvalue"] >= df.loc[finite, "jb_pvalue"] - 1e-12).all()


def test_fdr_flags_are_a_subset_of_the_uncorrected_ones(offline_screen):
    df = screen(_cfg(), use_cache=False)
    raw = df["bns_pvalue"] < 0.05
    assert not (df["bns_rejected_fdr"] & ~raw).any()


def test_fdr_can_be_switched_off(offline_screen):
    df = screen(_cfg(fdr_q=0.0), use_cache=False)
    assert "jb_qvalue" not in df.columns
    assert "jb_pvalue" in df.columns


def test_summarise_reports_both_jump_shares(offline_screen):
    s = summarise(screen(_cfg(), use_cache=False))
    assert s["share_jumps_fdr"] <= s["share_jumps_raw"]
