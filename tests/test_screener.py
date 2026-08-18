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
