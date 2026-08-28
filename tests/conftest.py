"""Shared fixtures.

The screener's data layer talks to Yahoo Finance and KRX KIND, neither of which
is reachable from CI. Everything here therefore builds *synthetic* inputs with
known (mu, sigma) so the full screen() path can be exercised offline and the
estimates checked against the truth that generated them.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from krxdrag.metrics import simulate_gbm

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# symbol -> (name, sector, market, mu, sigma, daily volume)
SYNTHETIC_SPEC: dict[str, tuple[str, str, str, float, float, float]] = {
    "000001.KS": ("가나전자", "전자부품", "KOSPI", 0.10, 0.20, 400_000.0),
    "000002.KS": ("나다전자", "전자부품", "KOSPI", 0.05, 0.45, 300_000.0),
    "000003.KS": ("다라전자", "전자부품", "KOSPI", 0.12, 0.30, 250_000.0),
    "000004.KS": ("라마화학", "화학", "KOSPI", 0.08, 0.55, 500_000.0),
    "000005.KS": ("마바화학", "화학", "KOSPI", -0.03, 0.35, 200_000.0),
    "000006.KQ": ("바사바이오", "바이오", "KOSDAQ", 0.20, 0.70, 350_000.0),
    "000007.KQ": ("사아바이오", "바이오", "KOSDAQ", 0.02, 0.60, 180_000.0),
    # deliberately illiquid: must be dropped by the turnover filter
    "000008.KQ": ("아자소형", "기타", "KOSDAQ", 0.04, 0.25, 1.0),
}


@pytest.fixture
def synthetic_universe() -> pd.DataFrame:
    """Same schema universe.load_universe() returns."""
    rows = [
        {
            "name": name,
            "code": symbol[:6],
            "sector": sector,
            "market": market,
            "symbol": symbol,
        }
        for symbol, (name, sector, market, _, _, _) in SYNTHETIC_SPEC.items()
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_prices() -> pd.DataFrame:
    """Same long-format frame data.load_prices() returns, OHLCV included.

    Closes come from simulate_gbm so the estimates the screen produces can be
    checked against the (mu, sigma) that generated them. The open/high/low
    around each close are drawn as a plausible bar -- enough for the range
    estimators to run end to end, not calibrated to reproduce sigma exactly;
    test_volatility.py owns that with properly simulated intraday paths.
    """
    n_days = 600
    start = date(2023, 1, 2)
    dates = pd.to_datetime([start + timedelta(days=i) for i in range(n_days + 1)])

    frames = []
    for seed, (symbol, (_, _, _, mu, sigma, volume)) in enumerate(SYNTHETIC_SPEC.items()):
        path = simulate_gbm(mu=mu, sigma=sigma, n_days=n_days, s0=50_000.0, seed=seed + 1)
        rng = np.random.default_rng(1000 + seed)

        close = path
        prev = np.concatenate([[path[0]], path[:-1]])
        daily = sigma / np.sqrt(252.0)
        open_ = prev * np.exp(rng.normal(0.0, daily * 0.3, size=close.size))
        wick = np.abs(rng.normal(0.0, daily * 0.6, size=close.size))
        high = np.maximum(open_, close) * np.exp(wick)
        low = np.minimum(open_, close) * np.exp(-wick)

        frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "symbol": symbol,
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def offline_screen(monkeypatch, synthetic_universe, synthetic_prices):
    """Patch the two network calls out of screener so screen() runs offline.

    Yields a dict the test can mutate to see what screen() asked for.
    """
    from krxdrag import screener

    calls: dict = {}

    def fake_universe(markets=("KOSPI", "KOSDAQ"), use_cache=True, day=None):
        calls["markets"] = markets
        return synthetic_universe[synthetic_universe["market"].isin(markets)].reset_index(
            drop=True
        )

    def fake_prices(symbols, lookback_days=504, batch_size=60, use_cache=True, end=None):
        calls["symbols"] = list(symbols)
        return synthetic_prices[synthetic_prices["symbol"].isin(symbols)].reset_index(
            drop=True
        )

    monkeypatch.setattr(screener, "load_universe", fake_universe)
    monkeypatch.setattr(screener, "load_prices", fake_prices)
    return calls


@pytest.fixture
def real_screen_df() -> pd.DataFrame:
    """The real KRX snapshot committed to the repo, for smoke-testing aggregates."""
    matches = sorted((PROJECT_ROOT / "reports").glob("krx_drag_*.csv"))
    if not matches:
        pytest.skip("no committed screen snapshot in reports/")
    return pd.read_csv(matches[-1])


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20260818)
