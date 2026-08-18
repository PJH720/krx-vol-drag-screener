"""Price loading with an on-disk parquet cache.

Primary source is yfinance with auto_adjust=True, so prices are adjusted for
dividends and splits. That matters here: an unadjusted series understates the
geometric drift g and therefore misstates the wedge against mu.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CACHE_DIR

log = logging.getLogger(__name__)


def _cache_key(symbols: list[str], start: date, end: date) -> Path:
    import hashlib

    h = hashlib.sha1("|".join(sorted(symbols)).encode()).hexdigest()[:12]
    return CACHE_DIR / f"px_{start:%Y%m%d}_{end:%Y%m%d}_{len(symbols)}_{h}.parquet"


def _download_batch(symbols: list[str], start: date, end: date) -> pd.DataFrame:
    import yfinance as yf

    raw = yf.download(
        tickers=symbols,
        start=start.isoformat(),
        end=end.isoformat(),
        auto_adjust=True,
        progress=False,
        threads=True,
        group_by="column",
    )
    if raw is None or raw.empty:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"].copy()
        volume = raw["Volume"].copy() if "Volume" in raw.columns.levels[0] else None
    else:  # single ticker collapses the column index
        close = raw[["Close"]].copy()
        close.columns = symbols[:1]
        volume = raw[["Volume"]].copy()
        volume.columns = symbols[:1]

    close = close.stack(future_stack=True).rename("close").to_frame()
    if volume is not None:
        vol = volume.stack(future_stack=True).rename("volume")
        close = close.join(vol, how="left")
    else:
        close["volume"] = np.nan

    close.index.names = ["date", "symbol"]
    return close.reset_index()


def load_prices(
    symbols: list[str],
    lookback_days: int = 504,
    batch_size: int = 60,
    use_cache: bool = True,
    end: date | None = None,
) -> pd.DataFrame:
    """Long-format frame with columns [date, symbol, close, volume]."""
    end = end or date.today()
    # Calendar span generous enough to contain `lookback_days` trading days.
    start = end - timedelta(days=int(lookback_days * 1.55) + 20)

    path = _cache_key(symbols, start, end)
    if use_cache and path.exists():
        log.info("prices: cache hit %s", path.name)
        return pd.read_parquet(path)

    frames: list[pd.DataFrame] = []
    total = (len(symbols) + batch_size - 1) // batch_size
    for i in range(0, len(symbols), batch_size):
        chunk = symbols[i : i + batch_size]
        idx = i // batch_size + 1
        try:
            df = _download_batch(chunk, start, end)
            if not df.empty:
                frames.append(df)
            log.info("prices: batch %d/%d (%d symbols)", idx, total, len(chunk))
        except Exception as exc:  # a bad batch must not sink the whole run
            log.warning("prices: batch %d/%d failed: %s", idx, total, exc)

    if not frames:
        return pd.DataFrame(columns=["date", "symbol", "close", "volume"])

    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=["close"])
    out = out[out["close"] > 0]
    out.to_parquet(path, index=False)
    log.info("prices: %d rows, %d symbols", len(out), out["symbol"].nunique())
    return out


def to_wide(prices: pd.DataFrame) -> pd.DataFrame:
    """Pivot long price frame to a date x symbol close matrix.

    Batches are not supposed to overlap, but a single retried or duplicated
    batch would make pivot() raise and sink the whole run, so collapse any
    duplicate (date, symbol) pair to its last observation first.
    """
    deduped = prices.drop_duplicates(subset=["date", "symbol"], keep="last")
    return deduped.pivot(index="date", columns="symbol", values="close").sort_index()


def median_turnover(prices: pd.DataFrame) -> pd.Series:
    """Median daily traded value (KRW) per symbol -- the liquidity filter."""
    df = prices.copy()
    df["turnover"] = df["close"] * df["volume"].fillna(0)
    return df.groupby("symbol")["turnover"].median()
