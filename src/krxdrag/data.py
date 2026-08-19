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


# Fields pulled from yfinance, in the order the long frame carries them.
# Open/High/Low are what the range-based volatility estimators run on;
# auto_adjust=True scales all four consistently, so the ratios they depend on
# (H/L, C/O) survive dividend and split adjustment unchanged.
OHLCV_FIELDS: tuple[tuple[str, str], ...] = (
    ("Open", "open"),
    ("High", "high"),
    ("Low", "low"),
    ("Close", "close"),
    ("Volume", "volume"),
)

PRICE_COLUMNS: tuple[str, ...] = tuple(name for _, name in OHLCV_FIELDS)


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

    multi = isinstance(raw.columns, pd.MultiIndex)
    available = set(raw.columns.levels[0]) if multi else set(raw.columns)

    frames: dict[str, pd.Series] = {}
    for source, name in OHLCV_FIELDS:
        if source not in available:
            continue
        if multi:
            field = raw[source].copy()
        else:  # a single ticker collapses the column index
            field = raw[[source]].copy()
            field.columns = symbols[:1]
        frames[name] = field.stack(future_stack=True).rename(name)

    if "close" not in frames:
        return pd.DataFrame()

    out = pd.concat(frames.values(), axis=1)
    for name in PRICE_COLUMNS:
        if name not in out.columns:
            out[name] = np.nan
    out = out[list(PRICE_COLUMNS)]
    out.index.names = ["date", "symbol"]
    return out.reset_index()


def load_prices(
    symbols: list[str],
    lookback_days: int = 504,
    batch_size: int = 60,
    use_cache: bool = True,
    end: date | None = None,
) -> pd.DataFrame:
    """Long-format frame with columns [date, symbol, open, high, low, close, volume].

    Open/High/Low may be NaN when the source omits them; every consumer must
    tolerate that rather than assume a full bar is present.
    """
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
        return pd.DataFrame(columns=["date", "symbol", *PRICE_COLUMNS])

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


def to_bars(prices: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """One symbol's OHLC bars, date-ordered, with incomplete bars dropped.

    A bar is usable by the range estimators only if all four prices are present
    and positive and the high/low actually bracket the open/close; a feed that
    reports a High below its own Close is corrupt, and silently squaring the
    resulting negative log would produce a plausible-looking variance.
    """
    rows = prices[prices["symbol"] == symbol]
    if rows.empty:
        return pd.DataFrame(columns=["date", *PRICE_COLUMNS])

    bars = rows.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    cols = ["open", "high", "low", "close"]
    if not set(cols) <= set(bars.columns):
        return bars.assign(**{c: np.nan for c in cols if c not in bars.columns})

    ok = bars[cols].notna().all(axis=1) & (bars[cols] > 0).all(axis=1)
    ok &= bars["high"] >= bars[["open", "close", "low"]].max(axis=1)
    ok &= bars["low"] <= bars[["open", "close", "high"]].min(axis=1)
    return bars[ok].reset_index(drop=True)


def median_turnover(prices: pd.DataFrame) -> pd.Series:
    """Median daily traded value (KRW) per symbol -- the liquidity filter."""
    df = prices.copy()
    df["turnover"] = df["close"] * df["volume"].fillna(0)
    return df.groupby("symbol")["turnover"].median()
