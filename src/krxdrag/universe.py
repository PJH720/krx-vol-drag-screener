"""KRX listed-company universe, sourced from the exchange's own KIND service."""

from __future__ import annotations

import io
import logging
from datetime import date
from pathlib import Path

import pandas as pd
import requests

from .config import CACHE_DIR, KIND_MARKETS, KIND_URL, USER_AGENT, YF_SUFFIX

log = logging.getLogger(__name__)

_VALID_CODE = r"^\d{6}$"


def _cache_path(day: date) -> Path:
    return CACHE_DIR / f"universe_{day:%Y%m%d}.parquet"


def _fetch_market(market: str, timeout: int = 60) -> pd.DataFrame:
    """Download one market's listing table from KIND."""
    url = f"{KIND_URL}&marketType={KIND_MARKETS[market]}"
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
    resp.raise_for_status()
    tables = pd.read_html(io.BytesIO(resp.content), encoding="euc-kr")
    df = tables[0]

    out = pd.DataFrame(
        {
            "name": df["회사명"].astype(str).str.strip(),
            "code": df["종목코드"].astype(str).str.zfill(6),
            "sector": df.get("업종", pd.Series(dtype=str)).astype(str).str.strip(),
        }
    )
    out["market"] = market
    # Preferred shares and provisional listings carry alphabetic codes; drop them
    # so every row maps cleanly onto a yfinance symbol.
    out = out[out["code"].str.match(_VALID_CODE)]
    out["symbol"] = out["code"] + YF_SUFFIX[market]
    return out.drop_duplicates(subset="code").reset_index(drop=True)


def load_universe(
    markets: tuple[str, ...] = ("KOSPI", "KOSDAQ"),
    use_cache: bool = True,
    day: date | None = None,
) -> pd.DataFrame:
    """Return the listed universe as columns [name, code, sector, market, symbol]."""
    day = day or date.today()
    path = _cache_path(day)

    if use_cache and path.exists():
        cached = pd.read_parquet(path)
        subset = cached[cached["market"].isin(markets)]
        if not subset.empty:
            log.info("universe: %d names from cache", len(subset))
            return subset.reset_index(drop=True)

    frames = [_fetch_market(m) for m in markets]
    universe = pd.concat(frames, ignore_index=True)
    universe.to_parquet(path, index=False)
    log.info("universe: %d names fetched from KIND", len(universe))
    return universe
