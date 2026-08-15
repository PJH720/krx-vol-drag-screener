"""Central configuration and cache paths."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parents[1]
CACHE_DIR = PROJECT_ROOT / "data" / "cache"
REPORT_DIR = PROJECT_ROOT / "reports"

CACHE_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

KIND_URL = "https://kind.krx.co.kr/corpgeneral/corpList.do?method=download"
KIND_MARKETS = {"KOSPI": "stockMkt", "KOSDAQ": "kosdaqMkt"}
YF_SUFFIX = {"KOSPI": ".KS", "KOSDAQ": ".KQ"}

USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"


@dataclass(frozen=True)
class ScreenConfig:
    """Parameters that define one screening run."""

    lookback_days: int = 504          # ~2 years of trading days
    min_obs: int = 250                # reject thin history
    min_median_turnover: float = 5e8  # KRW/day, liquidity floor
    markets: tuple[str, ...] = ("KOSPI", "KOSDAQ")
    max_names: int | None = None      # cap universe size (None = all)
    batch_size: int = 60              # yfinance tickers per request
    periods_per_year: int = 252
    confidence: float = 0.95
