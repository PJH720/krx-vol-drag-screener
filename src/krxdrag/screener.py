"""Cross-sectional variance-drag screen over the KRX universe."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .config import ScreenConfig
from .data import load_prices, median_turnover, to_wide
from .diagnostics import compute_diagnostics
from .jumps import decompose_jumps
from .metrics import compute_drag, log_returns
from .rolling import drag_trend
from .universe import load_universe

log = logging.getLogger(__name__)


def screen(cfg: ScreenConfig | None = None, use_cache: bool = True) -> pd.DataFrame:
    """Run the full screen and return one row per surviving symbol."""
    return screen_panel(cfg, use_cache=use_cache)[0]


def screen_panel(
    cfg: ScreenConfig | None = None,
    use_cache: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The screen, plus the price matrix it was computed from.

    The sector and rolling layers need that same date x symbol matrix. Handing
    it back costs nothing here and saves the caller a second download of the
    whole universe -- roughly 1,100 tickers under --no-cache.
    """
    cfg = cfg or ScreenConfig()

    universe = load_universe(markets=cfg.markets, use_cache=use_cache)
    if cfg.max_names:
        universe = universe.head(cfg.max_names)
    log.info("screen: %d candidate names", len(universe))

    prices = load_prices(
        universe["symbol"].tolist(),
        lookback_days=cfg.lookback_days,
        batch_size=cfg.batch_size,
        use_cache=use_cache,
    )
    if prices.empty:
        return pd.DataFrame(), pd.DataFrame()

    turnover = median_turnover(prices)
    wide = to_wide(prices).tail(cfg.lookback_days)
    meta = universe.set_index("symbol")

    rows: list[dict] = []
    for symbol in wide.columns:
        series = wide[symbol].dropna().to_numpy(dtype=float)
        if series.size < cfg.min_obs:
            continue

        m = compute_drag(
            series,
            periods_per_year=cfg.periods_per_year,
            confidence=cfg.confidence,
        )
        if m is None:
            continue

        tv = float(turnover.get(symbol, 0.0))
        if tv < cfg.min_median_turnover:
            continue

        returns = log_returns(series)
        d = compute_diagnostics(returns)
        info = meta.loc[symbol]

        row: dict = {
            "symbol": symbol,
            "code": info["code"],
            "name": info["name"],
            "market": info["market"],
            "sector": info["sector"],
            "median_turnover": tv,
            **m.to_dict(),
        }
        if d is not None:
            row.update(d.to_dict())
        if cfg.decompose_jumps:
            j = decompose_jumps(returns, periods_per_year=cfg.periods_per_year)
            if j is not None:
                # n_obs is already on the row from DragMetrics and agrees
                row.update({k: v for k, v in j.to_dict().items() if k != "n_obs"})
        if cfg.rolling_window:
            row["drag_trend"] = drag_trend(
                series,
                window=cfg.rolling_window,
                periods_per_year=cfg.periods_per_year,
            )
        rows.append(row)

    if not rows:
        return pd.DataFrame(), wide

    df = pd.DataFrame(rows)

    # Sanity guard: the Itô identity must hold exactly for every row. A bare
    # assert would vanish under `python -O`, and this check is load-bearing.
    residual = float(np.abs((df["mu"] - df["g"]) - df["drag"]).max())
    if not residual < 1e-10:
        raise ValueError(f"Ito identity violated, max residual {residual}")

    df = df.sort_values("drag", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    log.info("screen: %d names passed filters", len(df))
    return df, wide


def summarise(df: pd.DataFrame) -> dict[str, float]:
    """Headline statistics for the report header."""
    if df.empty:
        return {}
    positive_mu = df[df["mu"] > 0]
    return {
        "n_names": len(df),
        "median_sigma": float(df["sigma"].median()),
        "median_drag": float(df["drag"].median()),
        "median_drag_ratio": float(positive_mu["drag_ratio"].median())
        if not positive_mu.empty
        else float("nan"),
        "share_g_negative_mu_positive": float(
            ((df["mu"] > 0) & (df["g"] < 0)).mean()
        ),
        "median_gbm_score": float(df["gbm_score"].median())
        if "gbm_score" in df
        else float("nan"),
        "share_normal_ok": float(df["normal_ok"].mean())
        if "normal_ok" in df
        else float("nan"),
    }
