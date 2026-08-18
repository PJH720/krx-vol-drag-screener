"""Where the drag concentrates: aggregation by KRX listing sector.

The screen ranks 1,100-odd names individually, which answers "which stock bleeds
most" but not "which industry bleeds most". The KIND listing table already
carries an industry label for every name and universe.py already passes it
through, so the aggregation is cheap.

Two numbers are reported per sector and they are not the same thing:

- the **median constituent drag**, which describes a typical stock in the
  sector, and
- the **equal-weight portfolio drag**, which describes actually holding the
  sector.

The second is always smaller, because a portfolio's variance is damped by
imperfect correlation while its constituents' variances are not. The gap is the
diversification benefit, and reporting only the first would overstate what a
sector costs a holder.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import TRADING_DAYS, compute_drag

DEFAULT_MIN_NAMES: int = 5


def aggregate_sectors(
    df: pd.DataFrame,
    min_names: int = DEFAULT_MIN_NAMES,
) -> pd.DataFrame:
    """Per-sector summary of a completed screen, ranked by median drag.

    Sectors with fewer than `min_names` members are dropped: KRX has 117 listing
    sectors and 29 of them hold a single name, so an unfiltered table is topped
    by whichever lone stock happened to be most volatile.
    """
    if df.empty or "sector" not in df.columns:
        return pd.DataFrame()

    d = df.dropna(subset=["sector"]).copy()
    d = d[d["sector"].astype(str).str.strip() != ""]
    if d.empty:
        return pd.DataFrame()

    d["_is_trap"] = (d["mu"] > 0) & (d["g"] < 0)

    agg: dict[str, pd.Series] = {
        "n_names": d.groupby("sector")["symbol"].count(),
        "median_sigma": d.groupby("sector")["sigma"].median(),
        "median_drag": d.groupby("sector")["drag"].median(),
        "drag_iqr": (
            d.groupby("sector")["drag"].quantile(0.75)
            - d.groupby("sector")["drag"].quantile(0.25)
        ),
        "median_mu": d.groupby("sector")["mu"].median(),
        "median_g": d.groupby("sector")["g"].median(),
        "share_trap": d.groupby("sector")["_is_trap"].mean(),
    }
    for optional in ("drag_ratio", "gbm_score", "jump_ratio", "drag_trend"):
        if optional in d.columns:
            agg[f"median_{optional}"] = d.groupby("sector")[optional].median()

    out = pd.DataFrame(agg)
    out = out[out["n_names"] >= min_names]
    if out.empty:
        return out

    out = out.sort_values("median_drag", ascending=False)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out.reset_index()


def sector_portfolio_drag(
    wide: pd.DataFrame,
    universe: pd.DataFrame,
    min_names: int = DEFAULT_MIN_NAMES,
    periods_per_year: int = TRADING_DAYS,
) -> pd.DataFrame:
    """Drag of a daily-rebalanced equal-weight portfolio of each sector.

    `wide` is a date x symbol close matrix, `universe` needs symbol and sector
    columns. The portfolio's simple return is the cross-sectional mean of its
    constituents' simple returns, which is what equal weighting with daily
    rebalancing means.
    """
    if wide.empty or universe.empty:
        return pd.DataFrame()

    members = (
        universe.dropna(subset=["sector"])
        .assign(sector=lambda x: x["sector"].astype(str).str.strip())
        .query("sector != ''")
        .groupby("sector")["symbol"]
        .apply(lambda s: [sym for sym in s if sym in wide.columns])
    )

    simple = wide.astype(float).pct_change()

    rows: list[dict] = []
    for sector, symbols in members.items():
        if len(symbols) < min_names:
            continue

        port_returns = simple[symbols].mean(axis=1, skipna=True).dropna()
        if port_returns.size < 20:
            continue

        # Rebuild an index level series so the same estimator can be reused.
        # The leading 1.0 matters: without it the level series is one point
        # short and the estimate silently drops the first day's return.
        growth = pd.concat([pd.Series([1.0]), (1.0 + port_returns).reset_index(drop=True)])
        level = 100.0 * growth.cumprod()
        m = compute_drag(level.to_numpy(dtype=float), periods_per_year=periods_per_year)
        if m is None:
            continue

        rows.append(
            {
                "sector": sector,
                "n_names": len(symbols),
                "portfolio_sigma": m.sigma,
                "portfolio_drag": m.drag,
                "portfolio_g": m.g,
                "portfolio_mu": m.mu,
            }
        )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("portfolio_drag", ascending=False).reset_index(
        drop=True
    )


def diversification_benefit(
    sector_table: pd.DataFrame,
    portfolio_table: pd.DataFrame,
) -> pd.DataFrame:
    """Join the two views and report how much drag diversification removes.

    `drag_saved` is median constituent drag minus equal-weight portfolio drag --
    the part of a typical constituent's drag that holding the whole sector
    avoids.
    """
    if sector_table.empty or portfolio_table.empty:
        return pd.DataFrame()

    merged = sector_table.merge(
        portfolio_table.drop(columns=["n_names"]), on="sector", how="inner"
    )
    if merged.empty:
        return merged

    merged["drag_saved"] = merged["median_drag"] - merged["portfolio_drag"]
    merged["drag_saved_share"] = np.where(
        merged["median_drag"] > 0,
        merged["drag_saved"] / merged["median_drag"],
        np.nan,
    )
    return merged.sort_values("drag_saved", ascending=False).reset_index(drop=True)
