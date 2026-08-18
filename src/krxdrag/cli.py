"""Command line entry point:  python -m krxdrag ..."""

from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from .config import ScreenConfig
from .data import load_prices, to_wide
from .leverage import KRX_LEVERAGED_ETFS, audit_leveraged_etfs
from .report import (
    write_chart,
    write_csv,
    write_html,
    write_leverage_chart,
    write_markdown,
    write_rolling_chart,
    write_sector_chart,
)
from .rolling import cross_sectional_drag
from .screener import screen, summarise
from .sectors import aggregate_sectors, diversification_benefit, sector_portfolio_drag

log = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="krxdrag",
        description="Rank KRX listed equities by variance drag (Ito correction).",
    )
    p.add_argument("--lookback", type=int, default=504, help="trading days of history")
    p.add_argument("--min-obs", type=int, default=250, help="minimum observations")
    p.add_argument(
        "--min-turnover",
        type=float,
        default=5e8,
        help="minimum median daily turnover in KRW",
    )
    p.add_argument(
        "--markets",
        nargs="+",
        default=["KOSPI", "KOSDAQ"],
        choices=["KOSPI", "KOSDAQ"],
    )
    p.add_argument("--max-names", type=int, default=None, help="cap universe size")
    p.add_argument("--top", type=int, default=25, help="rows per report table")
    p.add_argument(
        "--rolling-window",
        type=int,
        default=126,
        help="trading days for the rolling drag window; 0 disables",
    )
    p.add_argument(
        "--min-sector-names",
        type=int,
        default=5,
        help="drop sectors holding fewer names than this",
    )
    p.add_argument("--no-sectors", action="store_true", help="skip sector aggregation")
    p.add_argument("--no-jumps", action="store_true", help="skip the jump decomposition")
    p.add_argument(
        "--etf",
        action="store_true",
        help="audit KRX leveraged ETFs (downloads extra price history)",
    )
    p.add_argument("--html", action="store_true", help="also write a self-contained HTML report")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--quiet", action="store_true")
    return p


def _etf_audit(cfg: ScreenConfig, use_cache: bool) -> pd.DataFrame:
    """Download the geared products and check their delivered multiples."""
    symbols = sorted(
        {e.symbol for e in KRX_LEVERAGED_ETFS}
        | {f"{e.underlying}.KS" for e in KRX_LEVERAGED_ETFS if e.market == "KOSPI"}
        | {f"{e.underlying}.KQ" for e in KRX_LEVERAGED_ETFS if e.market != "KOSPI"}
    )
    try:
        prices = load_prices(symbols, lookback_days=cfg.lookback_days, use_cache=use_cache)
    except Exception as exc:  # the ETF section is optional; never sink the run
        log.warning("etf: price download failed: %s", exc)
        return pd.DataFrame()

    if prices.empty:
        log.warning("etf: no prices returned, skipping the leverage audit")
        return pd.DataFrame()

    wide = to_wide(prices).tail(cfg.lookback_days)
    return audit_leveraged_etfs(wide, periods_per_year=cfg.periods_per_year)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    use_cache = not args.no_cache

    cfg = ScreenConfig(
        lookback_days=args.lookback,
        min_obs=args.min_obs,
        min_median_turnover=args.min_turnover,
        markets=tuple(args.markets),
        max_names=args.max_names,
        decompose_jumps=not args.no_jumps,
        rolling_window=args.rolling_window,
    )

    df = screen(cfg, use_cache=use_cache)
    if df.empty:
        print("No names passed the filters.", file=sys.stderr)
        return 1

    # --- optional analysis layers ---------------------------------------
    sectors = diversification = rolling = None
    sector_labels: dict[str, str] = {}

    if not args.no_sectors:
        sectors = aggregate_sectors(df, min_names=args.min_sector_names)

    wide = pd.DataFrame()
    if (not args.no_sectors) or args.rolling_window:
        prices = load_prices(
            df["symbol"].tolist(),
            lookback_days=cfg.lookback_days,
            batch_size=cfg.batch_size,
            use_cache=use_cache,
        )
        if not prices.empty:
            wide = to_wide(prices).tail(cfg.lookback_days)

    if not wide.empty:
        if sectors is not None and not sectors.empty:
            portfolios = sector_portfolio_drag(
                wide, df[["symbol", "sector"]], min_names=args.min_sector_names
            )
            diversification = diversification_benefit(sectors, portfolios)
        if args.rolling_window:
            rolling = cross_sectional_drag(
                wide, window=args.rolling_window, periods_per_year=cfg.periods_per_year
            )

    etfs = _etf_audit(cfg, use_cache) if args.etf else None

    # --- output ----------------------------------------------------------
    charts: dict[str, object] = {"main": write_chart(df)}
    if rolling is not None and not rolling.empty:
        charts["rolling"] = write_rolling_chart(rolling, window=args.rolling_window)
    if sectors is not None and not sectors.empty:
        result = write_sector_chart(sectors, diversification)
        if result is not None:
            charts["sectors"], sector_labels = result
    charts["leverage"] = write_leverage_chart()

    csv_path = write_csv(df)
    md_path = write_markdown(
        df,
        top_n=args.top,
        sectors=sectors,
        diversification=diversification,
        rolling=rolling,
        etfs=etfs,
        rolling_window=args.rolling_window,
    )
    html_path = None
    if args.html:
        html_path = write_html(
            df,
            top_n=args.top,
            sectors=sectors,
            diversification=diversification,
            rolling=rolling,
            etfs=etfs,
            rolling_window=args.rolling_window,
            charts=charts,
            sector_labels=sector_labels,
        )

    s = summarise(df)
    print()
    print(f"  names passed      : {s['n_names']:,}")
    print(f"  median sigma      : {s['median_sigma'] * 100:.1f}%")
    print(f"  median drag       : {s['median_drag'] * 100:.1f}%p")
    print(f"  median drag / mu  : {s['median_drag_ratio'] * 100:.1f}%")
    print(f"  mu>0 but g<0      : {s['share_g_negative_mu_positive'] * 100:.1f}%")
    if sectors is not None and not sectors.empty:
        print(f"  sectors reported  : {len(sectors):,}")
    if "drag_jump" in df.columns:
        print(f"  median jump share : {df['jump_ratio'].median() * 100:.1f}%")
    print()
    print(f"  csv  -> {csv_path}")
    print(f"  md   -> {md_path}")
    for label, path in charts.items():
        if path:
            print(f"  png  -> {path}  ({label})")
    if html_path:
        print(f"  html -> {html_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
