"""Command line entry point:  python -m krxdrag ..."""

from __future__ import annotations

import argparse
import logging
import sys

from .config import ScreenConfig
from .report import write_chart, write_csv, write_markdown
from .screener import screen, summarise


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
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = ScreenConfig(
        lookback_days=args.lookback,
        min_obs=args.min_obs,
        min_median_turnover=args.min_turnover,
        markets=tuple(args.markets),
        max_names=args.max_names,
    )

    df = screen(cfg, use_cache=not args.no_cache)
    if df.empty:
        print("No names passed the filters.", file=sys.stderr)
        return 1

    csv_path = write_csv(df)
    md_path = write_markdown(df, top_n=args.top)
    png_path = write_chart(df)

    s = summarise(df)
    print()
    print(f"  names passed      : {s['n_names']:,}")
    print(f"  median sigma      : {s['median_sigma'] * 100:.1f}%")
    print(f"  median drag       : {s['median_drag'] * 100:.1f}%p")
    print(f"  median drag / mu  : {s['median_drag_ratio'] * 100:.1f}%")
    print(f"  mu>0 but g<0      : {s['share_g_negative_mu_positive'] * 100:.1f}%")
    print()
    print(f"  csv  -> {csv_path}")
    print(f"  md   -> {md_path}")
    if png_path:
        print(f"  png  -> {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
