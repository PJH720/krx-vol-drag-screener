"""End-to-end runs through main(), with the network patched out."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from krxdrag import cli
from krxdrag.cli import build_parser, main


@pytest.fixture
def offline_cli(monkeypatch, tmp_path, synthetic_universe, synthetic_prices):
    """Patch both network calls and redirect every artefact into tmp_path."""
    from krxdrag import report, screener

    def fake_universe(markets=("KOSPI", "KOSDAQ"), use_cache=True, day=None):
        return synthetic_universe[synthetic_universe["market"].isin(markets)].reset_index(
            drop=True
        )

    def fake_prices(symbols, lookback_days=504, batch_size=60, use_cache=True, end=None):
        return synthetic_prices[synthetic_prices["symbol"].isin(symbols)].reset_index(
            drop=True
        )

    monkeypatch.setattr(screener, "load_universe", fake_universe)
    monkeypatch.setattr(screener, "load_prices", fake_prices)
    monkeypatch.setattr(cli, "load_prices", fake_prices)
    monkeypatch.setattr(report, "REPORT_DIR", tmp_path)
    return tmp_path


BASE_ARGS = ["--lookback", "600", "--min-turnover", "1e8", "--top", "3", "--quiet"]


def test_full_run_writes_every_artefact(offline_cli, capsys):
    rc = main(BASE_ARGS + ["--min-sector-names", "2", "--html"])
    assert rc == 0

    written = {p.suffix for p in offline_cli.iterdir()}
    assert {".csv", ".md", ".png", ".html"} <= written

    out = capsys.readouterr().out
    assert "names passed" in out
    assert "median jump share" in out
    assert "html ->" in out


def test_html_from_a_real_run_is_self_contained(offline_cli):
    assert main(BASE_ARGS + ["--min-sector-names", "2", "--html"]) == 0

    page = next(offline_cli.glob("*.html")).read_text(encoding="utf-8")
    assert "http://" not in page and "https://" not in page
    assert "data:image/png;base64," in page
    # the analysis layers really made it into the page
    assert "업종별 드래그" in page
    assert "드래그의 점프 성분" in page


def test_markdown_from_a_real_run_carries_the_sections(offline_cli):
    assert main(BASE_ARGS + ["--min-sector-names", "2"]) == 0

    text = next(offline_cli.glob("*.md")).read_text(encoding="utf-8")
    assert "업종별 드래그" in text
    assert "롤링" in text
    assert "투자 자문이 아니다" in text


def test_html_is_not_written_without_the_flag(offline_cli):
    assert main(BASE_ARGS) == 0
    assert list(offline_cli.glob("*.html")) == []


def test_no_sectors_and_no_jumps_shrink_the_output(offline_cli):
    assert main(BASE_ARGS + ["--no-sectors", "--no-jumps", "--rolling-window", "0"]) == 0

    csv = pd.read_csv(next(offline_cli.glob("*.csv")))
    assert "drag_jump" not in csv.columns
    assert "drag_trend" not in csv.columns

    text = next(offline_cli.glob("*.md")).read_text(encoding="utf-8")
    assert "업종별 드래그" not in text
    assert "드래그의 점프 성분" not in text


def test_markets_flag_restricts_the_universe(offline_cli):
    assert main(BASE_ARGS + ["--markets", "KOSPI"]) == 0
    csv = pd.read_csv(next(offline_cli.glob("*.csv")))
    assert set(csv["market"]) == {"KOSPI"}


def test_impossible_filters_exit_nonzero(offline_cli, capsys):
    assert main(BASE_ARGS + ["--min-obs", "99999"]) == 1
    assert "No names passed" in capsys.readouterr().err


def test_etf_audit_survives_an_unreachable_data_source(offline_cli, monkeypatch, capsys):
    """--etf must degrade to a missing section, never sink the run."""
    def boom(*a, **k):
        raise ConnectionError("yahoo unreachable")

    monkeypatch.setattr(cli, "load_prices", boom)
    # screener still uses its own patched loader, so only the ETF fetch fails
    assert main(BASE_ARGS + ["--no-sectors", "--rolling-window", "0", "--etf"]) == 0

    text = next(offline_cli.glob("*.md")).read_text(encoding="utf-8")
    assert "레버리지 ETF" not in text


def test_etf_audit_renders_when_prices_are_available(offline_cli, monkeypatch):
    """A geared product priced in the feed must reach the report."""
    from krxdrag.leverage import simulate_leveraged_path
    from krxdrag.metrics import simulate_gbm

    dates = pd.date_range("2023-01-02", periods=601, freq="D")
    underlying = simulate_gbm(mu=0.09, sigma=0.30, n_days=600, s0=30_000.0, seed=51)
    frames = [
        pd.DataFrame({"date": dates, "symbol": "069500.KS", "close": underlying, "volume": 1e6}),
        pd.DataFrame({
            "date": dates, "symbol": "122630.KS",
            "close": simulate_leveraged_path(underlying, 2.0), "volume": 1e6,
        }),
    ]
    etf_prices = pd.concat(frames, ignore_index=True)

    monkeypatch.setattr(cli, "load_prices", lambda symbols, **k: etf_prices[
        etf_prices["symbol"].isin(symbols)
    ].reset_index(drop=True))

    assert main(BASE_ARGS + ["--no-sectors", "--rolling-window", "0", "--etf"]) == 0
    text = next(offline_cli.glob("*.md")).read_text(encoding="utf-8")
    assert "레버리지 ETF" in text
    assert "KODEX 레버리지" in text


# --- parser ------------------------------------------------------------------

def test_parser_defaults_match_the_documented_ones():
    args = build_parser().parse_args([])
    assert args.lookback == 504
    assert args.min_obs == 250
    assert args.rolling_window == 126
    assert args.markets == ["KOSPI", "KOSDAQ"]
    assert not args.html and not args.etf
    assert not args.no_sectors and not args.no_jumps


def test_parser_rejects_an_unknown_market():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--markets", "NASDAQ"])


def test_the_universe_is_downloaded_once(monkeypatch, offline_cli, synthetic_prices):
    """screen() built a price panel and threw it away; main() then refetched it.

    Under --no-cache that was a second full download of ~1,100 tickers.
    """
    from krxdrag import screener

    calls = {"n": 0}
    real = screener.load_prices

    def counting(symbols, **kw):
        calls["n"] += 1
        return synthetic_prices[synthetic_prices["symbol"].isin(symbols)].reset_index(
            drop=True
        )

    monkeypatch.setattr(screener, "load_prices", counting)

    assert main(BASE_ARGS + ["--min-sector-names", "2", "--no-cache"]) == 0
    assert calls["n"] == 1, f"price panel fetched {calls['n']} times"


def test_screen_panel_returns_the_matrix_the_screen_used(offline_screen):
    from krxdrag.config import ScreenConfig
    from krxdrag.screener import screen_panel

    df, wide = screen_panel(
        ScreenConfig(lookback_days=600, min_median_turnover=1e8, batch_size=8),
        use_cache=False,
    )
    assert not df.empty and not wide.empty
    assert set(df["symbol"]) <= set(wide.columns)


def test_range_volatility_reaches_the_report(offline_cli):
    assert main(BASE_ARGS + ["--min-sector-names", "2", "--html"]) == 0

    csv = pd.read_csv(next(offline_cli.glob("*.csv")))
    assert "sigma_sq_yang_zhang" in csv.columns
    assert "limit_hit_share" in csv.columns

    text = next(offline_cli.glob("*.md")).read_text(encoding="utf-8")
    assert "변동성 추정량 비교" in text


def test_no_range_vol_flag_drops_the_columns(offline_cli):
    assert main(BASE_ARGS + ["--no-range-vol", "--no-sectors", "--rolling-window", "0"]) == 0

    csv = pd.read_csv(next(offline_cli.glob("*.csv")))
    assert "sigma_sq_yang_zhang" not in csv.columns
    assert "sigma_sq" in csv.columns

    text = next(offline_cli.glob("*.md")).read_text(encoding="utf-8")
    assert "변동성 추정량 비교" not in text


def test_persistence_is_omitted_on_too_narrow_a_cross_section(offline_cli):
    """Seven names cannot support a cross-sectional rank correlation."""
    assert main(BASE_ARGS + ["--min-sector-names", "2", "--validation-window", "126"]) == 0
    text = next(offline_cli.glob("*.md")).read_text(encoding="utf-8")
    assert "다음 기간까지 살아남는가" not in text


def test_persistence_measurement_reaches_the_report(monkeypatch, tmp_path):
    """End to end on a universe wide enough to rank: the measurement is
    computed from the screen's own price panel, not from a fixture table."""
    from krxdrag import report, screener
    from krxdrag.metrics import TRADING_DAYS

    rng = np.random.default_rng(21)
    n_names, n_days = 40, 760
    sigma = rng.uniform(0.25, 1.10, n_names)
    mu = rng.uniform(-0.3, 0.9, n_names)
    step = 1.0 / TRADING_DAYS
    inc = rng.normal(
        (mu - 0.5 * sigma**2) * step, sigma * np.sqrt(step), size=(n_days, n_names)
    )
    close = 50_000 * np.exp(np.vstack([np.zeros(n_names), np.cumsum(inc, axis=0)]))
    dates = pd.to_datetime(pd.date_range("2022-01-03", periods=n_days + 1))

    frames, rows = [], []
    for i in range(n_names):
        c = close[:, i]
        o = np.concatenate([[c[0]], c[:-1]])
        wick = np.abs(rng.normal(0, sigma[i] / np.sqrt(252) * 0.6, c.size))
        frames.append(
            pd.DataFrame(
                {
                    "date": dates, "symbol": f"{i + 1:06d}.KS", "open": o,
                    "high": np.maximum(o, c) * np.exp(wick),
                    "low": np.minimum(o, c) * np.exp(-wick),
                    "close": c, "volume": 1e6,
                }
            )
        )
        rows.append(
            {"name": f"종목{i + 1}", "code": f"{i + 1:06d}", "sector": "화학",
             "market": "KOSPI", "symbol": f"{i + 1:06d}.KS"}
        )

    prices = pd.concat(frames, ignore_index=True)
    universe = pd.DataFrame(rows)
    monkeypatch.setattr(screener, "load_universe", lambda **k: universe)
    monkeypatch.setattr(
        screener, "load_prices",
        lambda syms, **k: prices[prices["symbol"].isin(syms)].reset_index(drop=True),
    )
    monkeypatch.setattr(cli, "load_prices", lambda syms, **k: prices)
    monkeypatch.setattr(report, "REPORT_DIR", tmp_path)

    assert main(["--lookback", "761", "--min-turnover", "0", "--top", "3", "--quiet",
                 "--no-sectors", "--validation-window", "126", "--html"]) == 0

    text = next(tmp_path.glob("*.md")).read_text(encoding="utf-8")
    assert "다음 기간까지 살아남는가" in text
    assert "표본외 R² 가 음수" in text          # mu really is that bad
    assert next(tmp_path.glob("*.html")).read_text(encoding="utf-8").count(
        "다음 기간까지 살아남는가"
    ) == 1


def test_validation_window_zero_disables_it(offline_cli):
    assert main(BASE_ARGS + ["--validation-window", "0", "--no-sectors",
                             "--rolling-window", "0"]) == 0
    text = next(offline_cli.glob("*.md")).read_text(encoding="utf-8")
    assert "다음 기간까지 살아남는가" not in text
