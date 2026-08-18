"""The report writers must produce real files without touching the network."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from krxdrag.config import ScreenConfig
from krxdrag.report import write_chart, write_csv, write_markdown
from krxdrag.screener import screen

RUN_DATE = date(2026, 8, 18)


@pytest.fixture
def screened(offline_screen) -> pd.DataFrame:
    return screen(
        ScreenConfig(
            lookback_days=600, min_obs=250, min_median_turnover=1e8, batch_size=8
        ),
        use_cache=False,
    )


def test_write_csv_roundtrips(screened, tmp_path):
    path = write_csv(screened, out_dir=tmp_path, run_date=RUN_DATE)
    assert path.name == "krx_drag_20260818.csv"

    back = pd.read_csv(path)
    assert len(back) == len(screened)
    for col in ("rank", "symbol", "sigma", "mu", "g", "drag"):
        assert col in back.columns


def test_write_markdown_has_the_load_bearing_sections(screened, tmp_path):
    path = write_markdown(screened, top_n=3, out_dir=tmp_path, run_date=RUN_DATE)
    text = path.read_text(encoding="utf-8")

    assert "2026-08-18" in text
    assert "## 요약" in text
    assert "변동성 함정" in text
    # the interpretation caveats must never be silently dropped
    assert "투자 자문이 아니다" in text


def test_write_markdown_survives_an_empty_trap_table(tmp_path, screened):
    """No mu>0 & g<0 names must not crash the writer."""
    safe = screened[screened["g"] > 0].copy()
    path = write_markdown(safe, top_n=3, out_dir=tmp_path, run_date=RUN_DATE)
    assert path.exists()


def test_write_chart_produces_a_png(screened, tmp_path):
    path = write_chart(screened, out_dir=tmp_path, run_date=RUN_DATE)
    assert path is not None
    assert path.suffix == ".png"
    assert path.stat().st_size > 5_000  # a real figure, not a blank canvas


def test_writers_default_to_today(screened, tmp_path):
    path = write_csv(screened, out_dir=tmp_path)
    assert path.name == f"krx_drag_{date.today():%Y%m%d}.csv"


# ---------------------------------------------------------------------------
# the analysis sections added in v0.2
# ---------------------------------------------------------------------------

import html.parser
import re

import numpy as np

from krxdrag.data import to_wide
from krxdrag.leverage import LeveragedETF, audit_leveraged_etfs, simulate_leveraged_path
from krxdrag.metrics import simulate_gbm
from krxdrag.report import (
    _korean_font,
    write_html,
    write_leverage_chart,
    write_rolling_chart,
    write_sector_chart,
)
from krxdrag.rolling import cross_sectional_drag
from krxdrag.sectors import (
    aggregate_sectors,
    diversification_benefit,
    sector_portfolio_drag,
)


@pytest.fixture
def analysis(screened, synthetic_prices):
    """Every optional layer the report can render, built from synthetic data."""
    wide = to_wide(synthetic_prices)
    sectors = aggregate_sectors(screened, min_names=2)
    portfolios = sector_portfolio_drag(wide, screened[["symbol", "sector"]], min_names=2)
    return {
        "sectors": sectors,
        "diversification": diversification_benefit(sectors, portfolios),
        "rolling": cross_sectional_drag(wide, window=126, min_names=3),
        "etfs": _etf_table(),
    }


def _etf_table() -> pd.DataFrame:
    underlying = simulate_gbm(mu=0.09, sigma=0.30, n_days=900, seed=41)
    wide = pd.DataFrame(
        {
            "069500.KS": underlying,
            "122630.KS": simulate_leveraged_path(underlying, 2.0),
            "114800.KS": simulate_leveraged_path(underlying, -1.0),
        }
    )
    products = (
        LeveragedETF("122630", "KODEX 레버리지", 2.0, "069500"),
        LeveragedETF("114800", "KODEX 인버스", -1.0, "069500"),
    )
    return audit_leveraged_etfs(wide, products=products)


def test_markdown_includes_every_optional_section(screened, tmp_path, analysis):
    path = write_markdown(
        screened, top_n=3, out_dir=tmp_path, run_date=RUN_DATE, **analysis
    )
    text = path.read_text(encoding="utf-8")

    assert "업종별 드래그" in text
    assert "분산투자가 걷어내는 드래그" in text
    assert "드래그의 점프 성분" in text
    assert "레버리지 ETF" in text
    assert "롤링" in text
    # the caveat that the two sector numbers are not the same thing
    assert "업종을 보유했을 때의 드래그가 아니다" in text


def test_markdown_omits_sections_whose_data_is_absent(screened, tmp_path):
    """Only the optional layers disappear; drag_ratio is core and stays."""
    optional = ["drag_cont", "drag_jump", "jump_ratio", "bns_pvalue", "has_jumps"]
    bare = screened.drop(columns=[c for c in optional if c in screened.columns])
    text = write_markdown(
        bare, top_n=3, out_dir=tmp_path, run_date=RUN_DATE
    ).read_text(encoding="utf-8")

    assert "업종별 드래그" not in text
    assert "드래그의 점프 성분" not in text
    assert "레버리지 ETF" not in text


def test_sector_chart_writes_a_png_and_reports_its_labels(tmp_path, analysis):
    result = write_sector_chart(
        analysis["sectors"], analysis["diversification"], out_dir=tmp_path, run_date=RUN_DATE
    )
    assert result is not None
    path, mapping = result
    assert path.exists() and path.stat().st_size > 5_000

    # without a Korean font the chart must fall back to numeric labels
    if _korean_font() is None:
        assert mapping
        assert all(re.fullmatch(r"S\d{2}", k) for k in mapping)
        assert set(mapping.values()) <= set(analysis["sectors"]["sector"])
    else:
        assert mapping == {}


def test_sector_chart_on_empty_input_returns_none(tmp_path):
    assert write_sector_chart(pd.DataFrame(), out_dir=tmp_path) is None


def test_rolling_and_leverage_charts_write_pngs(tmp_path, analysis):
    rolling_path = write_rolling_chart(analysis["rolling"], out_dir=tmp_path, run_date=RUN_DATE)
    assert rolling_path is not None and rolling_path.stat().st_size > 5_000

    lev_path = write_leverage_chart(out_dir=tmp_path, run_date=RUN_DATE)
    assert lev_path is not None and lev_path.stat().st_size > 5_000


def test_rolling_chart_on_empty_input_returns_none(tmp_path):
    assert write_rolling_chart(pd.DataFrame(), out_dir=tmp_path) is None


# --- HTML -------------------------------------------------------------------

class _Collector(html.parser.HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags: list[str] = []
        self.rows = 0
        self.external: list[str] = []

    def handle_starttag(self, tag, attrs):
        self.tags.append(tag)
        if tag == "tr":
            self.rows += 1
        for key, value in attrs:
            if key in {"src", "href", "action"} and value and value.startswith(
                ("http://", "https://", "//")
            ):
                self.external.append(value)


def _parse(path) -> _Collector:
    c = _Collector()
    c.feed(path.read_text(encoding="utf-8"))
    return c


def test_html_is_well_formed_and_has_the_expected_rows(screened, tmp_path, analysis):
    path = write_html(screened, top_n=3, out_dir=tmp_path, run_date=RUN_DATE, **analysis)
    assert path.suffix == ".html"

    c = _parse(path)
    for tag in ("title", "style", "main", "table", "script", "footer"):
        assert tag in c.tags
    assert c.rows > 3


def test_html_references_nothing_off_the_machine(screened, tmp_path, analysis):
    """A self-contained page: no CDN, no remote images, no fetch targets."""
    path = write_html(screened, top_n=5, out_dir=tmp_path, run_date=RUN_DATE, **analysis)
    assert _parse(path).external == []

    text = path.read_text(encoding="utf-8")
    assert "http://" not in text and "https://" not in text
    assert "fetch(" not in text and "XMLHttpRequest" not in text


def test_html_embeds_charts_as_data_uris(screened, tmp_path, analysis):
    charts = {
        "main": write_chart(screened, out_dir=tmp_path, run_date=RUN_DATE),
        "leverage": write_leverage_chart(out_dir=tmp_path, run_date=RUN_DATE),
    }
    path = write_html(
        screened, top_n=3, out_dir=tmp_path, run_date=RUN_DATE, charts=charts, **analysis
    )
    text = path.read_text(encoding="utf-8")
    assert text.count("data:image/png;base64,") >= 2


def test_html_survives_missing_charts(screened, tmp_path):
    path = write_html(
        screened, top_n=3, out_dir=tmp_path, run_date=RUN_DATE,
        charts={"main": None, "sectors": None},
    )
    assert path.exists()
    assert "<img" not in path.read_text(encoding="utf-8")


def test_html_escapes_hostile_names(screened, tmp_path):
    """A name is data, not markup."""
    poisoned = screened.copy()
    poisoned.loc[poisoned.index[0], "name"] = '<script>alert("x")</script>'
    text = write_html(
        poisoned, top_n=3, out_dir=tmp_path, run_date=RUN_DATE
    ).read_text(encoding="utf-8")

    assert '<script>alert("x")</script>' not in text
    assert "&lt;script&gt;" in text


def test_html_flags_a_leverage_mismatch(screened, tmp_path):
    underlying = simulate_gbm(mu=0.09, sigma=0.30, n_days=600, seed=44)
    wide = pd.DataFrame(
        {"069500.KS": underlying, "999999.KS": simulate_leveraged_path(underlying, 2.0)}
    )
    etfs = audit_leveraged_etfs(
        wide, products=(LeveragedETF("999999", "Mislabelled 3x", 3.0, "069500"),)
    )
    text = write_html(
        screened, top_n=3, out_dir=tmp_path, run_date=RUN_DATE, etfs=etfs
    ).read_text(encoding="utf-8")

    assert "note warn" in text
    assert "Mislabelled 3x" in text


def test_html_marks_negative_numbers(screened, tmp_path):
    text = write_html(
        screened, top_n=10, out_dir=tmp_path, run_date=RUN_DATE
    ).read_text(encoding="utf-8")
    assert 'class="num neg"' in text


def test_html_cells_carry_numeric_sort_keys(screened, tmp_path):
    text = write_html(
        screened, top_n=5, out_dir=tmp_path, run_date=RUN_DATE
    ).read_text(encoding="utf-8")
    assert re.search(r'data-v="-?\d+(\.\d+)?"', text)


def test_tile_labels_are_not_uppercased(screened, tmp_path):
    """text-transform:uppercase would render sigma as Sigma and mu as M."""
    text = write_html(
        screened, top_n=3, out_dir=tmp_path, run_date=RUN_DATE
    ).read_text(encoding="utf-8")

    assert "text-transform: uppercase" not in text
    assert "중위 σ" in text and "드래그/μ" in text


def test_count_columns_render_as_integers(screened, tmp_path, analysis):
    """Ranks and name counts are counts, not measurements.

    Checked on the rank column specifically -- a measurement that happens to
    round to 1.00 is perfectly legitimate elsewhere on the page.
    """
    text = write_html(
        screened, top_n=5, out_dir=tmp_path, run_date=RUN_DATE, **analysis
    ).read_text(encoding="utf-8")

    ranks = re.findall(r"<tr><td class=\"num\" data-v=\"[\d.]+\">([\d.]+)</td>", text)
    assert ranks, "no rank cells found"
    assert all("." not in r for r in ranks), ranks
    assert ranks[0] == "1"
