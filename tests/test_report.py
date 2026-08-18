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
