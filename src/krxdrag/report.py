"""Markdown / CSV / chart output for a completed screen."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from .config import REPORT_DIR
from .screener import summarise

PCT = 100.0


def _fmt_table(df: pd.DataFrame, cols: list[tuple[str, str]]) -> str:
    view = df[[c for c, _ in cols]].copy()
    view.columns = [label for _, label in cols]
    return view.to_markdown(index=False, floatfmt=".2f")


def write_csv(
    df: pd.DataFrame,
    out_dir: Path | None = None,
    run_date: date | None = None,
) -> Path:
    out_dir = out_dir or REPORT_DIR
    run_date = run_date or date.today()
    path = out_dir / f"krx_drag_{run_date:%Y%m%d}.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def write_markdown(
    df: pd.DataFrame,
    top_n: int = 25,
    out_dir: Path | None = None,
    run_date: date | None = None,
) -> Path:
    out_dir = out_dir or REPORT_DIR
    run_date = run_date or date.today()
    path = out_dir / f"krx_drag_{run_date:%Y%m%d}.md"
    s = summarise(df)

    d = df.copy()
    for col in ("sigma", "g", "mu", "drag", "se_drag"):
        d[col + "_pct"] = d[col] * PCT
    d["drag_ratio_pct"] = d["drag_ratio"] * PCT

    cols = [
        ("rank", "#"),
        ("name", "종목"),
        ("market", "시장"),
        ("sigma_pct", "σ %"),
        ("mu_pct", "μ %"),
        ("g_pct", "g %"),
        ("drag_pct", "드래그 ½σ² %"),
        ("drag_ratio_pct", "드래그/μ %"),
        ("gbm_score", "GBM 적합도"),
    ]

    worst = d.head(top_n)
    best = d.sort_values("drag").head(top_n)
    trap = d[(d["mu"] > 0) & (d["g"] < 0)].sort_values("drag", ascending=False).head(top_n)

    lines = [
        f"# KRX 변동성 드래그 리포트 — {run_date:%Y-%m-%d}",
        "",
        "산술 드리프트 μ 와 기하 성장률 g 의 차이는 Itô 보정항 ½σ² 과 정확히 같다.",
        "이 값이 **변동성 드래그**이며, 기대수익률 중 복리 실현에 도달하지 못하는 부분이다.",
        "",
        "## 요약",
        "",
        f"- 통과 종목 수: **{s.get('n_names', 0):,}**",
        f"- 중위 연변동성 σ: **{s.get('median_sigma', float('nan')) * PCT:.1f}%**",
        f"- 중위 드래그 ½σ²: **{s.get('median_drag', float('nan')) * PCT:.1f}%p**",
        f"- 중위 드래그/μ 비율: **{s.get('median_drag_ratio', float('nan')) * PCT:.1f}%**",
        f"- μ>0 이지만 g<0 인 종목 비중: **{s.get('share_g_negative_mu_positive', float('nan')) * PCT:.1f}%**",
        f"- Jarque–Bera 정규성 미기각 비중: **{s.get('share_normal_ok', float('nan')) * PCT:.1f}%**",
        "",
        f"## 드래그 상위 {top_n} (변동성이 가장 많이 갉아먹는 종목)",
        "",
        _fmt_table(worst, cols),
        "",
        f"## 드래그 하위 {top_n} (복리 효율이 가장 좋은 종목)",
        "",
        _fmt_table(best, cols),
        "",
        "## 변동성 함정: μ > 0 이지만 g < 0",
        "",
        "기대수익률은 양수이나 변동성 드래그가 이를 완전히 상쇄하여,",
        "실제 매수 후 보유 시 원금이 줄어든 종목이다.",
        "",
        _fmt_table(trap, cols) if not trap.empty else "_해당 종목 없음_",
        "",
        "---",
        "",
        "### 해석 시 주의",
        "",
        "1. μ 의 표준오차는 매우 크다. 드리프트 추정은 본질적으로 어려우므로 순위를 투자 신호로 읽지 말 것.",
        "2. σ² 은 상대적으로 정확히 추정되므로 드래그 ½σ² 자체는 μ 보다 신뢰할 만하다.",
        "3. GBM 적합도가 낮은 종목은 정규성·독립성 가정이 크게 깨진 경우이므로 해석에 유의할 것.",
        "4. 과거 실현 변동성이며 미래 예측이 아니다. 본 자료는 투자 자문이 아니다.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_chart(
    df: pd.DataFrame,
    out_dir: Path | None = None,
    run_date: date | None = None,
) -> Path | None:
    """Scatter of sigma vs the mu/g wedge, with the theoretical 0.5*sigma^2 curve."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return None

    out_dir = out_dir or REPORT_DIR
    run_date = run_date or date.today()
    path = out_dir / f"krx_drag_{run_date:%Y%m%d}.png"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    ax1.scatter(df["sigma"] * PCT, df["drag"] * PCT, s=9, alpha=0.45, color="#2b6cb0")
    grid = np.linspace(0, df["sigma"].max() * 1.02, 200)
    ax1.plot(grid * PCT, 0.5 * grid**2 * PCT, color="#c53030", lw=1.6, label=r"$\frac{1}{2}\sigma^2$")
    ax1.set_xlabel("annualised volatility  σ  (%)")
    ax1.set_ylabel("variance drag  ½σ²  (%p)")
    ax1.set_title("Drag is quadratic in volatility")
    ax1.legend()
    ax1.grid(alpha=0.25)

    ax2.scatter(df["mu"] * PCT, df["g"] * PCT, s=9, alpha=0.45, color="#2f855a")
    lim = [
        min(df["mu"].min(), df["g"].min()) * PCT,
        max(df["mu"].max(), df["g"].max()) * PCT,
    ]
    ax2.plot(lim, lim, color="#4a5568", ls="--", lw=1.2, label="g = μ (no drag)")
    ax2.axhline(0, color="#c53030", lw=1.0)
    ax2.set_xlabel("arithmetic drift  μ  (%)")
    ax2.set_ylabel("geometric drift  g = μ − ½σ²  (%)")
    ax2.set_title("Every point sits below the 45° line")
    ax2.legend()
    ax2.grid(alpha=0.25)

    fig.suptitle("KRX variance drag — Itô decomposition", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
