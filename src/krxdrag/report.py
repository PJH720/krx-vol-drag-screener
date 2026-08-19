"""Markdown / CSV / chart / HTML output for a completed screen.

Charts label sectors with Korean industry names, and CI containers generally
have no Korean font installed -- matplotlib would silently render every label as
tofu boxes. _korean_font() looks for one and, failing that, the sector charts
fall back to numeric labels with the mapping printed alongside, so the output is
readable everywhere rather than quietly corrupt on some machines.
"""

from __future__ import annotations

import base64
import functools
import html
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .config import REPORT_DIR
from .screener import summarise
from .volatility import DEFAULT_METHOD, METHODS, liquidity_bias_report

PCT = 100.0

_KOREAN_FONT_CANDIDATES = (
    "NanumGothic",
    "NanumBarunGothic",
    "Malgun Gothic",
    "AppleGothic",
    "Noto Sans CJK KR",
    "Noto Sans KR",
    "UnDotum",
    "Source Han Sans KR",
)

TABLE_COLS: list[tuple[str, str]] = [
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


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def _korean_font() -> str | None:
    """Name of an installed font that can render Hangul, or None.

    Cached: the answer cannot change within a process, and a full run would
    otherwise materialise matplotlib's whole font list five times.
    """
    try:
        from matplotlib import font_manager
    except ImportError:
        return None

    available = {f.name for f in font_manager.fontManager.ttflist}
    for candidate in _KOREAN_FONT_CANDIDATES:
        if candidate in available:
            return candidate
    return None


def _fmt_table(df: pd.DataFrame, cols: list[tuple[str, str]]) -> str:
    """Markdown table with counts rendered as counts.

    tabulate applies a single `floatfmt` to every numeric column as soon as one
    float is present, which turns rank 1 and "8 names" into "1.00" and "8.00".
    Passing one format per column keeps integer columns integral.
    """
    present = [(c, label) for c, label in cols if c in df.columns]
    view = df[[c for c, _ in present]].copy()
    fmts = [
        ".0f" if pd.api.types.is_integer_dtype(view[c]) else ".2f" for c, _ in present
    ]
    view.columns = [label for _, label in present]
    return view.to_markdown(index=False, floatfmt=tuple(fmts))


def _with_percent_columns(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    for col in ("sigma", "g", "mu", "drag", "se_drag", "drag_cont", "drag_jump", "drag_trend"):
        if col in d.columns:
            d[col + "_pct"] = d[col] * PCT
    if "drag_ratio" in d.columns:
        d["drag_ratio_pct"] = d["drag_ratio"] * PCT
    return d


def _path(
    out_dir: Path | None,
    run_date: date | None,
    suffix: str,
    stem: str = "krx_drag",
) -> tuple[Path, date]:
    """Resolve an output path, defaulting both the directory and the date.

    Kept in one place so REPORT_DIR stays a single indirection (the tests
    monkeypatch it) and the chart writers do not each re-derive the defaults.
    """
    out_dir = out_dir or REPORT_DIR
    run_date = run_date or date.today()
    return out_dir / f"{stem}_{run_date:%Y%m%d}.{suffix}", run_date


# --------------------------------------------------------------------------
# CSV
# --------------------------------------------------------------------------

def write_csv(
    df: pd.DataFrame,
    out_dir: Path | None = None,
    run_date: date | None = None,
) -> Path:
    path, _ = _path(out_dir, run_date, "csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


# --------------------------------------------------------------------------
# Markdown
# --------------------------------------------------------------------------

def _sector_section(
    sectors: pd.DataFrame | None,
    diversification: pd.DataFrame | None,
    top_n: int,
) -> list[str]:
    if sectors is None or sectors.empty:
        return []

    s = sectors.copy()
    for col in ("median_sigma", "median_drag", "drag_iqr", "share_trap"):
        if col in s.columns:
            s[col + "_pct"] = s[col] * PCT

    cols = [
        ("rank", "#"),
        ("sector", "업종"),
        ("n_names", "종목수"),
        ("median_sigma_pct", "중위 σ %"),
        ("median_drag_pct", "중위 드래그 %p"),
        ("drag_iqr_pct", "드래그 IQR %p"),
        ("share_trap_pct", "함정 비중 %"),
    ]

    lines = [
        f"## 업종별 드래그 (상위 {top_n})",
        "",
        f"종목 {sectors['n_names'].sum():,}개가 {len(sectors)}개 업종에 걸쳐 있다.",
        "구성종목이 적은 업종은 제외했다 — KRX 업종 분류에는 단일 종목 업종이 다수 있어,",
        "필터 없이는 우연히 변동성이 컸던 한 종목이 표 상단을 차지한다.",
        "",
        _fmt_table(s.head(top_n), cols),
        "",
    ]

    if diversification is not None and not diversification.empty:
        d = diversification.copy()
        for col in ("median_drag", "portfolio_drag", "drag_saved"):
            d[col + "_pct"] = d[col] * PCT
        d["drag_saved_share_pct"] = d["drag_saved_share"] * PCT

        lines += [
            "### 분산투자가 걷어내는 드래그",
            "",
            "**업종의 중위 드래그는 그 업종을 보유했을 때의 드래그가 아니다.**",
            "포트폴리오의 분산은 종목 간 상관이 1보다 작은 만큼 감쇠하지만 개별 종목의 분산은 그렇지 않다.",
            "아래는 동일가중 업종 포트폴리오를 실제로 구성해 계산한 드래그와 구성종목 중위값의 차이다.",
            "",
            _fmt_table(
                d.head(top_n),
                [
                    ("sector", "업종"),
                    ("n_names", "종목수"),
                    ("median_drag_pct", "구성종목 중위 %p"),
                    ("portfolio_drag_pct", "동일가중 포트폴리오 %p"),
                    ("drag_saved_pct", "절감 %p"),
                    ("drag_saved_share_pct", "절감 비율 %"),
                ],
            ),
            "",
        ]
    return lines


def _jump_section(df: pd.DataFrame, top_n: int) -> list[str]:
    if "drag_jump" not in df.columns:
        return []

    d = _with_percent_columns(df)
    d["jump_ratio_pct"] = d["jump_ratio"] * PCT
    # The total shown beside the two components must be their sum. That is the
    # quadratic-variation drag, not DragMetrics.drag -- the latter is built from
    # the ddof=1 sample variance and differs by the mean-return term, so
    # printing it here would show a table whose columns visibly do not add up.
    d["drag_qv_pct"] = (d["drag_cont"] + d["drag_jump"]) * PCT
    worst = d.sort_values("drag_jump", ascending=False).head(top_n)
    share_jumpy = float(df["has_jumps"].mean()) * PCT if "has_jumps" in df else float("nan")

    return [
        "## 드래그의 점프 성분",
        "",
        "실현분산은 연속 확산과 불연속 점프를 모두 담고 있다. 이중멱변동(bipower variation)은",
        "각 수익률을 이웃과 곱해 점프의 영향을 억제하므로, `RV − BPV` 가 점프 분산이 된다.",
        "이에 따라 **실현 이차변동** `(A/n)Σr²` 이 `½σ²_연속 + ½σ²_점프` 로 정확히 분해된다.",
        "",
        "> 아래 표의 «총» 열은 이 이차변동 기준 드래그다. 표본분산(ddof=1) 기반의",
        "> 메인 표 `드래그 ½σ²` 와는 평균수익률 항만큼 미세하게 다르다.",
        "",
        f"- BNS 검정에서 유의한 점프가 확인된 종목: **{share_jumpy:.1f}%**",
        f"- 중위 점프 비중 (점프분산/실현분산): **{df['jump_ratio'].median() * PCT:.1f}%**",
        "",
        f"### 점프 드래그 상위 {top_n}",
        "",
        _fmt_table(
            worst,
            [
                ("rank", "#"),
                ("name", "종목"),
                ("sigma_pct", "σ %"),
                ("drag_qv_pct", "총 드래그 (QV) %p"),
                ("drag_cont_pct", "연속 %p"),
                ("drag_jump_pct", "점프 %p"),
                ("jump_ratio_pct", "점프 비중 %"),
                ("bns_pvalue", "BNS p"),
            ],
        ),
        "",
        "> 일간 데이터는 이 분해의 점근이론이 상정하는 것보다 성긴 격자다.",
        "> 분산이 어디서 오는지에 대한 지표로 읽되, 정밀한 측정값으로 보지 말 것.",
        "",
    ]


def _rolling_section(rolling: pd.DataFrame | None, window: int) -> list[str]:
    if rolling is None or rolling.empty:
        return []

    recent = rolling.iloc[-1]
    first = rolling.iloc[0]
    change = (recent["median"] - first["median"]) * PCT

    return [
        f"## 드래그 추이 ({window}거래일 롤링)",
        "",
        f"- 기간 시작 중위 드래그: **{first['median'] * PCT:.1f}%p**",
        f"- 기간 종료 중위 드래그: **{recent['median'] * PCT:.1f}%p** ({change:+.1f}%p)",
        f"- 최근 사분위 범위: **{recent['q1'] * PCT:.1f}%p ~ {recent['q3'] * PCT:.1f}%p**",
        "",
        "횡단면 중위값이 올라가면 한두 종목이 아니라 시장 전체가 더 많은 드래그를 치르고 있다는 뜻이다.",
        "",
    ]


def _volatility_section(df: pd.DataFrame, method: str = DEFAULT_METHOD) -> list[str]:
    """Range estimates beside close-to-close, and the liquidity gap between them."""
    col = f"sigma_sq_{method}"
    if col not in df.columns:
        return []

    label = METHODS.get(method, method)
    present = [m for m in METHODS if m == "close_to_close" or f"sigma_sq_{m}" in df.columns]

    rows = []
    for m in present:
        series = df["sigma_sq"] if m == "close_to_close" else df[f"sigma_sq_{m}"]
        series = series[np.isfinite(series) & (series > 0)]
        if series.empty:
            continue
        rows.append(
            {
                "추정량": METHODS[m],
                "중위 σ %": float(np.sqrt(series.median())) * PCT,
                "중위 드래그 %p": float(series.median()) * 0.5 * PCT,
            }
        )

    lines = [
        "## 변동성 추정량 비교",
        "",
        "드래그는 전적으로 `σ²` 의 함수인데, 메인 표의 `σ` 는 종가 대비 표본분산이다 —",
        "일중에 가격이 무엇을 했는지 전부 버리는, 가장 비효율적인 추정량이다.",
        "아래는 같은 데이터를 봉 전체로 읽었을 때의 값이다.",
        "",
        pd.DataFrame(rows).to_markdown(index=False, floatfmt=".2f"),
        "",
    ]

    if "limit_hit_share" in df.columns:
        limit = float(df["limit_hit_share"].mean()) * PCT
        lines += [f"- 가격제한폭(±30%) 도달 봉 비중: **{limit:.2f}%**", ""]

    table = liquidity_bias_report(df, method=method)
    if not table.empty:
        t = table.copy()
        t["median_gap_pct"] = t["median_gap"] * PCT
        t["median_turnover_bn"] = t["median_turnover"] / 1e8
        t["bucket"] = t["bucket"].astype(int)
        t["n_names"] = t["n_names"].astype(int)
        lines += [
            f"### {label} 과 종가 대비의 괴리 — 유동성 구간별",
            "",
            "범위 기반 추정량은 **모두 하방 편의**를 갖는다. 연속시간의 진짜 고·저가는 관측되지 않고,",
            "체결이 성길수록 관측 범위가 좁아지기 때문이다. 즉 **편의가 유동성과 상관된다.**",
            "아래에서 거래대금이 낮은 구간일수록 괴리가 커진다면 그것은 표본 추출의 산물이지,",
            "얇은 종목이 실제로 덜 변동적이라는 뜻이 아니다.",
            "",
            _fmt_table(
                t,
                [
                    ("bucket", "구간(1=최저 유동성)"),
                    ("n_names", "종목수"),
                    ("median_turnover_bn", "중위 거래대금(억)"),
                    ("median_gap_pct", "괴리 %"),
                ],
            ),
            "",
            "> 이 표가 단조롭게 기울어 있는 한, 범위 기반 추정량을 그대로 순위에 쓰면",
            "> **횡단면이 유동성 방향으로 왜곡된다.** 그래서 메인 순위는 여전히 종가 대비 값이다.",
            "",
        ]
    return lines


def _leverage_section(etfs: pd.DataFrame | None) -> list[str]:
    if etfs is None or etfs.empty:
        return []

    e = etfs.copy()
    for col in ("underlying_sigma", "underlying_drag", "theoretical_g", "actual_g",
                "drag_penalty", "critical_sigma"):
        e[col + "_pct"] = e[col] * PCT

    flagged = e[e["leverage_mismatch"]]
    lines = [
        "## 레버리지 ETF",
        "",
        "일간 리밸런싱 배율 L 펀드의 로그 성장률은 `g_L = L·μ − ½L²σ²`, 즉 `L·g − (L²−L)·D` 다.",
        "기초자산 성장률의 L배를 기대한 보유자는 정확히 `(L²−L)·D` 만큼 미달한다 —",
        "L=2 면 2D, L=3 과 L=−2 는 모두 6D. 인버스 2배가 3배 롱과 같은 벌점을 치른다.",
        "",
        _fmt_table(
            e,
            [
                ("name", "상품"),
                ("declared_leverage", "표방 배율"),
                ("realized_leverage", "실측 배율"),
                ("r_squared", "R²"),
                ("underlying_sigma_pct", "기초 σ %"),
                ("drag_penalty_pct", "드래그 벌점 %p"),
                ("theoretical_g_pct", "이론 g %"),
                ("actual_g_pct", "실현 g %"),
                ("critical_sigma_pct", "임계 σ %"),
            ],
        ),
        "",
        "실측 배율은 펀드의 일간 단순수익률을 기초지수에 회귀한 기울기다. 표방 배율을 그대로 믿지 않고",
        "데이터로 확인한다 — 임계 σ 는 `√(2μ/L)` 로, 기초자산 변동성이 이를 넘으면 `g_L < 0` 이 된다.",
        "",
    ]
    if not flagged.empty:
        names = ", ".join(flagged["name"].tolist())
        lines += [
            f"> ⚠️ 표방 배율과 실측 배율이 어긋난 상품: **{names}**. 상품 테이블 확인이 필요하다.",
            "",
        ]
    return lines


def write_markdown(
    df: pd.DataFrame,
    top_n: int = 25,
    out_dir: Path | None = None,
    run_date: date | None = None,
    *,
    sectors: pd.DataFrame | None = None,
    diversification: pd.DataFrame | None = None,
    rolling: pd.DataFrame | None = None,
    etfs: pd.DataFrame | None = None,
    rolling_window: int = 126,
) -> Path:
    path, run_date = _path(out_dir, run_date, "md")
    s = summarise(df)
    d = _with_percent_columns(df)

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
        _fmt_table(worst, TABLE_COLS),
        "",
        f"## 드래그 하위 {top_n} (복리 효율이 가장 좋은 종목)",
        "",
        _fmt_table(best, TABLE_COLS),
        "",
        "## 변동성 함정: μ > 0 이지만 g < 0",
        "",
        "기대수익률은 양수이나 변동성 드래그가 이를 완전히 상쇄하여,",
        "실제 매수 후 보유 시 원금이 줄어든 종목이다.",
        "",
        _fmt_table(trap, TABLE_COLS) if not trap.empty else "_해당 종목 없음_",
        "",
    ]

    lines += _rolling_section(rolling, rolling_window)
    lines += _volatility_section(df)
    lines += _sector_section(sectors, diversification, top_n)
    lines += _jump_section(df, top_n)
    lines += _leverage_section(etfs)

    lines += [
        "---",
        "",
        "### 해석 시 주의",
        "",
        "1. μ 의 표준오차는 매우 크다. 드리프트 추정은 본질적으로 어려우므로 순위를 투자 신호로 읽지 말 것.",
        "2. σ² 은 상대적으로 정확히 추정되므로 드래그 ½σ² 자체는 μ 보다 신뢰할 만하다.",
        "3. GBM 적합도가 낮은 종목은 정규성·독립성 가정이 크게 깨진 경우이므로 해석에 유의할 것.",
        "4. 업종 중위 드래그는 업종 포트폴리오의 드래그가 아니다. 둘을 혼동하지 말 것.",
        "5. 과거 실현 변동성이며 미래 예측이 아니다. 본 자료는 투자 자문이 아니다.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# charts
# --------------------------------------------------------------------------

def _pyplot():
    """matplotlib.pyplot on the Agg backend, or None when unavailable."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    font = _korean_font()
    if font:
        plt.rcParams["font.family"] = font
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def write_chart(
    df: pd.DataFrame,
    out_dir: Path | None = None,
    run_date: date | None = None,
) -> Path | None:
    """Scatter of sigma vs the mu/g wedge, with the theoretical 0.5*sigma^2 curve."""
    plt = _pyplot()
    if plt is None:
        return None
    path, _ = _path(out_dir, run_date, "png")

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


def write_sector_chart(
    sectors: pd.DataFrame,
    diversification: pd.DataFrame | None = None,
    out_dir: Path | None = None,
    run_date: date | None = None,
    top_n: int = 15,
) -> tuple[Path, dict[str, str]] | None:
    """Median constituent drag vs equal-weight portfolio drag, per sector.

    Returns the chart path and the label->sector mapping, which is non-empty
    only when no Korean font was available and numeric labels were substituted.
    """
    if sectors is None or sectors.empty:
        return None
    plt = _pyplot()
    if plt is None:
        return None

    path, run_date = _path(out_dir, run_date, "png", stem="krx_drag_sectors")

    top = sectors.head(top_n).iloc[::-1]  # largest at the top of a barh
    has_font = _korean_font() is not None
    if has_font:
        labels = top["sector"].tolist()
        mapping: dict[str, str] = {}
    else:
        labels = [f"S{i:02d}" for i in range(len(top), 0, -1)]
        mapping = dict(zip(labels, top["sector"].tolist()))

    portfolio = None
    if diversification is not None and not diversification.empty:
        lookup = diversification.set_index("sector")["portfolio_drag"]
        portfolio = [lookup.get(s, np.nan) for s in top["sector"]]

    fig, ax = plt.subplots(figsize=(10, max(4.0, 0.42 * len(top) + 1.5)))
    y = np.arange(len(top))

    ax.barh(y, top["median_drag"] * PCT, color="#2b6cb0", alpha=0.85,
            label="median constituent")
    if portfolio is not None and np.isfinite(portfolio).any():
        ax.scatter(np.asarray(portfolio) * PCT, y, color="#c53030", zorder=3, s=38,
                   label="equal-weight portfolio")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("variance drag  ½σ²  (%p)")
    ax.set_title("Sector drag: a typical constituent vs holding the sector")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path, mapping


def write_rolling_chart(
    rolling: pd.DataFrame,
    out_dir: Path | None = None,
    run_date: date | None = None,
    window: int = 126,
) -> Path | None:
    """Cross-sectional median drag through time, with the interquartile band."""
    if rolling is None or rolling.empty:
        return None
    plt = _pyplot()
    if plt is None:
        return None

    path, run_date = _path(out_dir, run_date, "png", stem="krx_drag_rolling")

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.fill_between(
        rolling.index, rolling["q1"] * PCT, rolling["q3"] * PCT,
        color="#2b6cb0", alpha=0.18, label="interquartile range",
    )
    ax.plot(rolling.index, rolling["median"] * PCT, color="#2b6cb0", lw=1.8,
            label="cross-sectional median")
    ax.set_xlabel("date")
    ax.set_ylabel("variance drag  ½σ²  (%p)")
    ax.set_title(f"Market-wide variance drag, {window}-day rolling window")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def write_leverage_chart(
    sigmas: tuple[float, ...] = (0.20, 0.40, 0.65),
    mu: float = 0.10,
    out_dir: Path | None = None,
    run_date: date | None = None,
) -> Path | None:
    """Log growth against the multiple: an inverted parabola with a zero crossing."""
    plt = _pyplot()
    if plt is None:
        return None
    from .leverage import leverage_curve, optimal_leverage

    path, run_date = _path(out_dir, run_date, "png", stem="krx_drag_leverage")

    grid = np.linspace(-3.0, 5.0, 401)
    colors = ("#2f855a", "#2b6cb0", "#c53030")

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    peak = 0.0
    for sigma, color in zip(sigmas, colors):
        curve = leverage_curve(mu, sigma, grid)
        ax.plot(curve["leverage"], curve["levered_g"] * PCT, color=color, lw=1.8,
                label=f"σ = {sigma * PCT:.0f}%")
        star = optimal_leverage(mu, sigma**2)
        if -3.0 <= star <= 5.0:
            g_star = mu * star - 0.5 * sigma**2 * star**2
            peak = max(peak, g_star)
            ax.scatter([star], [g_star * PCT], color=color, s=40, zorder=3)
        # right-hand zero crossing: beyond it, more leverage destroys capital
        crossing = 2.0 * mu / sigma**2
        if 0.0 < crossing <= 5.0:
            ax.scatter([crossing], [0.0], color=color, s=42, zorder=3,
                       marker="x", linewidths=1.8)

    ax.axhline(0, color="#4a5568", lw=1.0)
    ax.axvline(1, color="#4a5568", ls=":", lw=1.0)
    # A high-sigma parabola plunges fast and would flatten everything else into
    # the top pixel row, so frame the window on the region that carries the
    # message: the peaks and the zero crossings.
    ax.set_ylim(-6.0 * peak * PCT if peak > 0 else -60.0, 2.2 * peak * PCT + 4.0)
    ax.set_xlabel("leverage  L")
    ax.set_ylabel(r"log growth  $g_L = L\mu - \frac{1}{2}L^2\sigma^2$  (%)")
    ax.set_title(
        f"More leverage stops helping (μ = {mu * PCT:.0f}%; "
        "● Kelly L*, ✕ zero crossing)"
    )
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# HTML
# --------------------------------------------------------------------------

_CSS = """
:root {
  --bg: #ffffff; --fg: #1a202c; --muted: #4a5568; --line: #e2e8f0;
  --accent: #2b6cb0; --warn: #c53030; --panel: #f7fafc;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #14181d; --fg: #e6eaf0; --muted: #9aa5b4; --line: #2b3440;
    --accent: #6aa9e9; --warn: #f0776c; --panel: #1b2027;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0; padding: 2rem 1.25rem 4rem; background: var(--bg); color: var(--fg);
  font: 15px/1.65 -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans KR",
        "Malgun Gothic", "Apple SD Gothic Neo", sans-serif;
}
main { max-width: 1120px; margin: 0 auto; }
h1 { font-size: 1.75rem; margin: 0 0 .35rem; letter-spacing: -.02em; }
h2 { font-size: 1.2rem; margin: 2.5rem 0 .75rem; padding-bottom: .35rem;
     border-bottom: 1px solid var(--line); }
h3 { font-size: 1.02rem; margin: 1.75rem 0 .5rem; color: var(--muted); }
p, li { color: var(--fg); }
.sub { color: var(--muted); margin: 0 0 2rem; }
.tiles { display: grid; gap: .75rem; grid-template-columns: repeat(auto-fit, minmax(165px, 1fr)); }
.tile { background: var(--panel); border: 1px solid var(--line); border-radius: 9px; padding: .85rem 1rem; }
/* No text-transform here: uppercasing would turn σ into Σ and μ into Μ,
   destroying the very notation these labels name. */
.tile .k { display: block; font-size: .78rem; letter-spacing: .02em;
           color: var(--muted); margin-bottom: .3rem; }
.tile .v { font-size: 1.45rem; font-weight: 650; font-variant-numeric: tabular-nums; }
.scroll { overflow-x: auto; -webkit-overflow-scrolling: touch; margin: .5rem 0 1rem; }
table { border-collapse: collapse; width: 100%; font-size: .875rem; }
th, td { padding: .45rem .7rem; text-align: right; border-bottom: 1px solid var(--line);
         white-space: nowrap; }
th:nth-child(2), td:nth-child(2), th:first-child, td:first-child { text-align: left; }
th { position: sticky; top: 0; background: var(--bg); cursor: pointer;
     user-select: none; font-weight: 600; color: var(--muted); }
th:hover { color: var(--accent); }
th::after { content: " ⇅"; opacity: .35; font-size: .8em; }
tbody tr:hover { background: var(--panel); }
td.num { font-variant-numeric: tabular-nums; }
.neg { color: var(--warn); }
figure { margin: 1rem 0 1.5rem; }
img { max-width: 100%; height: auto; border: 1px solid var(--line); border-radius: 8px;
      background: #fff; }
figcaption { color: var(--muted); font-size: .82rem; margin-top: .4rem; }
.note { background: var(--panel); border-left: 3px solid var(--accent);
        padding: .8rem 1rem; border-radius: 0 6px 6px 0; margin: 1rem 0; }
.note.warn { border-left-color: var(--warn); }
footer { margin-top: 3rem; padding-top: 1rem; border-top: 1px solid var(--line);
         color: var(--muted); font-size: .84rem; }
"""

_JS = """
document.querySelectorAll('table').forEach(function (table) {
  table.querySelectorAll('th').forEach(function (th, col) {
    th.addEventListener('click', function () {
      var body = table.tBodies[0];
      var rows = Array.prototype.slice.call(body.rows);
      var asc = th.dataset.asc !== 'true';
      table.querySelectorAll('th').forEach(function (o) { delete o.dataset.asc; });
      th.dataset.asc = asc;
      rows.sort(function (a, b) {
        var x = a.cells[col].dataset.v, y = b.cells[col].dataset.v;
        var nx = parseFloat(x), ny = parseFloat(y);
        var both = !isNaN(nx) && !isNaN(ny);
        var cmp = both ? nx - ny : String(x).localeCompare(String(y), 'ko');
        return asc ? cmp : -cmp;
      });
      rows.forEach(function (r) { body.appendChild(r); });
    });
  });
});
"""


def _embed_png(path: Path | None) -> str | None:
    """Base64 data URI for a PNG, so the page carries its own images."""
    if path is None or not Path(path).exists():
        return None
    encoded = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _figure(src: str | None, caption: str, alt: str) -> str:
    if src is None:
        return ""
    return (
        f'<figure><img src="{src}" alt="{html.escape(alt)}">'
        f"<figcaption>{html.escape(caption)}</figcaption></figure>"
    )


# Columns holding counts, not measurements: rendering these as "1.00" is noise.
INT_COLUMNS: frozenset[str] = frozenset(
    {"rank", "n_names", "n_obs", "code"}
)


def _cell(value, fmt: str = "{:.2f}") -> str:
    """One <td>, carrying a sort key and a negative-number class."""
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return '<td class="num" data-v="">—</td>'
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        num = float(value)
        cls = "num neg" if num < 0 else "num"
        return f'<td class="{cls}" data-v="{num}">{fmt.format(num)}</td>'
    text = html.escape(str(value))
    return f'<td data-v="{text}">{text}</td>'


def _html_table(df: pd.DataFrame, cols: list[tuple[str, str]]) -> str:
    present = [(c, label) for c, label in cols if c in df.columns]
    if not present or df.empty:
        return "<p><em>해당 종목 없음</em></p>"

    head = "".join(f"<th>{html.escape(label)}</th>" for _, label in present)
    body = "".join(
        "<tr>"
        + "".join(
            _cell(row[c], "{:.0f}" if c in INT_COLUMNS else "{:.2f}")
            for c, _ in present
        )
        + "</tr>"
        for _, row in df.iterrows()
    )
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def _tile(label: str, value: str) -> str:
    return f'<div class="tile"><span class="k">{html.escape(label)}</span>' \
           f'<span class="v">{html.escape(value)}</span></div>'


def write_html(
    df: pd.DataFrame,
    top_n: int = 25,
    out_dir: Path | None = None,
    run_date: date | None = None,
    *,
    sectors: pd.DataFrame | None = None,
    diversification: pd.DataFrame | None = None,
    rolling: pd.DataFrame | None = None,
    etfs: pd.DataFrame | None = None,
    rolling_window: int = 126,
    charts: dict[str, Path | None] | None = None,
    sector_labels: dict[str, str] | None = None,
) -> Path:
    """Single self-contained HTML leaderboard.

    Every asset is inlined -- CSS, the sort script and the charts as base64 data
    URIs -- so the page works from a file:// URL with no network at all.
    """
    path, run_date = _path(out_dir, run_date, "html")
    charts = charts or {}
    s = summarise(df)
    d = _with_percent_columns(df)

    def pct(key: str, suffix: str = "%") -> str:
        v = s.get(key, float("nan"))
        return "—" if not np.isfinite(v) else f"{v * PCT:.1f}{suffix}"

    worst = d.head(top_n)
    best = d.sort_values("drag").head(top_n)
    trap = d[(d["mu"] > 0) & (d["g"] < 0)].sort_values("drag", ascending=False).head(top_n)

    parts: list[str] = [
        # Without an in-document charset there is no Content-Type either (the
        # page is opened from file://), so the browser decodes this almost
        # entirely Hangul document with its locale default and renders mojibake.
        "<!doctype html>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>KRX 변동성 드래그 — {run_date:%Y-%m-%d}</title>",
        f"<style>{_CSS}</style>",
        "<main>",
        f"<h1>KRX 변동성 드래그 리포트</h1>",
        f'<p class="sub">{run_date:%Y-%m-%d} · 산술 드리프트 μ 와 기하 성장률 g 의 차이는 '
        "Itô 보정항 ½σ² 과 정확히 같다. 그 차이가 복리 실현에 도달하지 못하는 부분이다.</p>",
        '<div class="tiles">',
        _tile("통과 종목", f"{s.get('n_names', 0):,}"),
        _tile("중위 σ", pct("median_sigma")),
        _tile("중위 드래그", pct("median_drag", "%p")),
        _tile("중위 드래그/μ", pct("median_drag_ratio")),
        _tile("μ>0 이지만 g<0", pct("share_g_negative_mu_positive")),
        _tile("정규성 미기각", pct("share_normal_ok")),
        "</div>",
        _figure(
            _embed_png(charts.get("main")),
            "드래그는 변동성의 제곱에 비례하며, 모든 종목이 45° 선 아래에 있다.",
            "sigma versus drag scatter and mu versus g scatter",
        ),
        f"<h2>드래그 상위 {top_n}</h2>",
        _html_table(worst, TABLE_COLS),
        f"<h2>드래그 하위 {top_n}</h2>",
        _html_table(best, TABLE_COLS),
        "<h2>변동성 함정: μ &gt; 0 이지만 g &lt; 0</h2>",
        '<div class="note">기대수익률은 양수이나 드래그가 이를 완전히 상쇄하여, '
        "매수 후 보유 시 원금이 줄어든 종목이다.</div>",
        _html_table(trap, TABLE_COLS),
    ]

    if rolling is not None and not rolling.empty:
        parts += [
            f"<h2>드래그 추이 ({rolling_window}거래일 롤링)</h2>",
            _figure(
                _embed_png(charts.get("rolling")),
                "횡단면 중위값이 올라가면 시장 전체가 더 많은 드래그를 치르고 있다는 뜻이다.",
                "cross-sectional median drag over time",
            ),
        ]

    if sectors is not None and not sectors.empty:
        sec = sectors.copy()
        for col in ("median_sigma", "median_drag", "drag_iqr", "share_trap"):
            if col in sec.columns:
                sec[col + "_pct"] = sec[col] * PCT
        parts += [
            "<h2>업종별 드래그</h2>",
            _figure(
                _embed_png(charts.get("sectors")),
                "막대는 구성종목 중위 드래그, 점은 동일가중 업종 포트폴리오의 드래그다. "
                "그 간격이 분산투자 이득이다.",
                "median constituent drag versus equal weight portfolio drag by sector",
            ),
        ]
        if sector_labels:
            # Built through _html_table so the cells carry the data-v keys the
            # sort script reads. Hand-rolled <td>s left every header looking
            # clickable (cursor, hover colour, the ⇅ affordance) while doing
            # nothing.
            parts.append(
                _html_table(
                    pd.DataFrame(
                        {"label": list(sector_labels), "sector": list(sector_labels.values())}
                    ),
                    [("label", "라벨"), ("sector", "업종")],
                )
            )
        parts.append(
            _html_table(
                sec.head(top_n),
                [
                    ("rank", "#"),
                    ("sector", "업종"),
                    ("n_names", "종목수"),
                    ("median_sigma_pct", "중위 σ %"),
                    ("median_drag_pct", "중위 드래그 %p"),
                    ("drag_iqr_pct", "드래그 IQR %p"),
                    ("share_trap_pct", "함정 비중 %"),
                ],
            )
        )
        if diversification is not None and not diversification.empty:
            div = diversification.copy()
            for col in ("median_drag", "portfolio_drag", "drag_saved"):
                div[col + "_pct"] = div[col] * PCT
            div["drag_saved_share_pct"] = div["drag_saved_share"] * PCT
            parts += [
                "<h3>분산투자가 걷어내는 드래그</h3>",
                '<div class="note">업종의 중위 드래그는 그 업종을 <em>보유</em>했을 때의 '
                "드래그가 아니다. 포트폴리오 분산은 상관이 1보다 작은 만큼 감쇠한다.</div>",
                _html_table(
                    div.head(top_n),
                    [
                        ("sector", "업종"),
                        ("n_names", "종목수"),
                        ("median_drag_pct", "구성종목 중위 %p"),
                        ("portfolio_drag_pct", "포트폴리오 %p"),
                        ("drag_saved_pct", "절감 %p"),
                        ("drag_saved_share_pct", "절감 비율 %"),
                    ],
                ),
            ]

    vol_col = f"sigma_sq_{DEFAULT_METHOD}"
    if vol_col in df.columns:
        rows = []
        for m in METHODS:
            series = df["sigma_sq"] if m == "close_to_close" else df.get(f"sigma_sq_{m}")
            if series is None:
                continue
            series = series[np.isfinite(series) & (series > 0)]
            if series.empty:
                continue
            rows.append(
                {
                    "method": METHODS[m],
                    "sigma_pct": float(np.sqrt(series.median())) * PCT,
                    "drag_pct": float(series.median()) * 0.5 * PCT,
                }
            )
        parts += [
            "<h2>변동성 추정량 비교</h2>",
            '<div class="note">드래그는 전적으로 σ² 의 함수인데, 메인 표의 σ 는 '
            "종가 대비 표본분산 — 일중 움직임을 전부 버리는 가장 비효율적인 추정량입니다.</div>",
            _html_table(
                pd.DataFrame(rows),
                [("method", "추정량"), ("sigma_pct", "중위 σ %"), ("drag_pct", "중위 드래그 %p")],
            ),
        ]
        table = liquidity_bias_report(df, method=DEFAULT_METHOD)
        if not table.empty:
            t = table.copy()
            t["median_gap_pct"] = t["median_gap"] * PCT
            t["median_turnover_bn"] = t["median_turnover"] / 1e8
            parts += [
                "<h3>유동성 구간별 괴리</h3>",
                '<div class="note warn">범위 기반 추정량은 <strong>모두 하방 편의</strong>를 '
                "갖고, 체결이 성길수록 커집니다. 아래가 단조롭게 기울어 있다면 그것은 표본 추출의 "
                "산물이지 얇은 종목이 실제로 덜 변동적이라는 뜻이 아닙니다. "
                "그래서 메인 순위는 여전히 종가 대비 값입니다.</div>",
                _html_table(
                    t,
                    [
                        ("bucket", "구간(1=최저 유동성)"),
                        ("n_names", "종목수"),
                        ("median_turnover_bn", "중위 거래대금(억)"),
                        ("median_gap_pct", "괴리 %"),
                    ],
                ),
            ]

    if "drag_jump" in df.columns:
        dj = d.copy()
        dj["jump_ratio_pct"] = dj["jump_ratio"] * PCT
        dj["drag_qv_pct"] = (dj["drag_cont"] + dj["drag_jump"]) * PCT
        share = float(df["has_jumps"].mean()) * PCT if "has_jumps" in df else float("nan")
        parts += [
            "<h2>드래그의 점프 성분</h2>",
            '<div class="note">이중멱변동은 각 수익률을 이웃과 곱해 점프의 영향을 억제한다. '
            "<code>RV − BPV</code> 가 점프 분산이며, <strong>실현 이차변동</strong>이 연속 성분과 "
            "점프 성분으로 정확히 분해된다. 아래 «총» 열은 그 이차변동 기준 드래그로, "
            "표본분산 기반의 메인 표 값과는 평균수익률 항만큼 다르다. "
            f"BNS 검정에서 유의한 점프가 확인된 종목: <strong>{share:.1f}%</strong>.</div>",
            _html_table(
                dj.sort_values("drag_jump", ascending=False).head(top_n),
                [
                    ("rank", "#"),
                    ("name", "종목"),
                    ("sigma_pct", "σ %"),
                    ("drag_qv_pct", "총 드래그 (QV) %p"),
                    ("drag_cont_pct", "연속 %p"),
                    ("drag_jump_pct", "점프 %p"),
                    ("jump_ratio_pct", "점프 비중 %"),
                    ("bns_pvalue", "BNS p"),
                ],
            ),
        ]

    parts += [
        "<h2>레버리지</h2>",
        '<div class="note">일간 리밸런싱 배율 L 펀드의 로그 성장률은 '
        "<code>g_L = L·μ − ½L²σ²</code>, 즉 <code>L·g − (L²−L)·D</code> 다. "
        "L=2 면 2D, L=3 과 L=−2 는 모두 6D — 인버스 2배가 3배 롱과 같은 벌점을 치른다.</div>",
        _figure(
            _embed_png(charts.get("leverage")),
            "레버리지를 올리면 성장률은 켈리 배율에서 정점을 찍고 다시 내려와 0을 통과한다.",
            "log growth against leverage for several volatility levels",
        ),
    ]
    if etfs is not None and not etfs.empty:
        e = etfs.copy()
        for col in ("underlying_sigma", "drag_penalty", "theoretical_g",
                    "actual_g", "critical_sigma"):
            e[col + "_pct"] = e[col] * PCT
        parts.append(
            _html_table(
                e,
                [
                    ("name", "상품"),
                    ("declared_leverage", "표방 배율"),
                    ("realized_leverage", "실측 배율"),
                    ("r_squared", "R²"),
                    ("underlying_sigma_pct", "기초 σ %"),
                    ("drag_penalty_pct", "드래그 벌점 %p"),
                    ("theoretical_g_pct", "이론 g %"),
                    ("actual_g_pct", "실현 g %"),
                    ("critical_sigma_pct", "임계 σ %"),
                ],
            )
        )
        flagged = e[e["leverage_mismatch"]]
        if not flagged.empty:
            names = html.escape(", ".join(flagged["name"].tolist()))
            parts.append(
                f'<div class="note warn">표방 배율과 실측 배율이 어긋난 상품: '
                f"<strong>{names}</strong>. 상품 테이블 확인이 필요하다.</div>"
            )

    parts += [
        "<footer>",
        "<p><strong>해석 시 주의.</strong> μ 의 표준오차는 매우 크다 — 순위를 투자 신호로 읽지 말 것. "
        "σ² 은 상대적으로 정확하므로 드래그 ½σ² 자체는 μ 보다 신뢰할 만하다. "
        "GBM 적합도가 낮은 종목은 정규성·독립성 가정이 크게 깨진 경우다. "
        "업종 중위 드래그는 업종 포트폴리오의 드래그가 아니다. "
        "과거 실현 변동성이며 미래 예측이 아니다.</p>",
        "<p><strong>본 자료는 교육·연구 목적이며 투자 자문이 아니다.</strong></p>",
        "</footer>",
        "</main>",
        f"<script>{_JS}</script>",
    ]

    path.write_text("\n".join(p for p in parts if p), encoding="utf-8")
    return path
