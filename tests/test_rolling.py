"""Rolling estimates must agree with the one-shot estimator and track regimes."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from krxdrag.metrics import TRADING_DAYS, compute_drag, simulate_gbm
from krxdrag.rolling import (
    cross_sectional_drag,
    drag_trend,
    rolling_drag,
    rolling_panel,
)


# --- the load-bearing consistency test ----------------------------------------

@pytest.mark.parametrize("sigma", [0.15, 0.35, 0.65])
def test_full_length_window_equals_compute_drag(sigma):
    """A window covering the whole series must reproduce compute_drag() exactly.

    This is the regression guard: the two implementations use the same formulas
    and must never drift apart.
    """
    path = simulate_gbm(mu=0.09, sigma=sigma, n_days=800, seed=13)
    n_returns = len(path) - 1

    last = rolling_drag(path, window=n_returns).iloc[-1]
    m = compute_drag(path)

    for field in ("sigma", "sigma_sq", "g", "mu", "drag"):
        assert last[field] == pytest.approx(getattr(m, field), abs=1e-12), field


def test_rolling_preserves_the_ito_identity_on_every_row():
    path = simulate_gbm(mu=0.07, sigma=0.4, n_days=900, seed=2)
    rd = rolling_drag(path, window=126)
    assert np.allclose(rd["mu"] - rd["g"], rd["drag"], atol=1e-12)
    assert np.allclose(rd["drag"], 0.5 * rd["sigma_sq"], atol=1e-12)


# --- regime tracking -----------------------------------------------------------

def _regime_path(sigma_a: float, sigma_b: float, days: int = 500, seed: int = 5):
    """Two GBM regimes glued together at a known break point."""
    first = simulate_gbm(mu=0.05, sigma=sigma_a, n_days=days, seed=seed)
    second = simulate_gbm(mu=0.05, sigma=sigma_b, n_days=days, s0=first[-1], seed=seed + 1)
    return np.concatenate([first, second[1:]])


def test_rolling_drag_tracks_a_volatility_regime_change():
    """Drag must move from ~0.5*0.20^2 to ~0.5*0.60^2 across the break."""
    sigma_a, sigma_b, days, window = 0.20, 0.60, 500, 126
    path = _regime_path(sigma_a, sigma_b, days=days)
    rd = rolling_drag(path, window=window)

    # windows entirely inside each regime
    early = rd["drag"].iloc[: days - window]
    late = rd["drag"].iloc[-(days - window):]

    assert early.median() == pytest.approx(0.5 * sigma_a**2, rel=0.30)
    assert late.median() == pytest.approx(0.5 * sigma_b**2, rel=0.30)
    assert late.median() > 4 * early.median()


def test_the_break_is_detected_within_one_window():
    """Drag should cross the midpoint between regimes inside `window` days of the break."""
    sigma_a, sigma_b, days, window = 0.20, 0.60, 500, 126
    rd = rolling_drag(_regime_path(sigma_a, sigma_b, days=days), window=window)

    midpoint = 0.5 * (0.5 * sigma_a**2 + 0.5 * sigma_b**2)
    crossed = np.argmax(rd["drag"].to_numpy() > midpoint)
    # rolling_drag's first row ends at return index window-1, so convert to
    # position in the return series
    crossing_day = crossed + window
    assert days <= crossing_day <= days + window


# --- drag_trend ----------------------------------------------------------------

# drag_trend compares the last `window` returns against the `window` before
# them, so the regime break has to sit exactly on that boundary: each regime
# contributes `window` returns.
WINDOW = 126


def test_drag_trend_is_positive_when_volatility_rises():
    path = _regime_path(0.20, 0.60, days=WINDOW, seed=9)
    assert drag_trend(path, window=WINDOW) > 0


def test_drag_trend_is_negative_when_volatility_falls():
    path = _regime_path(0.60, 0.20, days=WINDOW, seed=9)
    assert drag_trend(path, window=WINDOW) < 0


def test_drag_trend_is_near_zero_in_a_stable_regime():
    path = simulate_gbm(mu=0.05, sigma=0.30, n_days=2 * WINDOW, seed=9)
    # noise scale on the difference of two independent 0.5*sigma^2 estimates
    tol = 6 * 0.5 * 0.30**2 * np.sqrt(2.0 / (WINDOW - 1)) * np.sqrt(2)
    assert abs(drag_trend(path, window=WINDOW)) < tol


def test_drag_trend_recovers_the_regime_gap():
    """The reported change should be close to the true 0.5*(sig_b^2 - sig_a^2)."""
    sig_a, sig_b = 0.20, 0.60
    got = drag_trend(_regime_path(sig_a, sig_b, days=WINDOW, seed=9), window=WINDOW)
    assert got == pytest.approx(0.5 * (sig_b**2 - sig_a**2), rel=0.35)


def test_drag_trend_is_nan_without_two_windows():
    path = simulate_gbm(mu=0.05, sigma=0.3, n_days=150, seed=1)
    assert np.isnan(drag_trend(path, window=126))


# --- panel / cross-section ------------------------------------------------------

@pytest.fixture
def wide_prices() -> pd.DataFrame:
    specs = {"A": 0.20, "B": 0.40, "C": 0.60}
    n = 700
    idx = pd.date_range("2023-01-02", periods=n + 1, freq="B")
    return pd.DataFrame(
        {
            name: simulate_gbm(mu=0.06, sigma=s, n_days=n, seed=i + 1)
            for i, (name, s) in enumerate(specs.items())
        },
        index=idx,
    )


def test_rolling_panel_matches_per_column_rolling_drag(wide_prices):
    panel = rolling_panel(wide_prices, window=126)
    single = rolling_drag(wide_prices["B"], window=126)
    assert np.allclose(panel["B"].dropna(), single["drag"], atol=1e-12)


def test_rolling_panel_orders_columns_by_true_volatility(wide_prices):
    panel = rolling_panel(wide_prices, window=252)
    medians = panel.median()
    assert medians["A"] < medians["B"] < medians["C"]


@pytest.mark.parametrize("field", ["sigma", "sigma_sq", "g", "mu", "drag"])
def test_rolling_panel_fields_agree_with_rolling_drag(wide_prices, field):
    panel = rolling_panel(wide_prices, window=126, field=field)
    single = rolling_drag(wide_prices["A"], window=126)
    assert np.allclose(panel["A"].dropna(), single[field], atol=1e-12)


def test_rolling_panel_rejects_unknown_field(wide_prices):
    with pytest.raises(ValueError, match="unknown field"):
        rolling_panel(wide_prices, field="nonsense")


def test_cross_sectional_drag_brackets_the_median(wide_prices):
    cs = cross_sectional_drag(wide_prices, window=126, min_names=3)
    assert not cs.empty
    assert (cs["q1"] <= cs["median"]).all()
    assert (cs["median"] <= cs["q3"]).all()
    assert (cs["n_names"] == 3).all()


# --- degenerate input ------------------------------------------------------------

def test_series_shorter_than_window_yields_empty_frame():
    path = simulate_gbm(mu=0.05, sigma=0.3, n_days=50, seed=1)
    rd = rolling_drag(path, window=126)
    assert rd.empty
    assert list(rd.columns) == ["sigma", "sigma_sq", "g", "mu", "drag"]


def test_window_below_two_is_rejected():
    with pytest.raises(ValueError, match="at least 2"):
        rolling_drag(simulate_gbm(0.05, 0.3, 300, seed=1), window=1)


def test_series_index_is_preserved(wide_prices):
    rd = rolling_drag(wide_prices["A"], window=126)
    assert isinstance(rd.index, pd.DatetimeIndex)
    assert rd.index[-1] == wide_prices.index[-1]


def test_nonpositive_prices_are_dropped():
    prices = pd.Series([100.0, 0.0, 110.0, -3.0, 121.0, 130.0] + [140.0] * 200)
    rd = rolling_drag(prices, window=50)
    assert not rd.empty
    assert np.isfinite(rd["sigma"]).all()
