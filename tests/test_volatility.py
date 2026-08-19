"""Range estimators must recover a known sigma, and fail in the documented ways.

Every claim the module docstring makes -- drift invariance, efficiency, the
overnight blind spot, the liquidity-correlated sampling bias -- is asserted here
against simulated bars whose true sigma is known.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from krxdrag.volatility import (
    DEFAULT_METHOD,
    DRIFT_FREE,
    INTRADAY_ONLY,
    KRX_PRICE_LIMIT,
    METHODS,
    RELATIVE_EFFICIENCY,
    close_to_close,
    estimate_all,
    garman_klass_daily,
    limit_hit_mask,
    liquidity_bias_report,
    parkinson_daily,
    rogers_satchell_daily,
    simulate_ohlc,
    yang_zhang,
)

SIGMA = 0.40
TRUE_VAR = SIGMA**2
FINE = 2000          # densely traded day
DAYS = 20_000


def _est(mu=0.0, sigma=SIGMA, steps=FINE, days=DAYS, overnight=0.0, seed=1):
    return estimate_all(
        simulate_ohlc(mu, sigma, days, steps_per_day=steps,
                      overnight_share=overnight, seed=seed)
    )


# --- recovery ---------------------------------------------------------------

@pytest.mark.parametrize("method", sorted(METHODS))
def test_every_estimator_recovers_a_known_sigma(method):
    """Densely sampled, driftless, gapless: all five must land near the truth."""
    got = _est().get(method)
    assert got == pytest.approx(TRUE_VAR, rel=0.06), method


def test_estimators_are_all_biased_downward_not_upward():
    """The sampling bias has a direction: unobserved extremes narrow the range."""
    e = _est(steps=100)
    for method in sorted(INTRADAY_ONLY):
        assert e.get(method) < TRUE_VAR, method
    # close-to-close never sees a range, so it is unaffected
    assert e.close_to_close == pytest.approx(TRUE_VAR, rel=0.05)


# --- drift: the reason Rogers-Satchell and Yang-Zhang are the defaults -------

@pytest.mark.parametrize("method", sorted(DRIFT_FREE))
def test_drift_free_estimators_do_not_move_with_the_drift(method):
    flat = _est(mu=0.0, seed=5).get(method)
    steep = _est(mu=6.0, seed=5).get(method)
    assert steep == pytest.approx(flat, rel=0.05), method


def test_parkinson_bias_grows_with_the_drift():
    """Its error is correlated with mu -- which is what disqualifies it here.

    A sigma^2 error that tracks mu contaminates the drag-versus-mu relationship
    the screen exists to describe, which is worse than a constant bias.
    """
    biases = [_est(mu=mu, seed=5).parkinson / TRUE_VAR - 1.0 for mu in (0.0, 1.0, 3.0, 6.0)]
    assert biases == sorted(biases)          # monotone in drift
    assert biases[0] < 0.0                   # mildly low with no drift
    assert biases[-1] > 0.20                 # badly high once drift is large


def test_default_method_is_drift_free():
    assert DEFAULT_METHOD in DRIFT_FREE


# --- efficiency: the reason to bother at all --------------------------------

@pytest.mark.parametrize("method", ["parkinson", "garman_klass", "rogers_satchell"])
def test_daily_estimate_is_more_efficient_than_close_to_close(method):
    bars = simulate_ohlc(0.0, SIGMA, 60_000, steps_per_day=FINE, seed=3)
    o, h, l, c = (bars[k].to_numpy() for k in ("open", "high", "low", "close"))

    daily = {
        "parkinson": parkinson_daily(h, l),
        "garman_klass": garman_klass_daily(o, h, l, c),
        "rogers_satchell": rogers_satchell_daily(o, h, l, c),
    }[method]
    cc = np.diff(np.log(c)) ** 2

    measured = float(cc.var() / daily.var())
    assert measured == pytest.approx(RELATIVE_EFFICIENCY[method], rel=0.25), method
    assert measured > 3.0


# --- the overnight blind spot ------------------------------------------------

def test_intraday_estimators_miss_the_overnight_gap():
    """Bar-only estimators cannot see the move between close and next open."""
    e = _est(overnight=0.4, seed=4)
    for method in sorted(INTRADAY_ONLY):
        assert e.get(method) < 0.65 * TRUE_VAR, method


def test_close_to_close_and_yang_zhang_span_the_gap():
    e = _est(overnight=0.4, seed=4)
    assert e.close_to_close == pytest.approx(TRUE_VAR, rel=0.06)
    assert e.yang_zhang == pytest.approx(TRUE_VAR, rel=0.08)


def test_yang_zhang_tracks_a_growing_overnight_share():
    got = [_est(overnight=s, seed=6).yang_zhang for s in (0.0, 0.3, 0.6)]
    for v in got:
        assert v == pytest.approx(TRUE_VAR, rel=0.10)


# --- KRX price limits ---------------------------------------------------------

def test_limit_days_are_detected():
    close = np.array([100.0, 130.0, 130.0, 91.0, 91.0])
    high = np.array([100.0, 130.0, 131.0, 100.0, 91.0])
    low = np.array([100.0, 120.0, 129.0, 91.0, 90.0])

    mask = limit_hit_mask(close, high, low, limit=KRX_PRICE_LIMIT)
    assert mask.tolist() == [True, False, True, False]


def test_quiet_series_hits_no_limits():
    bars = simulate_ohlc(0.0, 0.20, 500, steps_per_day=200, seed=8)
    assert estimate_all(bars).limit_hit_share == 0.0


def test_limit_mask_needs_two_bars():
    assert limit_hit_mask(np.array([100.0]), np.array([100.0]), np.array([100.0])).size == 0


# --- the comparison object ------------------------------------------------------

def test_estimate_all_reports_the_gap_against_close_to_close():
    e = _est(steps=100, seed=9)
    assert e.range_gap == pytest.approx(e.rogers_satchell / e.close_to_close - 1.0)
    assert e.range_gap < 0  # coarse sampling understates


def test_get_rejects_an_unknown_method():
    with pytest.raises(ValueError, match="unknown method"):
        _est(days=200, steps=50).get("nonsense")


def test_to_dict_carries_every_method():
    d = _est(days=200, steps=50).to_dict()
    for method in METHODS:
        assert method in d


@pytest.mark.parametrize("n", [0, 5, 19])
def test_too_few_bars_returns_none(n):
    bars = simulate_ohlc(0.0, SIGMA, max(n, 1), steps_per_day=50, seed=1).head(n)
    assert estimate_all(bars) is None


def test_missing_columns_return_none():
    bars = simulate_ohlc(0.0, SIGMA, 100, steps_per_day=50, seed=1).drop(columns=["high"])
    assert estimate_all(bars) is None


def test_close_to_close_matches_metrics():
    """It must be the same number metrics.compute_drag would produce."""
    from krxdrag.metrics import compute_drag

    bars = simulate_ohlc(0.05, SIGMA, 600, steps_per_day=200, seed=12)
    c = bars["close"].to_numpy()
    assert close_to_close(c) == pytest.approx(compute_drag(c).sigma_sq, rel=1e-12)


def test_yang_zhang_needs_a_few_bars():
    assert np.isnan(yang_zhang(*[np.array([1.0, 1.0])] * 4))


# --- the liquidity-bias report ----------------------------------------------------

def test_liquidity_report_exposes_a_gap_that_widens_as_trading_thins():
    """The bias is correlated with liquidity; the report must show it.

    Thin names are simulated with a coarse intraday grid, liquid ones with a
    fine grid, so the range estimate should fall further below close-to-close
    as turnover drops -- exactly the artefact that would otherwise be misread
    as thin names being genuinely calmer.
    """
    rows = []
    for i, steps in enumerate([20, 60, 200, 800, 3000]):
        for k in range(6):
            e = estimate_all(
                simulate_ohlc(0.0, SIGMA, 900, steps_per_day=steps, seed=100 * i + k)
            )
            rows.append(
                {
                    "median_turnover": float(steps) * 1e6,
                    "sigma_sq": e.close_to_close,
                    "rogers_satchell": e.rogers_satchell,
                }
            )

    table = liquidity_bias_report(pd.DataFrame(rows), method="rogers_satchell", n_buckets=5)
    assert len(table) == 5
    assert table["median_gap"].is_monotonic_increasing   # thinnest bucket worst
    assert table["median_gap"].iloc[0] < -0.15           # and materially negative
    assert table["median_gap"].iloc[-1] > -0.10          # liquid bucket nearly clean


def test_liquidity_report_needs_the_columns_it_reads():
    assert liquidity_bias_report(pd.DataFrame()).empty
    assert liquidity_bias_report(pd.DataFrame({"sigma_sq": [1.0]})).empty


def test_liquidity_report_accepts_both_column_conventions():
    """screen() emits sigma_sq_<method>; a caller may pass the bare column."""
    base = pd.DataFrame(
        {
            "median_turnover": np.logspace(8, 11, 40),
            "sigma_sq": np.full(40, 0.16),
        }
    )
    prefixed = base.assign(sigma_sq_rogers_satchell=0.16 * 0.8)
    bare = base.assign(rogers_satchell=0.16 * 0.8)

    for frame in (prefixed, bare):
        table = liquidity_bias_report(frame, method="rogers_satchell", n_buckets=4)
        assert len(table) == 4
        assert table["median_gap"].iloc[0] == pytest.approx(-0.2, abs=1e-9)
