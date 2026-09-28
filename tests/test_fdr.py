"""Multiple testing across the cross-section.

Each diagnostic is a hypothesis test run once per name. At the 5% level across
1,100 names a test rejects ~55 of them by chance alone, so an uncorrected count
can be pure noise. These tests check the Benjamini-Hochberg procedure does what
it claims -- both that it controls the false-discovery rate and that it still
finds real effects.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from krxdrag.diagnostics import (
    DIAGNOSTIC_TESTS,
    FDR_Q,
    benjamini_hochberg,
    bh_adjusted,
    fdr_report,
)

M = 1106  # the real cross-section size, so the numbers mean something


def _mixture(n_true: int, m: int = M, shift: float = 3.0, seed: int = 0):
    """m one-sided p-values, the first n_true of them genuinely non-null."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=m)
    z[:n_true] += shift
    return stats.norm.sf(z), np.arange(m) < n_true


# --- error control ------------------------------------------------------------

def test_uncorrected_testing_rejects_about_five_percent_of_pure_noise():
    """The problem being solved: ~55 false positives out of 1,106."""
    rng = np.random.default_rng(1)
    counts = [(rng.uniform(size=M) < 0.05).sum() for _ in range(40)]
    assert 0.04 * M < np.mean(counts) < 0.06 * M


def test_bh_almost_never_rejects_under_the_global_null():
    """With every null true, BH must reject nothing in at least ~95% of runs."""
    rng = np.random.default_rng(2)
    any_rejection = sum(
        bool(benjamini_hochberg(rng.uniform(size=M)).any()) for _ in range(400)
    )
    assert any_rejection / 400 <= 0.09          # nominal 0.05, loose enough not to flake


def test_realised_false_discovery_rate_stays_under_q():
    """Among what BH rejects, the false share must sit at or below q."""
    rates = []
    for seed in range(60):
        p, truth = _mixture(200, seed=seed)
        rejected = benjamini_hochberg(p, q=FDR_Q)
        if rejected.sum():
            rates.append(float((rejected & ~truth).sum() / rejected.sum()))
    assert np.mean(rates) <= FDR_Q + 0.01


def test_the_procedure_still_finds_real_effects():
    """Controlling errors is worthless if it rejects nothing true."""
    powers = []
    for seed in range(30):
        p, truth = _mixture(200, seed=seed)
        rejected = benjamini_hochberg(p, q=FDR_Q)
        powers.append(float((rejected & truth).sum() / truth.sum()))
    assert np.mean(powers) > 0.5


# --- ordering between procedures -----------------------------------------------

def test_bh_is_stricter_than_no_correction_and_looser_than_bonferroni():
    p, _ = _mixture(200, seed=7)
    raw = p < FDR_Q
    bh = benjamini_hochberg(p, q=FDR_Q)
    bonferroni = p < FDR_Q / p.size

    assert not (bh & ~raw).any()             # BH never rejects what raw would not
    assert not (bonferroni & ~bh).any()      # Bonferroni never rejects what BH would not
    assert bonferroni.sum() < bh.sum() < raw.sum()


def test_more_signal_means_more_rejections():
    counts = [
        benjamini_hochberg(_mixture(n, seed=3)[0]).sum() for n in (0, 50, 200, 500)
    ]
    assert counts == sorted(counts)


# --- adjusted p-values -----------------------------------------------------------

def test_adjusted_pvalues_are_monotone_and_bounded():
    p, _ = _mixture(150, seed=4)
    q = bh_adjusted(p)

    assert np.all(q[np.isfinite(q)] <= 1.0)
    assert np.all(q[np.isfinite(q)] >= p[np.isfinite(q)])   # adjustment only inflates
    order = np.argsort(p)
    assert np.all(np.diff(q[order]) >= -1e-12)              # monotone in p


def test_adjusted_pvalues_agree_with_the_rejection_set():
    p, _ = _mixture(150, seed=5)
    assert np.array_equal(bh_adjusted(p) <= FDR_Q, benjamini_hochberg(p, q=FDR_Q))


def test_worked_example():
    p = np.array([0.001, 0.008, 0.02, 0.04, 0.2])
    # thresholds are q*k/m = 0.01, 0.02, 0.03, 0.04, 0.05
    assert benjamini_hochberg(p, 0.05).tolist() == [True, True, True, True, False]
    assert bh_adjusted(p) == pytest.approx([0.005, 0.02, 1 / 30, 0.05, 0.2], abs=1e-9)


# --- missing data ------------------------------------------------------------------

def test_non_finite_pvalues_never_reject_and_leave_m_unchanged():
    p = np.array([0.001, np.nan, 0.008, np.inf, 0.02])
    rejected = benjamini_hochberg(p, q=FDR_Q)

    assert not rejected[1] and not rejected[3]
    # m is 3, so thresholds are 0.0167 / 0.0333 / 0.05 -- all three finite ones pass
    assert rejected[[0, 2, 4]].tolist() == [True, True, True]
    assert np.isnan(bh_adjusted(p)[[1, 3]]).all()


@pytest.mark.parametrize("values", [[], [np.nan, np.nan]])
def test_degenerate_input(values):
    p = np.array(values, dtype=float)
    assert not benjamini_hochberg(p).any()
    assert np.isnan(bh_adjusted(p)).all()


# --- the report table ---------------------------------------------------------------

def test_fdr_report_contrasts_raw_against_controlled():
    p_jump, _ = _mixture(200, seed=6)
    frame = pd.DataFrame(
        {
            "jb_pvalue": np.random.default_rng(9).uniform(0, 1e-6, M),   # overwhelming
            "lb_sq_pvalue": np.random.default_rng(10).uniform(size=M),   # pure noise
            "bns_pvalue": p_jump,                                        # a real mixture
        }
    )
    table = fdr_report(frame).set_index("test")
    assert len(table) == 3

    overwhelming = table.loc[DIAGNOSTIC_TESTS["jb_pvalue"]]
    noise = table.loc[DIAGNOSTIC_TESTS["lb_sq_pvalue"]]
    mixture = table.loc[DIAGNOSTIC_TESTS["bns_pvalue"]]

    # a test that rejects everything is barely touched by the correction
    assert overwhelming["fdr_rejections"] == overwhelming["raw_rejections"] == M
    # pure noise: the raw count sits at the chance floor and FDR clears it out
    assert noise["raw_rejections"] == pytest.approx(noise["expected_by_chance"], rel=0.4)
    assert noise["fdr_rejections"] == 0
    # a real mixture keeps most of its rejections
    assert 0 < mixture["fdr_rejections"] < mixture["raw_rejections"]


def test_fdr_report_skips_absent_columns():
    assert fdr_report(pd.DataFrame()).empty
    table = fdr_report(pd.DataFrame({"jb_pvalue": [0.01, 0.02, 0.5]}))
    assert list(table["test"]) == [DIAGNOSTIC_TESTS["jb_pvalue"]]
