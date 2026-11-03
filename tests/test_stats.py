import math

import numpy as np
import pytest
from scipy import stats as st
from statsmodels.stats.power import NormalIndPower, TTestIndPower
from statsmodels.stats.proportion import proportion_effectsize, proportions_ztest

from pulso_exp.stats import (
    cohens_h,
    cuped,
    mde_proportions,
    sample_size_means,
    sample_size_proportions,
    two_proportion_ztest,
    welch_ttest,
)

# ----------------------------------------------------------------- power


@pytest.mark.parametrize(
    ("p", "mde", "alpha", "power", "ratio"),
    [
        (0.52, 0.03, 0.05, 0.8, 1.0),
        (0.10, 0.05, 0.05, 0.8, 1.0),
        (0.30, 0.10, 0.01, 0.9, 1.0),
        (0.05, 0.20, 0.05, 0.8, 2.0),
    ],
)
def test_sample_size_matches_statsmodels(p, mde, alpha, power, ratio):
    h = proportion_effectsize(p * (1 + mde), p)
    expected = NormalIndPower().solve_power(effect_size=h, alpha=alpha, power=power, ratio=ratio)
    ours = sample_size_proportions(p, mde, alpha=alpha, power=power, ratio=ratio)
    # statsmodels also counts the (negligible) opposite tail, so allow 0.5 %.
    assert ours == pytest.approx(expected, rel=5e-3)
    assert ours >= math.floor(expected)


def test_cohens_h_matches_statsmodels():
    assert cohens_h(0.4, 0.45) == pytest.approx(proportion_effectsize(0.45, 0.4))


def test_mde_inverts_sample_size():
    n = sample_size_proportions(0.52, 0.03)
    assert mde_proportions(0.52, n) == pytest.approx(0.03, rel=1e-3)


def test_sample_size_means_close_to_ttest_power():
    sd, delta = 120.0, 10.0
    expected = TTestIndPower().solve_power(effect_size=delta / sd, alpha=0.05, power=0.8)
    assert sample_size_means(sd, delta) == pytest.approx(expected, rel=0.01)


def test_sample_size_rejects_impossible_target():
    with pytest.raises(ValueError):
        sample_size_proportions(0.9, 0.2)


# ----------------------------------------------------------------- frequentist tests


def test_two_proportion_ztest_matches_statsmodels():
    res = two_proportion_ztest(5_200, 10_000, 5_400, 10_000)
    z, p = proportions_ztest([5_400, 5_200], [10_000, 10_000])
    assert res.statistic == pytest.approx(z)
    assert res.p_value == pytest.approx(p)
    assert res.diff == pytest.approx(0.02)
    assert res.ci_low < res.diff < res.ci_high


def test_welch_matches_scipy():
    rng = np.random.default_rng(1)
    a = rng.normal(10, 2, 500)
    b = rng.normal(10.4, 4, 800)
    res = welch_ttest(a, b)
    ref = st.ttest_ind(b, a, equal_var=False)
    assert res.statistic == pytest.approx(ref.statistic)
    assert res.p_value == pytest.approx(ref.pvalue)
    ci = ref.confidence_interval()
    assert (res.ci_low, res.ci_high) == pytest.approx((ci.low, ci.high))


# ----------------------------------------------------------------- CUPED


def _correlated(n=20_000, rho=0.7, seed=3):
    rng = np.random.default_rng(seed)
    x = rng.normal(100, 20, n)
    y = 50 + rho * (x - 100) + rng.normal(0, 20 * math.sqrt(1 - rho**2), n)
    return y, x


def test_cuped_reduces_variance_by_rho_squared():
    y, x = _correlated(rho=0.7)
    res = cuped(y, x)
    assert res.adjusted.var() < y.var()
    assert res.variance_reduction == pytest.approx(np.corrcoef(y, x)[0, 1] ** 2, rel=1e-6)
    assert res.variance_reduction == pytest.approx(0.49, abs=0.03)


def test_cuped_preserves_mean_and_treatment_effect():
    y, x = _correlated()
    arm = np.random.default_rng(9).random(y.size) < 0.5
    y = y + np.where(arm, 2.0, 0.0)
    res = cuped(y, x)
    assert res.adjusted.mean() == pytest.approx(y.mean())
    raw = welch_ttest(y[~arm], y[arm])
    adj = welch_ttest(res.adjusted[~arm], res.adjusted[arm])
    assert adj.diff == pytest.approx(2.0, abs=0.3)
    assert (adj.ci_high - adj.ci_low) < (raw.ci_high - raw.ci_low)


def test_cuped_with_useless_covariate_does_nothing():
    rng = np.random.default_rng(5)
    y = rng.normal(size=10_000)
    res = cuped(y, rng.normal(size=10_000))
    assert abs(res.variance_reduction) < 0.01
    assert cuped(y, np.ones_like(y)).theta == 0.0
