import math

import pytest
from statsmodels.stats.power import NormalIndPower, TTestIndPower
from statsmodels.stats.proportion import proportion_effectsize

from pulso_exp.stats import (
    cohens_h,
    mde_proportions,
    sample_size_means,
    sample_size_proportions,
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
