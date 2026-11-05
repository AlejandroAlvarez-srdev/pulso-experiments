"""Property-based checks: invariants that must hold for any valid input, not just examples."""

import numpy as np
from hypothesis import assume, given, settings
from hypothesis import strategies as stn

from pulso_exp.stats import msprt_proportions, srm_check, two_proportion_ztest

counts = stn.integers(min_value=20, max_value=200_000)


@stn.composite
def arm(draw):
    n = draw(counts)
    conv = draw(stn.integers(min_value=1, max_value=n - 1))
    return conv, n


@given(arm(), arm())
def test_ztest_is_antisymmetric_in_the_arms(a, b):
    fwd = two_proportion_ztest(*a, *b)
    rev = two_proportion_ztest(*b, *a)
    assert np.isclose(fwd.statistic, -rev.statistic)
    assert np.isclose(fwd.p_value, rev.p_value)
    assert np.isclose(fwd.diff, -rev.diff)


@given(arm(), arm(), stn.floats(min_value=0.001, max_value=0.2))
def test_ztest_ci_brackets_estimate_and_p_value_is_a_probability(a, b, alpha):
    res = two_proportion_ztest(*a, *b, alpha=alpha)
    assert res.ci_low <= res.diff <= res.ci_high
    assert 0.0 <= res.p_value <= 1.0


@given(stn.lists(stn.integers(min_value=1, max_value=10**6), min_size=2, max_size=6), stn.randoms())
def test_srm_does_not_depend_on_arm_order(observed, rnd):
    shuffled = observed[:]
    rnd.shuffle(shuffled)
    assert np.isclose(srm_check(observed).p_value, srm_check(shuffled).p_value)


@settings(max_examples=200)
@given(
    stn.lists(stn.tuples(arm(), arm()), min_size=1, max_size=15),
    stn.floats(min_value=1e-3, max_value=0.5),
)
def test_msprt_p_values_are_monotone_valid_and_symmetric(looks, tau):
    # Build cumulative counts from per-batch increments.
    cc = np.cumsum([c for (c, _), _ in looks])
    nc = np.cumsum([n for (_, n), _ in looks])
    ct = np.cumsum([c for _, (c, _) in looks])
    nt = np.cumsum([n for _, (_, n) in looks])
    assume(np.all(cc < nc) and np.all(ct < nt))
    fwd = msprt_proportions(cc, nc, ct, nt, tau)
    rev = msprt_proportions(ct, nt, cc, nc, tau)
    # p can underflow to exactly 0 when the evidence is overwhelming (Lambda = inf).
    assert np.all((fwd.p_value >= 0) & (fwd.p_value <= 1))
    assert np.all(np.diff(fwd.p_value) <= 0)
    assert np.all(fwd.likelihood_ratio > 0)
    # The mixture is centred on zero, so swapping arms changes nothing.
    assert np.allclose(fwd.likelihood_ratio, rev.likelihood_ratio)
