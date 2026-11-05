import math

import numpy as np
import pytest
from scipy import integrate
from scipy import stats as st

from pulso_exp.stats import compare_sequential, msprt_proportions


def _reference_mixture_lr(d: float, v: float, tau: float) -> float:
    """Ratio of marginal likelihoods by numerical integration, no closed form."""
    num, _ = integrate.quad(
        lambda th: st.norm.pdf(d, th, math.sqrt(v)) * st.norm.pdf(th, 0, tau),
        -12 * tau,
        12 * tau,
        points=[0.0, d],
    )
    return num / st.norm.pdf(d, 0, math.sqrt(v))


@pytest.mark.parametrize(
    ("cc", "nc", "ct", "nt", "tau"),
    [
        (520, 1_000, 560, 1_000, 0.02),
        (5_200, 10_000, 5_420, 10_000, 0.01),
        (300, 1_000, 280, 1_200, 0.05),
    ],
)
def test_closed_form_matches_numerical_mixture(cc, nc, ct, nt, tau):
    res = msprt_proportions([cc], [nc], [ct], [nt], tau=tau)
    pooled = (cc + ct) / (nc + nt)
    v = pooled * (1 - pooled) * (1 / nc + 1 / nt)
    expected = _reference_mixture_lr(ct / nt - cc / nc, v, tau)
    assert res.likelihood_ratio[0] == pytest.approx(expected, rel=1e-6)


def test_always_valid_p_value_is_a_running_minimum():
    # Evidence first grows, then the arms converge again: the p-value must not bounce back.
    res = msprt_proportions(
        [100, 200, 300, 400], [200, 400, 600, 800], [130, 250, 330, 410], [200, 400, 600, 800], 0.05
    )
    assert np.all(np.diff(res.p_value) <= 0)
    assert res.p_value[-1] < 1 / res.likelihood_ratio[-1]


def test_degenerate_looks_carry_no_evidence():
    res = msprt_proportions([0, 0], [10, 20], [0, 5], [10, 20], tau=0.05)
    assert res.likelihood_ratio[0] == 1.0
    assert res.p_value[0] == 1.0


def test_rejects_bad_inputs():
    with pytest.raises(ValueError):
        msprt_proportions([1], [10], [1], [10], tau=0)
    with pytest.raises(ValueError):
        msprt_proportions([0], [0], [1], [10], tau=0.02)


def test_first_rejection_index():
    res = msprt_proportions(
        [500, 1_000, 1_500], [1_000, 2_000, 3_000], [500, 1_150, 1_800], [1_000, 2_000, 3_000], 0.05
    )
    assert res.first_rejection(alpha=1e-12) is None
    look = res.first_rejection()
    assert look is not None and res.p_value[look - 1] < 0.05
    assert look == 1 or res.p_value[look - 2] >= 0.05


def test_msprt_keeps_alpha_under_continuous_peeking():
    # A/A, 28 daily looks: naive peeking inflates the error rate, mSPRT does not.
    res = compare_sequential(p=0.52, lift=0.0, n_per_arm=16_800, n_looks=28, n_sims=10_000, seed=7)
    assert res.reject_fixed == pytest.approx(0.05, abs=0.01)
    assert res.reject_peeking > 0.2
    # Ville's inequality bounds the probability of *ever* crossing 1/alpha by alpha.
    assert res.reject_msprt <= 0.05


def test_msprt_trades_power_at_the_mde_for_early_stopping_on_large_effects():
    at_mde = compare_sequential(p=0.52, lift=0.03, n_per_arm=16_800, n_looks=28, n_sims=4_000)
    big = compare_sequential(p=0.52, lift=0.045, n_per_arm=16_800, n_looks=28, n_sims=4_000)
    assert at_mde.reject_fixed == pytest.approx(0.8, abs=0.03)
    assert at_mde.reject_msprt < at_mde.reject_fixed
    assert big.reject_msprt > 0.85
    assert big.msprt_mean_stop_fraction < 0.6
