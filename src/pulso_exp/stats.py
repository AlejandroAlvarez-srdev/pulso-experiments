"""Reusable statistics for A/B tests.

Small, dependency-light functions (numpy + scipy) that the notebooks call and
the test-suite checks against statsmodels / closed-form answers.
"""

from __future__ import annotations

import math

from scipy import stats as st


def _z(alpha: float, two_sided: bool = True) -> float:
    return float(st.norm.ppf(1 - alpha / 2 if two_sided else 1 - alpha))


# --------------------------------------------------------------------------- power


def cohens_h(p1: float, p2: float) -> float:
    """Arcsine-transformed effect size between two proportions."""
    return 2 * math.asin(math.sqrt(p2)) - 2 * math.asin(math.sqrt(p1))


def sample_size_proportions(
    p_baseline: float,
    mde_relative: float,
    alpha: float = 0.05,
    power: float = 0.8,
    ratio: float = 1.0,
    two_sided: bool = True,
) -> int:
    """Units needed in the control arm (treatment gets ``ratio`` times as many).

    Uses the arcsine (Cohen's h) normal approximation, the same parametrisation
    as ``statsmodels.stats.power.NormalIndPower``.
    """
    if not 0 < p_baseline < 1:
        raise ValueError("p_baseline must be in (0, 1)")
    p_target = p_baseline * (1 + mde_relative)
    if not 0 < p_target < 1:
        raise ValueError("baseline * (1 + mde) must be in (0, 1)")
    h = abs(cohens_h(p_baseline, p_target))
    n = ((_z(alpha, two_sided) + st.norm.ppf(power)) / h) ** 2 * (1 + 1 / ratio)
    return math.ceil(n)


def mde_proportions(
    p_baseline: float, n_per_arm: int, alpha: float = 0.05, power: float = 0.8
) -> float:
    """Smallest relative uplift detectable with ``n_per_arm`` units per arm (two-sided)."""
    h = (_z(alpha) + st.norm.ppf(power)) * math.sqrt(2 / n_per_arm)
    p_target = math.sin(math.asin(math.sqrt(p_baseline)) + h / 2) ** 2
    return p_target / p_baseline - 1


def sample_size_means(
    sd: float, mde_absolute: float, alpha: float = 0.05, power: float = 0.8
) -> int:
    """Units per arm to detect an absolute difference in means (equal variances)."""
    n = 2 * ((_z(alpha) + st.norm.ppf(power)) * sd / mde_absolute) ** 2
    return math.ceil(n)
