"""Reusable statistics for A/B tests.

Small, dependency-light functions (numpy + scipy) that the notebooks call and
the test-suite checks against statsmodels / closed-form answers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
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


# --------------------------------------------------------------------------- frequentist tests


@dataclass(frozen=True)
class EffectEstimate:
    control: float
    treatment: float
    diff: float
    relative: float
    ci_low: float
    ci_high: float
    statistic: float
    p_value: float

    def significant(self, alpha: float = 0.05) -> bool:
        return self.p_value < alpha


def two_proportion_ztest(
    conv_c: int, n_c: int, conv_t: int, n_t: int, alpha: float = 0.05
) -> EffectEstimate:
    """Pooled two-proportion z-test with an unpooled Wald CI for the difference."""
    p_c, p_t = conv_c / n_c, conv_t / n_t
    pooled = (conv_c + conv_t) / (n_c + n_t)
    se_pooled = math.sqrt(pooled * (1 - pooled) * (1 / n_c + 1 / n_t))
    z = (p_t - p_c) / se_pooled
    p_value = 2 * st.norm.sf(abs(z))
    se = math.sqrt(p_c * (1 - p_c) / n_c + p_t * (1 - p_t) / n_t)
    half = _z(alpha) * se
    diff = p_t - p_c
    return EffectEstimate(p_c, p_t, diff, diff / p_c, diff - half, diff + half, z, p_value)


def welch_ttest(control: np.ndarray, treatment: np.ndarray, alpha: float = 0.05) -> EffectEstimate:
    """Welch's t-test (unequal variances) with a Welch-Satterthwaite CI."""
    a = np.asarray(control, dtype=float)
    b = np.asarray(treatment, dtype=float)
    va, vb = a.var(ddof=1) / a.size, b.var(ddof=1) / b.size
    se = math.sqrt(va + vb)
    df = (va + vb) ** 2 / (va**2 / (a.size - 1) + vb**2 / (b.size - 1))
    diff = b.mean() - a.mean()
    t = diff / se
    p_value = 2 * st.t.sf(abs(t), df)
    half = st.t.ppf(1 - alpha / 2, df) * se
    return EffectEstimate(
        a.mean(), b.mean(), diff, diff / a.mean(), diff - half, diff + half, t, p_value
    )


# --------------------------------------------------------------------------- CUPED


@dataclass(frozen=True)
class CupedResult:
    adjusted: np.ndarray
    theta: float
    variance_reduction: float


def cuped(y: np.ndarray, x: np.ndarray) -> CupedResult:
    """CUPED: y_adj = y - theta * (x - mean(x)), theta = cov(y, x) / var(x).

    ``x`` must be a pre-experiment covariate (unaffected by treatment) and
    ``theta`` is estimated on both arms pooled, so the adjusted metric keeps the
    same expectation in each arm while its variance shrinks by corr(y, x)^2.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    var_x = x.var(ddof=1)
    theta = 0.0 if var_x == 0 else float(np.cov(y, x, ddof=1)[0, 1] / var_x)
    adjusted = y - theta * (x - x.mean())
    reduction = 1 - adjusted.var(ddof=1) / y.var(ddof=1)
    return CupedResult(adjusted, theta, float(reduction))


# --------------------------------------------------------------------------- SRM


@dataclass(frozen=True)
class SrmResult:
    observed: tuple[int, ...]
    expected: tuple[float, ...]
    chi2: float
    p_value: float
    mismatch: bool


def srm_check(
    observed: list[int] | tuple[int, ...],
    expected_ratios: list[float] | None = None,
    threshold: float = 0.001,
) -> SrmResult:
    """Chi-square goodness-of-fit for sample ratio mismatch.

    The conventional threshold is strict (p < 0.001): SRM checks run on every
    experiment, and a real SRM usually produces p-values many orders smaller.
    """
    obs = np.asarray(observed, dtype=float)
    ratios = (
        np.full(obs.size, 1 / obs.size) if expected_ratios is None else np.asarray(expected_ratios)
    )
    expected = obs.sum() * ratios / ratios.sum()
    chi2, p_value = st.chisquare(obs, expected)
    return SrmResult(
        tuple(int(o) for o in obs),
        tuple(float(e) for e in expected),
        float(chi2),
        float(p_value),
        bool(p_value < threshold),
    )


# --------------------------------------------------------------------------- Bayesian


@dataclass(frozen=True)
class BayesResult:
    alpha_c: float
    beta_c: float
    alpha_t: float
    beta_t: float
    prob_treatment_better: float
    expected_loss_treatment: float
    uplift_ci: tuple[float, float]

    @property
    def mean_c(self) -> float:
        return self.alpha_c / (self.alpha_c + self.beta_c)

    @property
    def mean_t(self) -> float:
        return self.alpha_t / (self.alpha_t + self.beta_t)


def beta_binomial_ab(
    conv_c: int,
    n_c: int,
    conv_t: int,
    n_t: int,
    prior: tuple[float, float] = (1.0, 1.0),
    draws: int = 200_000,
    ci: float = 0.95,
    seed: int = 0,
) -> BayesResult:
    """Conjugate Beta-Binomial posterior for each arm, summarised by Monte Carlo.

    ``expected_loss_treatment`` is E[max(p_c - p_t, 0)]: the conversion we expect
    to give up if we ship treatment and it is actually worse.
    """
    a0, b0 = prior
    ac, bc = a0 + conv_c, b0 + n_c - conv_c
    at, bt = a0 + conv_t, b0 + n_t - conv_t
    rng = np.random.default_rng(seed)
    pc = rng.beta(ac, bc, size=draws)
    pt = rng.beta(at, bt, size=draws)
    uplift = pt / pc - 1
    tail = (1 - ci) / 2
    lo, hi = np.quantile(uplift, [tail, 1 - tail])
    return BayesResult(
        ac,
        bc,
        at,
        bt,
        prob_treatment_better=float((pt > pc).mean()),
        expected_loss_treatment=float(np.maximum(pc - pt, 0).mean()),
        uplift_ci=(float(lo), float(hi)),
    )


# --------------------------------------------------------------------------- peeking


@dataclass(frozen=True)
class PeekingResult:
    n_sims: int
    n_looks: int
    false_positive_fixed: float
    false_positive_peeking: float
    first_rejection_look: np.ndarray


def simulate_peeking(
    p: float = 0.5,
    n_per_arm: int = 10_000,
    n_looks: int = 20,
    n_sims: int = 2_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> PeekingResult:
    """A/A simulation: how often does 'stop at the first p < alpha' declare a winner?

    Each simulated test accrues ``n_per_arm`` units per arm in ``n_looks`` equal
    batches. Both arms share the same true rate, so every rejection is a false
    positive. ``fixed`` only tests once at the planned horizon.
    """
    rng = np.random.default_rng(seed)
    batch = n_per_arm // n_looks
    conv_c = rng.binomial(batch, p, size=(n_sims, n_looks)).cumsum(axis=1)
    conv_t = rng.binomial(batch, p, size=(n_sims, n_looks)).cumsum(axis=1)
    n = batch * np.arange(1, n_looks + 1)
    pooled = (conv_c + conv_t) / (2 * n)
    se = np.sqrt(pooled * (1 - pooled) * 2 / n)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(se > 0, (conv_t - conv_c) / n / se, 0.0)
    reject = np.abs(z) > _z(alpha)
    any_reject = reject.any(axis=1)
    first = np.where(any_reject, reject.argmax(axis=1) + 1, 0)
    return PeekingResult(
        n_sims, n_looks, float(reject[:, -1].mean()), float(any_reject.mean()), first[any_reject]
    )


# --------------------------------------------------------------------------- sequential (mSPRT)


@dataclass(frozen=True)
class MsprtResult:
    likelihood_ratio: np.ndarray
    p_value: np.ndarray
    diff: np.ndarray

    def first_rejection(self, alpha: float = 0.05) -> int | None:
        """1-based index of the first look where the always-valid p-value drops below alpha."""
        hits = np.flatnonzero(self.p_value < alpha)
        return int(hits[0]) + 1 if hits.size else None


def msprt_proportions(
    conv_c: np.ndarray | list[int],
    n_c: np.ndarray | list[int],
    conv_t: np.ndarray | list[int],
    n_t: np.ndarray | list[int],
    tau: float,
) -> MsprtResult:
    """Mixture SPRT for a difference in proportions, with always-valid p-values.

    Inputs are cumulative counts at each look. Under a normal approximation the
    estimate ``d = p_t - p_c`` has variance ``V`` (pooled, i.e. under H0), and
    mixing the alternative over ``N(0, tau^2)`` gives the closed-form ratio

        Lambda = sqrt(V / (V + tau^2)) * exp(tau^2 d^2 / (2 V (V + tau^2)))

    (Johari, Pekelis & Walsh, "Always valid inference", 2017). ``Lambda`` is a
    nonnegative martingale under H0, so by Ville's inequality
    ``P(sup Lambda >= 1/alpha) <= alpha``: the test can be checked after every
    batch and stopped at any time without inflating the false-positive rate.
    The always-valid p-value is the running minimum of ``1 / Lambda``.

    ``tau`` is the expected size of the absolute difference. Small values make
    the test sensitive to small effects but slow on large ones, and vice versa.
    """
    if tau <= 0:
        raise ValueError("tau must be positive")
    cc, nc = np.asarray(conv_c, dtype=float), np.asarray(n_c, dtype=float)
    ct, nt = np.asarray(conv_t, dtype=float), np.asarray(n_t, dtype=float)
    if np.any(nc <= 0) or np.any(nt <= 0):
        raise ValueError("every look needs at least one unit per arm")
    diff = ct / nt - cc / nc
    pooled = (cc + ct) / (nc + nt)
    v = pooled * (1 - pooled) * (1 / nc + 1 / nt)
    t2 = tau**2
    with np.errstate(divide="ignore", invalid="ignore"):
        log_lr = 0.5 * np.log(v / (v + t2)) + t2 * diff**2 / (2 * v * (v + t2))
    # A degenerate look (all or no conversions so far) carries no evidence.
    log_lr = np.where(v > 0, log_lr, 0.0)
    # Work in log space: overwhelming evidence overflows Lambda to inf, and the
    # p-value then underflows to 0 instead of raising a warning on 1 / inf.
    with np.errstate(over="ignore"):
        lr = np.exp(log_lr)
    p_value = np.minimum.accumulate(np.exp(-np.maximum(log_lr, 0.0)), axis=-1)
    return MsprtResult(lr, p_value, diff)


@dataclass(frozen=True)
class SequentialComparison:
    n_sims: int
    n_looks: int
    lift: float
    reject_fixed: float
    reject_peeking: float
    reject_msprt: float
    msprt_mean_stop_fraction: float


def compare_sequential(
    p: float = 0.3,
    lift: float = 0.0,
    n_per_arm: int = 10_000,
    n_looks: int = 20,
    n_sims: int = 2_000,
    tau: float = 0.02,
    alpha: float = 0.05,
    seed: int = 0,
) -> SequentialComparison:
    """Rejection rate of three stopping rules on the same simulated experiments.

    - ``fixed``: one z-test at the planned horizon.
    - ``peeking``: stop at the first look where the z-test gives p < alpha.
    - ``msprt``: stop at the first look where the always-valid p-value < alpha.

    With ``lift == 0`` every rejection is a false positive; with a positive
    relative ``lift`` on the treatment rate the rejection rate is power.
    ``msprt_mean_stop_fraction`` is the average share of the planned sample the
    mSPRT used, counting runs that never reject as using all of it.
    """
    rng = np.random.default_rng(seed)
    batch = n_per_arm // n_looks
    conv_c = rng.binomial(batch, p, size=(n_sims, n_looks)).cumsum(axis=1)
    conv_t = rng.binomial(batch, p * (1 + lift), size=(n_sims, n_looks)).cumsum(axis=1)
    n = np.broadcast_to(batch * np.arange(1, n_looks + 1), conv_c.shape)

    pooled = (conv_c + conv_t) / (2 * n)
    se = np.sqrt(pooled * (1 - pooled) * 2 / n)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(se > 0, (conv_t - conv_c) / n / se, 0.0)
    z_reject = np.abs(z) > _z(alpha)

    seq = msprt_proportions(conv_c, n, conv_t, n, tau=tau)
    seq_reject = seq.p_value < alpha
    any_seq = seq_reject.any(axis=1)
    stop_look = np.where(any_seq, seq_reject.argmax(axis=1) + 1, n_looks)

    return SequentialComparison(
        n_sims=n_sims,
        n_looks=n_looks,
        lift=lift,
        reject_fixed=float(z_reject[:, -1].mean()),
        reject_peeking=float(z_reject.any(axis=1).mean()),
        reject_msprt=float(any_seq.mean()),
        msprt_mean_stop_fraction=float((stop_look / n_looks).mean()),
    )
