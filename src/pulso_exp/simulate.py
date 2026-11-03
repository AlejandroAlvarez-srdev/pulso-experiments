"""Seeded simulator for Pulso's checkout experiment and delivery data.

Everything produced here is SIMULATED. The generator plants known ground truth
(a treatment effect on the checkout funnel, a novelty bump that decays, bot
traffic, duplicated rows, typos, missing values and stuck deliveries) so the
notebooks can show that the analysis recovers what was planted.

Tables
------
merchants  one row per merchant, with the experiment assignment.
sessions   one row per checkout session (with deliberately duplicated rows).
orders     one row per order created from a paid checkout session.
events     long event log (``checkout.viewed``, ``payment.authorized``,
           ``pulser.assigned``, ``refund.issued`` ...) derived from the above.

Column names are the snake_case form of the shared Pulso entities
(``merchantId`` -> ``merchant_id``, ``amountCents`` -> ``amount_cents``).
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass

import numpy as np
import pandas as pd

EXPERIMENT_NAME = "checkout-one-page"

# city -> (lat, lng, share of merchants, traffic congestion multiplier)
CITIES: dict[str, tuple[float, float, float, float]] = {
    "CDMX": (19.4326, -99.1332, 0.45, 1.25),
    "Guadalajara": (20.6597, -103.3496, 0.20, 1.10),
    "Monterrey": (25.6866, -100.3161, 0.20, 1.05),
    "Puebla": (19.0414, -98.2063, 0.15, 1.00),
}

# category -> (share of merchants, median basket in MXN, median prep minutes)
CATEGORIES: dict[str, tuple[float, float, float]] = {
    "grocery": (0.30, 450.0, 8.0),
    "restaurant": (0.25, 280.0, 12.0),
    "pharmacy": (0.20, 300.0, 5.0),
    "retail": (0.15, 900.0, 6.0),
    "flowers": (0.10, 600.0, 10.0),
}

VEHICLE_SPEED_KMH: dict[str, float] = {"BIKE": 14.0, "MOTO": 24.0, "CAR": 19.0}
DEVICES = ("web", "android", "ios")
FUNNEL_STEPS = ("checkout_view", "details", "payment_submitted", "paid")

# Relative weight of each hour of the day for checkout sessions (sums to 1).
_HOURLY = np.array(
    [1, 0.5, 0.3, 0.2, 0.2, 0.4, 1, 2, 3.5, 4.5, 5, 5.5, 7, 7.5, 6, 5, 5, 5.5, 7, 7.5, 6, 4, 3, 2]
)
HOUR_WEIGHTS = _HOURLY / _HOURLY.sum()


@dataclass(frozen=True)
class SimConfig:
    """Knobs of the simulation. The defaults are what the notebooks use."""

    seed: int = 42
    n_merchants: int = 60_000
    n_bot_merchants: int = 15
    n_pulsers: int = 3_000
    start_date: str = "2026-03-02"
    pre_days: int = 28
    exp_days: int = 28
    # funnel: checkout_view -> details -> payment_submitted -> paid
    view_to_details: float = 0.62
    details_to_payment: float = 0.55
    payment_auth_rate: float = 0.93
    # planted treatment effect, relative lift on details -> payment_submitted
    true_uplift: float = 0.04
    novelty_uplift: float = 0.08
    novelty_half_life_days: float = 4.0
    # data quality issues
    duplicate_session_rate: float = 0.005
    missing_device_rate: float = 0.02
    missing_latency_rate: float = 0.03
    latency_timeout_rate: float = 0.001
    amount_typo_rate: float = 0.002
    stuck_delivery_rate: float = 0.003
    missing_rating_rate: float = 0.04

    @property
    def experiment_start(self) -> pd.Timestamp:
        return pd.Timestamp(self.start_date) + pd.Timedelta(days=self.pre_days)


@dataclass
class Dataset:
    merchants: pd.DataFrame
    sessions: pd.DataFrame
    orders: pd.DataFrame
    events: pd.DataFrame
    config: SimConfig


def assign_variant(
    merchant_ids: pd.Series | list[str], experiment: str = EXPERIMENT_NAME
) -> np.ndarray:
    """Deterministic 50/50 hash bucketing, as a feature-flag service would do it."""
    buckets = np.array(
        [
            int(hashlib.md5(f"{experiment}:{m}".encode()).hexdigest()[:8], 16) % 100
            for m in merchant_ids
        ]
    )
    return np.where(buckets < 50, "control", "treatment")


def uplift_multiplier(day: np.ndarray, config: SimConfig) -> np.ndarray:
    """Planted multiplier on details -> payment for treated sessions on experiment day ``day``."""
    novelty = config.novelty_uplift * 0.5 ** (day / config.novelty_half_life_days)
    return 1.0 + config.true_uplift + novelty


def _uuids(rng: np.random.Generator, n: int) -> np.ndarray:
    raw = rng.bytes(16 * n)
    return np.array(
        [str(uuid.UUID(bytes=raw[i : i + 16], version=4)) for i in range(0, 16 * n, 16)]
    )


def _haversine_km(lat1, lng1, lat2, lng2) -> np.ndarray:
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    a = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lng2 - lng1) / 2) ** 2
    )
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def _merchants(
    rng: np.random.Generator, cfg: SimConfig
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    n = cfg.n_merchants + cfg.n_bot_merchants
    ids = _uuids(rng, n)
    city = rng.choice(list(CITIES), size=n, p=[c[2] for c in CITIES.values()])
    category = rng.choice(list(CATEGORIES), size=n, p=[c[0] for c in CATEGORIES.values()])
    signup = pd.Timestamp(cfg.start_date) - pd.to_timedelta(rng.integers(1, 900, size=n), unit="D")
    merchants = pd.DataFrame(
        {
            "merchant_id": ids,
            "category": category,
            "city": city,
            "signup_date": signup.normalize(),
            "variant": assign_variant(ids),
        }
    )
    # Hidden per-merchant traits that make behaviour persistent across periods
    # (this persistence is what CUPED exploits).
    is_bot = np.zeros(n, dtype=bool)
    is_bot[cfg.n_merchants :] = True
    activity = rng.lognormal(np.log(0.03), 1.0, size=n)  # checkout sessions per day
    activity[is_bot] = rng.uniform(15, 25, size=cfg.n_bot_merchants)
    propensity = np.exp(rng.normal(0.0, 0.35, size=n))
    basket = np.exp(rng.normal(0.0, 0.4, size=n))
    hidden = {"is_bot": is_bot, "activity": activity, "propensity": propensity, "basket": basket}
    return merchants, hidden


def _sessions(
    rng: np.random.Generator, cfg: SimConfig, merchants: pd.DataFrame, hidden: dict[str, np.ndarray]
) -> pd.DataFrame:
    frames = []
    for period, days, offset in (
        ("pre", cfg.pre_days, 0),
        ("experiment", cfg.exp_days, cfg.pre_days),
    ):
        counts = rng.poisson(hidden["activity"] * days)
        idx = np.repeat(np.arange(len(merchants)), counts)
        n = len(idx)
        day = rng.integers(0, days, size=n)
        hour = rng.choice(24, size=n, p=HOUR_WEIGHTS)
        started = (
            pd.Timestamp(cfg.start_date)
            + pd.to_timedelta(day + offset, unit="D")
            + pd.to_timedelta(hour, unit="h")
            + pd.to_timedelta(rng.integers(0, 3600, size=n), unit="s")
        )
        variant = merchants["variant"].to_numpy()[idx]
        treated = (variant == "treatment") & (period == "experiment")
        bot = hidden["is_bot"][idx]

        device = rng.choice(DEVICES, size=n, p=[0.45, 0.35, 0.20]).astype(object)
        device[bot] = "web"

        p_details = np.full(n, cfg.view_to_details)
        p_payment = cfg.details_to_payment * hidden["propensity"][idx]
        p_payment = np.where(treated, p_payment * uplift_multiplier(day, cfg), p_payment)
        p_payment = np.clip(p_payment, 0.0, 0.97)
        p_details[bot] = 0.05
        p_payment[bot] = 0.0
        reached_details = rng.random(n) < p_details
        submitted = reached_details & (rng.random(n) < p_payment)
        paid = submitted & (rng.random(n) < cfg.payment_auth_rate)
        step = np.select([paid, submitted, reached_details], [3, 2, 1], default=0)

        android = device == "android"
        latency = rng.lognormal(np.log(780), 0.35, size=n) * np.where(android, 1.15, 1.0)
        latency = np.where(treated, latency * 1.025, latency)  # the new page is a bit heavier
        latency[rng.random(n) < cfg.latency_timeout_rate] = 30_000.0

        frames.append(
            pd.DataFrame(
                {
                    "session_id": _uuids(rng, n),
                    "merchant_id": merchants["merchant_id"].to_numpy()[idx],
                    "variant": variant,
                    "period": period,
                    "started_at": started,
                    "device": device,
                    "funnel_step": np.array(FUNNEL_STEPS)[step],
                    "payment_failed": submitted & ~paid,
                    "page_load_ms": np.round(latency),
                }
            )
        )
    sessions = pd.concat(frames, ignore_index=True)
    n = len(sessions)
    sessions.loc[rng.random(n) < cfg.missing_device_rate, "device"] = np.nan
    sessions.loc[rng.random(n) < cfg.missing_latency_rate, "page_load_ms"] = np.nan
    dupes = sessions[rng.random(n) < cfg.duplicate_session_rate]
    sessions = pd.concat([sessions, dupes], ignore_index=True)
    return sessions.sort_values("started_at", kind="stable").reset_index(drop=True)


def _orders(
    rng: np.random.Generator,
    cfg: SimConfig,
    merchants: pd.DataFrame,
    sessions: pd.DataFrame,
    hidden: dict[str, np.ndarray],
) -> pd.DataFrame:
    paid = sessions[sessions["funnel_step"] == "paid"].drop_duplicates("session_id")
    m_index = pd.Series(np.arange(len(merchants)), index=merchants["merchant_id"])
    midx = m_index.loc[paid["merchant_id"]].to_numpy()
    n = len(paid)
    city = merchants["city"].to_numpy()[midx]
    category = merchants["category"].to_numpy()[midx]

    created = paid["started_at"].to_numpy() + pd.to_timedelta(
        rng.integers(120, 600, size=n), unit="s"
    )
    created = pd.DatetimeIndex(created)
    hour = created.hour.to_numpy()

    basket_median = np.array([CATEGORIES[c][1] for c in category])
    amount = basket_median * hidden["basket"][midx] * rng.lognormal(0.0, 0.5, size=n)
    amount_cents = np.round(amount * 100).astype(np.int64)
    typo = rng.random(n) < cfg.amount_typo_rate
    amount_cents[typo] *= 100

    center = np.array([CITIES[c][:2] for c in city])
    pickup_lat = center[:, 0] + rng.normal(0, 0.04, size=n)
    pickup_lng = center[:, 1] + rng.normal(0, 0.04, size=n)
    km = np.clip(rng.lognormal(np.log(3.2), 0.6, size=n), 0.3, 25.0)
    bearing = rng.uniform(0, 2 * np.pi, size=n)
    dropoff_lat = pickup_lat + (km * np.cos(bearing)) / 111.0
    dropoff_lng = pickup_lng + (km * np.sin(bearing)) / (111.0 * np.cos(np.radians(pickup_lat)))
    distance_km = _haversine_km(pickup_lat, pickup_lng, dropoff_lat, dropoff_lng)

    vehicle_p = np.where(
        (distance_km < 2)[:, None],
        [0.55, 0.35, 0.10],
        np.where((distance_km < 6)[:, None], [0.25, 0.55, 0.20], [0.05, 0.60, 0.35]),
    )
    u = rng.random(n)[:, None]
    vehicle = np.array(list(VEHICLE_SPEED_KMH))[(u > vehicle_p.cumsum(axis=1)).sum(axis=1)]

    peak = np.isin(hour, [13, 14, 18, 19, 20])
    pulser_ids = _uuids(rng, cfg.n_pulsers)
    pulser_rating_pool = np.clip(5.0 - rng.gamma(2.0, 0.18, size=cfg.n_pulsers), 2.5, 5.0).round(2)
    pulser_pick = rng.integers(0, cfg.n_pulsers, size=n)
    rating = pulser_rating_pool[pulser_pick]
    active_tasks = np.minimum(rng.poisson(np.where(peak, 1.6, 0.7)), 3)

    rain_table = rng.random((len(CITIES), cfg.pre_days + cfg.exp_days)) < 0.25
    day_index = (created.normalize() - pd.Timestamp(cfg.start_date)).days.to_numpy()
    city_index = np.array([list(CITIES).index(c) for c in city])
    raining = rain_table[city_index, np.clip(day_index, 0, rain_table.shape[1] - 1)]

    congestion = np.array([CITIES[c][3] for c in city])
    speed = np.array([VEHICLE_SPEED_KMH[v] for v in vehicle])
    rain_penalty = np.where(raining, np.where(vehicle == "CAR", 1.15, 1.25), 1.0)
    prep = np.array([CATEGORIES[c][2] for c in category]) * rng.lognormal(0, 0.25, size=n)
    wait = 3.0 + 2.5 * active_tasks + np.where(peak, 4.0, 0.0)
    travel = distance_km * 1.3 / speed * 60 * congestion * np.where(peak, 1.35, 1.0) * rain_penalty
    minutes = (prep + wait + travel) * (1 - 0.03 * (rating - 4.5)) * rng.lognormal(0, 0.15, size=n)
    stuck = rng.random(n) < cfg.stuck_delivery_rate
    minutes[stuck] += rng.uniform(180, 400, size=stuck.sum())

    status = np.full(n, "DELIVERED", dtype=object)
    roll = rng.random(n)
    status[roll < 0.03 + 0.02 * (active_tasks >= 3)] = "CANCELLED"
    status[roll > 0.99] = "FAILED"
    delivered = status == "DELIVERED"

    eta_quoted = np.round(10 + 4 * distance_km)
    late = delivered & (minutes > eta_quoted + 30)
    refund = (status == "FAILED") | (late & (rng.random(n) < 0.2))

    rating_obs = rating.astype(float)
    rating_obs[rng.random(n) < cfg.missing_rating_rate] = np.nan
    assigned_at = created + pd.to_timedelta(np.where(status == "CANCELLED", np.nan, wait), unit="m")

    return pd.DataFrame(
        {
            "order_id": _uuids(rng, n),
            "session_id": paid["session_id"].to_numpy(),
            "merchant_id": paid["merchant_id"].to_numpy(),
            "variant": paid["variant"].to_numpy(),
            "period": paid["period"].to_numpy(),
            "status": status,
            "created_at": created,
            "amount_cents": amount_cents,
            "currency": "MXN",
            "items_count": 1 + rng.poisson(2.0, size=n),
            "city": city,
            "category": category,
            "pickup_lat": pickup_lat.round(5),
            "pickup_lng": pickup_lng.round(5),
            "dropoff_lat": dropoff_lat.round(5),
            "dropoff_lng": dropoff_lng.round(5),
            "pulser_id": np.where(status == "CANCELLED", None, pulser_ids[pulser_pick]),
            "vehicle": np.where(status == "CANCELLED", None, vehicle),
            "pulser_rating": np.where(status == "CANCELLED", np.nan, rating_obs),
            "pulser_active_tasks": np.where(status == "CANCELLED", -1, active_tasks),
            "raining": raining,
            "assigned_at": assigned_at,
            "eta_minutes": eta_quoted,
            "delivery_minutes": np.where(delivered, minutes.round(1), np.nan),
            "refund_issued": refund,
        }
    )


def _events(sessions: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    s = sessions.drop_duplicates("session_id")
    step = s["funnel_step"].map({k: i for i, k in enumerate(FUNNEL_STEPS)})
    second = pd.to_timedelta(1, unit="s")
    parts = [
        s.assign(event_name="checkout.viewed", occurred_at=s["started_at"]),
        s[step >= 1].assign(
            event_name="checkout.details_completed", occurred_at=s["started_at"] + 40 * second
        ),
        s[step >= 2].assign(
            event_name="payment.submitted", occurred_at=s["started_at"] + 90 * second
        ),
        s[step == 3].assign(
            event_name="payment.authorized", occurred_at=s["started_at"] + 95 * second
        ),
        s[s["payment_failed"]].assign(
            event_name="payment.failed", occurred_at=s["started_at"] + 95 * second
        ),
    ]
    o = orders
    terminal = {
        "DELIVERED": "order.delivered",
        "CANCELLED": "order.cancelled",
        "FAILED": "assignment.failed",
    }
    delivered_at = o["created_at"] + pd.to_timedelta(o["delivery_minutes"].fillna(30), unit="m")
    parts += [
        o.assign(event_name="order.created", occurred_at=o["created_at"]),
        o[o["assigned_at"].notna()].assign(
            event_name="pulser.assigned", occurred_at=o["assigned_at"]
        ),
        o.assign(event_name=o["status"].map(terminal), occurred_at=delivered_at),
        o[o["refund_issued"]].assign(
            event_name="refund.issued", occurred_at=delivered_at + pd.to_timedelta(2, unit="h")
        ),
    ]
    cols = ["occurred_at", "event_name", "merchant_id", "session_id", "order_id"]
    events = pd.concat([p.reindex(columns=cols) for p in parts], ignore_index=True)
    return events.sort_values(["occurred_at", "event_name"], kind="stable").reset_index(drop=True)


def simulate(config: SimConfig | None = None) -> Dataset:
    """Generate the full simulated dataset. Same config -> identical output."""
    cfg = config or SimConfig()
    rng = np.random.default_rng(cfg.seed)
    merchants, hidden = _merchants(rng, cfg)
    sessions = _sessions(rng, cfg, merchants, hidden)
    orders = _orders(rng, cfg, merchants, sessions, hidden)
    events = _events(sessions, orders)
    return Dataset(merchants=merchants, sessions=sessions, orders=orders, events=events, config=cfg)
