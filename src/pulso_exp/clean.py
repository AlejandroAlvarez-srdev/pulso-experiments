"""Data quality rules for the simulated Pulso tables.

Each rule is a small, testable function. ``clean_sessions`` and ``clean_orders``
chain them and return a :class:`CleaningReport` so every decision taken in the
EDA notebook is explicit and countable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from pulso_exp.simulate import CATEGORIES

BOT_MIN_SESSIONS = 100
TYPO_FACTOR = 50
STUCK_DELIVERY_MINUTES = 240


@dataclass
class CleaningReport:
    rows: list[dict] = field(default_factory=list)

    def add(self, table: str, rule: str, affected: int, action: str) -> None:
        self.rows.append(
            {"table": table, "rule": rule, "rows_affected": affected, "action": action}
        )

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows, columns=["table", "rule", "rows_affected", "action"])


def find_bot_merchants(sessions: pd.DataFrame, min_sessions: int = BOT_MIN_SESSIONS) -> set[str]:
    """Merchants with an implausible session count in any period and not a single paid order."""
    per_period = sessions.groupby(["merchant_id", "period"]).agg(
        n=("session_id", "size"), paid=("funnel_step", lambda s: (s == "paid").sum())
    )
    flagged = per_period[(per_period["n"] >= min_sessions) & (per_period["paid"] == 0)]
    return set(flagged.index.get_level_values("merchant_id"))


def fix_amount_typos(
    orders: pd.DataFrame, factor: int = TYPO_FACTOR
) -> tuple[pd.Series, pd.Series]:
    """Detect amounts entered x100 (cents typed twice) and scale them back.

    An order is suspect when it exceeds ``factor`` times its category's median
    basket. Returns the corrected amounts and the boolean mask of fixed rows.
    """
    median_cents = orders["category"].map({k: v[1] * 100 for k, v in CATEGORIES.items()})
    suspect = orders["amount_cents"] > factor * median_cents
    fixed = orders["amount_cents"].where(~suspect, orders["amount_cents"] // 100)
    return fixed, suspect


def clean_sessions(sessions: pd.DataFrame, report: CleaningReport | None = None) -> pd.DataFrame:
    report = report if report is not None else CleaningReport()
    out = sessions.drop_duplicates("session_id")
    report.add("sessions", "exact duplicate session_id", len(sessions) - len(out), "dropped")

    bots = find_bot_merchants(out)
    is_bot = out["merchant_id"].isin(bots)
    report.add(
        "sessions",
        f"bot merchants (>= {BOT_MIN_SESSIONS} sessions/period, 0 paid): {len(bots)}",
        int(is_bot.sum()),
        "dropped (merchant excluded from every analysis)",
    )
    out = out[~is_bot]

    missing_device = out["device"].isna()
    report.add("sessions", "missing device", int(missing_device.sum()), "kept as 'unknown'")
    out = out.assign(device=out["device"].fillna("unknown"))

    report.add(
        "sessions",
        "missing page_load_ms",
        int(out["page_load_ms"].isna().sum()),
        "kept as NaN (excluded from latency guardrail only)",
    )
    report.add(
        "sessions",
        "page_load_ms >= 30 s (client timeout)",
        int((out["page_load_ms"] >= 30_000).sum()),
        "kept: real user pain, the p95 guardrail should see it",
    )
    return out.reset_index(drop=True)


def clean_orders(
    orders: pd.DataFrame,
    bot_merchants: set[str] | None = None,
    report: CleaningReport | None = None,
) -> pd.DataFrame:
    report = report if report is not None else CleaningReport()
    out = orders.drop_duplicates("order_id")
    if bot_merchants:
        out = out[~out["merchant_id"].isin(bot_merchants)]

    fixed, suspect = fix_amount_typos(out)
    report.add(
        "orders",
        f"amount_cents > {TYPO_FACTOR}x category median",
        int(suspect.sum()),
        "divided by 100 (x100 entry typo)",
    )
    out = out.assign(amount_cents=fixed, amount_fixed=suspect)

    stuck = out["delivery_minutes"] > STUCK_DELIVERY_MINUTES
    report.add(
        "orders",
        f"delivery_minutes > {STUCK_DELIVERY_MINUTES}",
        int(stuck.sum()),
        "flagged 'stuck'; excluded from ETA model training, kept for ops metrics",
    )
    out = out.assign(stuck=stuck)

    report.add(
        "orders",
        "missing pulser_rating on assigned orders",
        int((out["pulser_rating"].isna() & out["pulser_id"].notna()).sum()),
        "kept as NaN (new pulsers); the model handles missing natively",
    )
    report.add(
        "orders",
        "delivery_minutes missing (not DELIVERED)",
        int(out["delivery_minutes"].isna().sum()),
        "expected: cancelled/failed orders have no delivery time",
    )
    return out.reset_index(drop=True)
