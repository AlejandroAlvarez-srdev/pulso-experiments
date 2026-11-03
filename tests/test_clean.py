import pandas as pd
import pytest

from pulso_exp.clean import CleaningReport, clean_orders, clean_sessions, find_bot_merchants
from pulso_exp.simulate import SimConfig, simulate


@pytest.fixture(scope="module")
def data():
    return simulate(SimConfig(n_merchants=5_000, n_bot_merchants=4, n_pulsers=300))


def test_clean_sessions_dedupes_and_drops_bots(data):
    report = CleaningReport()
    out = clean_sessions(data.sessions, report)
    assert not out["session_id"].duplicated().any()
    bots = set(data.merchants["merchant_id"].iloc[-4:])
    assert find_bot_merchants(data.sessions) == bots
    assert not out["merchant_id"].isin(bots).any()
    assert out["device"].notna().all()
    assert set(report.to_frame()["table"]) == {"sessions"}


def test_bot_rule_spares_heavy_but_real_merchants():
    sessions = pd.DataFrame(
        {
            "merchant_id": ["real"] * 150 + ["bot"] * 150,
            "session_id": range(300),
            "period": "experiment",
            "funnel_step": ["paid"] + ["checkout_view"] * 149 + ["checkout_view"] * 150,
        }
    )
    assert find_bot_merchants(sessions) == {"bot"}


def test_clean_orders_fixes_x100_typos_and_flags_stuck(data):
    report = CleaningReport()
    out = clean_orders(data.orders, report=report)
    assert out["amount_fixed"].sum() > 0
    fixed = out[out["amount_fixed"]]
    original = data.orders.set_index("order_id").loc[fixed["order_id"], "amount_cents"]
    assert (fixed["amount_cents"].to_numpy() * 100 == original.to_numpy()).all()
    assert out["stuck"].sum() == (data.orders["delivery_minutes"] > 240).sum()
    assert len(report.to_frame()) == 4
