import numpy as np
import pandas as pd
import pytest

from pulso_exp.simulate import FUNNEL_STEPS, SimConfig, assign_variant, simulate, uplift_multiplier

SMALL = SimConfig(n_merchants=6_000, n_bot_merchants=3, n_pulsers=300)


@pytest.fixture(scope="module")
def data():
    return simulate(SMALL)


def test_same_seed_is_reproducible(data):
    again = simulate(SMALL)
    pd.testing.assert_frame_equal(data.sessions, again.sessions)
    pd.testing.assert_frame_equal(data.orders, again.orders)


def test_different_seed_changes_data(data):
    other = simulate(SimConfig(seed=7, n_merchants=6_000, n_bot_merchants=3, n_pulsers=300))
    assert not data.merchants["merchant_id"].equals(other.merchants["merchant_id"])


def test_assignment_is_deterministic_and_balanced(data):
    ids = data.merchants["merchant_id"]
    assert (assign_variant(ids) == data.merchants["variant"]).all()
    share = (data.merchants["variant"] == "treatment").mean()
    assert share == pytest.approx(0.5, abs=0.02)


def test_schema_and_domain_values(data):
    assert set(data.sessions["funnel_step"]) <= set(FUNNEL_STEPS)
    assert set(data.orders["status"]) <= {"DELIVERED", "CANCELLED", "FAILED"}
    assert (data.orders["currency"] == "MXN").all()
    assert data.orders["amount_cents"].dtype == np.int64
    delivered = data.orders["status"] == "DELIVERED"
    assert data.orders.loc[delivered, "delivery_minutes"].notna().all()
    assert data.orders.loc[~delivered, "delivery_minutes"].isna().all()


def test_every_paid_session_has_exactly_one_order(data):
    paid = data.sessions.loc[data.sessions["funnel_step"] == "paid", "session_id"].unique()
    assert len(paid) == len(data.orders)
    assert set(paid) == set(data.orders["session_id"])


def test_planted_data_quality_issues_exist(data):
    assert data.sessions["session_id"].duplicated().any()
    assert data.sessions["device"].isna().any()
    assert data.sessions["page_load_ms"].isna().any()
    assert (data.orders["delivery_minutes"] > 240).any()


def test_events_reconstruct_the_funnel(data):
    counts = data.events["event_name"].value_counts()
    sessions = data.sessions.drop_duplicates("session_id")
    assert counts["checkout.viewed"] == len(sessions)
    assert counts["payment.authorized"] == counts["order.created"] == len(data.orders)


def test_novelty_decays_to_true_uplift():
    cfg = SimConfig()
    mult = uplift_multiplier(np.array([0, 4, 60]), cfg)
    assert mult[0] == pytest.approx(1 + cfg.true_uplift + cfg.novelty_uplift)
    assert mult[1] == pytest.approx(1 + cfg.true_uplift + cfg.novelty_uplift / 2)
    assert mult[2] == pytest.approx(1 + cfg.true_uplift, abs=1e-4)


def test_treatment_converts_better_in_experiment_only():
    data = simulate(SimConfig(n_merchants=30_000, n_bot_merchants=0, n_pulsers=300))
    s = data.sessions.drop_duplicates("session_id")
    rate = (s["funnel_step"] == "paid").groupby([s["period"], s["variant"]]).mean()
    assert rate["experiment", "treatment"] > rate["experiment", "control"]
    assert rate["pre", "treatment"] == pytest.approx(rate["pre", "control"], abs=0.01)
