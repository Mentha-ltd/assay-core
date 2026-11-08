from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

import assay_core
from assay_core import Sealer

T0 = datetime(2026, 1, 10, tzinfo=UTC)
T1 = T0 + timedelta(hours=1)


def model():
    rng = np.random.default_rng(0)
    n = 400
    mode = rng.choice(["calm", "wild"], n)
    err = np.where(mode == "calm", rng.normal(0, 1, n), rng.normal(0, 5, n))
    df = pd.DataFrame({"f": 100.0, "a": 100.0 + err, "mode": mode})
    return assay_core.fit(df, "f", "a", condition="mode"), df


def sealer(path=":memory:"):
    return Sealer(
        path,
        agent_id="m1",
        model_version="1",
        target_name="load",
        unit="MW",
        resolution_rule="metered MW",
    )


def test_claim_from_model_then_resolve_and_check():
    m, df = model()
    s = sealer()
    cid = s.claim_from(
        m, 100.0, T0, T1, condition="calm", as_of=T0 - timedelta(hours=1), input_data=df
    )
    s.resolve(cid, 100.4)
    assert s.check().ok
    card = s.scorecard()
    assert card.loc["calm", "cases"] == 1
    assert card.loc["calm", "hit_rate"] == 1.0


def test_wild_claim_is_wider_than_calm_claim():
    m, _ = model()
    s = sealer()
    a = s.claim_from(m, 100.0, T0, T1, condition="calm", as_of=T0)
    b = s.claim_from(m, 100.0, T0, T1, condition="wild", as_of=T0)
    ca, cb = s.ledger.get_claim(a), s.ledger.get_claim(b)
    assert (cb.upper - cb.lower) > 3 * (ca.upper - ca.lower)


def test_unseen_condition_is_sealed_as_borrowed():
    m, _ = model()
    s = sealer()
    cid = s.claim_from(m, 100.0, T0, T1, condition="storm", as_of=T0)
    assert s.ledger.get_claim(cid).context["borrowed_range"] is True


def test_same_target_and_condition_cannot_be_sealed_twice():
    m, _ = model()
    s = sealer()
    s.claim_from(m, 100.0, T0, T1, condition="calm", as_of=T0)
    with pytest.raises(ValueError, match="already exists"):
        s.claim_from(m, 100.0, T0, T1, condition="calm", as_of=T0)


def test_claim_about_the_past_is_refused():
    m, _ = model()
    s = sealer()
    with pytest.raises(ValueError, match="forecasts the past"):
        s.claim_from(m, 100.0, T0, T1, condition="calm", as_of=T1)


def test_cannot_resolve_before_target_ends():
    m, _ = model()
    s = sealer()
    start = datetime.now(UTC) + timedelta(days=1)
    cid = s.claim_from(m, 100.0, start, start + timedelta(hours=1), condition="calm")
    with pytest.raises(ValueError, match="before the target"):
        s.resolve(cid, 100.0)


def test_editing_a_sealed_range_is_detected(tmp_path):
    m, _ = model()
    path = tmp_path / "c.duckdb"
    s = sealer(path)
    cid = s.claim_from(m, 100.0, T0, T1, condition="calm", as_of=T0)
    s.resolve(cid, 150.0)  # a miss
    head = s.head()
    assert s.check(head).ok
    s.ledger.con.execute(
        "UPDATE claims SET upper = upper + 100 WHERE claim_id = ?", [cid]
    )
    report = s.check(head)
    assert not report.ok
    assert any("changed" in p for p in report.problems)


def test_scorecard_counts_hits_and_misses():
    m, _ = model()
    s = sealer()
    for i, actual in enumerate([100.0, 100.2, 160.0]):
        t = T0 + timedelta(hours=2 * i)
        cid = s.claim_from(
            m, 100.0, t, t + timedelta(hours=1), condition="calm", as_of=t
        )
        s.resolve(cid, actual)
    row = s.scorecard().loc["calm"]
    assert row["cases"] == 3
    assert row["hit_rate"] == pytest.approx(2 / 3)


def test_same_data_gives_same_fingerprint():
    from assay_core.seal import _hash

    _, df = model()
    assert _hash(df) == _hash(df.copy())
    assert _hash(df) != _hash(df.assign(a=df["a"] + 1))
