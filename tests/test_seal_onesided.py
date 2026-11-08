from datetime import UTC, datetime, timedelta

import pytest
from test_onesided import make

import assay_core
from assay_core import Sealer
from assay_core.record.models import Claim
from assay_core.record.scoring import interval_score, is_covered

T0 = datetime(2026, 1, 10, tzinfo=UTC)
T1 = T0 + timedelta(hours=1)


def sealer():
    return Sealer(
        agent_id="m", model_version="1", target_name="x", unit="u", resolution_rule="r"
    )


def test_floor_claim_is_sealed_with_no_upper_limit_and_verifies():
    m = assay_core.fit(make(), "f", "a", condition="mode", side="floor")
    s = sealer()
    cid = s.claim_from(m, 100.0, T0, T1, condition="calm", as_of=T0)
    claim = s.ledger.get_claim(cid)
    assert claim.upper is None
    assert claim.lower is not None
    assert claim.context["side"] == "floor"
    s.resolve(cid, 100.0)
    assert s.check().ok


def test_floor_scorecard_counts_a_breach_as_a_miss():
    m = assay_core.fit(make(), "f", "a", condition="mode", side="floor")
    s = sealer()
    ok = s.claim_from(m, 100.0, T0, T1, condition="calm", as_of=T0)
    bad = s.claim_from(
        m, 100.0, T1, T1 + timedelta(hours=1), condition="calm", as_of=T1
    )
    s.resolve(ok, 100.0)
    s.resolve(bad, 50.0)
    assert s.scorecard().loc["calm", "hit_rate"] == pytest.approx(0.5)


def test_ceiling_claim_covered_logic():
    m = assay_core.fit(make(), "f", "a", condition="mode", side="ceiling")
    s = sealer()
    cid = s.claim_from(m, 100.0, T0, T1, condition="wild", as_of=T0)
    s.resolve(cid, 1000.0)
    assert not is_covered(s.ledger.get_claim(cid), s.ledger.latest_resolution(cid))


def test_claim_with_no_limits_is_refused():
    with pytest.raises(ValueError, match="at least one limit"):
        Claim(
            claim_id="x",
            agent_id="a",
            model_version="1",
            code_version="1",
            as_of=T0,
            data_cutoff=T0,
            target_name="t",
            unit="u",
            target_start=T0,
            target_end=T1,
            point=1.0,
            lower=None,
            upper=None,
            coverage=0.9,
            resolution_rule="r",
            input_data_hash="h",
        )


def test_interval_score_refuses_one_sided_claims():
    m = assay_core.fit(make(), "f", "a", side="floor")
    s = sealer()
    cid = s.claim_from(m, 100.0, T0, T1, as_of=T0)
    s.resolve(cid, 100.0)
    with pytest.raises(ValueError, match="one-sided"):
        interval_score(s.ledger.get_claim(cid), s.ledger.latest_resolution(cid))
