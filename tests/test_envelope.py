import json
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from assay_core import Sealer, assess, explain, fit, learn_scope
from assay_core.autonomy import autonomy_level
from assay_core.envelope import decide

T0 = datetime(2026, 1, 10, tzinfo=UTC)
T1 = T0 + timedelta(hours=1)


def history(n=1200, seed=3):
    """Calm hours have small errors. Wild hours have large ones."""
    rng = np.random.default_rng(seed)
    mode = rng.choice(["calm", "wild"], n, p=[0.7, 0.3])
    err = np.where(mode == "calm", rng.normal(0, 1, n), rng.normal(0, 6, n))
    wind = np.where(mode == "calm", rng.uniform(0, 5, n), rng.uniform(8, 14, n))
    return pd.DataFrame(
        {"t": np.arange(n), "f": 100.0, "a": 100.0 + err, "mode": mode, "wind": wind}
    )


def sealer():
    return Sealer(
        ":memory:", agent_id="m1", model_version="1", target_name="load", unit="MW",
        resolution_rule="metered MW",
    )  # fmt: skip


def parts():
    df = history()
    model = fit(df, "f", "a", condition="mode")
    report = assess(df, "f", "a", condition="mode", sort_by="t")
    return df, model, report


def test_envelope_joins_what_the_parts_say():
    _df, model, report = parts()
    env = decide(model, report, 100.0, condition="calm")
    lower, upper, borrowed = model.range(100.0, "calm")
    assert (env.lower, env.upper, env.borrowed) == (lower, upper, borrowed)
    n = int(report.by_condition.loc["calm", "cases"])
    assert env.record_cases == n
    assert env.record_hits == round(report.by_condition.loc["calm", "hit_rate"] * n)
    assert env.record_label == "calm"
    direct = autonomy_level(env.record_hits, n, model.claimed, tolerance=0.02)
    assert env.autonomy == direct


def test_no_condition_uses_all_cases():
    df = history()
    model = fit(df, "f", "a")
    report = assess(df, "f", "a", sort_by="t")
    env = decide(model, report, 100.0)
    assert env.record_cases == len(report.rows)
    assert env.record_label == "all cases"
    assert env.record_note == ""


def test_unseen_condition_falls_back_and_says_so():
    _df, model, report = parts()
    env = decide(model, report, 100.0, condition="storm")
    assert env.borrowed is True
    assert env.record_label == "all cases"
    assert "storm" in env.record_note
    assert env.record_cases == len(report.rows)


def test_scope_in_and_out():
    df, model, report = parts()
    scope = learn_scope(df, ["wind"])
    inside = decide(
        model, report, 100.0, condition="calm", inputs={"wind": 3.0}, scope=scope
    )
    assert inside.in_scope is True
    assert inside.scope_reasons == ""
    out = decide(
        model, report, 100.0, condition="calm", inputs={"wind": 40.0}, scope=scope
    )
    assert out.in_scope is False
    assert "wind above learned" in out.scope_reasons
    assert out.autonomy.level == 0
    assert "outside" in out.autonomy.reason


def test_scope_not_judged_does_not_lower_the_level():
    df, model, report = parts()
    env = decide(model, report, 100.0, condition="calm")
    assert env.in_scope is None
    scope = learn_scope(df, ["wind"])
    inside = decide(
        model, report, 100.0, condition="calm", inputs={"wind": 3.0}, scope=scope
    )
    assert env.autonomy.level == inside.autonomy.level


def test_scope_needs_both_inputs_and_scope():
    df, model, report = parts()
    with pytest.raises(ValueError, match="both inputs and scope"):
        decide(model, report, 100.0, inputs={"wind": 3.0})
    with pytest.raises(ValueError, match="both inputs and scope"):
        decide(model, report, 100.0, scope=learn_scope(df, ["wind"]))


def test_rare_condition_gets_a_lower_level_than_a_common_one():
    rng = np.random.default_rng(5)
    n = 1500
    mode = rng.choice(["calm", "storm"], n, p=[0.97, 0.03])
    err = np.where(mode == "calm", rng.normal(0, 1, n), rng.normal(0, 8, n))
    df = pd.DataFrame({"t": np.arange(n), "f": 100.0, "a": 100.0 + err, "mode": mode})
    model = fit(df, "f", "a", condition="mode")
    report = assess(df, "f", "a", condition="mode", sort_by="t")
    calm = decide(model, report, 100.0, condition="calm")
    storm = decide(model, report, 100.0, condition="storm")
    assert calm.record_cases > 10 * storm.record_cases
    assert calm.autonomy.level == 2
    assert storm.autonomy.level == 0
    assert storm.autonomy.level < calm.autonomy.level


def test_one_sided_envelope_has_a_missing_limit_and_is_json_safe():
    df = history()
    model = fit(df, "f", "a", condition="mode", side="floor")
    report = assess(df, "f", "a", condition="mode", sort_by="t", side="floor")
    env = decide(model, report, 100.0, condition="calm")
    assert env.upper is None
    assert env.lower is not None
    text = json.dumps(env.to_dict(), allow_nan=False)
    assert '"upper": null' in text
    assert "no limit" in str(env)


def test_to_dict_is_plain_json():
    df, model, report = parts()
    scope = learn_scope(df, ["wind"])
    expl = explain(report, df, features=["wind"])
    env = decide(
        model, report, 100.0, condition="wild", inputs={"wind": 10.0}, scope=scope,
        explanation=expl,
    )  # fmt: skip
    d = env.to_dict()
    json.dumps(d, allow_nan=False)
    assert d["range"]["claimed"] == model.claimed
    assert d["record"]["cases"] == env.record_cases
    assert d["autonomy"]["level"] == env.autonomy.level
    assert d["scope"]["in_scope"] is True
    assert d["findings"] == expl.findings
    assert d["claim_id"] is None


def test_findings_pass_through_unchanged():
    df, model, report = parts()
    expl = explain(report, df, features=["wind"])
    env = decide(model, report, 100.0, condition="calm", explanation=expl)
    assert env.findings == expl.findings


def test_sealing_needs_both_times():
    _df, model, report = parts()
    with pytest.raises(ValueError, match="target_start and target_end"):
        decide(model, report, 100.0, sealer=sealer())
    with pytest.raises(ValueError, match="target_start and target_end"):
        decide(model, report, 100.0, sealer=sealer(), target_start=T0)


def test_sealed_claim_carries_the_decision():
    df, model, report = parts()
    scope = learn_scope(df, ["wind"])
    s = sealer()
    env = decide(
        model, report, 100.0, condition="calm", inputs={"wind": 3.0}, scope=scope,
        sealer=s, target_start=T0, target_end=T1, as_of=T0 - timedelta(hours=1),
    )  # fmt: skip
    assert env.claim_id is not None
    claim = s.ledger.get_claim(env.claim_id)
    assert (claim.lower, claim.upper) == (env.lower, env.upper)
    assert claim.context["autonomy_level"] == env.autonomy.level
    assert claim.context["in_scope"] is True
    assert claim.context["record_cases"] == env.record_cases
    s.resolve(env.claim_id, 100.3)
    assert s.check().ok
    assert s.scorecard().loc["calm", "hit_rate"] == 1.0
    assert env.claim_id in str(env)


def test_out_of_scope_claim_is_still_sealed_and_marked():
    df, model, report = parts()
    scope = learn_scope(df, ["wind"])
    s = sealer()
    env = decide(
        model, report, 100.0, condition="calm", inputs={"wind": 40.0}, scope=scope,
        sealer=s, target_start=T0, target_end=T1, as_of=T0,
    )  # fmt: skip
    claim = s.ledger.get_claim(env.claim_id)
    assert claim.context["in_scope"] is False
    assert claim.context["autonomy_level"] == 0


def test_thin_record_stays_in_shadow():
    df = history(n=60)
    model = fit(df, "f", "a")
    report = assess(df, "f", "a", sort_by="t", min_group=0)
    env = decide(model, report, 100.0)
    assert env.autonomy.level == 0
    assert "cases" in env.autonomy.reason


def test_text_shows_every_part():
    df, model, report = parts()
    scope = learn_scope(df, ["wind"])
    env = decide(
        model, report, 100.0, condition="calm", inputs={"wind": 3.0}, scope=scope
    )
    text = str(env)
    for word in ("range", "claimed", "record (calm)", "in scope", "level"):
        assert word in text
