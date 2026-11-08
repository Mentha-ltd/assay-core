import numpy as np
import pandas as pd
import pytest

import assay_core
from assay_core.scope import learn_scope, walk_forward_scope


def uniform(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"x": rng.uniform(0, 10, n), "y": rng.normal(50, 5, n)})


def test_case_inside_is_in_scope_and_outside_is_flagged_with_reason():
    sc = learn_scope(uniform(), ["x"])
    out = sc.check(pd.DataFrame({"x": [5.0, -5.0, 20.0]}))
    assert out["in_scope"].tolist() == [True, False, False]
    assert "x below learned" in out["reasons"].iloc[1]
    assert "x above learned" in out["reasons"].iloc[2]


def test_missing_value_is_flagged():
    sc = learn_scope(uniform(), ["x"])
    out = sc.check(pd.DataFrame({"x": [np.nan]}))
    assert not out["in_scope"].iloc[0]
    assert "missing" in out["reasons"].iloc[0]


def test_two_columns_both_reasons_listed():
    sc = learn_scope(uniform(), ["x", "y"])
    out = sc.check(pd.DataFrame({"x": [99.0], "y": [-99.0]}))
    assert "x above" in out["reasons"].iloc[0]
    assert "y below" in out["reasons"].iloc[0]


def test_unknown_column_refused():
    with pytest.raises(ValueError, match="not in the table"):
        learn_scope(uniform(), ["nope"])


def test_no_columns_refused():
    with pytest.raises(ValueError, match="at least one"):
        learn_scope(uniform(), [])


@pytest.mark.parametrize("tail", [0.0, 0.5, -0.1, True])
def test_bad_tail_refused(tail):
    with pytest.raises(ValueError, match="tail"):
        learn_scope(uniform(), ["x"], tail=tail)


def test_too_few_learning_cases_refused():
    with pytest.raises(ValueError, match="at least 100"):
        learn_scope(uniform(50), ["x"])


def test_missing_values_in_learning_cases_refused():
    df = uniform()
    df.loc[3, "x"] = np.nan
    with pytest.raises(ValueError, match="missing"):
        learn_scope(df, ["x"])


def test_about_two_percent_of_normal_cases_are_flagged():
    sc = learn_scope(uniform(5000, 1), ["x"])
    share_out = 1 - sc.check(uniform(5000, 2))["in_scope"].mean()
    assert 0.01 <= share_out <= 0.03


def test_walk_forward_first_rows_are_unjudged_and_no_look_ahead():
    df = uniform(300)
    df.loc[150, "x"] = 1000.0
    out = walk_forward_scope(df, ["x"])
    assert (out["scope"].iloc[:100] == "unjudged").all()
    assert out["scope"].iloc[150] == "out"
    # the shock must not have changed the verdict on the rows before it
    clean = walk_forward_scope(df.iloc[:150], ["x"])
    assert out["scope"].iloc[:150].tolist() == clean["scope"].tolist()


def shocked(n=4000, seed=5):
    """Mostly normal. About 0.5% of cases have an unusual input and a 6x larger error."""
    rng = np.random.default_rng(seed)
    shock = rng.random(n) < 0.005
    x = np.where(shock, rng.uniform(40, 50, n), rng.uniform(0, 10, n))
    err = np.where(shock, rng.normal(0, 6, n), rng.normal(0, 1, n))
    err[:300] = rng.normal(0, 1, 300)
    x[:300] = rng.uniform(0, 10, 300)
    return pd.DataFrame({"t": range(n), "f": 100.0, "a": 100.0 + err, "x": x})


def test_flag_catches_every_shock_and_shocks_miss_their_range():
    df = shocked()
    r = assay_core.assess(df, "f", "a", sort_by="t", features=["x"])
    rows = r.rows.merge(df[["x"]], left_on="row", right_index=True)
    judged = rows[rows["scope"] != "unjudged"]
    shocks = judged[judged["x"] > 20]
    assert len(shocks) >= 10
    assert (shocks["scope"] == "out").all()  # every real shock was flagged
    assert shocks["hit"].mean() < 0.5  # and the range really did fail on them
    tbl = r.by_scope
    assert tbl.loc["in", "hit_rate"] >= 0.87
    assert tbl.loc["out", "hit_rate"] < tbl.loc["in", "hit_rate"] - 0.05


def test_false_alarm_cost_is_small():
    df = shocked()
    r = assay_core.assess(df, "f", "a", sort_by="t", features=["x"])
    judged = r.rows[r.rows["scope"] != "unjudged"]
    assert (judged["scope"] == "out").mean() < 0.03  # at most 3% of cases flagged


def test_assess_without_features_has_no_scope_table():
    r = assay_core.assess(shocked(), "f", "a", sort_by="t")
    assert r.by_scope is None
