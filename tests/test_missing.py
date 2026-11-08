import numpy as np
import pandas as pd
import pytest

import assay_core


def table(n=3000, seed=21):
    rng = np.random.default_rng(seed)
    temp = rng.normal(20, 6, n)
    err = rng.normal(0, 1, n) * np.where(temp > 30, 6, 1)
    return pd.DataFrame(
        {
            "t": range(n),
            "f": 100.0,
            "a": 100.0 + err,
            "temp": temp,
            "mode": rng.choice(["a", "b"], n),
        }
    )


def with_gaps(df, seed=1):
    rng = np.random.default_rng(seed)
    out = df.copy()
    out.loc[rng.random(len(out)) < 0.03, "a"] = np.nan
    out.loc[rng.random(len(out)) < 0.02, "f"] = np.nan
    return out


def test_assess_drops_gap_rows_and_says_how_many():
    gappy = with_gaps(table())
    r = assay_core.assess(gappy, "f", "a", condition="mode", sort_by="t")
    expected = int(gappy[["f", "a"]].isna().any(axis=1).sum())
    assert r.dropped == expected > 0
    assert f"{expected} rows dropped" in str(r)


def test_result_equals_the_table_with_gap_rows_removed_by_hand():
    gappy = with_gaps(table())
    by_hand = gappy.dropna(subset=["f", "a"])
    a = assay_core.assess(
        gappy, "f", "a", condition="mode", sort_by="t", features=["temp"]
    )
    b = assay_core.assess(
        by_hand, "f", "a", condition="mode", sort_by="t", features=["temp"]
    )
    pd.testing.assert_frame_equal(a.rows, b.rows)
    pd.testing.assert_frame_equal(a.by_condition, b.by_condition)
    pd.testing.assert_frame_equal(a.by_scope, b.by_scope)


def test_explain_lines_up_with_assess_when_rows_were_dropped():
    gappy = with_gaps(table())
    by_hand = gappy.dropna(subset=["f", "a"])
    ra = assay_core.assess(gappy, "f", "a", sort_by="t")
    rb = assay_core.assess(by_hand, "f", "a", sort_by="t")
    ea = ra.explain(gappy, features=["temp"])
    eb = rb.explain(by_hand, features=["temp"])
    pd.testing.assert_frame_equal(ea.misses, eb.misses)
    pd.testing.assert_frame_equal(ea.by_input, eb.by_input)
    assert any("temp" in f for f in ea.findings)


def test_no_gaps_means_nothing_reported():
    r = assay_core.assess(table(), "f", "a", sort_by="t")
    assert r.dropped == 0
    assert "dropped" not in str(r)


def test_refuse_mode_raises_with_the_count():
    gappy = with_gaps(table())
    n = int(gappy[["f", "a"]].isna().any(axis=1).sum())
    with pytest.raises(ValueError, match=f"{n} rows have a missing"):
        assay_core.assess(gappy, "f", "a", sort_by="t", missing="refuse")


def test_refuse_mode_is_fine_when_there_are_no_gaps():
    assay_core.assess(table(), "f", "a", sort_by="t", missing="refuse")


def test_bad_missing_value_refused():
    with pytest.raises(ValueError, match="missing must be"):
        assay_core.assess(table(), "f", "a", missing="ignore")


def test_input_gaps_are_not_dropped_they_make_the_case_out_of_scope():
    df = table()
    df.loc[1500, "temp"] = np.nan
    r = assay_core.assess(df, "f", "a", sort_by="t", features=["temp"])
    assert r.dropped == 0
    row = r.rows[r.rows["row"] == 1500].iloc[0]
    assert row["scope"] == "out"
    assert "missing" in row["reasons"]


def test_all_rows_missing_is_an_error_not_a_silent_empty_report():
    df = table(n=100)
    df["a"] = np.nan
    with pytest.raises(ValueError, match="not enough history"):
        assay_core.assess(df, "f", "a", sort_by="t")
