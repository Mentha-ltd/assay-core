import numpy as np
import pandas as pd
import pytest

import assay_core


def hidden_cause(n=6000, seed=11):
    """Errors are 6x larger when temp > 30 (about 10% of rows). 'noise_col' is unrelated.
    'mode' is unrelated too, so nothing should be blamed on it."""
    rng = np.random.default_rng(seed)
    temp = rng.normal(20, 6, n)
    hot = temp > 30
    err = np.where(hot, rng.normal(0, 6, n), rng.normal(0, 1, n))
    return pd.DataFrame(
        {
            "t": range(n),
            "f": 100.0,
            "a": 100.0 + err,
            "temp": temp,
            "noise_col": rng.normal(0, 1, n),
            "mode": rng.choice(["a", "b"], n),
        }
    )


def run(df, **kw):
    r = assay_core.assess(df, "f", "a", sort_by="t", **kw)
    return r, r.explain(df, features=["temp", "noise_col"], by=["mode"])


def test_finds_the_hidden_cause_and_blames_nothing_else():
    _, ex = run(hidden_cause())
    worse = ex.by_input[ex.by_input["verdict"] == "WORSE than claimed"]
    assert "temp" in set(worse["column"])
    assert "noise_col" not in set(worse["column"])
    assert (
        ex.by_condition["verdict"] != "WORSE than claimed"
    ).all()  # 'mode' is innocent
    assert any("temp" in f for f in ex.findings)


def test_the_top_temp_band_has_the_high_miss_rate():
    _, ex = run(hidden_cause())
    temp = ex.by_input[ex.by_input["column"] == "temp"].sort_values("miss_rate")
    assert temp["miss_rate"].iloc[-1] > 0.2
    assert temp["miss_rate"].iloc[0] < 0.1


def test_every_miss_has_a_row_with_side_and_size():
    r, ex = run(hidden_cause())
    assert len(ex.misses) == int((~r.rows["hit"]).sum())
    assert set(ex.misses["side"]) <= {"below", "above"}
    assert (ex.misses["miss_size"] > 0).all()


def test_an_input_is_named_only_when_extreme_and_it_is_the_right_one():
    _, ex = run(hidden_cause())
    named = ex.misses[ex.misses["unusual_input"].notna()]
    assert len(named) >= 20
    assert (named["unusual_input"] == "temp").mean() > 0.8
    pct = named["unusual_pct"]
    assert ((pct <= 5) | (pct >= 95)).all()  # never named for an ordinary value


def test_ordinary_misses_are_left_unexplained_by_inputs():
    _, ex = run(hidden_cause())
    assert ex.misses["unusual_input"].isna().sum() > 0


def test_a_group_with_larger_errors_is_found_by_condition_column():
    rng = np.random.default_rng(3)
    n = 5000
    mode = rng.choice(["calm", "wild"], n)
    err = np.where(mode == "wild", rng.normal(0, 6, n), rng.normal(0, 1, n))
    df = pd.DataFrame({"t": range(n), "f": 0.0, "a": err, "mode": mode})
    r = assay_core.assess(
        df, "f", "a", sort_by="t"
    )  # one pooled range: wild is under-covered
    ex = r.explain(df, by=["mode"])
    row = ex.by_condition.set_index("group").loc["wild"]
    assert row["verdict"] == "WORSE than claimed"
    assert (
        ex.by_condition.set_index("group").loc["calm", "verdict"]
        != "WORSE than claimed"
    )


def test_drift_in_the_last_period_is_found():
    rng = np.random.default_rng(4)
    n = 4000
    sd = np.where(np.arange(n) > 3000, 5.0, 1.0)
    df = pd.DataFrame({"t": range(n), "f": 0.0, "a": rng.normal(0, 1, n) * sd})
    r = assay_core.assess(df, "f", "a", sort_by="t")
    ex = r.explain(df)
    last = ex.by_period.iloc[-1]
    assert last["verdict"] == "WORSE than claimed"
    assert any("period" in f for f in ex.findings)


def test_clean_data_gives_no_findings():
    rng = np.random.default_rng(8)
    n = 3000
    df = pd.DataFrame(
        {"t": range(n), "f": 0.0, "a": rng.normal(0, 1, n), "x": rng.normal(0, 1, n)}
    )
    r = assay_core.assess(df, "f", "a", sort_by="t")
    ex = r.explain(df, features=["x"])
    assert ex.findings == []
    assert "No group is worse" in str(ex)


def test_percentile_uses_only_earlier_rows():
    rng = np.random.default_rng(5)
    n = 600
    x = np.arange(n, dtype=float)  # always the largest so far
    err = rng.normal(0, 1, n)
    err[500] = 50.0  # one big miss
    df = pd.DataFrame({"t": range(n), "f": 0.0, "a": err, "x": x})
    r = assay_core.assess(df, "f", "a", sort_by="t")
    ex = r.explain(df, features=["x"])
    big = ex.misses[ex.misses["row"] == 500].iloc[0]
    assert big["unusual_input"] == "x"
    assert big["unusual_pct"] == pytest.approx(100.0)


def test_floor_misses_are_breaches_below():
    df = hidden_cause()
    r = assay_core.assess(df, "f", "a", sort_by="t", side="floor")
    ex = r.explain(df, features=["temp"])
    assert (ex.misses["side"] == "below").all()


def test_unknown_column_refused():
    df = hidden_cause()
    r = assay_core.assess(df, "f", "a", sort_by="t")
    with pytest.raises(ValueError, match="not in the table"):
        r.explain(df, features=["nope"])


@pytest.mark.parametrize("extreme", [0.0, 50.0, -1.0])
def test_bad_extreme_refused(extreme):
    df = hidden_cause()
    r = assay_core.assess(df, "f", "a", sort_by="t")
    with pytest.raises(ValueError, match="extreme"):
        r.explain(df, extreme=extreme)


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.2])
def test_bad_alpha_refused(alpha):
    df = hidden_cause()
    r = assay_core.assess(df, "f", "a", sort_by="t")
    with pytest.raises(ValueError, match="alpha"):
        r.explain(df, alpha=alpha)


def test_scope_flag_is_carried_into_the_miss_table():
    df = hidden_cause()
    r = assay_core.assess(df, "f", "a", sort_by="t", features=["temp"])
    ex = r.explain(df)
    assert "scope" in ex.misses.columns
    assert set(ex.misses["scope"]) <= {"in", "out", "unjudged"}


def test_miss_size_is_in_the_users_units():
    df = hidden_cause()
    r = assay_core.assess(df, "f", "a", sort_by="t")
    ex = r.explain(df)
    m = ex.misses.iloc[0]
    row = r.rows[r.rows["row"] == m["row"]].iloc[0]
    actual = df.sort_values("t")["a"].iloc[int(m["row"])]
    expected = row["lower"] - actual if m["side"] == "below" else actual - row["upper"]
    assert m["miss_size"] == pytest.approx(expected)
