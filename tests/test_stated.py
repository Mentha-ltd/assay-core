# Tests for audit(): grading a range someone else stated.
import numpy as np
import pandas as pd
import pytest

from assay_core import audit


def make(n=400, seed=0, share=0.80):
    rng = np.random.default_rng(seed)
    p50 = rng.normal(100, 10, n)
    sd = 10.0
    z = {0.80: 1.2816, 0.90: 1.6449}[share]
    y = p50 + rng.normal(0, sd, n)
    return pd.DataFrame(
        {
            "t": np.arange(n),
            "site": np.where(np.arange(n) % 2 == 0, "a", "b"),
            "p50": p50,
            "p10": p50 - z * sd,
            "p90": p50 + z * sd,
            "y": y,
        }
    )


def test_calibrated_vendor_range_passes():
    df = make(2000)
    r = audit(df, forecast="p50", actual="y", lower="p10", upper="p90", claimed=0.80)
    rate = r.rows["hit"].mean()
    assert 0.77 < rate < 0.83
    row = r.by_condition.iloc[0]
    assert row["hit_low"] < 0.80 < row["hit_high"]


def test_overconfident_vendor_range_is_caught():
    df = make(2000)
    df["p10"] = df["p50"] - 5
    df["p90"] = df["p50"] + 5
    r = audit(df, forecast="p50", actual="y", lower="p10", upper="p90", claimed=0.80)
    assert r.by_condition.iloc[0]["hit_high"] < 0.80


def test_bounds_are_inclusive():
    df = pd.DataFrame(
        {"f": [1.0, 1.0], "lo": [0.0, 0.0], "hi": [2.0, 2.0], "y": [2.0, 0.0]}
    )
    r = audit(df, forecast="f", actual="y", lower="lo", upper="hi")
    assert r.rows["hit"].all()


def test_one_sided_ceiling_and_floor():
    df = make(200)
    c = audit(df, forecast="p50", actual="y", upper="p90", claimed=0.90)
    assert c.side == "ceiling"
    assert c.rows["hit"].equals(df["y"].le(df["p90"]).rename("hit"))
    f = audit(df, forecast="p50", actual="y", lower="p10", claimed=0.90)
    assert f.side == "floor"
    assert f.rows["hit"].mean() == pytest.approx((df["y"] >= df["p10"]).mean())


def test_condition_table():
    df = make(600)
    df.loc[df["site"] == "b", "y"] += 40
    r = audit(
        df,
        forecast="p50",
        actual="y",
        lower="p10",
        upper="p90",
        condition="site",
        claimed=0.80,
    )
    t = r.by_condition
    assert t.loc["a", "hit_rate"] > 0.7
    assert t.loc["b", "hit_rate"] < 0.5
    assert {"cases", "hit_low", "hit_high"} <= set(t.columns)


def test_missing_forecast_rows_are_dropped_and_counted():
    df = make(100)
    df.loc[3, "y"] = np.nan
    r = audit(df, forecast="p50", actual="y", lower="p10", upper="p90")
    assert r.dropped == 1
    assert len(r.rows) == 99
    with pytest.raises(ValueError, match="missing forecast or actual"):
        audit(
            df, forecast="p50", actual="y", lower="p10", upper="p90", missing="refuse"
        )


def test_missing_or_crossed_limits_are_refused():
    df = make(50)
    bad = df.copy()
    bad.loc[0, "p10"] = np.nan
    with pytest.raises(ValueError, match="missing p10"):
        audit(bad, forecast="p50", actual="y", lower="p10", upper="p90")
    cross = df.copy()
    cross.loc[0, "p10"], cross.loc[0, "p90"] = 200.0, 0.0
    with pytest.raises(ValueError, match="above"):
        audit(cross, forecast="p50", actual="y", lower="p10", upper="p90")


def test_bad_inputs():
    df = make(20)
    with pytest.raises(ValueError, match="lower, upper or both"):
        audit(df, forecast="p50", actual="y")
    with pytest.raises(ValueError, match="between 0 and 1"):
        audit(df, forecast="p50", actual="y", upper="p90", claimed=1.5)
    with pytest.raises(ValueError, match="not in the table"):
        audit(df, forecast="p50", actual="nope", upper="p90")


def test_sort_by_orders_rows():
    df = make(50).sample(frac=1, random_state=1)
    r = audit(df, forecast="p50", actual="y", lower="p10", upper="p90", sort_by="t")
    assert r.rows["row"].tolist() == list(range(50))
    expected = df.sort_values("t")["y"].between(
        df.sort_values("t")["p10"], df.sort_values("t")["p90"]
    )
    assert r.rows["hit"].tolist() == expected.tolist()


def test_explain_works_on_a_stated_range():
    df = make(800)
    df.loc[df["site"] == "b", "y"] += 40
    r = audit(
        df,
        forecast="p50",
        actual="y",
        lower="p10",
        upper="p90",
        condition="site",
        sort_by="t",
        claimed=0.80,
    )
    out = r.explain(df, features=["p50"], by=["site"])
    assert "b" in str(out)


def test_scope_flag():
    df = make(400)
    df.loc[300:, "p50"] += 500
    df.loc[300:, ["p10", "p90"]] += 500
    df.loc[300:, "y"] += 500
    r = audit(
        df,
        forecast="p50",
        actual="y",
        lower="p10",
        upper="p90",
        sort_by="t",
        features=["p50"],
    )
    assert r.by_scope is not None
    assert (r.rows["scope"] == "out").sum() > 0
    assert "out" in r.by_scope.index
