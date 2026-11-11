import numpy as np
import pandas as pd
import pytest

from assay_core.look import where_to_look


def make(n=200, seed=0):
    r = np.random.default_rng(seed)
    cond = np.where(np.arange(n) % 2 == 0, "calm", "storm")
    sd = np.where(cond == "calm", 1.0, 4.0)
    a = r.normal(0, sd)
    return pd.DataFrame({"a": a, "lo": -1.28, "hi": 1.28, "c": cond})


def test_storm_is_first_and_too_narrow():
    t = where_to_look(make(), "a", "lo", "hi", "c")
    assert t.index[0] == "storm" and t.loc["storm", "advice"] == "too narrow"
    assert t.loc["storm", "excess"] > 0 > t.loc["calm", "excess"]


def test_excess_sums_to_zero():
    t = where_to_look(make(), "a", "lo", "hi", "c")
    assert t["excess"].sum() == pytest.approx(0, abs=1e-9)


def test_too_wide_flagged():
    d = make()
    d.loc[d["c"] == "calm", ["lo", "hi"]] = [-10, 10]
    assert where_to_look(d, "a", "lo", "hi", "c").loc["calm", "advice"] == "too wide"


def test_width_share_is_fraction():
    t = where_to_look(make(), "a", "lo", "hi", "c")
    assert ((t["width_share"] > 0) & (t["width_share"] <= 1)).all()


def test_exact_score_arithmetic():
    d = pd.DataFrame({"a": [5.0], "lo": [0.0], "hi": [2.0], "c": ["x"]})
    t = where_to_look(d, "a", "lo", "hi", "c")
    assert t.loc["x", "mean_score"] == 2.0 + 10 * 3.0
    assert t.loc["x", "hit_rate"] == 0.0


def test_refuses_missing_and_crossed():
    d = make()
    d.loc[0, "a"] = np.nan
    with pytest.raises(ValueError):
        where_to_look(d, "a", "lo", "hi", "c")
    d = make()
    d.loc[0, ["lo", "hi"]] = [3, 1]
    with pytest.raises(ValueError):
        where_to_look(d, "a", "lo", "hi", "c")
