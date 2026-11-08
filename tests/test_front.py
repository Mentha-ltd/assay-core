import numpy as np
import pandas as pd
import pytest

import assay_core


def table(n=1200, seed=0):
    rng = np.random.default_rng(seed)
    mode = rng.choice(["calm", "wild"], n)
    err = np.where(mode == "calm", rng.normal(0, 1, n), rng.normal(0, 5, n))
    return pd.DataFrame({"t": range(n), "f": 50.0, "a": 50.0 + err, "mode": mode})


def test_assess_reports_each_condition_near_claim():
    r = assay_core.assess(table(), "f", "a", condition="mode", sort_by="t")
    assert set(r.by_condition.index) == {"calm", "wild"}
    assert (r.by_condition["hit_rate"].between(0.85, 0.95)).all()
    assert "claimed 90%" in str(r)


def test_assess_without_condition_gives_one_row():
    r = assay_core.assess(table(), "f", "a", sort_by="t")
    assert len(r.by_condition) == 1
    assert 0.86 <= r.overall_hit_rate <= 0.94


def test_assess_uses_users_column_names_and_units():
    df = table().rename(columns={"f": "pred_mw", "a": "meas_mw"})
    r = assay_core.assess(df, "pred_mw", "meas_mw", condition="mode", sort_by="t")
    assert r.rows["lower"].lt(r.rows["upper"]).all()


def test_fit_gives_wider_range_for_wild():
    m = assay_core.fit(table(), "f", "a", condition="mode")
    lo_c, hi_c, b_c = m.range(100.0, "calm")
    lo_w, hi_w, b_w = m.range(100.0, "wild")
    assert (hi_w - lo_w) > 3 * (hi_c - lo_c)
    assert not b_c
    assert not b_w


def test_fit_unseen_condition_is_borrowed():
    m = assay_core.fit(table(), "f", "a", condition="mode")
    assert m.range(100.0, "storm")[2]


def test_fit_without_condition_works():
    lo, hi, borrowed = assay_core.fit(table(), "f", "a").range(100.0)
    assert lo < 100.0 < hi
    assert not borrowed


def test_assess_too_little_history_is_refused():
    with pytest.raises(ValueError):
        assay_core.assess(table(n=10), "f", "a", sort_by="t")


def test_range_is_one_sided_when_forecast_is_biased():
    df = table()
    df["a"] = df["a"] - 20  # forecaster always too high
    _lo, hi, _ = assay_core.fit(df, "f", "a").range(50.0)
    assert hi < 50.0
