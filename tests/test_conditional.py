import numpy as np
import pandas as pd
import pytest

from assay_core.conditional import (
    conditional_bounds,
    range_for,
    walk_forward_conditional,
)


def mixed(seed=0, n_calm=300, n_wild=300):
    rng = np.random.default_rng(seed)
    e = np.concatenate([rng.normal(0, 1, n_calm), rng.normal(0, 5, n_wild)])
    c = np.array(["calm"] * n_calm + ["wild"] * n_wild)
    perm = rng.permutation(len(e))
    return e[perm], c[perm]


def test_wild_range_is_wider_than_calm_range():
    e, c = mixed()
    r = conditional_bounds(e, c)
    assert (r["wild"].upper - r["wild"].lower) > 3 * (r["calm"].upper - r["calm"].lower)


def test_small_group_borrows_the_pooled_range():
    e, c = mixed()
    c = c.copy()
    c[:5] = "rare"
    r = conditional_bounds(e, c)
    assert r["rare"].borrowed
    assert (r["rare"].lower, r["rare"].upper) == (
        r["__pooled__"].lower,
        r["__pooled__"].upper,
    )
    assert not r["calm"].borrowed


def test_unseen_condition_gets_pooled_range_borrowed():
    e, c = mixed()
    got = range_for(conditional_bounds(e, c), "never seen")
    assert got.borrowed


def test_min_group_forces_borrowing():
    e, c = mixed(n_calm=40, n_wild=300)
    assert not conditional_bounds(e, c)["calm"].borrowed
    assert conditional_bounds(e, c, min_group=100)["calm"].borrowed


def test_length_mismatch_is_refused():
    with pytest.raises(ValueError):
        conditional_bounds(np.zeros(30), np.array(["a"] * 29))


def test_too_few_errors_overall_is_refused():
    with pytest.raises(ValueError):
        conditional_bounds(np.zeros(10), np.array(["a"] * 10))


def test_numeric_condition_labels_work():
    e, c = mixed()
    r = conditional_bounds(e, np.where(c == "calm", 1, 2))
    assert "1" in r
    assert "2" in r


def test_walk_forward_covers_each_condition_near_90_percent():
    e, c = mixed(seed=1, n_calm=1500, n_wild=1500)
    df = pd.DataFrame({"f": 100.0, "a": 100.0 + e, "cond": c, "t": range(len(e))})
    out = walk_forward_conditional(df, "f", "a", "cond", sort_by="t")
    by = out.groupby("cond")["hit"].mean()
    assert 0.87 <= by["calm"] <= 0.93
    assert 0.87 <= by["wild"] <= 0.93


def test_pooled_range_misses_wild_days_but_conditional_does_not():
    e, c = mixed(seed=2, n_calm=1500, n_wild=1500)
    df = pd.DataFrame({"f": 0.0, "a": e, "cond": c, "t": range(len(e))})
    pooled = walk_forward_conditional(
        df.assign(cond="all"), "f", "a", "cond", sort_by="t"
    )
    pooled["real"] = c[pooled["row"]]
    wild_pooled = pooled.loc[pooled["real"] == "wild", "hit"].mean()
    cond = walk_forward_conditional(df, "f", "a", "cond", sort_by="t")
    wild_cond = cond.loc[cond["cond"] == "wild", "hit"].mean()
    assert wild_pooled < 0.85
    assert wild_cond > wild_pooled + 0.04


def test_no_look_ahead_first_row_needs_history():
    e, c = mixed()
    df = pd.DataFrame({"f": 0.0, "a": e[:60], "cond": c[:60], "t": range(60)})
    out = walk_forward_conditional(df, "f", "a", "cond", sort_by="t")
    assert out["row"].min() == 19
