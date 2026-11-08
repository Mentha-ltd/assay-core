"""The fast walk-forward range check must give exactly the same answers as the simple one.

`slow_reference` is the first version: sort the past again for every row.
"""

import numpy as np
import pandas as pd
import pytest

from assay_core.conditional import (
    _limits,
    _need,
    _signed_errors,
    walk_forward_conditional,
)


def slow_reference(df, forecast, actual, condition, coverage=0.90, sort_by=None, window=None,
                   min_group=0, side="both"):  # fmt: skip
    d = df.sort_values(sort_by, kind="stable") if sort_by else df
    d = d.reset_index(drop=True)
    err = _signed_errors(d, forecast, actual)
    labels = d[condition].astype(str).to_numpy()
    base = _need(coverage, side)
    need = max(base, min_group)
    rows = []
    for i in range(len(d)):
        start = 0 if window is None else max(0, i - window)
        past_e, past_l = err[start:i], labels[start:i]
        if len(past_e) < base:
            continue
        own = past_e[past_l == labels[i]]
        borrowed = len(own) < need
        lo, hi = _limits(past_e if borrowed else own, coverage, side)
        point = float(d.loc[i, forecast])
        y = float(d.loc[i, actual])
        low = np.nan if lo is None else point + lo
        high = np.nan if hi is None else point + hi
        hit = (lo is None or y >= low) and (hi is None or y <= high)
        rows.append(
            {
                "row": i,
                condition: d.loc[i, condition],
                "forecast": point,
                "lower": low,
                "upper": high,
                "borrowed": borrowed,
                "n_used": len(past_e) if borrowed else len(own),
                "hit": bool(hit),
            }
        )
    return pd.DataFrame(rows)


def table(kind, n, seed):
    rng = np.random.default_rng(seed)
    mode = rng.choice(
        ["calm", "windy", "storm", "rare"], n, p=[0.5, 0.35, 0.145, 0.005]
    )
    scale = np.select(
        [mode == "calm", mode == "windy", mode == "storm"], [1.0, 3.0, 8.0], 2.0
    )
    if kind == "normal":
        err = rng.normal(0, 1, n) * scale
    elif kind == "ties":  # many equal errors
        err = rng.integers(-3, 4, n).astype(float) * scale
    elif kind == "heavy":
        err = rng.standard_cauchy(n)
    elif kind == "biased":
        err = rng.normal(-5, 1, n) * scale
    else:
        raise ValueError(kind)
    forecast = rng.uniform(50, 150, n)
    return pd.DataFrame(
        {"t": rng.permutation(n), "f": forecast, "a": forecast + err, "mode": mode,
         "k": rng.integers(0, 4, n)}
    )  # fmt: skip


@pytest.mark.parametrize("kind", ["normal", "ties", "heavy", "biased"])
@pytest.mark.parametrize("side", ["both", "floor", "ceiling"])
@pytest.mark.parametrize("window", [None, 120, 25])
def test_fast_equals_slow(kind, side, window):
    seed = len(kind) * 10 + len(side) + (window or 0)
    df = table(kind, 600, seed)
    kw = {"sort_by": "t", "window": window, "side": side}
    fast = walk_forward_conditional(df, "f", "a", "mode", **kw)
    slow = slow_reference(df, "f", "a", "mode", **kw)
    pd.testing.assert_frame_equal(fast, slow)


@pytest.mark.parametrize("coverage", [0.8, 0.9, 0.95])
@pytest.mark.parametrize("min_group", [0, 60])
def test_coverage_and_min_group_match(coverage, min_group):
    df = table("normal", 700, 5)
    kw = {"sort_by": "t", "coverage": coverage, "min_group": min_group}
    pd.testing.assert_frame_equal(
        walk_forward_conditional(df, "f", "a", "mode", **kw),
        slow_reference(df, "f", "a", "mode", **kw),
    )


@pytest.mark.parametrize("seed", range(10))
def test_many_random_tables(seed):
    rng = np.random.default_rng(seed)
    df = table(
        ["normal", "ties", "heavy", "biased"][seed % 4],
        int(rng.integers(80, 500)),
        seed,
    )
    kw = {
        "sort_by": "t",
        "window": [None, 40, 200][seed % 3],
        "side": ["both", "floor", "ceiling"][(seed // 3) % 3],
        "min_group": [0, 30][seed % 2],
    }
    pd.testing.assert_frame_equal(
        walk_forward_conditional(df, "f", "a", "mode", **kw),
        slow_reference(df, "f", "a", "mode", **kw),
    )


def test_integer_condition_column_keeps_its_values():
    df = table("normal", 400, 1)
    pd.testing.assert_frame_equal(
        walk_forward_conditional(df, "f", "a", "k", sort_by="t"),
        slow_reference(df, "f", "a", "k", sort_by="t"),
    )


def test_without_sort_column_keeps_table_order():
    df = table("normal", 300, 2)
    pd.testing.assert_frame_equal(
        walk_forward_conditional(df, "f", "a", "mode"),
        slow_reference(df, "f", "a", "mode"),
    )


def test_missing_values_are_refused_with_the_count():
    df = table("normal", 300, 3)
    df.loc[[10, 50], "a"] = np.nan
    with pytest.raises(ValueError, match="2 rows have a missing"):
        walk_forward_conditional(df, "f", "a", "mode", sort_by="t")


def test_too_little_history_gives_an_empty_table():
    df = table("normal", 10, 4)
    assert walk_forward_conditional(df, "f", "a", "mode", sort_by="t").empty


def test_a_long_table_runs_in_seconds():
    import time

    rng = np.random.default_rng(0)
    n = 50_000
    df = pd.DataFrame(
        {"f": 0.0, "a": rng.normal(0, 1, n), "mode": rng.choice(list("abc"), n)}
    )
    start = time.perf_counter()
    out = walk_forward_conditional(df, "f", "a", "mode")
    assert time.perf_counter() - start < 15
    assert len(out) == n - 19
