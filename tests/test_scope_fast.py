"""The fast walk-forward scope check must give exactly the same answers as the simple one.

`slow_reference` is the first version: recompute the percentiles from scratch for every row.
"""

import numpy as np
import pandas as pd
import pytest

from assay_core.scope import _check_tail, _columns, walk_forward_scope


def slow_reference(df, features, tail=0.01, min_history=None, window=None):
    _check_tail(tail)
    cols = _columns(df, features)
    need = round(1 / tail) if min_history is None else min_history
    data = df[cols].to_numpy(float)
    scope, why = [], []
    for i in range(len(df)):
        past = data[(0 if window is None else max(0, i - window)) : i]
        past = past[~np.isnan(past).any(axis=1)]
        if len(past) < need:
            scope.append("unjudged")
            why.append("")
            continue
        lo = np.quantile(past, tail, axis=0)
        hi = np.quantile(past, 1 - tail, axis=0)
        parts = []
        for j, c in enumerate(cols):
            x = data[i, j]
            if np.isnan(x):
                parts.append(f"{c} is missing")
            elif x < lo[j]:
                parts.append(f"{c} below learned {lo[j]:.4g}")
            elif x > hi[j]:
                parts.append(f"{c} above learned {hi[j]:.4g}")
        scope.append("out" if parts else "in")
        why.append("; ".join(parts))
    return pd.DataFrame({"scope": scope, "reasons": why}, index=df.index)


def table(kind, n, seed):
    rng = np.random.default_rng(seed)
    if kind == "normal":
        data = {"a": rng.normal(0, 1, n), "b": rng.uniform(0, 10, n)}
    elif kind == "ties":  # many equal values
        data = {
            "a": rng.integers(0, 6, n).astype(float),
            "b": rng.integers(0, 3, n).astype(float),
        }
    elif kind == "heavy":  # extreme values
        data = {"a": rng.standard_cauchy(n), "b": rng.lognormal(0, 2, n)}
    elif kind == "trend":  # always growing: every new value is the largest so far
        data = {
            "a": np.arange(n, dtype=float),
            "b": rng.normal(0, 1, n) + np.arange(n) / 50,
        }
    else:
        raise ValueError(kind)
    df = pd.DataFrame(data)
    if kind != "trend":  # sprinkle missing values
        for c in df.columns:
            df.loc[rng.random(n) < 0.03, c] = np.nan
    return df


@pytest.mark.parametrize("kind", ["normal", "ties", "heavy", "trend"])
@pytest.mark.parametrize("window", [None, 150, 7])
@pytest.mark.parametrize("tail", [0.01, 0.05, 0.25])
def test_fast_equals_slow(kind, window, tail):
    seed = len(kind) * 100 + (window or 0) + int(tail * 1000)
    df = table(kind, 700, seed=seed)
    fast = walk_forward_scope(df, ["a", "b"], tail=tail, window=window, min_history=20)
    slow = slow_reference(df, ["a", "b"], tail=tail, window=window, min_history=20)
    pd.testing.assert_frame_equal(fast, slow)


@pytest.mark.parametrize("seed", range(12))
def test_fast_equals_slow_on_many_random_tables(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(120, 500))
    df = table(["normal", "ties", "heavy"][seed % 3], n, seed)
    window = [None, 40, 130][seed % 3]
    tail = [0.01, 0.02, 0.1][(seed // 3) % 3]
    fast = walk_forward_scope(df, ["a", "b"], tail=tail, window=window)
    slow = slow_reference(df, ["a", "b"], tail=tail, window=window)
    pd.testing.assert_frame_equal(fast, slow)


def test_default_min_history_matches():
    df = table("normal", 400, 99)
    pd.testing.assert_frame_equal(
        walk_forward_scope(df, ["a"]), slow_reference(df, ["a"])
    )


def test_one_column_and_tiny_table():
    df = table("normal", 30, 1)
    pd.testing.assert_frame_equal(
        walk_forward_scope(df, ["a"], min_history=5),
        slow_reference(df, ["a"], min_history=5),
    )


def test_all_rows_missing_in_one_column():
    df = table("normal", 300, 2)
    df["b"] = np.nan
    pd.testing.assert_frame_equal(
        walk_forward_scope(df, ["a", "b"], min_history=5),
        slow_reference(df, ["a", "b"], min_history=5),
    )


def test_a_long_table_runs_in_seconds():
    import time

    rng = np.random.default_rng(0)
    df = pd.DataFrame({c: rng.normal(0, 1, 50_000) for c in "abc"})
    start = time.perf_counter()
    out = walk_forward_scope(df, ["a", "b", "c"])
    assert time.perf_counter() - start < 10
    assert (out["scope"] != "unjudged").sum() == 49_900
