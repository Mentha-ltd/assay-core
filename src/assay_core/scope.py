"""Scope check: is this case unlike the cases the range was learned from?

A range is only promised for cases that look like the past. Here "look like" is simple:
each named input column must lie between its own 1st and 99th percentile of the past
(the tail share is a choice, not a theorem). A case outside is flagged, with the reason.

Known blind spot: each column is checked alone. An unusual COMBINATION of normal values
passes. Nothing here knows what the columns mean.
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd


def _check_tail(tail: float) -> None:
    if isinstance(tail, bool) or not 0 < tail < 0.5:
        raise ValueError(f"tail must be between 0 and 0.5, got {tail}")


def _columns(df: pd.DataFrame, features: Sequence[str]) -> list[str]:
    cols = list(features)
    if not cols:
        raise ValueError("name at least one input column")
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"input columns not in the table: {missing}")
    return cols


@dataclass(frozen=True)
class Scope:
    low: dict[str, float]  # tail-quantile of each column over the learning cases
    high: dict[str, float]
    tail: float
    n: int  # learning cases

    def check(self, df: pd.DataFrame) -> pd.DataFrame:
        """For each row: in_scope (bool) and reasons (text, empty when in scope)."""
        cols = _columns(df, list(self.low))
        reasons = pd.Series([""] * len(df), index=df.index, dtype=object)
        for c in cols:
            v = df[c].to_numpy(float)
            for mask, text in (
                (np.isnan(v), f"{c} is missing"),
                (v < self.low[c], f"{c} below learned {self.low[c]:.4g}"),
                (v > self.high[c], f"{c} above learned {self.high[c]:.4g}"),
            ):
                hit = pd.Series(mask, index=df.index)
                reasons = reasons.where(
                    ~hit, reasons + np.where(reasons == "", "", "; ") + text
                )
        return pd.DataFrame({"in_scope": reasons == "", "reasons": reasons})


def learn_scope(df: pd.DataFrame, features: Sequence[str], tail: float = 0.01) -> Scope:
    """Learn the normal range of each column from the learning cases."""
    _check_tail(tail)
    cols = _columns(df, features)
    need = round(1 / tail)
    if len(df) < need:
        raise ValueError(
            f"need at least {need} learning cases for tail {tail}, got {len(df)}"
        )
    data = df[cols].to_numpy(float)
    if np.isnan(data).any():
        raise ValueError("learning cases contain missing values in the input columns")
    return Scope(
        low={c: float(np.quantile(data[:, j], tail)) for j, c in enumerate(cols)},
        high={c: float(np.quantile(data[:, j], 1 - tail)) for j, c in enumerate(cols)},
        tail=tail,
        n=len(df),
    )


def _quantile(sorted_values: list[float], q: float) -> float:
    """Same number as np.quantile(values, q) (linear method), from an already sorted list."""
    m = len(sorted_values)
    virtual = (m - 1) * q
    prev = math.floor(virtual)
    if virtual >= m - 1:  # numpy clips both indexes to the last value
        prev = m - 1
        nxt = m - 1
    else:
        nxt = prev + 1
    gamma = virtual - prev
    a = sorted_values[prev]
    b = sorted_values[nxt]
    diff = b - a
    # numpy's own lerp: adds from the left below the midpoint, subtracts from the right above
    return b - diff * (1 - gamma) if gamma >= 0.5 else a + diff * gamma


def walk_forward_scope(
    df: pd.DataFrame,
    features: Sequence[str],
    tail: float = 0.01,
    min_history: int | None = None,
    window: int | None = None,
) -> pd.DataFrame:
    """Judge each row against the rows BEFORE it only (rows must already be in time order).

    Returns one row per input row: scope ("in", "out" or "unjudged"), reasons.
    A row is "unjudged" while fewer than min_history earlier rows exist
    (default: 1 / tail, so 100 for a 1% tail).

    Rows with a missing value in any input column are not used to learn the range.
    Each column keeps a sorted list of its past values: one insert (and one removal when
    `window` is set) per row, so a long table takes seconds. The numbers are identical to
    recomputing the percentiles from scratch for every row."""
    _check_tail(tail)
    cols = _columns(df, features)
    need = round(1 / tail) if min_history is None else min_history
    data = df[cols].to_numpy(float)
    n = len(data)
    complete = ~np.isnan(data).any(axis=1)
    lists: list[list[float]] = [[] for _ in cols]
    scope, why = [], []
    for i in range(n):
        if i > 0:
            new = i - 1
            if complete[new]:
                for j in range(len(cols)):
                    bisect.insort(lists[j], data[new, j])
            old = new - window if window is not None else -1
            if old >= 0 and complete[old]:
                for j in range(len(cols)):
                    del lists[j][bisect.bisect_left(lists[j], data[old, j])]
        if len(lists[0]) < need:
            scope.append("unjudged")
            why.append("")
            continue
        parts = []
        for j, c in enumerate(cols):
            x = data[i, j]
            if np.isnan(x):
                parts.append(f"{c} is missing")
                continue
            lo = _quantile(lists[j], tail)
            hi = _quantile(lists[j], 1 - tail)
            if x < lo:
                parts.append(f"{c} below learned {lo:.4g}")
            elif x > hi:
                parts.append(f"{c} above learned {hi:.4g}")
        scope.append("out" if parts else "in")
        why.append("; ".join(parts))
    return pd.DataFrame({"scope": scope, "reasons": why}, index=df.index)
