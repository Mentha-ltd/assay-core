"""Ranges that depend on the condition (engine 3, per-condition version).

One pooled range mixes easy and hard cases: it is too wide for the easy ones and too narrow
for the hard ones. Here each condition (any label: day type, weather band, machine mode) gets
a range learned only from its own past errors (a "Mondrian" conformal range).

A condition with too few past errors borrows the pooled range and says so (borrowed=True).
side="floor" or "ceiling" gives one limit instead of two.
Nothing here knows what the numbers mean: units are whatever the user's columns use.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass

import numpy as np
import pandas as pd

from assay_core.bounds import asymmetric_bounds, min_errors, tail_rank
from assay_core.onesided import SIDES, ceiling_limit, floor_limit, min_errors_one_side


@dataclass(frozen=True)
class GroupRange:
    lower: float | None  # signed error limits: add to the forecast to get the range
    upper: float | None  # None: no limit on that side (floor or ceiling only)
    n_used: int  # past errors the limits were learned from
    borrowed: bool  # True: too few of its own errors, so the pooled limits were used


def _need(coverage: float, side: str) -> int:
    if side not in SIDES:
        raise ValueError(f"side must be one of {SIDES}, got {side!r}")
    return min_errors(coverage) if side == "both" else min_errors_one_side(coverage)


def _limits(
    errors: np.ndarray, coverage: float, side: str
) -> tuple[float | None, float | None]:
    if side == "floor":
        return floor_limit(errors, coverage), None
    if side == "ceiling":
        return None, ceiling_limit(errors, coverage)
    return asymmetric_bounds(errors, coverage)


def _signed_errors(df: pd.DataFrame, forecast: str, actual: str) -> np.ndarray:
    return (df[actual] - df[forecast]).to_numpy(float)


def conditional_bounds(
    errors: np.ndarray,
    conditions: np.ndarray,
    coverage: float = 0.90,
    min_group: int = 0,
    side: str = "both",
) -> dict[str, GroupRange]:
    """One range per condition, plus the key '__pooled__' for all cases together.

    A condition needs at least max(minimum for the side, min_group) errors of its own;
    otherwise it takes the pooled limits (borrowed=True)."""
    errors = np.asarray(errors, dtype=float)
    labels = np.asarray(conditions).astype(str)
    if len(errors) != len(labels):
        raise ValueError("errors and conditions must have the same length")
    need = max(_need(coverage, side), min_group)
    pooled_lo, pooled_hi = _limits(errors, coverage, side)
    out = {"__pooled__": GroupRange(pooled_lo, pooled_hi, len(errors), False)}
    for label in sorted(set(labels)):
        own = errors[labels == label]
        if len(own) >= need:
            lo, hi = _limits(own, coverage, side)
            out[label] = GroupRange(lo, hi, len(own), False)
        else:
            out[label] = GroupRange(pooled_lo, pooled_hi, len(own), True)
    return out


def range_for(fitted: dict[str, GroupRange], condition: object) -> GroupRange:
    """The range for one condition. An unseen condition gets the pooled range, borrowed."""
    found = fitted.get(str(condition))
    if found is not None and str(condition) != "__pooled__":
        return found
    p = fitted["__pooled__"]
    return GroupRange(p.lower, p.upper, 0, True)


def _pick(
    sorted_errors: list[float], coverage: float, side: str
) -> tuple[float | None, float | None]:
    """The same limits as _limits(), read from an already sorted list of past errors."""
    n = len(sorted_errors)
    if side == "both":
        k = tail_rank(n, coverage)
        return sorted_errors[n - k], sorted_errors[k - 1]
    k = tail_rank(n, 2 * coverage - 1)
    if side == "floor":
        return sorted_errors[n - k], None
    return None, sorted_errors[k - 1]


def walk_forward_conditional(
    df: pd.DataFrame,
    forecast: str,
    actual: str,
    condition: str,
    coverage: float = 0.90,
    sort_by: str | None = None,
    window: int | None = None,
    min_group: int = 0,
    side: str = "both",
) -> pd.DataFrame:
    """For each row, build the range from earlier rows only (no look-ahead).

    Returns the rows in time order with: forecast, lower, upper (in the user's units; NaN
    where a floor or ceiling has no limit), borrowed, n_used, hit.
    Rows with too little history for even the pooled range are dropped.

    Each condition keeps a sorted list of its past errors, and one more list holds all of
    them: one insert (and one removal when `window` is set) per row, so a long table takes
    seconds. The numbers are identical to sorting the past again for every row."""
    d = df.sort_values(sort_by, kind="stable") if sort_by else df
    d = d.reset_index(drop=True)
    err = _signed_errors(d, forecast, actual)
    if np.isnan(err).any():
        raise ValueError(
            f"{int(np.isnan(err).sum())} rows have a missing forecast or actual: drop them first"
        )
    labels = d[condition].astype(str).to_numpy()
    shown = d[condition].to_numpy()
    points = d[forecast].to_numpy(float)
    actuals = d[actual].to_numpy(float)
    base = _need(coverage, side)
    need = max(base, min_group)
    pooled: list[float] = []
    own_lists: dict[str, list[float]] = {}
    out: dict[str, list] = {
        k: []
        for k in (
            "row",
            condition,
            "forecast",
            "lower",
            "upper",
            "borrowed",
            "n_used",
            "hit",
        )
    }
    for i in range(len(d)):
        if i > 0:
            new = i - 1
            bisect.insort(pooled, err[new])
            bisect.insort(own_lists.setdefault(labels[new], []), err[new])
            old = new - window if window is not None else -1
            if old >= 0:
                del pooled[bisect.bisect_left(pooled, err[old])]
                mine = own_lists[labels[old]]
                del mine[bisect.bisect_left(mine, err[old])]
        if len(pooled) < base:
            continue
        own = own_lists.get(labels[i], [])
        borrowed = len(own) < need
        group = pooled if borrowed else own
        lo, hi = _pick(group, coverage, side)
        point = float(points[i])
        y = float(actuals[i])
        low = np.nan if lo is None else point + lo
        high = np.nan if hi is None else point + hi
        out["row"].append(i)
        out[condition].append(shown[i])
        out["forecast"].append(point)
        out["lower"].append(low)
        out["upper"].append(high)
        out["borrowed"].append(borrowed)
        out["n_used"].append(len(group))
        out["hit"].append(bool((lo is None or y >= low) and (hi is None or y <= high)))
    return pd.DataFrame(out)
