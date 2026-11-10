"""Grade a range that someone else stated.

A vendor, a model or an agent gives a lower and an upper limit for each case, with a
claimed share of cases they should hold (80% for a P10 to P90 band, 90% for P5 to P95).
assess() builds its own ranges from a point forecast. audit() takes the stated ones as
they are and counts how often the actual value landed inside.

    report = audit(df, forecast="p50", actual="load", lower="p10", upper="p90",
                   claimed=0.80, condition="site", sort_by="time")
    print(report)               # hit rate per condition, with a 95% interval
    print(report.explain(df))   # where and why the misses happened

Nothing is learned here, so there is no look-ahead to worry about: each limit was stated
before the outcome. Give claimed as the share the source says (P10 to P90 is 0.80).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from assay_core.autonomy import accuracy_range
from assay_core.front import ALL, Report
from assay_core.scope import walk_forward_scope


def _side(lower: str | None, upper: str | None) -> str:
    if lower is None and upper is None:
        raise ValueError("give lower, upper or both")
    if lower is None:
        return "ceiling"
    if upper is None:
        return "floor"
    return "both"


def audit(
    df: pd.DataFrame,
    forecast: str,
    actual: str,
    lower: str | None = None,
    upper: str | None = None,
    condition: str | None = None,
    sort_by: str | None = None,
    claimed: float = 0.90,
    features: Sequence[str] | None = None,
    tail: float = 0.01,
    window: int | None = None,
    missing: str = "drop",
    confidence: float = 0.95,
) -> Report:
    """Count how often the actual value landed inside the stated range.

    forecast: the point forecast (for example the P50). It is not graded. It is used to
    show the gap and to explain the misses.
    lower, upper: columns with the stated limits. Give one for a floor or a ceiling.
    claimed: the share of cases the stated range should hold.
    features: input columns. If given, each case is also judged in or out of scope.
    missing: rows with a missing forecast or actual are dropped ("drop") or refused
    ("refuse"). A missing or crossed limit is always refused: fix those rows first."""
    if missing not in ("drop", "refuse"):
        raise ValueError(f'missing must be "drop" or "refuse", got {missing!r}')
    if not 0.0 < claimed < 1.0:
        raise ValueError(f"claimed must be between 0 and 1, got {claimed}")
    side = _side(lower, upper)
    limits = [c for c in (lower, upper) if c is not None]
    needed = [forecast, actual, *limits, *([condition] if condition else [])]
    absent = [c for c in needed if c not in df.columns]
    if absent:
        raise ValueError(f"columns not in the table: {absent}")
    gaps = df[[forecast, actual]].isna().any(axis=1)
    dropped = int(gaps.sum())
    if dropped and missing == "refuse":
        raise ValueError(f"{dropped} rows have a missing forecast or actual")
    d = df.loc[~gaps]
    d = d.sort_values(sort_by, kind="stable") if sort_by else d
    d = d.reset_index(drop=True)
    if d.empty:
        raise ValueError("no rows to judge")
    for c in limits:
        n_bad = int(d[c].isna().sum())
        if n_bad:
            raise ValueError(f"{n_bad} rows have a missing {c}: fix or drop them first")
    if side == "both":
        crossed = int((d[lower] > d[upper]).sum())
        if crossed:
            raise ValueError(
                f"{crossed} rows have {lower} above {upper}: sort the quantiles"
            )
    col = condition if condition else ALL
    shown = d[condition].to_numpy() if condition else np.full(len(d), ALL, dtype=object)
    y = d[actual].to_numpy(float)
    low = d[lower].to_numpy(float) if lower else np.full(len(d), np.nan)
    high = d[upper].to_numpy(float) if upper else np.full(len(d), np.nan)
    hit = np.ones(len(d), dtype=bool)
    if lower:
        hit &= y >= low
    if upper:
        hit &= y <= high
    rows = pd.DataFrame(
        {
            "row": np.arange(len(d)),
            col: shown,
            "forecast": d[forecast].to_numpy(float),
            "lower": low,
            "upper": high,
            "borrowed": False,
            "n_used": np.nan,
            "hit": hit,
        }
    )
    point = rows["forecast"]
    if side == "floor":
        gap = point - rows["lower"]
    elif side == "ceiling":
        gap = rows["upper"] - point
    else:
        gap = rows["upper"] - rows["lower"]
    g = rows.assign(gap=gap).groupby(col)
    table = pd.DataFrame(
        {
            "cases": g.size(),
            "hit_rate": g["hit"].mean(),
            "mean_gap": g["gap"].mean(),
            "borrowed_share": g["borrowed"].mean(),
        }
    )
    bounds = [
        accuracy_range(int(h), int(n), confidence)
        for h, n in zip(g["hit"].sum(), table["cases"], strict=True)
    ]
    table["hit_low"] = [b[0] for b in bounds]
    table["hit_high"] = [b[1] for b in bounds]
    by_scope = None
    if features:
        judged = walk_forward_scope(d, features, tail=tail, window=window)
        rows = rows.assign(
            scope=judged.loc[rows["row"], "scope"].to_numpy(),
            reasons=judged.loc[rows["row"], "reasons"].to_numpy(),
        )
        sg = rows.groupby("scope")["hit"]
        by_scope = pd.DataFrame({"cases": sg.size(), "hit_rate": sg.mean()})
    columns = {
        "forecast": forecast,
        "actual": actual,
        "condition": condition,
        "sort_by": sort_by,
    }
    return Report(
        claimed=claimed,
        rows=rows,
        by_condition=table,
        side=side,
        by_scope=by_scope,
        columns=columns,
        dropped=dropped,
    )
