"""The front door: two calls an engineer can use on their own table.

    report = assess(df, forecast="f", actual="a", condition="mode", sort_by="time")
    print(report)                      # honest hit rate per condition, no look-ahead

    model = fit(df, forecast="f", actual="a", condition="mode", side="floor")
    lower, upper, borrowed = model.range(point=120.0, condition="calm")
    # side="floor": upper is None. side="ceiling": lower is None. Default "both".

Units are the user's units. Nothing here knows what the numbers mean.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from assay_core.conditional import (
    GroupRange,
    conditional_bounds,
    range_for,
    walk_forward_conditional,
)

ALL = "all cases"


@dataclass(frozen=True)
class Report:
    claimed: float
    rows: pd.DataFrame  # one row per case judged: lower, upper, hit, borrowed
    by_condition: pd.DataFrame  # cases, hit_rate, mean_gap, borrowed_share
    side: str = "both"

    @property
    def overall_hit_rate(self) -> float:
        return float(self.rows["hit"].mean())

    def __str__(self) -> str:
        t = self.by_condition.round(3).to_string()
        kind = {"both": "range", "floor": "floor", "ceiling": "ceiling"}[self.side]
        return (
            f"claimed {self.claimed:.0%} {kind} | judged {len(self.rows)} cases | "
            f"actual hit rate {self.overall_hit_rate:.1%}\n{t}"
        )


@dataclass(frozen=True)
class Model:
    claimed: float
    groups: dict[str, GroupRange]
    side: str = "both"

    def range(
        self, point: float, condition: object = None
    ) -> tuple[float | None, float | None, bool]:
        """(lower, upper, borrowed) for a new forecast. None means no limit on that side.
        borrowed=True: too few cases of this condition, so the pooled limits were used."""
        if condition is None:
            g = self.groups["__pooled__"]
        else:
            g = range_for(self.groups, condition)
        lower = None if g.lower is None else point + g.lower
        upper = None if g.upper is None else point + g.upper
        return lower, upper, g.borrowed


def _prepare(df: pd.DataFrame, condition: str | None) -> tuple[pd.DataFrame, str]:
    if condition is None:
        return df.assign(**{ALL: ALL}), ALL
    return df, condition


def assess(
    df: pd.DataFrame,
    forecast: str,
    actual: str,
    condition: str | None = None,
    sort_by: str | None = None,
    coverage: float = 0.90,
    window: int | None = None,
    min_group: int = 0,
    side: str = "both",
) -> Report:
    """Walk through the table in time order. Each case gets a range built only from earlier
    cases. Then count how often the actual value landed inside (or above the floor, or below
    the ceiling, when side is "floor" or "ceiling")."""
    d, col = _prepare(df, condition)
    rows = walk_forward_conditional(
        d, forecast, actual, col, coverage, sort_by, window, min_group, side
    )
    if rows.empty:
        raise ValueError("not enough history to judge any case")
    if side == "floor":
        gap = rows["forecast"] - rows["lower"]
    elif side == "ceiling":
        gap = rows["upper"] - rows["forecast"]
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
    return Report(claimed=coverage, rows=rows, by_condition=table, side=side)


def fit(
    df: pd.DataFrame,
    forecast: str,
    actual: str,
    condition: str | None = None,
    coverage: float = 0.90,
    min_group: int = 0,
    side: str = "both",
) -> Model:
    """Learn the ranges from every case in df (use only cases whose outcome is known)."""
    d, col = _prepare(df, condition)
    errors = (d[actual] - d[forecast]).to_numpy(float)
    groups = conditional_bounds(errors, d[col].to_numpy(), coverage, min_group, side)
    return Model(coverage, groups, side)
