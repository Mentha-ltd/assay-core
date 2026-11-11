"""Look: say where a range loses score, so the user knows where to spend time."""

from __future__ import annotations

import numpy as np
import pandas as pd


def where_to_look(
    df: pd.DataFrame,
    actual: str,
    lower: str,
    upper: str,
    condition: str,
    day: str | None = None,
    claimed: float = 0.80,
    penalty: float = 10.0,
    confidence: float = 0.95,
    min_cases: int = 30,
    n_boot: int = 1000,
    seed: int = 0,
) -> pd.DataFrame:
    """Rank conditions by score points lost above average, counting days not rows.

    Row score = (upper - lower) + penalty * distance outside (Predico range score,
    penalty 10 for the 80% range). Lower is better.
    day: column naming the independent case (for example the delivery date). The 96
    slots of one day move together, so they count as one case. None = each row is a case.
    cases: independent cases (days) in the condition.
    hit_low / hit_high: day-resampled interval for the hit rate.
    excess: points the condition costs above the average case (sums to zero).
    advice: 'too narrow' if the interval is below claimed, 'too wide' if above,
    'ok' if it contains claimed, 'not enough data' under min_cases.
    """
    a, lo, hi = (
        df[actual].to_numpy(float),
        df[lower].to_numpy(float),
        df[upper].to_numpy(float),
    )
    if np.isnan(a).any() or np.isnan(lo).any() or np.isnan(hi).any():
        raise ValueError("missing values: drop or fill them first")
    if (lo > hi).any():
        raise ValueError("lower is above upper in some rows")
    miss = np.maximum(lo - a, 0.0) + np.maximum(a - hi, 0.0)
    work = pd.DataFrame(
        {
            "g": df[condition].to_numpy(),
            "d": df[day].to_numpy() if day else np.arange(len(df)),
            "width": hi - lo,
            "score": (hi - lo) + penalty * miss,
            "hit": (miss == 0).astype(float),
        }
    )
    cell = work.groupby(["g", "d"]).agg(score=("score", "mean"), hit=("hit", "mean"))
    overall = cell["score"].mean()
    rng = np.random.default_rng(seed)
    q = [(1 - confidence) / 2, 1 - (1 - confidence) / 2]
    rows = {}
    for g, c in cell.groupby(level=0):
        h = c["hit"].to_numpy()
        boots = h[rng.integers(0, len(h), size=(n_boot, len(h)))].mean(axis=1)
        low, high = np.quantile(boots, q)
        rows[g] = {
            "cases": len(h),
            "hit_rate": h.mean(),
            "hit_low": low,
            "hit_high": high,
            "mean_score": c["score"].mean(),
            "excess": (c["score"] - overall).sum(),
        }
    out = pd.DataFrame.from_dict(rows, orient="index")
    parts = work.groupby("g")
    out["width_share"] = parts["width"].sum() / parts["score"].sum()
    out["advice"] = np.select(
        [out["cases"] < min_cases, out["hit_high"] < claimed, out["hit_low"] > claimed],
        ["not enough data", "too narrow", "too wide"],
        default="ok",
    )
    return out.sort_values("excess", ascending=False)
