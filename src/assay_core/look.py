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
    claimed: float = 0.80,
    penalty: float = 10.0,
    tolerance: float = 0.03,
) -> pd.DataFrame:
    """Rank conditions by score points lost above the average row.

    Row score = (upper - lower) + penalty * distance outside (Predico range score,
    penalty 10 for the 80% range). Lower is better.
    excess: points this condition costs above what average rows would have cost.
    width_share: part of its own score that is width (high = maybe too cautious).
    advice: 'too narrow' / 'too wide' / 'ok' from hit rate against claimed.
    """
    a, lo, hi = df[actual].to_numpy(float), df[lower].to_numpy(float), df[upper].to_numpy(float)
    if np.isnan(a).any() or np.isnan(lo).any() or np.isnan(hi).any():
        raise ValueError("missing values: drop or fill them first")
    if (lo > hi).any():
        raise ValueError("lower is above upper in some rows")
    width = hi - lo
    miss = np.maximum(lo - a, 0.0) + np.maximum(a - hi, 0.0)
    work = pd.DataFrame(
        {
            "g": df[condition].to_numpy(),
            "width": width,
            "pen": penalty * miss,
            "hit": (miss == 0).astype(float),
        }
    )
    work["score"] = work["width"] + work["pen"]
    mean_all = work["score"].mean()
    g = work.groupby("g")
    out = pd.DataFrame(
        {
            "cases": g.size(),
            "hit_rate": g["hit"].mean(),
            "mean_score": g["score"].mean(),
            "width_share": g["width"].sum() / g["score"].sum(),
        }
    )
    out["excess"] = (out["mean_score"] - mean_all) * out["cases"]
    gap = out["hit_rate"] - claimed
    out["advice"] = np.where(
        gap < -tolerance, "too narrow", np.where(gap > tolerance, "too wide", "ok")
    )
    return out.sort_values("excess", ascending=False)
