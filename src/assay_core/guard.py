"""Guard: check a day-ahead submission before the gate, and price what a failure costs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Check:
    """Result of check_submission.

    ok: True when the file can be sent (possibly after the repair).
    problems: plain-words list of what is wrong. Empty when clean.
    fixed: copy with each row sorted low <= mid <= high. None if not sendable.
    """

    ok: bool
    problems: list[str]
    fixed: pd.DataFrame | None


def check_submission(
    df: pd.DataFrame,
    slots: int = 96,
    cols: tuple[str, str, str] = ("q10", "q50", "q90"),
    history: pd.Series | None = None,
    margin: float = 0.5,
) -> Check:
    """Check one day's file: slot count, missing values, order, zero width, size.

    history: past actual values. If given, any forecast further than `margin` times
    the past range outside the past min/max is flagged (a likely unit or sign bug).
    """
    problems: list[str] = []
    if len(df) != slots:
        problems.append(f"{len(df)} rows, expected {slots}")
    missing = [c for c in cols if c not in df.columns]
    if missing:
        problems.append(f"missing columns: {missing}")
        return Check(False, problems, None)
    x = df[list(cols)].astype(float)
    bad = ~np.isfinite(x.to_numpy())
    if bad.any():
        problems.append(f"{int(bad.any(axis=1).sum())} rows have missing or infinite values")
        return Check(False, problems, None)
    v = x.to_numpy()
    crossed = int((np.diff(v, axis=1) < 0).any(axis=1).sum())
    if crossed:
        problems.append(f"{crossed} rows have quantiles in the wrong order (repaired by sorting)")
    fixed = pd.DataFrame(np.sort(v, axis=1), index=df.index, columns=list(cols))
    flat = int((fixed[cols[2]] == fixed[cols[0]]).sum())
    if flat:
        problems.append(f"{flat} rows have zero width: one miss costs the full penalty")
    if history is not None and len(history):
        lo, hi = float(history.min()), float(history.max())
        pad = margin * (hi - lo)
        out = int(((fixed < lo - pad) | (fixed > hi + pad)).any(axis=1).sum())
        if out:
            problems.append(f"{out} rows are far outside the past range: check units and sign")
    ok = len(df) == slots
    return Check(ok, problems, fixed if ok else None)


def fall_cost(prizes: list[float], rank: int, places: int = 1) -> float:
    """Euros lost per month if you drop `places` ranks. prizes[0] is rank 1; beyond the list pays 0."""

    def pay(r: int) -> float:
        return float(prizes[r - 1]) if 1 <= r <= len(prizes) else 0.0

    return pay(rank) - pay(rank + places)


def money_table(prizes: list[float], rank: int, width: int = 3) -> pd.DataFrame:
    """Euros per month at each rank near yours, and the change against your rank now."""
    ranks = range(max(1, rank - width), rank + width + 1)
    rows = [
        {
            "rank": r,
            "euros": (e := float(prizes[r - 1]) if r <= len(prizes) else 0.0),
            "change": e - (float(prizes[rank - 1]) if rank <= len(prizes) else 0.0),
        }
        for r in ranks
    ]
    return pd.DataFrame(rows)


def missed_days(prizes: list[float], rank: int, missed: int, allowance: int = 5) -> dict:
    """Days left before the league cliff, and the euros of league prize lost if you cross it."""
    prize = float(prizes[rank - 1]) if 1 <= rank <= len(prizes) else 0.0
    return {
        "days_left": max(0, allowance - missed),
        "at_cliff": missed >= allowance,
        "euros_at_stake": prize,
    }
