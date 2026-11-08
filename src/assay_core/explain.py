"""Explain and learn (engine 7): after misses, where do they concentrate, and why?

Two levels, both plain counting (no model of the cause):

1. Each miss: side, size in the user's units, whether the range was borrowed, scope flag,
   misses just before it, and which input looked most unusual for that case.
2. Groups: miss rate with a 95% interval for each condition, each input band, and each
   time period. A group is "WORSE than claimed" when even the low end of its interval
   is above the claimed miss rate.

Many groups are tested at once, so some would look worse by chance. The intervals are
widened by the number of groups (Bonferroni). That is conservative: a real, mild problem
can be missed. A cluster of misses is a lead, not proof of the cause.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import NormalDist
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from assay_core.front import Report

MISSING = "missing"


def _wilson(k: int, n: int, z: float) -> tuple[float, float]:
    """Wilson score interval for a share k / n (Brown, Cai and DasGupta, 2001)."""
    if n == 0:
        return math.nan, math.nan
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def _verdict(k: int, n: int, expected: float, z: float, min_cases: int) -> str:
    if n < min_cases:
        return "few cases"
    low, high = _wilson(k, n, z)
    if low > expected:
        return "WORSE than claimed"
    if high < expected:
        return "tighter than needed"
    return "ok"


def _table(key: pd.Series, miss: pd.Series, size: pd.Series, expected, z, min_cases):
    g = pd.DataFrame({"key": key.astype(str), "miss": miss, "size": size}).groupby(
        "key"
    )
    out = pd.DataFrame({"cases": g.size(), "misses": g["miss"].sum().astype(int)})
    out["miss_rate"] = out["misses"] / out["cases"]
    bounds = [
        _wilson(int(m), int(n), z)
        for m, n in zip(out["misses"], out["cases"], strict=True)
    ]
    out["miss_low"] = [b[0] for b in bounds]
    out["miss_high"] = [b[1] for b in bounds]
    out["verdict"] = [
        _verdict(int(m), int(n), expected, z, min_cases)
        for m, n in zip(out["misses"], out["cases"], strict=True)
    ]
    out["miss_size_total"] = g["size"].sum()
    out["worst_miss"] = g["size"].max()
    return out


@dataclass(frozen=True)
class Explanation:
    expected_miss_rate: float
    groups_tested: int
    misses: pd.DataFrame  # one row per miss
    by_condition: pd.DataFrame  # one row per group of a category column
    by_input: pd.DataFrame  # one row per band of each input column
    by_period: pd.DataFrame  # one row per time period
    findings: list[str]

    def __str__(self) -> str:
        head = (
            f"{len(self.misses)} misses | claimed miss rate {self.expected_miss_rate:.0%} | "
            f"{self.groups_tested} groups tested (intervals widened for that)"
        )
        if not self.findings:
            return head + "\nNo group is worse than claimed."
        return head + "\n" + "\n".join(f"- {f}" for f in self.findings)


def explain(
    report: Report,
    df: pd.DataFrame,
    features: Sequence[str] | None = None,
    by: Sequence[str] | None = None,
    bins: int = 4,
    periods: int = 4,
    min_cases: int = 30,
    alpha: float = 0.05,
    extreme: float = 5.0,
) -> Explanation:
    """Explain the misses in a report from assess().

    df: the same table that was given to assess().
    features: numeric input columns to cut into bands (quartiles by default).
    by: extra category columns to break the misses down by (the condition used in assess
        is always included).
    extreme: an input is named for a miss only if its value is in the lowest or highest
        `extreme` percent of earlier cases (default 5). Otherwise the miss is left
        unexplained by inputs: naming the "most unusual" of ordinary values would mislead.
    """
    cols = report.columns
    if not cols:
        raise ValueError("this report has no column names; build it with assess()")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be between 0 and 1, got {alpha}")
    if not 0 < extreme < 50:
        raise ValueError(f"extreme must be between 0 and 50, got {extreme}")
    if bins < 2 or periods < 2:
        raise ValueError("bins and periods must be at least 2")
    sort_by = cols["sort_by"]
    ordered = df.sort_values(sort_by, kind="stable") if sort_by else df
    ordered = ordered.reset_index(drop=True)
    feats = list(features or [])
    extra = [c for c in (cols["condition"], *(by or [])) if c]
    missing = [c for c in [*feats, *extra, cols["actual"]] if c not in ordered.columns]
    if missing:
        raise ValueError(f"columns not in the table: {missing}")

    rows = report.rows.reset_index(drop=True)
    idx = rows["row"].to_numpy()
    actual = ordered[cols["actual"]].to_numpy(float)[idx]
    miss = ~rows["hit"].to_numpy(bool)
    low = rows["lower"].to_numpy(float)
    high = rows["upper"].to_numpy(float)
    point = rows["forecast"].to_numpy(float)

    below = ~np.isnan(low) & (actual < low)
    above = ~np.isnan(high) & (actual > high)
    size = np.where(below, low - actual, np.where(above, actual - high, 0.0))
    size = np.where(miss, size, 0.0)
    if report.side == "floor":
        gap = point - low
    elif report.side == "ceiling":
        gap = high - point
    else:
        gap = high - low

    expected = 1 - report.claimed
    n_groups = sum(ordered[c].nunique() for c in extra) + len(feats) * bins + periods
    z = NormalDist().inv_cdf(1 - alpha / (2 * max(n_groups, 1)))
    size_s = pd.Series(size)
    miss_s = pd.Series(miss)

    # ---- groups ----
    cond_tables = []
    for c in extra:
        key = pd.Series(ordered[c].to_numpy()[idx]).where(lambda s: s.notna(), MISSING)
        t = _table(key, miss_s, size_s, expected, z, min_cases)
        t.insert(0, "column", c)
        cond_tables.append(t.reset_index().rename(columns={"key": "group"}))
    by_condition = (
        pd.concat(cond_tables, ignore_index=True)
        if cond_tables
        else pd.DataFrame(columns=["column", "group", "cases", "misses", "miss_rate"])
    )

    input_tables = []
    for f in feats:
        values = pd.Series(ordered[f].to_numpy(float)[idx])
        band = pd.qcut(values, q=bins, duplicates="drop").astype(object)
        band = band.where(values.notna(), MISSING).astype(str)
        t = _table(band, miss_s, size_s, expected, z, min_cases)
        t.insert(0, "column", f)
        input_tables.append(t.reset_index().rename(columns={"key": "group"}))
    by_input = (
        pd.concat(input_tables, ignore_index=True)
        if input_tables
        else pd.DataFrame(columns=["column", "group", "cases", "misses", "miss_rate"])
    )

    chunk = np.empty(len(rows), dtype=int)
    for p, part in enumerate(np.array_split(np.arange(len(rows)), periods)):
        chunk[part] = p + 1
    if sort_by:
        stamp = ordered[sort_by].to_numpy()[idx]
        names = {
            p: f"{pd.Timestamp(stamp[chunk == p][0]).date()} to {pd.Timestamp(stamp[chunk == p][-1]).date()}"
            if np.issubdtype(np.asarray(stamp).dtype, np.datetime64)
            else f"{stamp[chunk == p][0]} to {stamp[chunk == p][-1]}"
            for p in range(1, periods + 1)
        }
    else:
        names = {
            p: f"rows {idx[chunk == p][0]} to {idx[chunk == p][-1]}"
            for p in range(1, periods + 1)
        }
    period_key = pd.Series([f"{p}: {names[p]}" for p in chunk])
    by_period = _table(period_key, miss_s, size_s, expected, z, min_cases)
    by_period = by_period.reset_index().rename(columns={"key": "group"})

    # ---- one row per miss ----
    prior = np.array([miss[max(0, i - 5) : i].sum() for i in range(len(rows))])
    records = []
    for i in np.flatnonzero(miss):
        rec = {
            "row": int(idx[i]),
            "when": ordered[sort_by].iloc[idx[i]] if sort_by else None,
            "side": "below" if below[i] else "above",
            "miss_size": float(size[i]),
            "miss_vs_gap": float(size[i] / gap[i]) if gap[i] > 0 else math.nan,
            "borrowed": bool(rows["borrowed"].iloc[i]),
            "scope": rows["scope"].iloc[i] if "scope" in rows else None,
            "misses_in_last_5": int(prior[i]),
        }
        for c in extra:
            rec[c] = ordered[c].iloc[idx[i]]
        if feats:
            best = (-1.0, None, math.nan, math.nan)
            for f in feats:
                col = ordered[f].to_numpy(float)
                x = col[idx[i]]
                earlier = col[: idx[i]]
                earlier = earlier[~np.isnan(earlier)]
                if np.isnan(x):
                    score, pct = 100.0, math.nan
                elif len(earlier) >= 20:
                    pct = float((earlier <= x).mean() * 100)
                    score = abs(pct - 50) * 2
                else:
                    continue
                if score > best[0]:
                    best = (score, f, x, pct)
            if best[0] >= 100 - 2 * extreme:
                rec["unusual_input"], rec["unusual_value"], rec["unusual_pct"] = best[
                    1:
                ]
            else:
                rec["unusual_input"], rec["unusual_value"], rec["unusual_pct"] = (
                    None,
                    math.nan,
                    math.nan,
                )
        records.append(rec)
    misses = pd.DataFrame(records)

    # ---- plain-language findings ----
    findings: list[tuple[float, str]] = []

    def add(table: pd.DataFrame, kind: str, advice: str) -> None:
        for _, r in table.iterrows():
            if r.get("verdict") != "WORSE than claimed":
                continue
            strength = (r["miss_rate"] - expected) / math.sqrt(
                expected * (1 - expected) / r["cases"]
            )
            what = (
                f"{kind} {r['column']} = {r['group']}"
                if "column" in r
                else f"{kind} {r['group']}"
            )
            findings.append(
                (
                    strength,
                    (
                        f"{what}: {int(r['misses'])} misses in {int(r['cases'])} cases "
                        f"({r['miss_rate']:.0%} against {expected:.0%} claimed; "
                        f"total miss size {r['miss_size_total']:.4g}). {advice}"
                    ),
                )
            )

    add(
        by_condition,
        "group",
        "Its own errors still vary: look at the input bands for this group.",
    )
    add(
        by_input,
        "input band",
        "Consider using this input as the condition, or treat it as out of scope.",
    )
    add(
        by_period,
        "period",
        "Errors changed over time: try a shorter window or the adaptive method.",
    )
    if len(misses) >= 20:
        share = float((misses["misses_in_last_5"] >= 1).mean())
        indep = 1 - (1 - expected) ** 5
        if share > indep + 0.15:
            findings.append(
                (
                    0.0,
                    (
                        f"misses come in streaks: {share:.0%} of misses had another miss in the 5 cases "
                        f"before, against about {indep:.0%} if misses were independent. "
                        "The errors are not independent, so the real hit rate is less certain "
                        "than the intervals show."
                    ),
                )
            )
    findings.sort(key=lambda t: -t[0])
    return Explanation(
        expected_miss_rate=expected,
        groups_tested=n_groups,
        misses=misses,
        by_condition=by_condition,
        by_input=by_input,
        by_period=by_period,
        findings=[t[1] for t in findings],
    )
