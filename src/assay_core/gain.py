"""Does a new input make the range narrower at the same hit rate?

    g = information_gain(history, "forecast", "actual", inputs=["wind", "temp"],
                         sort_by="time", claimed=0.90)
    print(g)          # one line per input, narrower or not
    g.table           # the numbers

How it works. The history is cut in time: the early part builds ranges, the late part
judges them. Range A uses all early errors together. Range B splits the early errors
into bands of the input (equal-count bands for numbers, one band per value for text)
and gives each band its own range. Both are built to hold the same share of cases.
On the late part we measure the average width and the hit rate of each.

An input is called helpful only if B is clearly narrower (the 95% interval, widened for
the number of inputs tried, sits below zero and the change is at least min_gain) AND B
did not buy that width by missing more often. Narrower and worse hit rate is not a gain.

Limits: this finds leads, not causes. A lead found here is a candidate for a physical
check. It assumes late cases are independent for the interval, so streaks make it
look more certain than it is."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Gain:
    claimed: float
    fit_cases: int
    test_cases: int
    table: pd.DataFrame
    findings: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        head = (
            f"claimed {self.claimed:.0%} | {self.fit_cases} early cases build the ranges, "
            f"{self.test_cases} late cases judge them"
        )
        return "\n".join([head, *[f"- {f}" for f in self.findings]])


def _range(err: np.ndarray, claimed: float) -> tuple[float, float]:
    a = 1.0 - claimed
    return float(np.quantile(err, a / 2)), float(np.quantile(err, 1 - a / 2))


def _bands(x_fit: pd.Series, x_test: pd.Series, bins: int, min_cases: int):
    """Band number per early and late case. -1 means the late case fits no band.
    Returns None when no split with at least min_cases early cases per band exists."""
    if pd.api.types.is_numeric_dtype(x_fit):
        k = bins
        while k >= 2:
            edges = np.unique(np.quantile(x_fit, np.linspace(0, 1, k + 1)[1:-1]))
            fit_b = np.searchsorted(edges, x_fit.to_numpy(), side="right")
            counts = np.bincount(fit_b, minlength=len(edges) + 1)
            if len(edges) >= 1 and counts.min() >= min_cases:
                return fit_b, np.searchsorted(edges, x_test.to_numpy(), side="right"), k
            k -= 1
        return None
    codes = {v: i for i, v in enumerate(sorted(x_fit.unique(), key=str))}
    fit_b = x_fit.map(codes).to_numpy()
    counts = np.bincount(fit_b)
    if len(codes) < 2 or counts.min() < min_cases:
        keep = [v for v, i in codes.items() if counts[i] >= min_cases]
        if len(keep) < 2:
            return None
        codes = {v: i for i, v in enumerate(keep)}
        fit_b = x_fit.map(codes).fillna(-1).astype(int).to_numpy()
    test_b = x_test.map(codes).fillna(-1).astype(int).to_numpy()
    return fit_b, test_b, len(codes)


def information_gain(
    df: pd.DataFrame,
    forecast: str,
    actual: str,
    inputs: Sequence[str],
    sort_by: str | None = None,
    claimed: float = 0.90,
    bins: int = 4,
    min_cases: int = 30,
    test_share: float = 0.4,
    min_gain: float = 0.05,
    tolerance: float = 0.02,
    confidence: float = 0.95,
    n_boot: int = 1000,
    seed: int = 0,
) -> Gain:
    """Compare a pooled range with a per-band range for each input, on late data.

    min_gain: smallest narrowing that counts (0.05 is 5% narrower). Design setting.
    tolerance: how far the hit rate of the narrower range may fall below the pooled
    range's hit rate. Design setting."""
    if not 0.0 < claimed < 1.0:
        raise ValueError(f"claimed must be between 0 and 1, got {claimed}")
    if not 0.0 < test_share < 1.0:
        raise ValueError(f"test_share must be between 0 and 1, got {test_share}")
    if not inputs:
        raise ValueError("give at least one input column")
    absent = [c for c in [forecast, actual, *inputs] if c not in df.columns]
    if absent:
        raise ValueError(f"columns not in the table: {absent}")
    d = df.sort_values(sort_by, kind="stable") if sort_by else df
    d = d.dropna(subset=[forecast, actual]).reset_index(drop=True)
    cut = round(len(d) * (1 - test_share))
    early, late = d.iloc[:cut], d.iloc[cut:]
    if len(early) < 2 * min_cases or len(late) < min_cases:
        raise ValueError(
            f"need at least {2 * min_cases} early and {min_cases} late cases, "
            f"have {len(early)} and {len(late)}"
        )
    e_err = (early[actual] - early[forecast]).to_numpy(float)
    l_err = (late[actual] - late[forecast]).to_numpy(float)
    lo0, hi0 = _range(e_err, claimed)
    base_width = hi0 - lo0
    rng = np.random.default_rng(seed)
    level = 1 - (1 - confidence) / len(inputs)  # widened for the number of inputs tried
    rows, findings = [], []
    for name in inputs:
        e_ok, l_ok = early[name].notna().to_numpy(), late[name].notna().to_numpy()
        found = _bands(early.loc[e_ok, name], late.loc[l_ok, name], bins, min_cases)
        if found is None:
            rows.append({"input": name, "verdict": "not enough data"})
            findings.append(
                f"{name}: not enough early cases to split into bands of {min_cases} or more."
            )
            continue
        fb, tb, used = found
        e_use = e_err[e_ok]
        edges = {}
        for b in np.unique(fb[fb >= 0]):
            edges[int(b)] = _range(e_use[fb == b], claimed)
        t_err = l_err[l_ok]
        lo = np.array([edges.get(int(b), (lo0, hi0))[0] for b in tb])
        hi = np.array([edges.get(int(b), (lo0, hi0))[1] for b in tb])
        width = hi - lo
        hit = float(np.mean((t_err >= lo) & (t_err <= hi)))
        pooled_hit = float(np.mean((t_err >= lo0) & (t_err <= hi0)))
        rel = width / base_width - 1.0
        idx = rng.integers(0, len(rel), size=(n_boot, len(rel)))
        boot = rel[idx].mean(axis=1)
        low_ci, high_ci = np.quantile(boot, [(1 - level) / 2, 1 - (1 - level) / 2])
        change = float(rel.mean())
        narrower = high_ci < 0 and change <= -min_gain
        same_hit = hit >= pooled_hit - tolerance
        if narrower and same_hit:
            verdict = "narrower, same hit rate"
        elif narrower:
            verdict = "narrower but hits less often: not a gain"
        else:
            verdict = "no gain"
        rows.append(
            {
                "input": name,
                "bands": used,
                "width_pooled": base_width,
                "width_with_input": float(width.mean()),
                "change": change,
                "change_low": float(low_ci),
                "change_high": float(high_ci),
                "hit_pooled": pooled_hit,
                "hit_with_input": hit,
                "verdict": verdict,
            }
        )
        if verdict.startswith("narrower, same"):
            findings.append(
                f"{name}: range {-change:.0%} narrower on late cases "
                f"(width {base_width:.4g} to {float(width.mean()):.4g}), hit rate "
                f"{pooled_hit:.0%} to {hit:.0%}. A lead worth a physical check, not a proven cause."
            )
        elif verdict.startswith("narrower but"):
            findings.append(
                f"{name}: narrower by {-change:.0%} but the hit rate fell from "
                f"{pooled_hit:.0%} to {hit:.0%}. That is false confidence, not information."
            )
        else:
            findings.append(f"{name}: no clear gain (change {change:+.0%}).")
    table = pd.DataFrame(rows).set_index("input")
    return Gain(
        claimed=claimed,
        fit_cases=len(early),
        test_cases=len(late),
        table=table,
        findings=findings,
    )
