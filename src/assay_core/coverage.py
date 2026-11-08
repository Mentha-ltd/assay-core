"""Check whether a claimed range covers what it claims.

check_coverage: the user already has lower and upper columns.
walk_forward_check: build a range from past errors only, then check it.

Error = actual - predicted, in the units of the user's data.
Assumption: rows are in time order and independent enough for a
hit-rate interval. Strong autocorrelation makes the interval too narrow.
Interval: Wilson score interval for a hit rate (95%).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from assay_core.bounds import asymmetric_bounds, min_errors

Z95 = 1.959964
UNDER = "undercovers"
OK = "consistent"
OVER = "wider than needed"
FEW = "too few rows"


def wilson_interval(hits: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Return a 95% interval for a hit rate of hits out of n."""
    if n < 1 or hits < 0 or hits > n:
        raise ValueError(f"need 0 <= hits <= n and n >= 1, got {hits}, {n}")
    p = hits / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


@dataclass(frozen=True)
class CoverageResult:
    n: int
    claimed: float
    real: float
    real_low: float
    real_high: float
    miss_above: float
    miss_below: float
    mean_width: float  # same units as the user's data
    verdict: str
    message: str


def _claimed(value: float) -> float:
    if isinstance(value, bool) or not math.isfinite(float(value)):
        raise ValueError("claimed must be a number strictly between 0 and 1")
    if not 0.0 < float(value) < 1.0:
        raise ValueError(f"claimed must be strictly between 0 and 1, got {value!r}")
    return float(value)


def _numbers(df: pd.DataFrame, columns: list[str]) -> np.ndarray:
    for name in columns:
        if name not in df.columns:
            raise ValueError(
                f"column {name!r} not found. Columns are: {list(df.columns)}"
            )
    try:
        values = df[columns].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"columns {columns} must hold numbers") from exc
    if values.shape[0] < 1:
        raise ValueError("the table has no rows")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"columns {columns} contain NaN or infinite values")
    return values


def check_coverage(
    df: pd.DataFrame,
    actual: str,
    lower: str,
    upper: str,
    claimed: float = 0.90,
) -> CoverageResult:
    """Compare the claimed coverage with the real hit rate."""
    claim = _claimed(claimed)
    a, lo, up = _numbers(df, [actual, lower, upper]).T
    if np.any(lo > up):
        raise ValueError(f"column {lower!r} is above column {upper!r} on some rows")

    n = int(a.size)
    inside = (a >= lo) & (a <= up)
    hits = int(inside.sum())
    real = hits / n
    low, high = wilson_interval(hits, n)

    if high < claim:
        verdict = UNDER
    elif low > claim:
        verdict = OVER
    else:
        verdict = OK

    message = (
        f"Claimed {claim:.0%}. Real {real:.1%} on {n} rows "
        f"(95% interval {low:.1%} to {high:.1%}): {verdict}."
    )
    return CoverageResult(
        n=n,
        claimed=claim,
        real=real,
        real_low=low,
        real_high=high,
        miss_above=float(np.mean(a > up)),
        miss_below=float(np.mean(a < lo)),
        mean_width=float(np.mean(up - lo)),
        verdict=verdict,
        message=message,
    )


def check_coverage_by(
    df: pd.DataFrame,
    actual: str,
    lower: str,
    upper: str,
    condition: str,
    claimed: float = 0.90,
    min_rows: int = 20,
) -> pd.DataFrame:
    """Run check_coverage for each value of a condition column."""
    if condition not in df.columns:
        raise ValueError(f"column {condition!r} not found")
    if df[condition].isna().any():
        raise ValueError(f"column {condition!r} contains missing labels")
    rows = []
    for label, part in df.groupby(condition, sort=True):
        r = check_coverage(part, actual, lower, upper, claimed)
        rows.append(
            {
                "condition": label,
                "n": r.n,
                "real": r.real,
                "real_low": r.real_low,
                "real_high": r.real_high,
                "mean_width": r.mean_width,
                "verdict": FEW if r.n < min_rows else r.verdict,
            }
        )
    return pd.DataFrame(rows)


def walk_forward_check(
    df: pd.DataFrame,
    predicted: str,
    actual: str,
    coverage: float = 0.90,
    sort_by: str | None = None,
) -> tuple[pd.DataFrame, CoverageResult]:
    """Build a range for each row from earlier rows only, then check it.

    The first min_errors(coverage) rows are history only (19 for 90%).
    """
    cov = _claimed(coverage)
    data = df.sort_values(sort_by) if sort_by else df
    pred, act = _numbers(data, [predicted, actual]).T
    errors = act - pred

    start = min_errors(cov)
    if errors.size <= start:
        raise ValueError(
            f"need more than {start} rows for coverage {cov}, got {errors.size}"
        )

    lowers, uppers = [], []
    for t in range(start, errors.size):
        lo, up = asymmetric_bounds(errors[:t], cov)  # earlier rows only
        lowers.append(pred[t] + lo)
        uppers.append(pred[t] + up)

    table = pd.DataFrame(
        {
            "predicted": pred[start:],
            "actual": act[start:],
            "lower": lowers,
            "upper": uppers,
        },
        index=data.index[start:],
    )
    return table, check_coverage(table, "actual", "lower", "upper", cov)
