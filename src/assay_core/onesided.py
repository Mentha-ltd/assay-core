"""One-sided limits: a floor ("at least X") or a ceiling ("at most X").

Floor at coverage c: the actual value lands at or above the floor with probability c.
This is one tail, so it needs fewer past cases than a two-sided range: 9 for c = 0.90, not 19.
It uses the same rank rule as bounds.py with the tail share 1 - c, not (1 - c) / 2.
"""

from __future__ import annotations

import numpy as np

from assay_core.bounds import min_errors, tail_rank

SIDES = ("both", "floor", "ceiling")


def _check(errors: np.ndarray, coverage: float) -> np.ndarray:
    if isinstance(coverage, bool) or not 0.5 < coverage < 1:
        raise ValueError(
            f"coverage for one side must be between 0.5 and 1, got {coverage}"
        )
    e = np.asarray(errors, dtype=float)
    if np.isnan(e).any():
        raise ValueError("errors contain NaN")
    need = min_errors(2 * coverage - 1)
    if len(e) < need:
        raise ValueError(f"need at least {need} errors, got {len(e)}")
    return e


def floor_limit(errors: np.ndarray, coverage: float) -> float:
    """Signed error to add to a forecast: the actual value is above it with probability c."""
    e = _check(errors, coverage)
    k = tail_rank(len(e), 2 * coverage - 1)  # ceil((n + 1) * c)
    return float(np.sort(e)[len(e) - k])  # k-th largest


def ceiling_limit(errors: np.ndarray, coverage: float) -> float:
    """Signed error to add to a forecast: the actual value is below it with probability c."""
    e = _check(errors, coverage)
    k = tail_rank(len(e), 2 * coverage - 1)
    return float(np.sort(e)[k - 1])  # k-th smallest


def min_errors_one_side(coverage: float) -> int:
    return min_errors(2 * coverage - 1)
