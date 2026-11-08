"""Adaptive conformal inference (ACI) on top of asymmetric_bounds.

Method: Gibbs and Candes, "Adaptive Conformal Inference Under
Distribution Shift", arXiv 2106.00170.

Each day we build a range at coverage 1 - alpha_t. After the day we
update alpha_t from whether the real error fell outside the range.
Errors are signed: error = actual - forecast (for example GBP/MWh).

Capped days: with n past errors the widest finite range has coverage
(n - 1) / (n + 1). If ACI asks for more, we return that widest range
and mark the day as capped. The range is infinite only with fewer
than 2 past errors.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from assay_core.bounds import asymmetric_bounds, min_errors

MIN_COVERAGE = 0.01  # our clip when alpha_t >= 1 (heuristic, not in the paper)
CAP_TOLERANCE = 1e-9


def aci_update(
    alpha_t: float, missed: bool | int, target_alpha: float, gamma: float
) -> float:
    """Return alpha_next = alpha_t + gamma * (target_alpha - missed).

    missed is 1 if the real error fell outside the range, else 0.
    A miss lowers alpha_t, so the next range is wider.
    """
    if missed not in (0, 1):
        raise ValueError(f"missed must be 0 or 1, got {missed!r}")
    if not math.isfinite(gamma) or gamma <= 0.0:
        raise ValueError(f"gamma must be a positive number, got {gamma!r}")
    if not math.isfinite(target_alpha) or not 0.0 < target_alpha < 1.0:
        raise ValueError(f"target_alpha must be in (0, 1), got {target_alpha!r}")
    if not math.isfinite(alpha_t):
        raise ValueError("alpha_t must be a finite number")
    return float(alpha_t + gamma * (target_alpha - int(missed)))


def max_coverage(n_errors: int) -> float:
    """Highest coverage with a finite range from n past errors."""
    if n_errors < 2:
        return 0.0
    return (n_errors - 1) / (n_errors + 1)


def aci_interval(past_errors: np.ndarray, alpha_t: float) -> tuple[float, float, bool]:
    """Return (lower, upper, capped) at coverage 1 - alpha_t.

    capped is True when the asked coverage was above what the past
    errors can give, so the widest finite range was returned instead.
    The range is infinite only with fewer than 2 past errors.
    """
    n = len(past_errors)
    if n < 2:
        return -math.inf, math.inf, True
    cap = max_coverage(n)
    coverage = 1.0 - alpha_t
    capped = coverage > cap + CAP_TOLERANCE
    if capped:
        coverage = cap
    coverage = max(coverage, MIN_COVERAGE)
    if n < min_errors(coverage):  # cannot happen for n >= 2; kept as a guard
        return -math.inf, math.inf, True
    lower, upper = asymmetric_bounds(past_errors, coverage)
    return lower, upper, capped


@dataclass(frozen=True)
class AciRun:
    """Result of a walk-forward ACI run. Arrays have one entry per tested day."""

    lower: np.ndarray
    upper: np.ndarray
    alpha: np.ndarray  # alpha_t used on that day
    missed: np.ndarray  # True if the real error was outside the range
    capped: np.ndarray  # True if the asked coverage could not be given

    @property
    def coverage(self) -> float:
        return float(1.0 - self.missed.mean())

    @property
    def capped_share(self) -> float:
        return float(self.capped.mean())


def run_aci(errors: np.ndarray, coverage: float, gamma: float, window: int) -> AciRun:
    """Walk forward over the errors, using only past errors each day.

    Day t uses errors[t - window : t]. The first tested day is index `window`.
    """
    values = np.asarray(errors, dtype=float)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("errors must be a 1-D list of finite numbers")
    if window < min_errors(coverage):
        raise ValueError(f"window must be at least {min_errors(coverage)}")
    if len(values) <= window:
        raise ValueError("need more errors than the window size")

    target_alpha = 1.0 - coverage
    alpha = target_alpha
    lower, upper, alphas, missed, capped = [], [], [], [], []
    for t in range(window, len(values)):
        lo, up, cap = aci_interval(values[t - window : t], alpha)
        miss = not (lo <= values[t] <= up)
        lower.append(lo)
        upper.append(up)
        alphas.append(alpha)
        missed.append(miss)
        capped.append(cap)
        alpha = aci_update(alpha, miss, target_alpha, gamma)

    return AciRun(
        lower=np.array(lower),
        upper=np.array(upper),
        alpha=np.array(alphas),
        missed=np.array(missed),
        capped=np.array(capped),
    )
