"""How much may an agent do alone, given its resolved record?

The idea. A record of k right answers out of n cases does not prove the true
accuracy equals k / n. It only supports a floor: the lowest accuracy that the
record cannot rule out. The floor comes from the Wilson score interval
(Brown, Cai and DasGupta, 2001). With zero misses the floor is n / (n + z^2).

The cases must be independent. Windows cut from one recording are not. Count
the independent unit (a cycle, a day), not the rows.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist

LEVELS = {
    0: "shadow: the agent proposes, a human decides",
    1: "small actions: the record nearly supports the claim",
    2: "larger actions: the record supports the claim",
}


def _z(confidence: float) -> float:
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    return NormalDist().inv_cdf(0.5 + confidence / 2)


def _check(hits: int, n: int) -> None:
    if n < 0 or hits < 0 or hits > n:
        raise ValueError("need 0 <= hits <= n")


def accuracy_range(hits: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson range for a share of right answers. Both ends are in 0..1."""
    _check(hits, n)
    if n == 0:
        return 0.0, 1.0
    z = _z(confidence)
    p = hits / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def supported_accuracy(hits: int, n: int, confidence: float = 0.95) -> float:
    """The floor the record supports: the low end of the Wilson range."""
    return accuracy_range(hits, n, confidence)[0]


def cases_needed(
    target: float, confidence: float = 0.95, misses: int = 0, limit: int = 1_000_000
) -> int:
    """Fewest cases whose floor reaches `target`, if `misses` of them are wrong."""
    if not 0.0 < target < 1.0:
        raise ValueError("target must be between 0 and 1")
    if misses < 0:
        raise ValueError("misses cannot be negative")
    n = max(misses, 1)
    while n <= limit:
        if supported_accuracy(n - misses, n, confidence) >= target:
            return n
        n += 1
    raise ValueError("target cannot be reached within the case limit")


@dataclass(frozen=True)
class Autonomy:
    """The answer: a level, the evidence behind it, and the reason in words."""

    level: int
    hits: int
    n: int
    claimed: float
    floor: float
    ceiling: float
    needed_for_claim: int
    reason: str

    def __str__(self) -> str:
        return (
            f"level {self.level} ({LEVELS[self.level]})\n"
            f"  {self.hits} right of {self.n}; supported accuracy "
            f"{self.floor:.3f} to {self.ceiling:.3f}; claim {self.claimed:.2f}\n"
            f"  {self.reason}"
        )


def autonomy_level(
    hits: int,
    n: int,
    claimed: float,
    *,
    confidence: float = 0.95,
    margin: float = 0.05,
    tolerance: float = 0.0,
    scope_ok: bool = True,
) -> Autonomy:
    """Level 0, 1 or 2 from the record. A drop in the record drops the level.

    Level 2: the floor reaches the claim, less `tolerance`.
    Level 1: the floor is within `margin` of the claim.
    Level 0: otherwise, or the record contradicts the claim, or scope is not ok.

    tolerance: for an accuracy claim leave it at 0. A range claimed at 90% is
    expected to miss 10% of the time, so a well calibrated range sits at the
    claim and its floor sits a little below. Allow a small tolerance there
    (the envelope uses 0.02).
    """
    _check(hits, n)
    if not 0.0 < claimed < 1.0:
        raise ValueError("claimed must be between 0 and 1")
    if not 0.0 <= margin < claimed:
        raise ValueError("margin must be between 0 and the claim")
    if not 0.0 <= tolerance <= margin:
        raise ValueError("tolerance must be between 0 and the margin")
    floor, ceiling = accuracy_range(hits, n, confidence)
    misses = n - hits
    target = claimed - tolerance
    need = cases_needed(target, confidence, misses=0)
    need_now = cases_needed(target, confidence, misses=misses)
    if not scope_ok:
        level, why = 0, "today's input is outside what the record covers"
    elif n > 0 and ceiling < claimed:
        level, why = (
            0,
            "the record contradicts the claim: even the best case is below it",
        )
    elif floor >= target:
        level, why = 2, "the record supports the claim"
    elif floor >= claimed - margin:
        level, why = (
            1,
            f"with no further misses, {need_now - n} more cases would support the claim",
        )
    else:
        level, why = (
            0,
            f"with {misses} misses the claim needs {need_now} cases in all, {n} so far",
        )
    return Autonomy(level, hits, n, claimed, floor, ceiling, need, why)
