"""Rank: turn a score into a place on the leaderboard, and the place into euros."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Daily pool shares for days 1 to 5 (Predico rewards page). Rank 6 and below get 0.
DAILY_SHARES = (0.40, 0.27, 0.18, 0.10, 0.05)


@dataclass(frozen=True)
class Standing:
    """Where one score sits among the others, and what moving costs or pays.

    rank: place of my score (1 is best). Ties share the better place.
    euros: prize at this rank.
    gap_up: score to remove to pass the next better forecaster (0 at rank 1).
    cushion_down: score I can add before the next worse forecaster passes me.
    euros_if_up / euros_if_down: prize one place better / one place worse.
    """

    rank: int
    euros: float
    gap_up: float
    cushion_down: float
    euros_if_up: float
    euros_if_down: float


def _pay(prizes: list[float], r: int) -> float:
    return float(prizes[r - 1]) if 1 <= r <= len(prizes) else 0.0


def standing(
    mine: float, others: list[float], prizes: list[float], lower_is_better: bool = True
) -> Standing:
    """Place `mine` among `others` and price one step up or down.

    Scores are the monthly average on one track (for example MWI or RMSE).
    prizes: euros per month, prizes[0] is rank 1. Beyond the list pays 0.
    """
    o = np.sort(np.asarray(others, dtype=float))
    if not lower_is_better:
        o, mine = -o, -mine
    better = o[o < mine]
    rank = int((o < mine).sum()) + 1
    worse = o[o > mine]
    gap_up = float(mine - better.max()) if len(better) else 0.0
    cushion = float(worse.min() - mine) if len(worse) else float("inf")
    return Standing(
        rank=rank,
        euros=_pay(prizes, rank),
        gap_up=gap_up,
        cushion_down=cushion,
        euros_if_up=_pay(prizes, max(1, rank - 1)),
        euros_if_down=_pay(prizes, rank + 1),
    )


def daily_value(rank: int, slice_euros: float) -> float:
    """Euros won on one day: rank 1 to 5 take their share of that day's slice."""
    return slice_euros * DAILY_SHARES[rank - 1] if 1 <= rank <= len(DAILY_SHARES) else 0.0
