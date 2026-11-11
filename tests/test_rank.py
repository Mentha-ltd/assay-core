import math

import pytest

from assay_core.rank import daily_value, standing

PRIZES = [300, 200, 150, 120, 105, 70]


def test_rank_and_euros():
    s = standing(10.0, [8.0, 9.0, 11.0, 12.0], PRIZES)
    assert s.rank == 3 and s.euros == 150


def test_gaps_in_score_units():
    s = standing(10.0, [8.0, 9.0, 11.0, 12.0], PRIZES)
    assert s.gap_up == 1.0 and s.cushion_down == 1.0
    assert s.euros_if_up == 200 and s.euros_if_down == 120


def test_first_place_has_no_gap_up():
    s = standing(5.0, [8.0, 9.0], PRIZES)
    assert s.rank == 1 and s.gap_up == 0.0 and s.euros_if_up == 300


def test_last_place_cushion_is_infinite():
    s = standing(20.0, [8.0, 9.0], PRIZES)
    assert s.rank == 3 and math.isinf(s.cushion_down)


def test_ties_share_better_place():
    assert standing(9.0, [9.0, 9.0, 12.0], PRIZES).rank == 1


def test_higher_is_better():
    s = standing(10.0, [8.0, 12.0], PRIZES, lower_is_better=False)
    assert s.rank == 2 and s.gap_up == 2.0


def test_beyond_prize_list_pays_zero():
    s = standing(99.0, list(range(12)), PRIZES)
    assert s.rank == 13 and s.euros == 0.0


def test_daily_value():
    assert daily_value(1, 500) == pytest.approx(200)
    assert daily_value(5, 500) == pytest.approx(25)
    assert daily_value(6, 500) == 0.0
