import math

import numpy as np
import pytest

from assay_core.adaptive import aci_interval, aci_update, max_coverage, run_aci
from assay_core.bounds import asymmetric_bounds


def test_miss_lowers_alpha_so_range_gets_wider():
    assert aci_update(0.10, 1, 0.10, 0.01) == pytest.approx(0.091)


def test_no_miss_raises_alpha_so_range_gets_narrower():
    assert aci_update(0.10, 0, 0.10, 0.01) == pytest.approx(0.101)


@pytest.mark.parametrize(
    "args",
    [
        (0.1, 2, 0.1, 0.01),
        (0.1, 1, 0.1, 0.0),
        (0.1, 1, 0.1, -0.01),
        (0.1, 1, 0.0, 0.01),
        (0.1, 1, 1.0, 0.01),
        (float("nan"), 1, 0.1, 0.01),
    ],
)
def test_bad_inputs_are_refused(args):
    with pytest.raises(ValueError):
        aci_update(*args)


def test_max_coverage_by_hand():
    assert max_coverage(19) == pytest.approx(0.90)
    assert max_coverage(39) == pytest.approx(0.95)
    assert max_coverage(1) == 0.0


def test_one_error_gives_infinite_range():
    lower, upper, capped = aci_interval(np.array([1.0]), 0.10)
    assert lower == -math.inf and upper == math.inf and capped


def test_few_errors_give_widest_finite_range_and_flag_it():
    errors = np.arange(10, dtype=float)
    lower, upper, capped = aci_interval(errors, 0.10)
    assert (lower, upper) == (0.0, 9.0)  # smallest and largest error
    assert capped


def test_alpha_below_zero_is_capped_not_infinite():
    lower, upper, capped = aci_interval(np.arange(50, dtype=float), -0.1)
    assert math.isfinite(lower) and math.isfinite(upper) and capped


def test_not_capped_matches_piece_1_when_enough_errors():
    errors = np.random.default_rng(3).normal(0, 10, 100)
    lower, upper, capped = aci_interval(errors, 0.10)
    assert (lower, upper) == asymmetric_bounds(errors, 0.90)
    assert not capped


def jumping_errors(n=2400, jump_at=600, seed=0):
    """Made-up errors. Spread jumps from 10 to 25 (GBP/MWh) at jump_at."""
    rng = np.random.default_rng(seed)
    spread = np.where(np.arange(n) < jump_at, 10.0, 25.0)
    return rng.normal(0.0, spread)


def test_fixed_range_falls_but_aci_stays_near_90_percent():
    errors = jumping_errors()
    window = 200

    lower, upper = asymmetric_bounds(errors[:window], 0.90)
    later = errors[window:]
    fixed_coverage = float(np.mean((later >= lower) & (later <= upper)))

    aci = run_aci(errors, coverage=0.90, gamma=0.02, window=window)

    assert fixed_coverage < 0.75
    assert 0.87 <= aci.coverage <= 0.93


def test_aci_stays_near_90_percent_when_errors_do_not_drift():
    errors = np.random.default_rng(5).normal(0.0, 10.0, 2400)
    aci = run_aci(errors, coverage=0.90, gamma=0.02, window=200)
    assert 0.87 <= aci.coverage <= 0.93


def test_small_window_is_never_infinite_and_flags_capped_days():
    aci = run_aci(jumping_errors(), coverage=0.90, gamma=0.02, window=19)
    assert np.all(np.isfinite(aci.lower)) and np.all(np.isfinite(aci.upper))
    assert aci.capped.any()
    assert 0.0 < aci.capped_share <= 1.0


def test_run_refuses_window_that_is_too_small():
    with pytest.raises(ValueError):
        run_aci(np.arange(100, dtype=float), 0.90, 0.02, window=10)
