import math

import pytest

from assay_core.autonomy import (
    accuracy_range,
    autonomy_level,
    cases_needed,
    supported_accuracy,
)
from assay_core.explain import _wilson


def test_clean_record_floor_is_n_over_n_plus_z_squared():
    z = 1.959963984540054
    for n in (6, 24, 35, 73, 500):
        assert supported_accuracy(n, n) == pytest.approx(n / (n + z * z), abs=1e-12)


def test_numbers_quoted_to_buyers():
    assert cases_needed(0.90) == 35
    assert cases_needed(0.95) == 73
    assert supported_accuracy(24, 24) == pytest.approx(0.8617, abs=5e-4)
    assert supported_accuracy(6, 6) == pytest.approx(0.6097, abs=5e-4)


def test_cases_needed_is_the_exact_edge():
    for target in (0.8, 0.9, 0.95, 0.99):
        for misses in (0, 1, 3):
            n = cases_needed(target, misses=misses)
            assert supported_accuracy(n - misses, n) >= target
            assert supported_accuracy(n - 1 - misses, n - 1) < target


def test_matches_the_wilson_in_explain():
    z = 1.959963984540054
    for n in (1, 5, 30, 200):
        for hits in range(0, n + 1, max(1, n // 7)):
            a = accuracy_range(hits, n)
            b = _wilson(hits, n, z)
            assert a == pytest.approx(b, abs=1e-12)


def test_empty_record_knows_nothing():
    assert accuracy_range(0, 0) == (0.0, 1.0)
    result = autonomy_level(0, 0, 0.90)
    assert result.level == 0
    assert "35" in result.reason
    assert "0 misses" in result.reason


def test_thin_clean_record_stays_in_shadow():
    assert autonomy_level(24, 24, 0.95).level == 0
    assert autonomy_level(24, 24, 0.90).level == 1
    assert autonomy_level(35, 35, 0.90).level == 2


def test_contradicted_claim_is_level_zero():
    result = autonomy_level(50, 100, 0.90)
    assert result.level == 0
    assert "contradicts" in result.reason


def test_scope_flag_forces_shadow():
    assert autonomy_level(500, 500, 0.90, scope_ok=False).level == 0


def test_more_clean_cases_never_lower_the_level():
    for claimed in (0.8, 0.9, 0.95):
        levels = [autonomy_level(n, n, claimed).level for n in range(200)]
        assert levels == sorted(levels)


def test_same_miss_rate_more_cases_never_lowers_the_level():
    for claimed in (0.8, 0.9):
        for rate in (0.0, 0.02, 0.05):
            levels = []
            for n in range(50, 2000, 50):
                misses = math.floor(rate * n)
                levels.append(autonomy_level(n - misses, n, claimed).level)
            assert levels == sorted(levels)


def test_bad_inputs_are_refused():
    with pytest.raises(ValueError, match="hits"):
        accuracy_range(5, 3)
    with pytest.raises(ValueError, match="confidence"):
        accuracy_range(1, 2, confidence=1.5)
    with pytest.raises(ValueError, match="claimed"):
        autonomy_level(1, 2, 1.2)
    with pytest.raises(ValueError, match="margin"):
        autonomy_level(1, 2, 0.9, margin=0.95)


def test_reason_counts_the_misses():
    result = autonomy_level(45, 50, 0.90)
    assert result.level == 0
    assert "5 misses" in result.reason
    need = cases_needed(0.90, misses=5)
    assert f"needs {need} cases in all" in result.reason
    assert need > 50
