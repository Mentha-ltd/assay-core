import pytest

from assay_core.autonomy import autonomy_level, cases_needed


def test_calibrated_range_reaches_level_two_only_with_tolerance():
    # 1312 inside of 1436: a well calibrated 90% range, floor just under 0.90
    strict = autonomy_level(1312, 1436, 0.90)
    allowed = autonomy_level(1312, 1436, 0.90, tolerance=0.02)
    assert strict.level == 1
    assert allowed.level == 2


def test_tolerance_never_lowers_the_level():
    for n in (30, 100, 400, 1500):
        for rate in (0.80, 0.88, 0.90, 0.92, 0.97):
            hits = round(rate * n)
            a = autonomy_level(hits, n, 0.90).level
            b = autonomy_level(hits, n, 0.90, tolerance=0.02).level
            assert b >= a


def test_tolerance_changes_the_cases_needed_target():
    result = autonomy_level(0, 0, 0.90, tolerance=0.02)
    assert result.needed_for_claim == cases_needed(0.88)
    assert result.claimed == 0.90


def test_tolerance_cannot_exceed_the_margin():
    with pytest.raises(ValueError, match="tolerance"):
        autonomy_level(10, 10, 0.90, margin=0.05, tolerance=0.06)
    with pytest.raises(ValueError, match="tolerance"):
        autonomy_level(10, 10, 0.90, tolerance=-0.01)


def test_contradicted_claim_still_level_zero_with_tolerance():
    assert autonomy_level(50, 100, 0.90, tolerance=0.02).level == 0
