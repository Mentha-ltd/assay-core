import numpy as np
import pandas as pd
import pytest

from assay_core.coverage import (
    FEW,
    OK,
    OVER,
    UNDER,
    Z95,
    check_coverage,
    check_coverage_by,
    walk_forward_check,
    wilson_interval,
)


def frame(actual, lower=-1.0, upper=1.0):
    n = len(actual)
    return pd.DataFrame({"a": actual, "lo": np.full(n, lower), "up": np.full(n, upper)})


def test_wilson_ten_out_of_ten_by_hand():
    low, high = wilson_interval(10, 10)
    assert low == pytest.approx(10 / (10 + Z95**2))
    assert high == pytest.approx(1.0, abs=1e-9)


def test_wilson_zero_out_of_ten_by_hand():
    low, high = wilson_interval(0, 10)
    assert low == pytest.approx(0.0, abs=1e-9)
    assert high == pytest.approx(Z95**2 / (10 + Z95**2))


@pytest.mark.parametrize("args", [(1, 0), (-1, 10), (11, 10)])
def test_wilson_bad_inputs_refused(args):
    with pytest.raises(ValueError):
        wilson_interval(*args)


def test_hand_count():
    actual = [0.0] * 18 + [5.0, -5.0]
    r = check_coverage(frame(actual), "a", "lo", "up")
    assert r.n == 20
    assert r.real == pytest.approx(0.90)
    assert r.miss_above == pytest.approx(0.05)
    assert r.miss_below == pytest.approx(0.05)
    assert r.mean_width == pytest.approx(2.0)


def test_undercovers_is_flagged():
    actual = [0.0] * 70 + [5.0] * 30
    assert check_coverage(frame(actual), "a", "lo", "up").verdict == UNDER


def test_consistent_is_not_flagged():
    actual = [0.0] * 90 + [5.0] * 10
    assert check_coverage(frame(actual), "a", "lo", "up").verdict == OK


def test_wider_than_needed_is_flagged():
    actual = [0.0] * 100
    assert check_coverage(frame(actual), "a", "lo", "up").verdict == OVER


def test_missing_column_refused():
    with pytest.raises(ValueError, match="not found"):
        check_coverage(frame([0.0] * 30), "nope", "lo", "up")


def test_nan_refused():
    df = frame([0.0] * 30)
    df.loc[3, "a"] = np.nan
    with pytest.raises(ValueError):
        check_coverage(df, "a", "lo", "up")


def test_lower_above_upper_refused():
    with pytest.raises(ValueError):
        check_coverage(frame([0.0] * 30, lower=2.0, upper=1.0), "a", "lo", "up")


def test_text_column_refused():
    df = frame([0.0] * 30)
    df["a"] = "x"
    with pytest.raises(ValueError):
        check_coverage(df, "a", "lo", "up")


def test_empty_table_refused():
    with pytest.raises(ValueError):
        check_coverage(frame([]), "a", "lo", "up")


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5, float("nan"), True])
def test_bad_claim_refused(bad):
    with pytest.raises(ValueError):
        check_coverage(frame([0.0] * 30), "a", "lo", "up", claimed=bad)


def test_by_condition_flags_the_bad_group():
    df = pd.concat(
        [
            frame([0.0] * 40).assign(c="a"),
            frame([0.0] * 20 + [5.0] * 20).assign(c="b"),
            frame([0.0] * 5).assign(c="c"),
        ],
        ignore_index=True,
    )
    out = check_coverage_by(df, "a", "lo", "up", "c").set_index("condition")
    assert out.loc["a", "verdict"] == OVER
    assert out.loc["b", "verdict"] == UNDER
    assert out.loc["c", "verdict"] == FEW


def made_up(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    pred = 50 + 10 * np.sin(np.arange(n) / 20)
    actual = pred + rng.gamma(2.0, 10.0, n) - 15.0
    return pd.DataFrame({"day": np.arange(n), "pred": pred, "act": actual})


def test_walk_forward_tests_all_rows_after_history():
    table, result = walk_forward_check(made_up(), "pred", "act")
    assert result.n == 1500 - 19
    assert len(table) == 1500 - 19


def test_walk_forward_coverage_is_near_90_percent():
    _, result = walk_forward_check(made_up(), "pred", "act")
    assert 0.87 <= result.real <= 0.93


def test_walk_forward_never_looks_ahead():
    base = made_up()
    changed = base.copy()
    k = 100
    changed.loc[k, "act"] += 1e6
    t1, _ = walk_forward_check(base, "pred", "act")
    t2, _ = walk_forward_check(changed, "pred", "act")
    same = k - 19 + 1
    pd.testing.assert_frame_equal(
        t1.iloc[:same][["lower", "upper"]], t2.iloc[:same][["lower", "upper"]]
    )


def test_walk_forward_sort_by_matches_sorted_input():
    base = made_up(400)
    shuffled = base.sample(frac=1.0, random_state=1)
    a, _ = walk_forward_check(base, "pred", "act")
    b, _ = walk_forward_check(shuffled, "pred", "act", sort_by="day")
    pd.testing.assert_frame_equal(a, b)


def test_walk_forward_too_few_rows_refused():
    with pytest.raises(ValueError):
        walk_forward_check(made_up(19), "pred", "act")
