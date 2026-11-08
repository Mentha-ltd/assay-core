import numpy as np
import pandas as pd
import pytest

import assay_core
from assay_core.onesided import ceiling_limit, floor_limit, min_errors_one_side


def test_minimum_cases_one_side_is_9_for_90_percent():
    assert min_errors_one_side(0.90) == 9


def test_floor_with_9_errors_is_the_smallest():
    e = np.arange(9.0)
    assert floor_limit(e, 0.90) == 0.0  # k = ceil(10 * 0.9) = 9: 9th largest = smallest


def test_ceiling_with_9_errors_is_the_largest():
    e = np.arange(9.0)
    assert ceiling_limit(e, 0.90) == 8.0


def test_too_few_errors_refused():
    with pytest.raises(ValueError, match="at least 9"):
        floor_limit(np.zeros(8), 0.90)


@pytest.mark.parametrize("c", [0.5, 0.3, 1.0, True])
def test_bad_coverage_refused(c):
    with pytest.raises(ValueError):
        floor_limit(np.zeros(50), c)


def test_nan_refused():
    with pytest.raises(ValueError, match="NaN"):
        floor_limit(np.array([np.nan] * 20), 0.90)


def test_floor_is_closer_to_forecast_than_a_two_sided_lower_limit():
    rng = np.random.default_rng(0)
    e = rng.normal(0, 1, 500)
    from assay_core.bounds import asymmetric_bounds

    assert floor_limit(e, 0.90) > asymmetric_bounds(e, 0.90)[0]


def make(n=3000, seed=3):
    rng = np.random.default_rng(seed)
    mode = rng.choice(["calm", "wild"], n)
    err = np.where(mode == "calm", rng.normal(0, 1, n), rng.normal(0, 5, n))
    return pd.DataFrame({"t": range(n), "f": 100.0, "a": 100.0 + err, "mode": mode})


@pytest.mark.parametrize("side", ["floor", "ceiling"])
def test_walk_forward_one_sided_coverage_near_90(side):
    r = assay_core.assess(make(), "f", "a", condition="mode", sort_by="t", side=side)
    assert r.by_condition["hit_rate"].between(0.87, 0.93).all()
    assert side in str(r)


def test_floor_model_has_no_upper_limit():
    m = assay_core.fit(make(), "f", "a", condition="mode", side="floor")
    lower, upper, _ = m.range(100.0, "calm")
    assert upper is None
    assert lower < 100.0


def test_ceiling_model_has_no_lower_limit():
    m = assay_core.fit(make(), "f", "a", condition="mode", side="ceiling")
    lower, upper, _ = m.range(100.0, "wild")
    assert lower is None
    assert upper > 100.0


def test_unknown_side_refused():
    with pytest.raises(ValueError, match="side"):
        assay_core.fit(make(), "f", "a", side="middle")


def test_floor_is_needed_with_only_ten_cases_per_condition():
    df = make(n=60)
    assert assay_core.fit(df, "f", "a", side="floor").range(100.0)[0] is not None
    with pytest.raises(ValueError):
        assay_core.fit(df.head(12), "f", "a", side="both")
