# Tests for information_gain(): does an input narrow the range at the same hit rate?
import numpy as np
import pandas as pd
import pytest

from assay_core import information_gain


def data(n=3000, seed=1):
    """Error size depends on wind. temp is unrelated noise."""
    rng = np.random.default_rng(seed)
    wind = rng.uniform(0, 14, n)
    sd = 0.5 + 0.8 * wind
    return pd.DataFrame(
        {
            "t": np.arange(n),
            "f": 100.0,
            "a": 100.0 + rng.normal(0, sd),
            "wind": wind,
            "temp": rng.normal(15, 5, n),
            "kind": rng.choice(["x", "y", "z"], n),
        }
    )


def test_useful_input_is_found():
    g = information_gain(data(), "f", "a", ["wind"], sort_by="t")
    row = g.table.loc["wind"]
    assert row["verdict"] == "narrower, same hit rate"
    assert row["change"] < -0.10
    assert row["hit_with_input"] >= row["hit_pooled"] - 0.02
    assert "wind" in str(g)


def test_noise_input_shows_no_gain():
    g = information_gain(data(), "f", "a", ["temp"], sort_by="t")
    assert g.table.loc["temp", "verdict"] == "no gain"


def test_noise_is_rarely_called_helpful_across_many_seeds():
    helped = 0
    for seed in range(40):
        df = data(1500, seed)
        g = information_gain(df, "f", "a", ["temp"], sort_by="t", n_boot=200)
        helped += g.table.loc["temp", "verdict"] == "narrower, same hit rate"
    assert helped <= 3


def test_ranks_several_inputs():
    g = information_gain(data(), "f", "a", ["wind", "temp", "kind"], sort_by="t")
    v = g.table["verdict"]
    assert v["wind"] == "narrower, same hit rate"
    assert v["temp"] == "no gain"
    assert v["kind"] == "no gain"


def test_text_input_with_real_signal():
    df = data()
    df["site"] = np.where(df["wind"] > 7, "windy", "calm")
    g = information_gain(df, "f", "a", ["site"], sort_by="t")
    assert g.table.loc["site", "verdict"] == "narrower, same hit rate"


def test_uses_late_data_for_judging():
    df = data(2000)
    g = information_gain(df, "f", "a", ["wind"], sort_by="t", test_share=0.4)
    assert g.fit_cases == 1200
    assert g.test_cases == 800


def test_false_confidence_is_not_a_gain():
    df = data(3000)
    # late errors become much larger than early ones: bands built early miss too often
    df.loc[1800:, "a"] = 100.0 + (df.loc[1800:, "a"] - 100.0) * 3
    g = information_gain(df, "f", "a", ["wind"], sort_by="t")
    assert g.table.loc["wind", "verdict"] != "narrower, same hit rate"


def test_not_enough_data_is_said_plainly():
    df = data(70)
    with pytest.raises(ValueError, match="need at least"):
        information_gain(df, "f", "a", ["wind"], sort_by="t")
    flat = data(400)
    flat["const"] = 5.0
    g = information_gain(flat, "f", "a", ["const"], sort_by="t")
    assert g.table.loc["const", "verdict"] == "not enough data"
    assert "not enough" in str(g)


def test_missing_input_values_are_skipped():
    df = data()
    df.loc[::7, "wind"] = np.nan
    g = information_gain(df, "f", "a", ["wind"], sort_by="t")
    assert g.table.loc["wind", "verdict"] == "narrower, same hit rate"


def test_bad_arguments():
    df = data(300)
    with pytest.raises(ValueError, match="at least one input"):
        information_gain(df, "f", "a", [])
    with pytest.raises(ValueError, match="not in the table"):
        information_gain(df, "f", "a", ["nope"])
    with pytest.raises(ValueError, match="between 0 and 1"):
        information_gain(df, "f", "a", ["wind"], claimed=1.0)
    with pytest.raises(ValueError, match="test_share"):
        information_gain(df, "f", "a", ["wind"], test_share=0.0)


def test_same_seed_gives_same_answer():
    a = information_gain(data(), "f", "a", ["wind"], sort_by="t").table
    b = information_gain(data(), "f", "a", ["wind"], sort_by="t").table
    pd.testing.assert_frame_equal(a, b)
