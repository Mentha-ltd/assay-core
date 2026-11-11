import numpy as np
import pandas as pd

from assay_core.guard import check_submission, fall_cost, missed_days, money_table


def day(n=96):
    mid = np.linspace(-50, 50, n)
    return pd.DataFrame({"q10": mid - 10, "q50": mid, "q90": mid + 10})


def test_clean_file_is_ok():
    c = check_submission(day())
    assert c.ok and c.problems == []


def test_wrong_row_count_not_ok():
    c = check_submission(day(95))
    assert not c.ok and c.fixed is None and "95 rows" in c.problems[0]


def test_missing_column_not_ok():
    c = check_submission(day().drop(columns="q90"))
    assert not c.ok and "missing columns" in c.problems[0]


def test_nan_not_ok():
    d = day()
    d.loc[3, "q50"] = np.nan
    c = check_submission(d)
    assert not c.ok and "missing or infinite" in c.problems[0]


def test_crossed_rows_repaired():
    d = day()
    d.loc[5, ["q10", "q90"]] = d.loc[5, ["q90", "q10"]].to_numpy()
    c = check_submission(d)
    assert c.ok and "wrong order" in c.problems[0]
    assert (c.fixed["q10"] <= c.fixed["q50"]).all() and (c.fixed["q50"] <= c.fixed["q90"]).all()


def test_zero_width_flagged():
    d = day()
    d.loc[0, ["q10", "q50", "q90"]] = 1.0
    assert any("zero width" in p for p in check_submission(d).problems)


def test_far_outside_history_flagged():
    hist = pd.Series(np.linspace(-100, 100, 50))
    d = day() * 1000
    assert any("far outside" in p for p in check_submission(d, history=hist).problems)
    assert not any("far outside" in p for p in check_submission(day(), history=hist).problems)


def test_fall_cost():
    prizes = [300, 200, 150, 120, 105, 70]
    assert fall_cost(prizes, 1) == 100
    assert fall_cost(prizes, 6) == 70
    assert fall_cost(prizes, 12) == 0


def test_money_table_change_against_now():
    t = money_table([300, 200, 150, 120, 105, 70], rank=4, width=2)
    assert list(t["rank"]) == [2, 3, 4, 5, 6]
    assert t.loc[t["rank"] == 4, "change"].item() == 0
    assert t.loc[t["rank"] == 2, "change"].item() == 80


def test_missed_days():
    m = missed_days([300, 200], rank=2, missed=3)
    assert m == {"days_left": 2, "at_cliff": False, "euros_at_stake": 200.0}
    assert missed_days([300], rank=1, missed=5)["at_cliff"]
