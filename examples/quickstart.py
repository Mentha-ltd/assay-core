"""Quickstart: run Assay on your own table. Uses made-up data so it runs anywhere.

Your table needs: a forecast column and an actual column. Optional: a time column, a
condition column (a label for each case), and input columns. Use your own names and units.
"""

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd

import assay_core

# --- made-up table: a forecast that is 5 MW too high, and much noisier in windy hours ---
rng = np.random.default_rng(0)
n = 4000
weather = rng.choice(["calm", "windy"], n)
noise = np.where(weather == "calm", rng.normal(0, 2, n), rng.normal(0, 8, n))
df = pd.DataFrame(
    {
        "hour": pd.date_range("2026-01-01", periods=n, freq="h"),
        "forecast_mw": 100.0,
        "actual_mw": 95.0 + noise,
        "weather": weather,
        "temp_c": rng.normal(12, 6, n),
    }
)

# 1. CHECK: how honest is a 90% range, judged only on cases it did not learn from?
#    (one range for all cases: no condition given)
report = assay_core.assess(
    df, forecast="forecast_mw", actual="actual_mw", sort_by="hour"
)
print(report)

# 2. EXPLAIN: where do the misses concentrate? Name the columns you want to look at.
print()
print(report.explain(df, by=["weather"], features=["temp_c"]))

# 3. RANGE: one range per condition. Learn from all known cases, then ask for a new forecast.
model = assay_core.fit(
    df, forecast="forecast_mw", actual="actual_mw", condition="weather"
)
lower, upper, borrowed = model.range(point=100.0, condition="windy")
print(f"\nwindy hour, forecast 100 MW -> 90% range {lower:.1f} to {upper:.1f} MW")

# 4. FLOOR: "at least X, 9 times in 10" (one side only, needs fewer past cases)
floor_model = assay_core.fit(
    df, "forecast_mw", "actual_mw", condition="weather", side="floor"
)
floor, _, _ = floor_model.range(point=100.0, condition="windy")
print(
    f"windy hour, forecast 100 MW -> floor {floor:.1f} MW (90% of the time it is above)"
)

# 5. SEAL: write the claim into a tamper-evident record, then check it later.
seal = assay_core.Sealer(
    agent_id="demo",
    model_version="1",
    target_name="wind_output",
    unit="MW",
    resolution_rule="metered MW, hourly mean",
)
start = datetime(2026, 6, 1, 12, tzinfo=UTC)
claim_id = seal.claim_from(
    model,
    point=100.0,
    condition="windy",
    target_start=start,
    target_end=start + timedelta(hours=1),
    as_of=start - timedelta(hours=2),
    input_data=df,
)
seal.resolve(claim_id, actual=93.0)
print(f"\nrecord intact: {seal.check().ok}")
print(seal.scorecard())
