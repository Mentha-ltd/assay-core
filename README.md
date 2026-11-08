# Assay (`assay-core`)

Honest ranges for forecasts, checked against what really happened.

**Status: early. Built and tested. Not yet reviewed by a second person. Not yet tried on a
real customer's data.**

## The problem

A forecast gives one number. Whoever acts on it needs to know how wrong it can be.
Most ranges are too narrow on the hard cases, and nobody checks. Assay does four things:

1. **Checks** how often the real value lands inside the claimed range, on cases the range
   did not learn from (no look-ahead), and reports it per condition.
2. **Explains** the misses: which condition, input band or time period they concentrate in.
3. **Gives** a range that holds up per condition: two-sided, or one side only
   (a floor "at least X" or a ceiling "at most X").
4. **Seals** each claim in a hash-chained record. Editing a claim after the result is
   known is detected.

It also flags new cases that look unlike the past (the range is not promised there).

Everything works on your own table: your column names, your units. Nothing in the code
knows what the numbers mean.

## Install

```bash
uv add git+https://github.com/Mentha-ltd/assay-core
```

(The repository is private for now. You need access to it.)

## Use it

```python
import assay_core

# 1. CHECK (add condition="weather" to give each condition its own range)
report = assay_core.assess(
    df, forecast="forecast_mw", actual="actual_mw", sort_by="hour"
)
print(report)

# 2. EXPLAIN: where and why the misses happened
print(report.explain(df, by=["weather"], features=["temp_c"]))

# 3. RANGE (or side="floor" / side="ceiling")
model = assay_core.fit(df, "forecast_mw", "actual_mw", condition="weather")
lower, upper, borrowed = model.range(point=100.0, condition="windy")

# 4. SEAL
seal = assay_core.Sealer(
    agent_id="demo",
    model_version="1",
    target_name="wind_output",
    unit="MW",
    resolution_rule="metered MW, hourly mean",
)
claim_id = seal.claim_from(
    model, 100.0, target_start=t0, target_end=t1, condition="windy", as_of=now
)
seal.resolve(claim_id, actual=93.0)
seal.check().ok  # False if any sealed row was changed
```

A complete example that runs on made-up data: `examples/quickstart.py`.
Add `features=["temp", "load"]` to `assess` to judge each case in or out of scope.

## What it needs

- A table with one row per past case: forecast, actual, optionally a time column, a
  condition column and input columns. At least 19 past cases for a two-sided 90% range
  (9 for a floor or ceiling). A condition with fewer cases borrows the pooled range and
  says so (`borrowed=True`).
- Rows with a missing forecast or actual are left out, and the report says how many
  (`missing="refuse"` raises an error instead).
- The forecast must have been made before the actual value was known. Assay cannot check this.

## Speed

Measured on made-up data, 100,000 rows, on the author's test machine (not on a customer's
data or machine): `assess` 1.3 seconds, with the scope check 3.0 seconds, `explain` 3.6
seconds. The fast versions are proven identical to the simple versions by tests that compare
them on random tables.

## Known limits (read these)

- Ranges assume future errors look like past errors. If the world changes, the hit rate
  falls. `assess` and `explain` show this as hit rate by time period, not as a promise.
- The scope check looks at each input column alone. An unusual combination of normal values
  passes. The 1% tail cut is a choice, not a theorem. On test data it flagged about 1.4% of
  harmless cases for every real shock it caught.
- `explain` finds where misses cluster. A cluster is a lead, not proof of the cause. It counts
  many groups at once and widens its intervals for that, so a mild real problem can be
  missed: on test data it found a cause that doubled the errors every time, and one that made
  them 1.5 times larger in 15 of 20 runs.
- Only numeric errors and floors or ceilings. Yes/no and probability forecasts need
  different methods and are not here yet.
- Hit rates are measured on rows treated as independent. Rows close in time are not.
- No physics check and no money value yet. Those are planned.

## Method (sources)

- Split conformal prediction with the finite-sample rank correction, one tail at a time:
  Angelopoulos and Bates, "A Gentle Introduction to Conformal Prediction" (arXiv 2107.07511).
- Per-condition ranges (Mondrian conformal): Vovk et al., 2005.
- Adaptive version: Gibbs and Candes, 2021 (arXiv 2106.00170), with a cap so the range
  never becomes infinite.
- Intervals on hit rates: Wilson score interval (Brown, Cai and DasGupta, 2001), widened by
  the number of groups tested (Bonferroni).
- Hash chain: each record's SHA-256 depends on the one before it.

## Develop

```bash
uv sync
uv run pytest -q
uv run ruff check . && uv run ruff format --check .
```
