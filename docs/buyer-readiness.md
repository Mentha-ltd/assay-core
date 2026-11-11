# Assay: what a reviewer needs to know

Status: draft. Every claim below names the check behind it. Checks were run on 2026-11-11.

## 1. Data it needs

- A table with three columns: the forecast, the real outcome, and the time.
- Optional columns: a stated lower and upper limit, a condition label, input values.
- Format: anything pandas can read (CSV, Parquet). A file in, a report out.
- No live connection to any system is needed.

## 2. Where it runs and what it touches

- It runs on the customer's own machine, inside Python 3.12 or later.
- It opens no network connection and runs no shell command.
  Check: `tests/test_no_network.py` fails the build if any source file imports a network
  or shell module, calls eval or exec, or imports an undeclared package.
- It writes one thing: a DuckDB file, at a path the customer chooses, and only if the
  sealed record is used. Without it, nothing is written to disk.

## 3. What it depends on

- Four packages at run time: duckdb, numpy, pandas, pydantic (and what they pull in).
- Licences of all installed packages are permissive (MIT, BSD, Apache-2.0, PSF, Zlib,
  0BSD, CC0). No GPL, LGPL or AGPL. Check: `pip-licenses`, 2026-11-11.
- No known vulnerabilities in the installed packages. Check: `pip-audit`, 2026-11-11.
  This is true on that date only. Rerun it before each release.
- Licence of assay-core itself: to be confirmed.

## 4. What the sealed record shows, and what it does not

- Each claim is stored before its outcome, with a time stamp, and chained to the one
  before it with SHA-256. Rows are only added. The code has no update or delete method.
- It shows: the claim was recorded in this order, and no stored row was changed since
  the chain head was last noted.
- It does not show: that the claim was true, that the input data was correct, or that the
  time stamp came from a trusted clock.
- Limit: a person with write access to the file could rebuild the whole chain. The record
  is tamper-evident only if the chain head is also kept somewhere that person cannot
  change (for example, sent to the customer when each claim is made).

## 5. A pilot

- Length: 30 days, shadow mode. Assay reads forecasts and outcomes. It acts on nothing.
- Customer effort: export one table. Target: under one engineer-day. Not yet measured.
- What comes back: the hit rate of each stated range with its interval, where the misses
  cluster, which inputs narrow the range, and the autonomy level the evidence supports.

## 6. Not yet supported

- Checking the stated confidence of a classifier, prediction by prediction.
- Scope checks across several inputs at once (today each input is checked alone).
- Any result on a customer's real data. All tests so far use synthetic data.
