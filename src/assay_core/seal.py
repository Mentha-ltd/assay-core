"""Seal: write each range into the tamper-evident record, and check it later.

    s = Sealer("claims.duckdb", agent_id="my-model", model_version="3",
               target_name="load", unit="MW", resolution_rule="metered MW, hourly mean")
    cid = s.claim_from(model, point=120.0, condition="calm",
                       target_start=t0, target_end=t1, as_of=now)
    s.resolve(cid, actual=131.2)
    print(s.check())        # True/False plus the problems found
    print(s.head())         # publish this hash: it proves nothing before it was changed

Times without a timezone are taken as UTC.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import pandas as pd

from assay_core.front import Model
from assay_core.record.chain import ChainReport
from assay_core.record.models import Claim, Resolution
from assay_core.record.store import Ledger


def _utc(t: datetime) -> datetime:
    return t.replace(tzinfo=UTC) if t.tzinfo is None else t


def _hash(obj: object) -> str:
    """Fingerprint of the data a claim used. Same data gives the same text."""
    if obj is None:
        return "not given"
    if isinstance(obj, pd.DataFrame | pd.Series):
        raw = pd.util.hash_pandas_object(obj, index=True).to_numpy().tobytes()
        return hashlib.sha256(raw).hexdigest()
    return hashlib.sha256(repr(obj).encode()).hexdigest()


def _code_version() -> str:
    try:
        return version("assay-core")
    except PackageNotFoundError:
        return "unknown"


class Sealer:
    def __init__(
        self,
        path: str | Path = ":memory:",
        *,
        agent_id: str,
        model_version: str,
        target_name: str,
        unit: str,
        resolution_rule: str,
    ) -> None:
        self.ledger = Ledger(path)
        self.agent_id = agent_id
        self.model_version = model_version
        self.target_name = target_name
        self.unit = unit
        self.resolution_rule = resolution_rule

    def claim(
        self,
        point: float,
        lower: float | None,
        upper: float | None,
        coverage: float,
        target_start: datetime,
        target_end: datetime,
        *,
        as_of: datetime | None = None,
        data_cutoff: datetime | None = None,
        condition: str | None = None,
        input_data: object = None,
        context: dict[str, Any] | None = None,
    ) -> str:
        """Seal one range. Returns its claim id. One claim per target period and condition."""
        start, end = _utc(target_start), _utc(target_end)
        made = _utc(as_of) if as_of is not None else datetime.now(UTC)
        cid = f"{self.agent_id}:{start:%Y%m%dT%H%M%SZ}:{condition or '-'}"
        self.ledger.add_claim(
            Claim(
                claim_id=cid,
                agent_id=self.agent_id,
                model_version=self.model_version,
                code_version=_code_version(),
                as_of=made,
                data_cutoff=_utc(data_cutoff) if data_cutoff is not None else made,
                target_name=self.target_name,
                unit=self.unit,
                target_start=start,
                target_end=end,
                point=point,
                lower=lower,
                upper=upper,
                coverage=coverage,
                reference_class=condition,
                resolution_rule=self.resolution_rule,
                input_data_hash=_hash(input_data),
                context=context or {},
            )
        )
        return cid

    def claim_from(
        self,
        model: Model,
        point: float,
        target_start: datetime,
        target_end: datetime,
        condition: str | None = None,
        **kw: Any,
    ) -> str:
        """Take the range from a fitted model and seal it."""
        lower, upper, borrowed = model.range(point, condition)
        ctx = dict(kw.pop("context", None) or {})
        ctx["borrowed_range"] = borrowed
        ctx["side"] = model.side
        return self.claim(
            point, lower, upper, model.claimed, target_start, target_end,
            condition=condition, context=ctx, **kw,
        )  # fmt: skip

    def resolve(
        self, claim_id: str, actual: float, truth_data: object = None
    ) -> Resolution:
        """Record what really happened. The target period must have ended."""
        return self.ledger.resolve(claim_id, actual, _hash(truth_data))

    def check(self, expected_head: str | None = None) -> ChainReport:
        """Recompute every hash. report.ok is False if any row was changed, removed or added."""
        return self.ledger.verify(expected_head)

    def head(self) -> str:
        return self.ledger.head()

    def scorecard(self) -> pd.DataFrame:
        """Hit rate per condition, from resolved claims in the record."""
        rows = self.ledger.con.execute(
            """
            SELECT coalesce(c.reference_class, 'all cases') AS condition,
                   (coalesce(r.realised_value >= c.lower, true)
                    AND coalesce(r.realised_value <= c.upper, true)) AS hit
            FROM claims c JOIN resolutions r ON r.claim_id = c.claim_id
            WHERE c.agent_id = ?
              AND r.resolution_version = (
                  SELECT max(resolution_version) FROM resolutions WHERE claim_id = c.claim_id)
            """,
            [self.agent_id],
        ).df()
        if rows.empty:
            return pd.DataFrame(columns=["cases", "hit_rate"])
        g = rows.groupby("condition")["hit"]
        return pd.DataFrame({"cases": g.size(), "hit_rate": g.mean()})

    def close(self) -> None:
        self.ledger.close()
