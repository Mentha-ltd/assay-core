"""The decision envelope: everything an agent should know before it commits to a number.

    model = fit(history, "forecast", "actual", condition="weather")
    report = assess(history, "forecast", "actual", condition="weather", sort_by="time")
    env = decide(model, report, point=100.0, condition="calm")
    print(env)            # plain text
    env.to_dict()         # plain numbers and text, safe for json

It only joins results the other parts already produce. It adds no new statistics,
except the staged-autonomy level from autonomy.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd

from assay_core.autonomy import Autonomy, autonomy_level
from assay_core.explain import Explanation
from assay_core.front import ALL, Model, Report
from assay_core.scope import Scope
from assay_core.seal import Sealer


@dataclass(frozen=True)
class Envelope:
    point: float
    lower: float | None
    upper: float | None
    claimed: float
    borrowed: bool
    record_hits: int
    record_cases: int
    record_label: str  # which part of the record the hit rate comes from
    record_note: str  # empty unless the record had to fall back to all cases
    in_scope: bool | None  # None: not judged
    scope_reasons: str
    autonomy: Autonomy
    findings: list[str] = field(default_factory=list)
    claim_id: str | None = None

    @property
    def historical_hit_rate(self) -> float | None:
        if self.record_cases == 0:
            return None
        return self.record_hits / self.record_cases

    def to_dict(self) -> dict[str, Any]:
        a = self.autonomy
        return {
            "point": self.point,
            "range": {
                "lower": self.lower,
                "upper": self.upper,
                "claimed": self.claimed,
                "borrowed": self.borrowed,
            },
            "record": {
                "from": self.record_label,
                "cases": self.record_cases,
                "hits": self.record_hits,
                "hit_rate": self.historical_hit_rate,
                "note": self.record_note,
            },
            "scope": {"in_scope": self.in_scope, "reasons": self.scope_reasons},
            "autonomy": {
                "level": a.level,
                "meaning": str(a).split("\n", 1)[0],
                "supported_accuracy_low": a.floor,
                "supported_accuracy_high": a.ceiling,
                "cases_needed_clean": a.needed_for_claim,
                "reason": a.reason,
            },
            "findings": list(self.findings),
            "claim_id": self.claim_id,
        }

    def __str__(self) -> str:
        def fmt(x: float | None) -> str:
            return "no limit" if x is None else f"{x:.4g}"

        scope = {True: "in scope", False: "OUT OF SCOPE", None: "scope not judged"}[
            self.in_scope
        ]
        if self.scope_reasons:
            scope += f" ({self.scope_reasons})"
        lines = [
            f"point {self.point:.4g} | range {fmt(self.lower)} to {fmt(self.upper)} "
            f"| claimed {self.claimed:.0%}" + (" | borrowed" if self.borrowed else ""),
            f"record ({self.record_label}): {self.record_hits} of {self.record_cases} inside",
            scope,
            str(self.autonomy),
        ]
        if self.record_note:
            lines.append(self.record_note)
        lines += [f"- {f}" for f in self.findings]
        if self.claim_id:
            lines.append(f"sealed as {self.claim_id}")
        return "\n".join(lines)


def _record(report: Report, condition: object) -> tuple[int, int, str, str]:
    """Hits and cases for this condition. Falls back to all cases, and says so."""
    table = report.by_condition
    used = report.columns is not None and report.columns.get("condition") is not None
    if used and condition is not None and condition in table.index:
        n = int(table.loc[condition, "cases"])
        hits = round(float(table.loc[condition, "hit_rate"]) * n)
        return hits, n, str(condition), ""
    hits = int(report.rows["hit"].sum())
    n = len(report.rows)
    note = ""
    if used and condition is not None:
        note = f"condition {condition!r} is not in the record: all cases used"
    return hits, n, ALL, note


def decide(
    model: Model,
    report: Report,
    point: float,
    *,
    condition: object = None,
    inputs: dict[str, float] | None = None,
    scope: Scope | None = None,
    explanation: Explanation | None = None,
    sealer: Sealer | None = None,
    target_start: datetime | None = None,
    target_end: datetime | None = None,
    margin: float = 0.05,
    tolerance: float = 0.02,
    confidence: float = 0.95,
    **seal_kw: Any,
) -> Envelope:
    """Join the range, the record, the scope result and the autonomy level.

    inputs and scope: both are needed to judge scope. Without them it stays "not judged"
    and does not lower the autonomy level.
    sealer, target_start, target_end: all three are needed to seal the claim.
    tolerance: a range claimed at 90% is expected to miss 10% of the time, so a well
    calibrated range sits at the claim. Level 2 allows this many points below it."""
    if sealer is not None and (target_start is None or target_end is None):
        raise ValueError("sealing needs target_start and target_end")
    if (inputs is None) != (scope is None):
        raise ValueError("judging scope needs both inputs and scope")
    lower, upper, borrowed = model.range(point, condition)
    hits, n, label, note = _record(report, condition)
    in_scope: bool | None = None
    reasons = ""
    if scope is not None and inputs is not None:
        judged = scope.check(pd.DataFrame([inputs]))
        in_scope = bool(judged["in_scope"].iloc[0])
        reasons = str(judged["reasons"].iloc[0])
    level = autonomy_level(
        hits, n, model.claimed, confidence=confidence, margin=margin,
        tolerance=tolerance, scope_ok=in_scope is not False,
    )  # fmt: skip
    claim_id = None
    if sealer is not None and target_start is not None and target_end is not None:
        ctx = dict(seal_kw.pop("context", None) or {})
        ctx.update(
            autonomy_level=level.level,
            record_hits=hits,
            record_cases=n,
            in_scope=in_scope,
            scope_reasons=reasons,
        )
        claim_id = sealer.claim_from(
            model, point, target_start, target_end, condition, context=ctx, **seal_kw
        )
    return Envelope(
        point=float(point),
        lower=lower,
        upper=upper,
        claimed=model.claimed,
        borrowed=borrowed,
        record_hits=hits,
        record_cases=n,
        record_label=label,
        record_note=note,
        in_scope=in_scope,
        scope_reasons=reasons,
        autonomy=level,
        findings=list(explanation.findings) if explanation is not None else [],
        claim_id=claim_id,
    )
