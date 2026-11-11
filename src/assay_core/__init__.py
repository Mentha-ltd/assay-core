"""Assay: honest ranges for forecasts, checked against what really happened."""

from assay_core.autonomy import Autonomy, autonomy_level
from assay_core.envelope import Envelope, decide
from assay_core.explain import Explanation, explain
from assay_core.front import Model, Report, assess, fit
from assay_core.gain import Gain, information_gain
from assay_core.scope import Scope, learn_scope, walk_forward_scope
from assay_core.seal import Sealer
from assay_core.stated import audit

__all__ = [
    "Autonomy",
    "Envelope",
    "Explanation",
    "Gain",
    "Model",
    "Report",
    "Scope",
    "Sealer",
    "assess",
    "audit",
    "autonomy_level",
    "decide",
    "explain",
    "fit",
    "information_gain",
    "learn_scope",
    "walk_forward_scope",
]
