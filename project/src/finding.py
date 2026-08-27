"""The finding contract.

What a check returns. Kept in its own module so `checks.py` and `detect.py` can
both depend on it without depending on each other.

A finding carries not just "this is wrong" but a **suggested repair** where the
system knows one. That link is what makes the system prescriptive rather than
merely descriptive, and it is what the задание means by *автоматизирани процедури*.
A finding with no suggestion is shown as such rather than hidden -- some defects
need a human decision, and some rules cannot judge the column they ran on.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

#: Ordered worst first, so `sorted` puts what matters on top.
SEVERITIES = ("critical", "high", "medium", "low", "info")

#: The four mandated theoretical topics, plus `structure` for defects belonging to
#: none of them. Carried on every finding so coverage per topic can be reported
#: rather than asserted -- see writing/00-zadanie.md.
TOPICS = ("missing", "anomalies", "duplicates", "features", "structure")


@dataclass
class Suggestion:
    """A repair the system knows how to perform."""

    action: str
    params: dict[str, Any] = field(default_factory=dict)
    rationale: str = ""


@dataclass
class Finding:
    """One detected problem."""

    check: str
    severity: str
    topic: str
    message: str
    columns: list[str] = field(default_factory=list)
    affected_rows: int = 0
    affected_share: float = 0.0
    evidence: dict[str, Any] = field(default_factory=dict)
    suggestion: Suggestion | None = None

    def __post_init__(self) -> None:
        if self.severity not in SEVERITIES:
            raise ValueError(f"unknown severity {self.severity!r}")
        if self.topic not in TOPICS:
            raise ValueError(f"unknown topic {self.topic!r}")

    @property
    def rank(self) -> int:
        return SEVERITIES.index(self.severity)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
