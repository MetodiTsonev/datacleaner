"""What a check returns.

Its own module so checks.py and detect.py can both import it without a cycle.

A finding carries a suggested repair where we know one - that link is what makes this
prescriptive rather than just descriptive. No suggestion is shown as such, not hidden:
some defects need a human, and some rules can't judge the column they ran on.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

SEVERITIES = ("critical", "high", "medium", "low", "info")  # worst first

# The four mandated topics, plus structure for what belongs to none of them. On every
# finding so topic coverage can be reported rather than claimed.
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
