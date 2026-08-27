"""Stage 3 - the runner.

Runs every registered check and orders the results. Deliberately thin: the checks
live in `checks.py` and the finding contract in `finding.py`, so this module is only
about sequencing and presentation.

Stage 2 described the data. This stage judges it. The separation is deliberate: every
action the system later takes can be traced back to a specific finding, and every
finding to a specific measurement.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pandas as pd

from src.checks import CHECKS
from src.finding import SEVERITIES, TOPICS, Finding, Suggestion
from src.profile import ColumnProfile

__all__ = [
    "CHECKS",
    "SEVERITIES",
    "TOPICS",
    "Finding",
    "Suggestion",
    "as_table",
    "detect",
    "protect_target",
    "summarise",
]

def detect(
    frame: pd.DataFrame,
    profiles: list[ColumnProfile],
    *,
    target: str | None = None,
) -> list[Finding]:
    """Run every check, worst findings first.

    Args:
        frame: the loaded data.
        profiles: its column profiles, from stage 2.
        target: the column to be predicted, when there is one. Findings that would
            modify it are rewritten -- see :func:`protect_target`.
    """
    duplicated = sorted({c for c in frame.columns if list(frame.columns).count(c) > 1})
    if duplicated:
        raise ValueError(
            f"These column names appear more than once: {duplicated}. Selecting such "
            "a column returns a DataFrame rather than a Series, so no check can run "
            "on it. Read the file through src.loader.read_table, which makes names "
            "unique and reports what it changed."
        )

    findings: list[Finding] = []
    for check in CHECKS:
        findings.extend(check(frame, profiles))
    if target:
        findings = protect_target(findings, target)
    return sorted(findings, key=lambda f: (f.rank, f.check, f.columns))


def protect_target(findings: list[Finding], target: str) -> list[Finding]:
    """Rewrite any repair that would alter the column being predicted.

    Two things must never happen to a target, and nothing else in the pipeline
    prevents them:

    **It must not be imputed.** A filled-in label is a guess presented as an
    observation, and a model trained on it learns to reproduce the imputer. The row
    is dropped instead: losing a row costs information, inventing a label adds wrong
    information, and the second is worse.

    **It must not be dropped as a feature.** A target that is highly correlated with
    another column, or that happens to be unique per row, would otherwise be
    proposed for removal by the redundancy and identifier checks -- removing the
    thing you are trying to predict.

    Both rewrites are visible in the message, not silent, so the cost is stated.
    """
    protected: list[Finding] = []
    for finding in findings:
        suggestion = finding.suggestion
        if suggestion is None:
            protected.append(finding)
            continue

        touches = target in (suggestion.params.get("columns") or finding.columns)
        if not touches:
            protected.append(finding)
            continue

        if suggestion.action == "impute":
            finding = replace(
                finding,
                severity="high",
                message=(
                    finding.message
                    + f" '{target}' is the column being predicted, so it is not "
                    "filled in: a guessed label is worse than a missing row."
                ),
                suggestion=Suggestion(
                    action="drop_rows_missing_target",
                    params={"target": target},
                    rationale=(
                        "Drop the rows whose label is unknown. Imputing a target "
                        "fabricates the very thing the model is meant to learn."
                    ),
                ),
            )
        elif suggestion.action in {"drop_columns", "parse_numeric", "log_transform",
                                  "cap_outliers", "normalise_categories"}:
            remaining = [
                c for c in (suggestion.params.get("columns") or []) if c != target
            ]
            note = (
                f" '{target}' is the column being predicted and is excluded from "
                "this repair."
            )
            finding = replace(
                finding,
                message=finding.message + note,
                suggestion=None
                if not remaining
                else Suggestion(
                    action=suggestion.action,
                    params={**suggestion.params, "columns": remaining},
                    rationale=suggestion.rationale,
                ),
            )
        protected.append(finding)
    return protected


def summarise(findings: list[Finding]) -> dict[str, Any]:
    """Counts by severity and by mandated topic, for the UI and the report."""
    by_severity = {s: 0 for s in SEVERITIES}
    by_topic = {t: 0 for t in TOPICS}
    for finding in findings:
        by_severity[finding.severity] += 1
        by_topic[finding.topic] += 1
    return {
        "total": len(findings),
        "by_severity": {k: v for k, v in by_severity.items() if v},
        "by_topic": {k: v for k, v in by_topic.items() if v},
        "blocking": [f.message for f in findings if f.severity == "critical"],
        "unrepairable": [f.check for f in findings if f.suggestion is None],
    }


def as_table(findings: list[Finding]) -> pd.DataFrame:
    """Findings as a display table."""
    return pd.DataFrame(
        [
            {
                "severity": f.severity,
                "topic": f.topic,
                "check": f.check,
                "columns": ", ".join(f.columns) or "(whole table)",
                "rows affected": f.affected_rows or None,
                "%": round(100 * f.affected_share, 2) if f.affected_share else None,
                "repair": f.suggestion.action if f.suggestion else "— none —",
            }
            for f in findings
        ]
    )



