"""Runs the checks and orders the results.

Thin on purpose - checks live in checks.py, the contract in finding.py. Profiling
described the data; this judges it, so every later action traces back to a finding.
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
    """Rewrite any repair that would touch the column being predicted.

    Two things nothing else prevents. It must not be imputed - a filled-in label is a
    guess dressed as an observation, and the model just learns the imputer, so we drop
    the row instead. And it must not be dropped as redundant or identifier-like, which
    would remove the thing we're predicting.

    Both rewrites show up in the message rather than happening silently.
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
    by_severity = dict.fromkeys(SEVERITIES, 0)
    by_topic = dict.fromkeys(TOPICS, 0)
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



