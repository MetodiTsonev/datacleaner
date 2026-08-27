"""Findings -> an ordered, executable sequence of repairs.

Detection says what's wrong. This decides what to do about it, in what order, and
which side of the train/test split each step belongs on.

Three things happen here that a detector can't do:

  merge     three disguised-missing findings become one operation
  derive    converting "?" to null *creates* missing values, so an imputation step
            becomes necessary that no detector asked for
  order     a fixed sequence, because the order is a design decision and not
            something to compute per file

No topological sort. The order below is written down once, argued for in
writing/04-implementation/04-plan.md, and the same for every file. A sort over
declared constraints would produce the same answer with more machinery and less
explanation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

from src.finding import Finding

PRE_SPLIT = "pre_split"
POST_SPLIT = "post_split"

# The order, and why each step sits where it does. Position in this tuple IS the
# execution order; steps whose action no finding proposed are simply absent.
ORDER: tuple[tuple[str, str, str], ...] = (
    ("clean_column_names", PRE_SPLIT,
     "Headers first: every later step addresses columns by name."),
    ("drop_empty_rows", PRE_SPLIT,
     "Blank rows count toward every total, so remove them before anything is counted."),
    ("drop_rows", PRE_SPLIT,
     "A totals row is the maximum of every numeric column, so it would set the "
     "outlier bounds and move the mean."),
    ("parse_numeric", PRE_SPLIT,
     "Turn '1 234,56' into a number early: parsing changes what a column is, and "
     "later steps judge it on what it has become."),
    ("drop_columns", PRE_SPLIT,
     "Removing dead columns early means every later step does less work and cannot "
     "learn from them."),
    ("normalise_categories", PRE_SPLIT,
     "Trim and unify case before looking for duplicates, so 'Sofia' and 'sofia ' "
     "are recognised as the same value."),
    ("replace_disguised_missing", PRE_SPLIT,
     "Convert '?' to a real null, so the missingness becomes visible to everything "
     "downstream."),
    ("drop_duplicate_rows", PRE_SPLIT,
     "After normalising, so formatting duplicates are caught too; before the split, "
     "because copies straddling the boundary cannot be removed afterwards."),
    ("drop_rows_missing_target", PRE_SPLIT,
     "A row with no label cannot be used and cannot be filled in. Dropped before "
     "the split so the split is stratified on real labels."),
    ("impute", POST_SPLIT,
     "Fill the gaps using values computed from the training half only."),
    ("cap_outliers", POST_SPLIT,
     "Bounds from the training half only, and capped rather than deleted - an "
     "extreme value is often real."),
    ("log_transform", POST_SPLIT,
     "Compress the long tail once the data is otherwise clean."),
)

_POSITION = {action: i for i, (action, _, _) in enumerate(ORDER)}
_STAGE = {action: stage for action, stage, _ in ORDER}
_WHY = {action: why for action, _, why in ORDER}

# Actions that turn a value into a null, so imputation becomes necessary afterwards.
CREATES_NULLS = ("replace_disguised_missing", "parse_numeric")


@dataclass
class Step:
    action: str
    stage: str
    why: str
    params: dict[str, Any] = field(default_factory=dict)
    from_checks: list[str] = field(default_factory=list)
    # Set when no finding asked for this step and it follows from another one.
    derived_from: str | None = None

    @property
    def columns(self) -> list[str]:
        return list(self.params.get("columns") or [])

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Plan:
    steps: list[Step]
    # Findings the system reported but cannot repair. Kept, not dropped: a plan that
    # silently omits them looks complete when it isn't.
    unaddressed: list[Finding] = field(default_factory=list)

    def stage(self, name: str) -> list[Step]:
        return [s for s in self.steps if s.stage == name]

    def summary(self) -> dict[str, Any]:
        return {
            "steps": len(self.steps),
            "pre_split": len(self.stage(PRE_SPLIT)),
            "post_split": len(self.stage(POST_SPLIT)),
            "derived": sum(1 for s in self.steps if s.derived_from),
            "unaddressed": len(self.unaddressed),
        }


def build(findings: list[Finding], *, target: str | None = None) -> Plan:
    """Turn findings into an ordered plan."""
    merged: dict[str, Step] = {}
    unaddressed: list[Finding] = []

    for finding in findings:
        suggestion = finding.suggestion
        if suggestion is None:
            unaddressed.append(finding)
            continue
        if suggestion.action not in _POSITION:
            unaddressed.append(finding)
            continue
        _merge(merged, finding, suggestion.action, suggestion.params)

    _resolve_parse_versus_drop(merged)
    derived = _derive_imputation(merged, target=target)
    if derived:
        merged.setdefault("impute", derived)
        if derived is not merged["impute"]:
            # An impute step already existed; widen it and note the derivation.
            _absorb(merged["impute"], derived)

    steps = sorted(merged.values(), key=lambda s: _POSITION[s.action])
    steps = _drop_removed_columns(steps)
    return Plan(steps=steps, unaddressed=unaddressed)


def _drop_removed_columns(steps: list[Step]) -> list[Step]:
    """Stop later steps acting on columns an earlier step deletes.

    Without this the plan says "drop order_id" at step 4 and "impute order_id" at
    step 10, which is incoherent even though each half came from a real finding.
    """
    gone: set[str] = set()
    kept: list[Step] = []
    for step in steps:
        if step.action == "drop_columns":
            gone |= set(step.columns)
            kept.append(step)
            continue
        if step.columns:
            remaining = [c for c in step.columns if c not in gone]
            if not remaining:
                continue  # nothing left to act on
            step.params["columns"] = remaining
        kept.append(step)
    return kept


def _merge(merged: dict[str, Step], finding: Finding, action: str, params: dict) -> None:
    """One step per action, with the columns of every finding that asked for it."""
    step = merged.get(action)
    if step is None:
        merged[action] = Step(
            action=action,
            stage=_STAGE[action],
            why=_WHY[action],
            params=dict(params),
            from_checks=[finding.check],
        )
        return
    if finding.check not in step.from_checks:
        step.from_checks.append(finding.check)
    _absorb(step, Step(action, step.stage, step.why, dict(params)))


def _absorb(step: Step, other: Step) -> None:
    """Fold another step's params into this one. Lists union, scalars keep the first."""
    for key, value in other.params.items():
        if isinstance(value, list):
            existing = step.params.setdefault(key, [])
            for item in value:
                if item not in existing:
                    existing.append(item)
        else:
            step.params.setdefault(key, value)
    if other.derived_from and not step.derived_from:
        step.derived_from = other.derived_from


def _resolve_parse_versus_drop(merged: dict[str, Step]) -> None:
    """A column that can be parsed into a number is not a dead column.

    invoice_total holds values like "1 403,17", so it profiles as an identifier - its
    values are near-unique strings - and gets proposed for removal. It is a quantity
    nobody has parsed yet. Two findings contradict, and parsing wins: dropping loses a
    real feature, parsing recovers one.
    """
    parse, drop = merged.get("parse_numeric"), merged.get("drop_columns")
    if not parse or not drop:
        return
    rescued = [c for c in drop.columns if c in parse.columns]
    if not rescued:
        return
    remaining = [c for c in drop.columns if c not in rescued]
    if remaining:
        drop.params["columns"] = remaining
    else:
        del merged["drop_columns"]
    parse.why += (
        f" Keeps {', '.join(rescued)}, which would otherwise be dropped as dead - "
        "it is a quantity, not an identifier."
    )


def _derive_imputation(merged: dict[str, Step], *, target: str | None) -> Step | None:
    """Imputation nothing asked for.

    Replacing "?" with a null, or parsing "1 234,56" into a number, turns values that
    looked present into missing ones. Nothing downstream handles a null, and no
    detector could have reported it because it didn't exist when they ran. This is the
    step the system has to work out for itself.

    The target is excluded: its rows are dropped instead, never filled.
    """
    columns: list[str] = []
    causes: list[str] = []
    for action in CREATES_NULLS:
        step = merged.get(action)
        if step is None:
            continue
        causes.append(action)
        for column in step.columns:
            if column != target and column not in columns:
                columns.append(column)
    if not columns:
        return None
    return Step(
        action="impute",
        stage=POST_SPLIT,
        why=_WHY["impute"],
        params={"columns": columns, "strategy": "auto"},
        from_checks=[],
        derived_from=" and ".join(causes),
    )


def _provenance(step: Step) -> str:
    """Which findings asked for this step, and whether it was also derived."""
    parts = list(step.from_checks)
    if step.derived_from:
        parts.append(f"derived from {step.derived_from}")
    return ", ".join(parts)


def as_table(plan: Plan) -> pd.DataFrame:
    """The plan as a display table, in execution order."""
    rows = []
    for i, step in enumerate(plan.steps, start=1):
        rows.append(
            {
                "#": i,
                "stage": "before split" if step.stage == PRE_SPLIT else "after split",
                "step": step.action,
                "columns": ", ".join(step.columns) or "(whole table)",
                "from": _provenance(step),
            }
        )
    return pd.DataFrame(rows)
