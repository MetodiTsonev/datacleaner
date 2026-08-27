"""Validation against declared rules - the "валидиране" verb.

Detection asks "what looks wrong here?". This asks "does this meet my requirements?"
The rules come from a person, because a declared rule carries information the data
doesn't: that an age can't be negative isn't derivable from the values.

Rules are inferred as a *draft* and marked as such - inference can't tell intent from
an accident of one batch. Output is a split, not a score: rejected rows are kept with
their reason, because binning data that failed a rule you wrote a minute ago is how
datasets get quietly destroyed.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import pandas as pd

from src.profile import ColumnProfile
from src.text import DISGUISED_TOKENS, numeric_share_after_cleaning, plain

RuleKind = Literal[
    "not_null", "unique", "range", "allowed_values", "pattern", "type", "compare"
]

RANGE_PADDING = 0.10  # the observed minimum is almost never the true limit
MAX_INFERRED_CATEGORIES = 25  # above this the observed set is probably incomplete
SUSPICIOUS_REJECTION_SHARE = 0.10  # above this, suspect the rules, not the data

# The one place we guess at meaning from a name. Offered as a draft, never applied.
NON_NEGATIVE_HINTS = (
    "age", "amount", "price", "cost", "total", "quantity", "qty", "count",
    "weight", "height", "length", "duration", "days", "hours", "salary",
    "income", "sum", "възраст", "сума", "цена", "количество", "брой",
)


@dataclass
class Rule:
    """One declared constraint."""

    column: str
    kind: RuleKind
    # min/max for range, values for allowed_values, regex for pattern,
    # expected for type, other+op for compare.
    params: dict[str, Any] = field(default_factory=dict)
    inferred: bool = True  # False once a person wrote or reviewed it
    note: str = ""  # why it exists; shown beside it

    @property
    def label(self) -> str:
        if self.kind == "not_null":
            return f"{self.column} must not be empty"
        if self.kind == "unique":
            return f"{self.column} must be unique"
        if self.kind == "range":
            lo, hi = self.params.get("min"), self.params.get("max")
            if lo is not None and hi is not None:
                return f"{self.column} must be between {lo:g} and {hi:g}"
            if lo is not None:
                return f"{self.column} must be at least {lo:g}"
            if hi is not None:
                return f"{self.column} must be at most {hi:g}"
            # A range with neither bound constrains nothing. Reachable from the UI by
            # submitting the form with both fields blank, which crashed on formatting
            # None. Say what it is instead.
            return f"{self.column}: a range with no bounds (no effect)"
        if self.kind == "allowed_values":
            values = self.params.get("values", [])
            if not values:
                return f"{self.column}: an empty list of allowed values (no effect)"
            shown = ", ".join(map(str, values[:4]))
            more = f" (+{len(values) - 4} more)" if len(values) > 4 else ""
            return f"{self.column} must be one of: {shown}{more}"
        if self.kind == "pattern":
            return f"{self.column} must match {self.params.get('regex')}"
        if self.kind == "type":
            return f"{self.column} must be {self.params.get('expected')}"
        if self.kind == "compare":
            op = self.params.get("op", "<=")
            return f"{self.column} must be {op} {self.params.get('other')}"
        return f"{self.column}: {self.kind}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Violation:
    """One rule broken, and by how much."""

    rule: Rule
    failing_rows: list[int]
    examples: list[Any] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.failing_rows)


@dataclass
class ValidationResult:
    """The split, plus why each rejected row was rejected."""

    valid: pd.DataFrame
    rejected: pd.DataFrame
    violations: list[Violation]
    rules_checked: int
    # Rules that couldn't run at all. Separate because "passed" and "never ran" look
    # identical otherwise, and someone who declares a rule should know it did nothing.
    inapplicable: list[Rule] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.violations

    def summary(self) -> dict[str, Any]:
        total = len(self.valid) + len(self.rejected)
        share = len(self.rejected) / total if total else 0.0
        inferred_broken = sum(1 for v in self.violations if v.rule.inferred)
        return {
            "rules_checked": self.rules_checked,
            "rules_broken": len(self.violations),
            "rows_valid": len(self.valid),
            "rows_rejected": len(self.rejected),
            "rejected_share": share,
            "rules_inapplicable": len(self.inapplicable),
            "caution": self._caution(share, inferred_broken),
        }

    def _caution(self, share: float, inferred_broken: int) -> str | None:
        """A warning about the rules, not the data.

        Quarantining a big share of a file nearly always means a draft rule is wrong.
        """
        if share <= SUSPICIOUS_REJECTION_SHARE or not self.violations:
            return None
        if inferred_broken:
            return (
                f"{share:.0%} of rows were rejected, and {inferred_broken} of the "
                f"{len(self.violations)} broken rules were **inferred** rather than "
                "declared. A draft rule read off one batch is a description of that "
                "batch, not a requirement. Review the rules before trusting this "
                "split."
            )
        return (
            f"{share:.0%} of rows were rejected by rules you declared. Worth "
            "confirming that is intended before the quarantine is discarded."
        )


# --------------------------------------------------------------------- inference


def infer_rules(
    frame: pd.DataFrame, profiles: list[ColumnProfile], *, target: str | None = None
) -> list[Rule]:
    """Draft contract from what this batch happens to look like.

    Every rule is a guess about intent, marked inferred, with a note saying what it
    was based on so a reviewer can disagree.
    """
    rules: list[Rule] = []
    for profile in profiles:
        column = profile.name
        series = frame[column]

        # An identifier whose values are numbers written for people is a quantity
        # that has not been parsed yet, not a key. Inferring "must be unique" for it
        # would be a rule about a defect. Skip the column until stage 5 has parsed it.
        if profile.semantic_type == "identifier" and _unparsed_number(series):
            continue

        if profile.n_missing == 0 and profile.semantic_type != "empty":
            rules.append(
                Rule(
                    column,
                    "not_null",
                    note="nothing is missing in this batch, so it may be required",
                )
            )

        if profile.semantic_type == "identifier":
            rules.append(
                Rule(column, "unique", note="every value in this batch is distinct")
            )

        if profile.semantic_type == "numeric":
            values = pd.to_numeric(series, errors="coerce").dropna()
            if not values.empty:
                low, high = float(values.min()), float(values.max())
                span = high - low or abs(high) or 1.0
                floor = 0.0 if _looks_non_negative(column, low) else low - span * RANGE_PADDING
                rules.append(
                    Rule(
                        column,
                        "range",
                        {"min": round(floor, 4), "max": round(high + span * RANGE_PADDING, 4)},
                        note=(
                            f"observed {low:g} to {high:g}"
                            + (
                                "; the name suggests a quantity that cannot be "
                                "negative, so the lower bound is set to 0"
                                if floor == 0.0 and low >= 0
                                else f", padded by {RANGE_PADDING:.0%}"
                            )
                        ),
                    )
                )

        if (
            profile.semantic_type in {"categorical", "boolean"}
            and profile.n_distinct <= MAX_INFERRED_CATEGORIES
        ):
            observed = sorted(
                {
                    str(v)
                    for v in series.dropna().unique()
                    if str(v).strip().casefold() not in DISGUISED_TOKENS
                }
            )
            if observed:
                rules.append(
                    Rule(
                        column,
                        "allowed_values",
                        {"values": observed},
                        note=(
                            f"{len(observed)} distinct values in this batch - a new "
                            "value in a later file is not necessarily an error"
                        ),
                    )
                )

    # A label can't be imputed, so requiring it is the only honest rule.
    if target and not any(r.column == target and r.kind == "not_null" for r in rules):
        rules.append(
            Rule(
                target,
                "not_null",
                note="this is the column being predicted; a row without a label "
                "cannot be used and cannot be filled in",
            )
        )

    return rules


def _unparsed_number(series: pd.Series) -> bool:
    """Text that is really a number with human formatting -- a quantity, not a key."""
    text = series.dropna().astype("string").str.strip()
    return not text.empty and numeric_share_after_cleaning(text) >= 0.9


def _looks_non_negative(column: str, observed_min: float) -> bool:
    """Whether the column name suggests a quantity with no negative meaning."""
    if observed_min < 0:
        return False
    name = column.strip().casefold()
    return any(hint in name for hint in NON_NEGATIVE_HINTS)


# -------------------------------------------------------------------- validation


def is_applicable(frame: pd.DataFrame, rule: Rule) -> bool:
    """Whether this rule can be evaluated against this frame at all."""
    if rule.column not in frame.columns:
        return False
    return _failing_mask(frame, frame[rule.column], rule) is not None


def check_rule(frame: pd.DataFrame, rule: Rule) -> Violation | None:
    """Apply one rule. Returns the violation, or None when it holds or cannot run."""
    if rule.column not in frame.columns:
        return None
    series = frame[rule.column]
    failed = _failing_mask(frame, series, rule)
    if failed is None or not failed.any():
        return None
    positions = [int(i) for i in frame.index[failed]]
    return Violation(
        rule=rule,
        failing_rows=positions,
        examples=[plain(v) for v in series[failed].head(5)],
    )


def _failing_mask(
    frame: pd.DataFrame, series: pd.Series, rule: Rule
) -> pd.Series | None:
    """Rows that break the rule. None when the rule cannot be applied here."""
    if rule.kind == "not_null":
        return series.isna()

    if rule.kind == "unique":
        present = series.notna()
        return series.duplicated(keep=False) & present

    if rule.kind == "range":
        low, high = rule.params.get("min"), rule.params.get("max")
        if low is None and high is None:
            return None  # constrains nothing; not an error, just inert
        values = pd.to_numeric(series, errors="coerce")
        failed = pd.Series(False, index=series.index)
        if low is not None:
            failed |= values < low
        if high is not None:
            failed |= values > high
        # A value that is not a number at all fails a numeric range.
        return failed | (values.isna() & series.notna())

    if rule.kind == "allowed_values":
        allowed = {str(v) for v in rule.params.get("values", [])}
        if not allowed:
            # "Nothing is allowed" would reject every row. Almost certainly an
            # unfinished rule rather than an intention.
            return None
        return series.notna() & ~series.astype("string").isin(allowed)

    if rule.kind == "pattern":
        regex = rule.params.get("regex") or ""
        try:
            compiled = re.compile(regex)
        except re.error:
            return None
        return series.notna() & ~series.astype("string").str.match(compiled, na=False)

    if rule.kind == "type":
        expected = rule.params.get("expected")
        text = series.dropna().astype("string")
        if expected == "numeric":
            bad = pd.to_numeric(text, errors="coerce").isna()
        elif expected == "datetime":
            bad = pd.to_datetime(text, errors="coerce", format="mixed").isna()
        else:
            return None
        failed = pd.Series(False, index=series.index)
        failed.loc[bad.index[bad]] = True
        return failed

    if rule.kind == "compare":
        other = rule.params.get("other")
        if other not in frame.columns:
            return None
        left = _comparable(series)
        right = _comparable(frame[other])
        if left is None or right is None:
            return None
        op = rule.params.get("op", "<=")
        both = left.notna() & right.notna()
        if op == "<=":
            return both & ~(left <= right)
        if op == "<":
            return both & ~(left < right)
        if op == ">=":
            return both & ~(left >= right)
        if op == ">":
            return both & ~(left > right)
        if op == "==":
            return both & ~(left == right)
        return None

    return None


def _comparable(series: pd.Series) -> pd.Series | None:
    """Coerce to something orderable - numbers or dates - or give up."""
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        return numeric
    dates = pd.to_datetime(series, errors="coerce", format="mixed")
    return dates if dates.notna().any() else None


def validate(frame: pd.DataFrame, rules: list[Rule]) -> ValidationResult:
    """Apply every rule and split.

    The reason lists every rule a row broke, not just the first - fixing one and
    resubmitting to fail the next is how a validation loop wastes a day.
    """
    violations = [v for v in (check_rule(frame, r) for r in rules) if v is not None]
    inapplicable = [r for r in rules if not is_applicable(frame, r)]

    reasons: dict[int, list[str]] = {}
    for violation in violations:
        for row in violation.failing_rows:
            reasons.setdefault(row, []).append(violation.rule.label)

    failing = sorted(reasons)
    rejected = frame.loc[failing].copy()
    if not rejected.empty:
        rejected.insert(
            0, "__rejected_because", [" | ".join(reasons[i]) for i in failing]
        )
    valid = frame.drop(index=failing)
    return ValidationResult(
        valid=valid,
        rejected=rejected,
        violations=sorted(violations, key=lambda v: (-v.count, v.rule.column)),
        rules_checked=len(rules),
        inapplicable=inapplicable,
    )


# -------------------------------------------------------------------- rendering


def rules_table(rules: list[Rule]) -> pd.DataFrame:
    """Rules as an editable display table."""
    return pd.DataFrame(
        [
            {
                "column": r.column,
                "rule": r.kind,
                "constraint": r.label,
                "source": "inferred" if r.inferred else "declared",
                "why": r.note,
            }
            for r in rules
        ]
    )


def violations_table(violations: list[Violation]) -> pd.DataFrame:
    """Violations as a display table, worst first."""
    return pd.DataFrame(
        [
            {
                "column": v.rule.column,
                "constraint": v.rule.label,
                "rows failing": v.count,
                "examples": ", ".join(str(e) for e in v.examples[:3]),
            }
            for v in violations
        ]
    )
