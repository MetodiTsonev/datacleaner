"""What was done, as a file you can keep.

The cleaned CSV on its own is not a result: it is a table that looks fine and carries no
account of how it got that way. Six months later nobody can say which columns were
filled, with what, or why 52 rows disappeared. This writes that account next to the
data.

Markdown rather than JSON because the audience is a person, and because it pastes into
the thesis. Every number here comes from a `summary()` that already exists - this module
assembles, it does not compute, and nothing here may derive a figure of its own.
"""

from __future__ import annotations

from src.clean import CleanResult
from src.detect import summarise
from src.evaluate import Comparison, verdict
from src.features import FeatureReport
from src.finding import SEVERITIES, Finding
from src.loader import LoadResult
from src.plan import PRE_SPLIT, Plan
from src.validate import ValidationResult


def _bullets(lines: list[str]) -> str:
    return "\n".join(f"- {line}" for line in lines) if lines else "- (none)"


def _reading(load: LoadResult) -> str:
    s = load.summary()
    lines = [
        f"**{s['rows']:,}** rows x **{s['columns']}** columns",
        f"encoding `{s['encoding']}`, delimiter {s['delimiter']}, decimal `{s['decimal']}`",
        f"header row: {s['header_row']}",
    ]
    if s["skipped_rows"]:
        lines.append(f"skipped **{s['skipped_rows']}** line(s) above the header")
    lines.extend(s["notes"])
    return _bullets(lines)


def _found(findings: list[Finding]) -> str:
    if not findings:
        return "Nothing was flagged."
    counts = summarise(findings)["by_severity"]
    head = ", ".join(f"**{counts[s]}** {s}" for s in SEVERITIES if counts.get(s))
    serious = [f for f in findings if f.severity in ("critical", "high")]
    shown = serious[:10]
    lines = [f"{f.severity}: {f.message}" for f in shown]
    # Truncation is stated rather than silent - a list that stops without saying so
    # reads as the complete list.
    tail = "" if len(shown) == len(serious) else (
        f"\n\n...and {len(serious) - len(shown)} more; the full list is in the app."
    )
    return f"{head}.\n\n{_bullets(lines)}{tail}"


def _planned(plan: Plan) -> str:
    lines = []
    for i, step in enumerate(plan.steps, 1):
        when = "before the split" if step.stage == PRE_SPLIT else "after the split"
        mark = " *(derived - no check asked for this)*" if step.derived_from else ""
        lines.append(f"{i}. `{step.action}` ({when}){mark} - {step.why}")
    body = "\n".join(lines) if lines else "- (nothing to do)"
    if plan.unaddressed:
        body += "\n\n**Reported but not repaired** (named rather than dropped):\n"
        body += _bullets([f.message for f in plan.unaddressed])
    return body


def _done(result: CleanResult) -> str:
    s = result.summary()
    lines = []
    for i, a in enumerate(result.applied, 1):
        change = a.detail
        if a.rows_removed:
            change += f"; {a.rows_removed:,} row(s) removed"
        if a.columns_removed:
            change += f"; {a.columns_removed} column(s) removed"
        lines.append(f"{i}. `{a.step.action}` - {change}")
    head = (
        f"**{s['cells_changed']:,}** cells changed, **{s['rows_removed']:,}** rows "
        f"removed, **{s['nulls_remaining']}** nulls remaining. "
        f"Split into **{s['train_rows']:,}** training and **{s['test_rows']:,}** "
        "held-out rows."
    )
    body = f"{head}\n\n" + ("\n".join(lines) if lines else "- (nothing applied)")
    if result.skipped:
        body += "\n\n**Skipped** (planned but not performed):\n"
        body += _bullets([f"`{s.action}` - {s.why}" for s in result.skipped])
    return body


def _validated(validation: ValidationResult | None) -> str:
    if validation is None:
        return "Not run."
    s = validation.summary()
    lines = [
        f"**{s['rules_checked']}** rules checked, **{s['rules_broken']}** broken",
        f"**{s['rows_valid']:,}** rows valid, **{s['rows_rejected']:,}** rejected "
        f"({s['rejected_share']:.1%})",
    ]
    if s.get("caution"):
        lines.append(f"caution: {s['caution']}")
    return _bullets(lines)


def _prepared(features: FeatureReport | None) -> str:
    if features is None:
        return "Not run."
    s = features.summary()
    lines = [f"**{s['columns_in']}** columns in, **{s['columns_out']}** out"]
    lines.extend(features.steps)
    for column, reason in features.log_declined.items():
        lines.append(f"declined to transform `{column}` - {reason}")
    return _bullets(lines)


def _evidence(comparison: Comparison | None) -> str:
    if comparison is None:
        return (
            "Not measured. Without a column to predict there is nothing to score, so "
            "the only claim this report supports is that the data *changed*."
        )
    kind, sentence = verdict(comparison)
    if kind == "unscorable":
        return f"Could not be measured: {sentence}"
    s = comparison.summary()
    return _bullets([
        f"as uploaded: **{s['raw_auc']:.4f}** ROC AUC",
        f"after this pipeline: **{s['cleaned_auc']:.4f}** ({s['difference']:+.4f})",
        f"**{sentence}**",
        f"both scored on the same **{comparison.cleaned.rows:,}**-row training set and "
        "the same held-out rows",
        f"if you had instead deleted every incomplete row: **{comparison.naive_rows:,}** "
        "rows would remain",
    ])


LIMITS = """- Everything learned - fill values, category lists, outlier bounds, scaling -
  came from the training half only. The held-out rows were never fitted on.
- One model and one file. A different model may rank the two arms differently.
- A single run's difference is smaller than the variation between random splits, so
  one number here is weak evidence on its own.
- A transformation can improve the statistic it was applied for and still reduce how
  useful the data is."""


def build(
    load: LoadResult,
    findings: list[Finding],
    plan: Plan,
    result: CleanResult,
    *,
    target: str | None = None,
    validation: ValidationResult | None = None,
    features: FeatureReport | None = None,
    comparison: Comparison | None = None,
) -> str:
    """The whole account, as Markdown."""
    return f"""# What was done to `{load.name}`

Produced by DataCleaner. Column to predict: {f"`{target}`" if target else "none chosen"}.

## 1. How the file was read

{_reading(load)}

## 2. What was found

{_found(findings)}

## 3. What was planned

{_planned(plan)}

## 4. What was done

{_done(result)}

## 5. Validation against declared rules

{_validated(validation)}

## 6. Preparation for a model

{_prepared(features)}

## 7. Did it help?

{_evidence(comparison)}

## 8. What this does not tell you

{LIMITS}
"""
