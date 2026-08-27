"""Findings -> ordered plan.

Three behaviours a detector can't provide, and each has to be checked: merging
several findings into one step, deriving a step nobody asked for, and resolving two
findings that contradict.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.detect import detect
from src.finding import Finding, Suggestion
from src.loader import read_table
from src.plan import ORDER, POST_SPLIT, PRE_SPLIT, as_table, build
from src.profile import profile_frame

CENSUS = Path(__file__).parent.parent / "data" / "input" / "adult-census.csv"
MESSY = Path(__file__).parent.parent / "data" / "input" / "messy-orders.csv"


def finding(check: str, action: str | None, columns=None, **params) -> Finding:
    return Finding(
        check=check, severity="medium", topic="structure", message="m",
        columns=columns or [],
        suggestion=None if action is None
        else Suggestion(action=action, params={"columns": columns or [], **params}),
    )


def plan_for(path: Path, target: str | None = None):
    load = read_table(path)
    profiles = profile_frame(load.frame)
    return build(detect(load.frame, profiles, target=target), target=target)


# --------------------------------------------------------------------------- order

def test_steps_come_out_in_the_declared_order():
    findings = [
        finding("high_skew", "log_transform", ["a"]),
        finding("column_names", "clean_column_names"),
        finding("missing_values", "impute", ["b"]),
        finding("empty_rows", "drop_empty_rows"),
    ]
    actions = [s.action for s in build(findings).steps]
    positions = [[a for a, _, _ in ORDER].index(a) for a in actions]
    assert positions == sorted(positions)
    assert actions[0] == "clean_column_names", "headers first"


def test_every_step_is_marked_pre_or_post_split():
    for step in plan_for(MESSY).steps:
        assert step.stage in (PRE_SPLIT, POST_SPLIT)


def test_learning_steps_are_after_the_split():
    """Anything that computes a value from data must be fitted on train only."""
    plan = plan_for(MESSY)
    for step in plan.steps:
        if step.action in ("impute", "cap_outliers", "log_transform"):
            assert step.stage == POST_SPLIT, step.action


def test_deduplication_happens_before_the_split():
    """Copies straddling the boundary cannot be removed afterwards."""
    plan = plan_for(MESSY)
    dedup = [s for s in plan.steps if s.action == "drop_duplicate_rows"]
    assert dedup and dedup[0].stage == PRE_SPLIT


# -------------------------------------------------------------------------- merge

def test_findings_on_three_columns_become_one_step():
    findings = [
        finding("disguised_missing", "replace_disguised_missing", ["a"]),
        finding("disguised_missing", "replace_disguised_missing", ["b"]),
        finding("disguised_missing", "replace_disguised_missing", ["c"]),
    ]
    steps = build(findings).steps
    replace = [s for s in steps if s.action == "replace_disguised_missing"]
    assert len(replace) == 1, "three findings, one operation"
    assert replace[0].columns == ["a", "b", "c"]


def test_merging_records_every_check_that_asked():
    findings = [
        finding("uninformative_columns", "drop_columns", ["a"]),
        finding("identifier_columns", "drop_columns", ["b"]),
    ]
    step = build(findings).steps[0]
    assert set(step.from_checks) == {"uninformative_columns", "identifier_columns"}
    assert step.columns == ["a", "b"]


# ------------------------------------------------------------------------- derive

def test_converting_tokens_to_nulls_derives_an_imputation_step():
    """Nothing reported this: the nulls did not exist when the detectors ran."""
    findings = [finding("disguised_missing", "replace_disguised_missing", ["a", "b"])]
    plan = build(findings)
    impute = [s for s in plan.steps if s.action == "impute"]
    assert impute, "imputation must be derived"
    assert impute[0].derived_from == "replace_disguised_missing"
    assert impute[0].columns == ["a", "b"]
    assert impute[0].stage == POST_SPLIT


def test_parsing_numbers_also_derives_imputation():
    findings = [finding("numeric_in_text", "parse_numeric", ["a"])]
    step = [s for s in build(findings).steps if s.action == "impute"][0]
    assert step.derived_from == "parse_numeric"


def test_a_derived_step_merges_with_one_a_detector_asked_for():
    findings = [
        finding("disguised_missing", "replace_disguised_missing", ["a"]),
        finding("missing_values", "impute", ["b"]),
    ]
    step = [s for s in build(findings).steps if s.action == "impute"][0]
    assert set(step.columns) == {"a", "b"}
    assert step.derived_from and step.from_checks


def test_the_target_is_never_imputed_by_derivation():
    """Its rows are dropped instead - a guessed label is worse than a missing row."""
    findings = [
        finding("disguised_missing", "replace_disguised_missing", ["a", "label"])
    ]
    step = [s for s in build(findings, target="label").steps if s.action == "impute"][0]
    assert step.columns == ["a"]


def test_nothing_is_derived_when_no_step_creates_nulls():
    findings = [finding("exact_duplicates", "drop_duplicate_rows")]
    assert not [s for s in build(findings).steps if s.action == "impute"]


# ----------------------------------------------------------------------- conflicts

def test_a_column_is_not_imputed_after_being_dropped():
    """The plan said "drop order_id" then "impute order_id"."""
    findings = [
        finding("identifier_columns", "drop_columns", ["gone"]),
        finding("missing_values", "impute", ["gone"]),
        finding("missing_values", "impute", ["kept"]),
    ]
    plan = build(findings)
    impute = [s for s in plan.steps if s.action == "impute"][0]
    assert impute.columns == ["kept"]


def test_a_step_left_with_no_columns_is_removed_entirely():
    findings = [
        finding("identifier_columns", "drop_columns", ["gone"]),
        finding("missing_values", "impute", ["gone"]),
    ]
    assert [s.action for s in build(findings).steps] == ["drop_columns"]


def test_a_parseable_column_is_not_dropped_as_dead():
    """invoice_total holds "1 403,17", so it profiles as an identifier.

    It is an unparsed quantity, not a key. Two findings contradict and parsing wins:
    dropping loses a real feature.
    """
    findings = [
        finding("identifier_columns", "drop_columns", ["invoice_total", "order_id"]),
        finding("numeric_in_text", "parse_numeric", ["invoice_total"]),
    ]
    plan = build(findings)
    drop = [s for s in plan.steps if s.action == "drop_columns"][0]
    assert drop.columns == ["order_id"]
    parse = [s for s in plan.steps if s.action == "parse_numeric"][0]
    assert "invoice_total" in parse.why, "the rescue should be explained"


def test_parsing_comes_before_dropping():
    """Parsing changes what a column is, so the drop decision must follow it."""
    actions = [a for a, _, _ in ORDER]
    assert actions.index("parse_numeric") < actions.index("drop_columns")


def test_a_drop_step_that_loses_every_column_disappears():
    findings = [
        finding("identifier_columns", "drop_columns", ["x"]),
        finding("numeric_in_text", "parse_numeric", ["x"]),
    ]
    assert "drop_columns" not in [s.action for s in build(findings).steps]


# --------------------------------------------------------------------- unaddressed

def test_findings_without_a_repair_are_reported_not_dropped():
    findings = [
        finding("mixed_types", None, ["a"]),
        finding("encoding_damage", None, ["b"]),
        finding("exact_duplicates", "drop_duplicate_rows"),
    ]
    plan = build(findings)
    assert len(plan.steps) == 1
    assert {f.check for f in plan.unaddressed} == {"mixed_types", "encoding_damage"}


def test_an_unknown_action_is_reported_rather_than_silently_skipped():
    """A check proposing a repair the plan has no slot for must not vanish."""
    findings = [finding("something", "teleport_the_data", ["a"])]
    plan = build(findings)
    assert not plan.steps
    assert plan.unaddressed


# ------------------------------------------------------------------- integration

@pytest.mark.skipif(not CENSUS.exists(), reason="census sample not present")
def test_census_plan_matches_the_acceptance_criterion():
    """PLAN.md Step 4: an ordered plan containing the derived imputation step."""
    plan = plan_for(CENSUS, target="income")
    actions = [s.action for s in plan.steps]
    assert actions == [
        "drop_columns", "replace_disguised_missing", "drop_duplicate_rows",
        "impute", "cap_outliers", "log_transform",
    ]
    impute = [s for s in plan.steps if s.action == "impute"][0]
    assert impute.derived_from == "replace_disguised_missing"
    assert impute.columns == ["native_country", "occupation", "workclass"]


@pytest.mark.skipif(not MESSY.exists(), reason="messy sample not present")
def test_no_column_is_acted_on_after_being_dropped():
    plan = plan_for(MESSY, target="returned")
    dropped: set[str] = set()
    for step in plan.steps:
        if step.action == "drop_columns":
            dropped |= set(step.columns)
        else:
            assert not (set(step.columns) & dropped), step.action


@pytest.mark.skipif(not MESSY.exists(), reason="messy sample not present")
def test_as_table_lists_steps_in_execution_order():
    plan = plan_for(MESSY, target="returned")
    table = as_table(plan)
    assert list(table["#"]) == list(range(1, len(plan.steps) + 1))
    assert all(table["from"] != "")
