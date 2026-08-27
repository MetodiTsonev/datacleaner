"""Stage 4 tests - validation against declared rules.

Discovery (stage 3) asks "what looks wrong here?". Validation asks "does this meet my
requirements?" -- and the requirements come from a person, not from the data. These
tests cover both halves: that a declared rule is applied exactly, and that an
*inferred* rule is treated as the guess it is.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.loader import read_table
from src.profile import profile_frame
from src.validate import (
    SUSPICIOUS_REJECTION_SHARE,
    Rule,
    check_rule,
    infer_rules,
    rules_table,
    validate,
    violations_table,
)

MESSY = Path(__file__).parent.parent / "data" / "input" / "messy-orders.csv"


# ----------------------------------------------------------------- each rule kind

def test_not_null():
    frame = pd.DataFrame({"a": [1.0, None, 3.0]})
    violation = check_rule(frame, Rule("a", "not_null"))
    assert violation.failing_rows == [1]


def test_unique_flags_every_copy():
    """Both members of a duplicate pair are reported: neither is 'the' offender."""
    frame = pd.DataFrame({"id": ["A", "B", "A", "C"]})
    assert check_rule(frame, Rule("id", "unique")).count == 2


def test_range_lower_and_upper():
    frame = pd.DataFrame({"age": [-5, 30, 200, 40]})
    rule = Rule("age", "range", {"min": 0, "max": 120})
    assert check_rule(frame, rule).failing_rows == [0, 2]


def test_range_treats_a_non_number_as_a_failure():
    """A value that is not a number cannot satisfy a numeric range."""
    frame = pd.DataFrame({"age": ["30", "abc", "40"]})
    assert check_rule(frame, Rule("age", "range", {"min": 0})).failing_rows == [1]


def test_allowed_values():
    frame = pd.DataFrame({"c": ["Sofia", "Paris", "Varna"]})
    rule = Rule("c", "allowed_values", {"values": ["Sofia", "Varna"]})
    assert check_rule(frame, rule).failing_rows == [1]


def test_allowed_values_ignores_nulls():
    """Absence is `not_null`'s business, not this rule's."""
    frame = pd.DataFrame({"c": ["Sofia", None, "Varna"]})
    rule = Rule("c", "allowed_values", {"values": ["Sofia", "Varna"]})
    assert check_rule(frame, rule) is None


def test_pattern():
    frame = pd.DataFrame({"code": ["BG-01", "XX", "BG-02"]})
    rule = Rule("code", "pattern", {"regex": r"^BG-\d{2}$"})
    assert check_rule(frame, rule).failing_rows == [1]


def test_an_invalid_pattern_is_skipped_not_crashed():
    frame = pd.DataFrame({"code": ["a", "b"]})
    assert check_rule(frame, Rule("code", "pattern", {"regex": "([unclosed"})) is None


def test_type_numeric():
    frame = pd.DataFrame({"n": ["1", "two", "3"]})
    rule = Rule("n", "type", {"expected": "numeric"})
    assert check_rule(frame, rule).failing_rows == [1]


def test_compare_across_columns():
    """end_date before start_date - the classic cross-column constraint."""
    frame = pd.DataFrame({
        "start": ["2024-01-01", "2024-05-01", "2024-03-01"],
        "end": ["2024-02-01", "2024-04-01", "2024-04-01"],
    })
    rule = Rule("start", "compare", {"op": "<=", "other": "end"})
    assert check_rule(frame, rule).failing_rows == [1]


def test_compare_on_a_missing_column_is_skipped():
    frame = pd.DataFrame({"a": [1, 2]})
    assert check_rule(frame, Rule("a", "compare", {"other": "nope"})) is None


def test_a_rule_for_an_absent_column_is_skipped():
    assert check_rule(pd.DataFrame({"a": [1]}), Rule("b", "not_null")) is None


# --------------------------------------------------------------------- the split

def test_validate_splits_and_records_every_reason():
    """A row that breaks two rules must say so, or fixing one wastes a round trip."""
    frame = pd.DataFrame({"age": [-5, 30, 40], "city": ["Paris", "Sofia", "Varna"]})
    result = validate(frame, [
        Rule("age", "range", {"min": 0}, inferred=False),
        Rule("city", "allowed_values", {"values": ["Sofia", "Varna"]}, inferred=False),
    ])
    assert len(result.valid) == 2
    assert len(result.rejected) == 1
    reason = result.rejected["__rejected_because"].iloc[0]
    assert "age" in reason and "city" in reason


def test_valid_rows_keep_their_original_columns():
    frame = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    result = validate(frame, [Rule("a", "range", {"min": 0}, inferred=False)])
    assert list(result.valid.columns) == ["a", "b"]
    assert "__rejected_because" not in result.valid.columns


def test_nothing_broken_means_nothing_rejected():
    frame = pd.DataFrame({"age": [30, 40, 50]})
    result = validate(frame, [Rule("age", "range", {"min": 0, "max": 120})])
    assert result.passed
    assert result.rejected.empty
    assert result.summary()["caution"] is None


def test_no_rules_rejects_nothing():
    frame = pd.DataFrame({"a": [1, 2, 3]})
    result = validate(frame, [])
    assert len(result.valid) == 3
    assert result.passed


# ----------------------------------------------- inference is a guess, and says so

def test_inferred_rules_are_marked_as_drafts():
    frame = pd.DataFrame({"age": [30, 40, 50], "city": ["Sofia", "Varna", "Sofia"]})
    rules = infer_rules(frame, profile_frame(frame))
    assert rules, "something should be proposed"
    assert all(r.inferred for r in rules)
    assert all(r.note for r in rules), "each guess must say what it was based on"


def test_a_quantity_gets_a_padded_range():
    frame = pd.DataFrame({"score": [10.0, 20.0, 30.0]})
    rule = next(r for r in infer_rules(frame, profile_frame(frame)) if r.kind == "range")
    assert rule.params["max"] > 30.0, "the observed maximum is rarely the true limit"


def test_a_column_named_like_a_quantity_gets_a_floor_of_zero():
    frame = pd.DataFrame({"amount": [10.0, 20.0, 30.0]})
    rule = next(r for r in infer_rules(frame, profile_frame(frame)) if r.kind == "range")
    assert rule.params["min"] == 0.0
    assert "cannot be negative" in rule.note


def test_a_quantity_that_is_actually_negative_keeps_a_negative_floor():
    """The name is a hint, not an override. Observed data wins."""
    frame = pd.DataFrame({"amount": [-10.0, 20.0, 30.0]})
    rule = next(r for r in infer_rules(frame, profile_frame(frame)) if r.kind == "range")
    assert rule.params["min"] < 0


def test_allowed_values_are_not_inferred_for_high_cardinality():
    frame = pd.DataFrame({"c": [f"v{i}" for i in range(200)]})
    assert not [r for r in infer_rules(frame, profile_frame(frame))
                if r.kind == "allowed_values"]


def test_disguised_blanks_are_not_inferred_as_allowed_values():
    """Otherwise the contract legitimises the very defect stage 3 reported."""
    frame = pd.DataFrame({"c": ["Sofia", "?", "Varna", "Sofia"] * 5})
    rule = next(r for r in infer_rules(frame, profile_frame(frame))
                if r.kind == "allowed_values")
    assert "?" not in rule.params["values"]


def test_a_named_target_must_not_be_null():
    frame = pd.DataFrame({"x": [1, 2, 3], "y": ["a", None, "b"]})
    rules = infer_rules(frame, profile_frame(frame), target="y")
    assert any(r.column == "y" and r.kind == "not_null" for r in rules)


def test_numbers_written_for_people_do_not_get_a_uniqueness_rule():
    """`invoice_total` of "1 403,17" values profiles as an identifier.

    It is a quantity that has not been parsed yet, so "must be unique" would be a
    rule about a defect rather than a requirement.
    """
    frame = pd.DataFrame({"invoice_total": [f"{i} 40{i},1{i}" for i in range(60)]})
    assert not [r for r in infer_rules(frame, profile_frame(frame))
                if r.kind == "unique"]


# ------------------------------------------------- the warning is about the rules

def test_a_large_rejection_share_warns_about_the_rules_not_the_data():
    frame = pd.DataFrame({"c": ["Sofia"] * 5 + ["Paris"] * 5})
    result = validate(frame, [Rule("c", "allowed_values", {"values": ["Sofia"]})])
    caution = result.summary()["caution"]
    assert caution and "inferred" in caution
    assert result.summary()["rejected_share"] > SUSPICIOUS_REJECTION_SHARE


def test_a_declared_rule_gets_a_different_warning():
    frame = pd.DataFrame({"c": ["Sofia"] * 5 + ["Paris"] * 5})
    result = validate(
        frame,
        [Rule("c", "allowed_values", {"values": ["Sofia"]}, inferred=False)],
    )
    caution = result.summary()["caution"]
    assert caution and "you declared" in caution


# ---------------------------------------------------------------------- rendering

def test_rule_labels_read_as_sentences():
    assert Rule("age", "range", {"min": 0, "max": 120}).label == (
        "age must be between 0 and 120"
    )
    assert Rule("id", "unique").label == "id must be unique"
    assert Rule("a", "compare", {"op": "<=", "other": "b"}).label == "a must be <= b"


def test_tables_have_one_row_each():
    frame = pd.DataFrame({"age": [-5, 30]})
    rules = [Rule("age", "range", {"min": 0}, inferred=False)]
    result = validate(frame, rules)
    assert len(rules_table(rules)) == 1
    assert len(violations_table(result.violations)) == 1


# ------------------------------------------------------------------- integration

@pytest.mark.skipif(not MESSY.exists(), reason="messy sample not present")
def test_the_messy_sample_produces_a_reviewable_draft_contract():
    load = read_table(MESSY)
    profiles = profile_frame(load.frame)
    rules = infer_rules(load.frame, profiles, target="returned")
    result = validate(load.frame, rules)
    assert rules, "a draft contract should be proposed"
    assert result.violations, "the messy sample should break some of its own draft"
    assert result.summary()["caution"], "and the caution should fire"


@pytest.mark.skipif(not MESSY.exists(), reason="messy sample not present")
def test_the_defence_question_works():
    """"Could you add a constraint that amount must be positive?" -- ten seconds."""
    load = read_table(MESSY)
    result = validate(
        load.frame,
        [Rule("amount", "range", {"min": 0.01}, inferred=False,
              note="declared during the demonstration")],
    )
    assert result.summary()["rows_rejected"] == 16, "the -999 sentinel rows"
    assert "must be at least 0.01" in result.rejected["__rejected_because"].iloc[0]


# --------------------------------- degenerate rules the form can actually produce
#
# All three were reachable by submitting the add-rule form with its default or blank
# values, and none was covered by the tests above. The first crashed the whole page.

def test_a_range_with_no_bounds_does_not_crash_its_own_label():
    """Submitting the form with both bounds blank raised
    `TypeError: unsupported format string passed to NoneType.__format__`
    while rendering the contract table, taking the page down."""
    assert Rule("amount", "range", {}).label == (
        "amount: a range with no bounds (no effect)"
    )


def test_an_empty_allowed_list_rejects_nothing():
    """An empty set literally means "no value is allowed", so the naive reading
    quarantined 261 of 262 rows on a blank textarea."""
    frame = pd.DataFrame({"c": ["Sofia", "Varna"]})
    result = validate(frame, [Rule("c", "allowed_values", {"values": []})])
    assert len(result.rejected) == 0
    assert len(result.inapplicable) == 1


@pytest.mark.parametrize("rule", [
    Rule("amount", "range", {}),
    Rule("amount", "allowed_values", {"values": []}),
    Rule("amount", "compare", {"op": "<=", "other": "label"}),
    Rule("amount", "pattern", {"regex": "([unclosed"}),
    Rule("absent", "not_null"),
])
def test_a_rule_that_cannot_run_is_reported_not_counted_as_passing(rule):
    """A rule that never ran looks identical to one that passed.

    `compare` against a non-orderable column silently did nothing, and the user had
    no way to tell their constraint was ignored.
    """
    frame = pd.DataFrame({"amount": [1.0, 2.0], "label": ["a", "b"]})
    result = validate(frame, [rule])
    assert result.inapplicable == [rule]
    assert result.summary()["rules_inapplicable"] == 1
