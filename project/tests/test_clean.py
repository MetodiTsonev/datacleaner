"""Running the plan.

The tests that matter here are the ones about the split. Everything else is a
transformation you can read; the leakage guarantee is a property, and a property has
to be asserted -- including a positive control showing the assertion can fail.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.clean import (
    apply_impute,
    clean_column_names,
    drop_duplicate_rows,
    drop_empty_rows,
    drop_rows,
    drop_rows_missing_target,
    fit_impute,
    normalise_categories,
    parse_numeric,
    replace_disguised_missing,
    run,
    split,
)
from src.clean import as_table as clean_table
from src.clean import drop_columns as op_drop_columns
from src.detect import detect
from src.loader import read_table
from src.plan import build
from src.profile import profile_frame

CENSUS = Path(__file__).parent.parent / "data" / "input" / "adult-census.csv"
MESSY = Path(__file__).parent.parent / "data" / "input" / "messy-orders.csv"


def cleaned(path: Path, target: str | None = None, **kwargs):
    load = read_table(path)
    profiles = profile_frame(load.frame)
    plan = build(detect(load.frame, profiles, target=target), target=target)
    return load.frame, run(load.frame, plan, target=target, **kwargs)


# ------------------------------------------------------ the leakage guarantee

def test_the_fill_value_comes_from_the_training_half_only():
    """The guarantee, on data where the two halves disagree by design."""
    rng = np.random.default_rng(1)
    frame = pd.DataFrame({
        "x": np.concatenate([rng.normal(10, 1, 500), rng.normal(100, 1, 500)]),
        "y": ["a"] * 500 + ["b"] * 500,
    })
    frame.loc[rng.choice(1000, 50, replace=False), "x"] = np.nan

    train, test = split(frame, target="y", test_size=0.2, seed=7)
    learned = fit_impute(train, {"columns": ["x"], "strategy": "median"})
    value = learned["fill_values"]["x"]

    assert value == pytest.approx(float(train["x"].median()))
    assert value != pytest.approx(float(frame["x"].median())), "must not see the test half"


def test_a_deliberate_leak_is_caught_by_the_same_assertion():
    """Positive control.

    Without this, the test above could pass because the two medians happen to be
    equal rather than because the imputer is honest. Fitting on everything must fail
    the same comparison.
    """
    rng = np.random.default_rng(1)
    frame = pd.DataFrame({
        "x": np.concatenate([rng.normal(10, 1, 500), rng.normal(100, 1, 500)]),
        "y": ["a"] * 500 + ["b"] * 500,
    })
    frame.loc[rng.choice(1000, 50, replace=False), "x"] = np.nan
    train, _ = split(frame, target="y", test_size=0.2, seed=7)

    leaked = fit_impute(frame, {"columns": ["x"], "strategy": "median"})
    assert leaked["fill_values"]["x"] != pytest.approx(float(train["x"].median()))


def test_changing_only_the_test_half_does_not_change_what_is_learned():
    """The strongest form: test rows cannot reach a fitted value."""
    # A unique id per row, so the test rows can be identified exactly. Matching on
    # values instead picks up identical training rows and perturbs those too.
    frame = pd.DataFrame({
        "id": range(200),
        "x": [1.0, 2.0, 3.0, 4.0, 5.0] * 40,
        "y": ["a", "b"] * 100,
    })
    train, test = split(frame, target="y", seed=3)
    before = fit_impute(train, {"columns": ["x"], "strategy": "median"})

    perturbed = frame.copy()
    perturbed.loc[perturbed["id"].isin(test["id"]), "x"] = 9999.0
    train_again, test_again = split(perturbed, target="y", seed=3)

    assert train_again["id"].tolist() == train["id"].tolist(), "same split"
    assert test_again["x"].eq(9999.0).all(), "the test half really was perturbed"
    after = fit_impute(train_again, {"columns": ["x"], "strategy": "median"})
    assert before["fill_values"] == after["fill_values"]


# ----------------------------------------------------------------------- split

def test_the_split_is_stratified():
    frame = pd.DataFrame({"v": range(1000), "y": ["a"] * 950 + ["b"] * 50})
    train, test = split(frame, target="y", test_size=0.2, seed=5)
    for part in (train, test):
        assert (part["y"] == "b").mean() == pytest.approx(0.05, abs=0.01)


def test_the_split_loses_no_rows():
    frame = pd.DataFrame({"v": range(101), "y": ["a", "b"] * 50 + ["a"]})
    train, test = split(frame, target="y", seed=1)
    assert len(train) + len(test) == 101


def test_the_split_is_reproducible():
    frame = pd.DataFrame({"v": range(200), "y": ["a", "b"] * 100})
    a = split(frame, target="y", seed=11)[1]["v"].tolist()
    b = split(frame, target="y", seed=11)[1]["v"].tolist()
    assert a == b


def test_without_a_target_the_split_is_random_but_still_complete():
    frame = pd.DataFrame({"v": range(50)})
    train, test = split(frame, seed=2)
    assert len(train) + len(test) == 50
    assert set(train["v"]) | set(test["v"]) == set(range(50))


def test_a_one_row_frame_does_not_break_the_split():
    train, test = split(pd.DataFrame({"v": [1]}))
    assert len(train) == 1 and test.empty


# ------------------------------------------------------ structural operations

def test_column_names_are_tidied_and_made_unique():
    frame = pd.DataFrame([[1, 2, 3]], columns=["city ", "rating\n", "city"])
    res = clean_column_names(frame, {})
    assert len(set(res.frame.columns)) == 3
    assert "city" in res.frame.columns and "rating" in res.frame.columns
    assert "distinct" in res.detail
    assert res.renames["city "] == "city", "later steps need the new name"


def test_blank_rows_go():
    frame = pd.DataFrame({"a": [1.0, None, 3.0], "b": ["x", None, "z"]})
    assert len(drop_empty_rows(frame, {}).frame) == 2


def test_rows_are_dropped_by_position():
    frame = pd.DataFrame({"a": [1, 2, 3, 4]})
    assert drop_rows(frame, {"positions": [3]}).frame["a"].tolist() == [1, 2, 3]


def test_out_of_range_positions_are_ignored():
    frame = pd.DataFrame({"a": [1, 2]})
    res = drop_rows(frame, {"positions": [99]})
    assert len(res.frame) == 2 and "no matching" in res.detail


def test_human_written_numbers_are_parsed():
    frame = pd.DataFrame({"amount": ["1 234,56", "$100", "12 lv", "oops"]})
    res = parse_numeric(frame, {"columns": ["amount"]})
    assert res.frame["amount"].tolist()[:3] == [1234.56, 100.0, 12.0]
    assert pd.isna(res.frame["amount"].iloc[3]), "unparseable becomes null"
    assert "did not parse" in res.detail


def test_spellings_collapse_onto_the_commonest_form():
    """Not lowercase: the output should stay readable."""
    frame = pd.DataFrame({"city": ["Sofia", "Sofia", "sofia", " Sofia ", "Varna"]})
    res = normalise_categories(frame, {"columns": ["city"]})
    assert res.frame["city"].tolist() == ["Sofia"] * 4 + ["Varna"]
    assert res.cells_changed == 2


def test_normalising_leaves_nulls_null():
    frame = pd.DataFrame({"c": ["Sofia", None, "sofia"]})
    assert normalise_categories(frame, {"columns": ["c"]}).frame["c"].isna().sum() == 1


def test_tokens_become_real_nulls():
    frame = pd.DataFrame({"c": ["Sofia", "?", "N/A", "Varna"]})
    res = replace_disguised_missing(frame, {"columns": ["c"]})
    assert res.frame["c"].isna().sum() == 2
    assert res.cells_changed == 2


def test_numeric_sentinels_become_nulls():
    frame = pd.DataFrame({"amount": [10.0, -999.0, 20.0, -999.0]})
    res = replace_disguised_missing(
        frame, {"columns": ["amount"], "numeric_values": [-999.0]}
    )
    assert res.frame["amount"].isna().sum() == 2
    assert res.cells_changed == 2


def test_duplicate_rows_go():
    frame = pd.DataFrame({"a": [1, 2, 1, 2], "b": ["x", "y", "x", "y"]})
    res = drop_duplicate_rows(frame, {})
    assert len(res.frame) == 2 and "removed 2" in res.detail


def test_rows_with_no_label_are_dropped_not_filled():
    frame = pd.DataFrame({"x": [1, 2, 3], "y": ["a", None, "b"]})
    res = drop_rows_missing_target(frame, {"target": "y"})
    assert len(res.frame) == 2
    assert "no label" in res.detail


def test_dropping_columns_ignores_absent_ones():
    frame = pd.DataFrame({"a": [1], "b": [2]})
    assert list(op_drop_columns(frame, {"columns": ["b", "nope"]}).frame.columns) == ["a"]


# ------------------------------------------------------------------- imputation

def test_auto_strategy_uses_median_for_numbers_and_mode_for_labels():
    train = pd.DataFrame({"n": [1.0, 2.0, 3.0, None], "c": ["a", "a", "b", None]})
    fitted = fit_impute(train, {"columns": ["n", "c"], "strategy": "auto"})
    assert fitted["fill_values"]["n"] == 2.0
    assert fitted["fill_values"]["c"] == "a"


def test_an_indicator_column_records_what_was_filled():
    train = pd.DataFrame({"n": [1.0, None, 3.0]})
    fitted = fit_impute(train, {"columns": ["n"]})
    out, _, filled = apply_impute(train, {"add_indicator": True}, fitted)
    assert out["n_was_missing"].tolist() == [0, 1, 0]
    assert filled == 1


def test_a_group_median_is_used_where_one_was_asked_for():
    train = pd.DataFrame({
        "x": [10.0, 12.0, None, 100.0, 102.0, None],
        "g": ["a", "a", "a", "b", "b", "b"],
    })
    fitted = fit_impute(train, {"columns": ["x"], "strategy": "median", "group_by": "g"})
    out, _, _ = apply_impute(train, {"group_by": "g"}, fitted)
    assert out["x"].tolist() == [10.0, 12.0, 11.0, 100.0, 102.0, 101.0]


def test_a_group_with_no_training_data_falls_back_to_the_global_value():
    train = pd.DataFrame({"x": [10.0, 12.0, None], "g": ["a", "a", "zzz"]})
    fitted = fit_impute(train, {"columns": ["x"], "strategy": "median", "group_by": "g"})
    out, _, _ = apply_impute(train, {"group_by": "g"}, fitted)
    assert out["x"].iloc[2] == 11.0


def test_an_all_null_column_is_left_alone_rather_than_guessed():
    train = pd.DataFrame({"x": [None, None, None]})
    fitted = fit_impute(train, {"columns": ["x"]})
    assert fitted["fill_values"]["x"] is None
    out, _, _ = apply_impute(train, {}, fitted)
    assert out["x"].isna().all()


# ------------------------------------------------------------------- the runner

def test_unimplemented_steps_are_named_not_silently_skipped():
    """log_transform arrives in step 7. cap_outliers landed in step 6."""
    _, result = cleaned(CENSUS, target="income")
    assert {s.action for s in result.skipped} == {"log_transform"}


@pytest.mark.skipif(not CENSUS.exists(), reason="census sample not present")
def test_census_ends_with_no_nulls_and_the_duplicates_gone():
    """PLAN.md Step 5 acceptance criterion."""
    before, result = cleaned(CENSUS, target="income")
    assert int(before.isna().to_numpy().sum()) == 0, "pandas saw nothing wrong"
    assert int((before == "?").any(axis=1).sum()) == 3620

    dedup = [a for a in result.applied if a.step.action == "drop_duplicate_rows"][0]
    assert dedup.rows_removed == 52
    assert result.summary()["nulls_remaining"] == 0
    assert result.train.isna().to_numpy().sum() == 0
    assert result.test is not None and result.test.isna().to_numpy().sum() == 0


@pytest.mark.skipif(not CENSUS.exists(), reason="census sample not present")
def test_the_target_survives_cleaning_untouched():
    before, result = cleaned(CENSUS, target="income")
    assert "income" in result.train.columns
    assert set(result.train["income"].unique()) <= set(before["income"].unique())


@pytest.mark.skipif(not MESSY.exists(), reason="messy sample not present")
def test_the_messy_sample_runs_end_to_end():
    _, result = cleaned(MESSY, target="returned")
    assert result.summary()["steps_applied"] >= 7
    assert result.summary()["nulls_remaining"] == 0
    assert len(clean_table(result)) == len(result.applied)


@pytest.mark.skipif(not MESSY.exists(), reason="messy sample not present")
def test_every_applied_step_reports_what_it_did():
    _, result = cleaned(MESSY, target="returned")
    for a in result.applied:
        assert a.detail and a.detail != "", a.step.action


# --------------------------------------------- renaming must reach the later steps

def test_renaming_a_column_updates_every_later_step():
    """The bug: clean_column_names renamed "city " to "city", and every later step
    still held the old name. normalise_categories reported "nothing to unify" and
    replace_disguised_missing missed 18 "?" values, while the run reported success.
    """
    frame = pd.DataFrame({"city ": ["Sofia", "sofia", "?", "Varna"], "n": [1, 2, 3, 4]})
    from src.finding import Finding, Suggestion
    from src.plan import build as build_plan

    findings = [
        Finding(check="column_names", severity="medium", topic="structure",
                message="m", suggestion=Suggestion("clean_column_names")),
        Finding(check="inconsistent_categories", severity="medium", topic="structure",
                message="m", columns=["city "],
                suggestion=Suggestion("normalise_categories", {"columns": ["city "]})),
        Finding(check="disguised_missing", severity="critical", topic="missing",
                message="m", columns=["city "],
                suggestion=Suggestion("replace_disguised_missing",
                                      {"columns": ["city "], "tokens": ["?"]})),
    ]
    result = run(frame, build_plan(findings))
    by_action = {a.step.action: a for a in result.applied}
    assert "nothing to unify" not in by_action["normalise_categories"].detail
    nulled = by_action["replace_disguised_missing"]
    assert "city" in nulled.detail
    assert nulled.cells_changed == 1, "the ? should have been turned into a null"
    # The derived imputation then fills it, which is why the final frame has no nulls.
    assert result.frame["city"].isna().sum() == 0
    # Order is not preserved - .frame is train and test concatenated.
    assert sorted(result.frame["city"].tolist()) == ["Sofia", "Sofia", "Sofia", "Varna"]


def test_whitespace_variants_pool_their_votes():
    """A trailing space must never be able to win the vote for canonical form."""
    frame = pd.DataFrame({"c": ["Varna", "varna ", " varna", "Varna "]})
    assert normalise_categories(frame, {"columns": ["c"]}).frame["c"].tolist() == (
        ["Varna"] * 4
    )


def test_case_is_still_decided_by_count():
    """"US" is not "us", so case is left to the vote rather than folded."""
    frame = pd.DataFrame({"c": ["US", "US", "us"]})
    assert normalise_categories(frame, {"columns": ["c"]}).frame["c"].tolist() == (
        ["US"] * 3
    )


def test_drop_rows_still_finds_its_row_after_an_earlier_step_removed_one():
    """The checks report row labels, not positions.

    Regression: a blank row removed by an earlier step shifts every later position by
    one. Read positionally, the last row's label falls out of range, drop_rows reports
    "no matching rows", and the totals row it was meant to remove survives into the
    cleaned data as the maximum of every numeric column.
    """
    frame = pd.DataFrame({"item": ["a", None, "b", "TOTAL"], "n": [1.0, None, 2.0, 3.0]})
    after_blank = drop_empty_rows(frame, {}).frame
    assert len(after_blank) == 3

    result = drop_rows(after_blank, {"positions": [3]})  # label 3, now at position 2
    assert "TOTAL" not in result.frame["item"].tolist(), result.detail
    assert len(result.frame) == 2


def test_drop_rows_ignores_a_label_that_is_already_gone():
    frame = pd.DataFrame({"v": [1, 2, 3]})
    assert drop_rows(frame, {"positions": [99]}).detail == "no matching rows"
