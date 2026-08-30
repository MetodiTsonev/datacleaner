"""Preparing a matrix a model can consume.

Cleaning fixes what is wrong; this transforms what is right but unusable. Every step
is fitted on the training half, so the tests that matter most are the ones checking
the held-back half never influenced an encoding or a mean.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.clean import run
from src.detect import detect
from src.features import (
    CORRELATION_LIMIT,
    ONE_HOT_MAX_CATEGORIES,
    build,
    encode_categories,
    expand_dates,
    log_skewed,
    prune_correlated,
    scale,
    skew_table,
)
from src.loader import read_table
from src.plan import build as build_plan
from src.profile import profile_frame

CENSUS = Path(__file__).parent.parent / "data" / "input" / "adult-census.csv"
MESSY = Path(__file__).parent.parent / "data" / "input" / "messy-orders.csv"


def prepared(path: Path, target: str):
    load = read_table(path)
    profiles = profile_frame(load.frame)
    cleaned = run(
        load.frame,
        build_plan(detect(load.frame, profiles, target=target), target=target),
        target=target,
    )
    # Re-profile: types change once "?" is gone and numbers are parsed.
    return build(cleaned.train, cleaned.test, profile_frame(cleaned.train), target=target)


# ------------------------------------------------------------------ log transform

def test_a_long_tail_is_straightened():
    values = pd.Series(np.random.default_rng(0).lognormal(3, 1.2, 2000))
    frame = pd.DataFrame({"x": values})
    before, after, declined = log_skewed(frame, [], profile_frame(frame))
    assert before["x"] > 4
    assert abs(after["x"]) < 1
    assert not declined


def test_a_zero_inflated_column_is_measured_and_reverted():
    """capital_loss is 95.3% zeros: its skew IS the spike at zero, and log1p(0) is 0.

    Applying the transform anyway would be a step reporting success without producing
    any.
    """
    values = pd.Series([0.0] * 950 + list(np.random.default_rng(1).lognormal(6, 0.4, 50)))
    frame = pd.DataFrame({"x": values})
    original = frame["x"].copy()
    before, after, declined = log_skewed(frame, [], profile_frame(frame))
    assert "x" not in before, "the transform should not be claimed"
    assert "x" in declined
    assert "spike at zero" in declined["x"]
    pd.testing.assert_series_equal(frame["x"], original), "must be reverted exactly"


def test_a_negative_column_is_left_alone():
    """The log of a negative number does not exist."""
    frame = pd.DataFrame({"x": list(np.random.default_rng(2).lognormal(3, 1.2, 500) * -1)})
    before, _, _ = log_skewed(frame, [], profile_frame(frame))
    assert not before


def test_the_target_is_never_transformed():
    values = pd.Series(np.random.default_rng(3).lognormal(3, 1.2, 500))
    frame = pd.DataFrame({"y": values})
    before, _, _ = log_skewed(frame, [], profile_frame(frame), target="y")
    assert not before


def test_the_transform_reaches_the_test_half_too():
    train = pd.DataFrame({"x": np.random.default_rng(4).lognormal(3, 1.2, 500)})
    test = pd.DataFrame({"x": np.random.default_rng(5).lognormal(3, 1.2, 100)})
    log_skewed(train, [test], profile_frame(train))
    assert test["x"].max() < 20, "the test half must get the same transform"


# ------------------------------------------------------------------------- dates

def test_a_date_becomes_its_useful_parts():
    """A timestamp as a number is seconds since 1970, which means nothing to a model."""
    frame = pd.DataFrame({"d": pd.date_range("2024-01-01", periods=40).astype(str)})
    made = expand_dates(frame, [], profile_frame(frame))
    assert set(made) == {"d_year", "d_month", "d_weekday", "d_is_weekend"}
    assert "d" not in frame.columns, "the raw date is replaced"
    assert frame["d_is_weekend"].isin([0, 1]).all()
    assert frame["d_weekday"].between(0, 6).all()


def test_weekends_are_identified_correctly():
    frame = pd.DataFrame({"d": ["2024-01-06", "2024-01-07", "2024-01-08"] * 5})
    expand_dates(frame, [], profile_frame(frame))
    assert frame["d_is_weekend"].tolist()[:3] == [1, 1, 0]  # Sat, Sun, Mon


# ---------------------------------------------------------------------- encoding

def test_few_categories_become_one_column_each():
    frame = pd.DataFrame({"c": ["a", "b", "c"] * 40})
    chosen = encode_categories(frame, [], profile_frame(frame))
    assert "one-hot" in chosen["c"]
    assert {"c=a", "c=b", "c=c"} <= set(frame.columns)
    assert "c" not in frame.columns


def test_many_categories_become_a_frequency():
    """One-hot on a hundred categories is a hundred columns of almost all zeros."""
    frame = pd.DataFrame({"c": [f"v{i % 60}" for i in range(3000)]})
    chosen = encode_categories(frame, [], profile_frame(frame))
    assert "frequency" in chosen["c"]
    assert pd.api.types.is_numeric_dtype(frame["c"])


def test_rare_categories_are_pooled():
    """A category seen twice in a thousand rows cannot support an estimate."""
    frame = pd.DataFrame({"c": ["common"] * 990 + ["rare1"] * 5 + ["rare2"] * 5})
    encode_categories(frame, [], profile_frame(frame))
    assert "c=other" in frame.columns
    assert frame["c=other"].sum() == 10


def test_a_category_only_in_the_test_half_does_not_create_a_column():
    """The encoding is learned from train, so an unseen label falls into 'other'."""
    train = pd.DataFrame({"c": ["a", "b"] * 50})
    test = pd.DataFrame({"c": ["a", "b", "zzz"] * 10})
    encode_categories(train, [test], profile_frame(train))
    assert set(train.columns) == set(test.columns), "same columns on both sides"
    assert "c=zzz" not in test.columns


def test_the_threshold_between_the_two_methods_is_where_it_says():
    small = pd.DataFrame({"c": [f"v{i % ONE_HOT_MAX_CATEGORIES}" for i in range(2000)]})
    assert "one-hot" in encode_categories(small, [], profile_frame(small))["c"]


# ----------------------------------------------------------------------- scaling

def test_scaling_uses_the_training_mean_and_spread():
    train = pd.DataFrame({"x": [10.0, 20.0, 30.0, 40.0]})
    test = pd.DataFrame({"x": [1000.0]})
    learned = scale(train, [test])
    mean, sd = learned["x"]
    assert mean == pytest.approx(25.0)
    assert test["x"].iloc[0] == pytest.approx((1000.0 - mean) / sd), (
        "the test half is transformed by the training statistics, not its own"
    )


def test_a_column_with_no_spread_is_not_divided_by_zero():
    frame = pd.DataFrame({"x": [5.0] * 20})
    assert scale(frame, []) == {}
    assert frame["x"].tolist() == [5.0] * 20


def test_the_target_is_not_scaled():
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [10.0, 20.0, 30.0]})
    scale(frame, [], target="y")
    assert frame["y"].tolist() == [10.0, 20.0, 30.0]


# ------------------------------------------------------------------------ pruning

def test_a_duplicated_feature_is_dropped():
    values = list(np.random.default_rng(6).normal(0, 1, 200))
    other = list(np.random.default_rng(7).normal(0, 1, 200))
    frame = pd.DataFrame({"a": values, "b": values, "c": other})
    dropped = prune_correlated(frame, [])
    assert len(dropped) == 1
    left, right, corr = dropped[0]
    assert {left, right} == {"a", "b"}
    assert corr >= CORRELATION_LIMIT
    assert right not in frame.columns and left in frame.columns


def test_uncorrelated_features_are_kept():
    rng = np.random.default_rng(8)
    frame = pd.DataFrame({"a": rng.normal(0, 1, 300), "b": rng.normal(0, 1, 300)})
    assert prune_correlated(frame, []) == []


def test_the_target_is_never_pruned():
    values = list(np.random.default_rng(9).normal(0, 1, 200))
    frame = pd.DataFrame({"x": values, "y": values})
    prune_correlated(frame, [], target="y")
    assert "y" in frame.columns


# --------------------------------------------------------------------- the stage

def test_each_step_can_be_switched_off():
    frame = pd.DataFrame({"c": ["a", "b"] * 50, "n": list(range(100))})
    _, _, report = build(frame, None, profile_frame(frame), do_encode=False)
    assert not report.encoded


@pytest.mark.skipif(not CENSUS.exists(), reason="census sample not present")
def test_census_becomes_an_all_numeric_matrix():
    """PLAN.md Step 7 acceptance criterion."""
    X, X_test, report = prepared(CENSUS, "income")
    leftover = [
        c for c in X.columns
        if c != "income" and not pd.api.types.is_numeric_dtype(X[c])
    ]
    assert leftover == [], leftover
    assert report.columns_out > report.columns_in, "encoding widens the matrix"
    assert list(X.columns) == list(X_test.columns), "both halves must match"


@pytest.mark.skipif(not CENSUS.exists(), reason="census sample not present")
def test_census_skew_is_reported_before_and_after():
    """PLAN.md predicted capital_gain 11.89 -> ~0. It reaches 3.12, because 91.7% of
    the column is zero and a log cannot move that. Reported, not glossed."""
    _, _, report = prepared(CENSUS, "income")
    assert report.skew_before["capital_gain"] > 11
    assert 2 < report.skew_after["capital_gain"] < 4
    assert "capital_loss" in report.log_declined
    assert not skew_table(report).empty


@pytest.mark.skipif(not MESSY.exists(), reason="messy sample not present")
def test_the_messy_sample_gets_its_dates_expanded():
    X, _, report = prepared(MESSY, "returned")
    assert any("date" in s for s in report.steps)
    assert any(c.endswith("_is_weekend") for c in X.columns)
    leftover = [
        c for c in X.columns
        if c != "returned" and not pd.api.types.is_numeric_dtype(X[c])
    ]
    assert leftover == [], leftover
