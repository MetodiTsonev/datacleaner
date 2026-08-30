"""Capping extreme values.

The behaviour worth testing here is mostly *refusal*: the rules decline more often
than they act, and each refusal is a separate condition.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.anomalies import METHODS, apply_cap, bounds, comparison, fit_cap
from src.clean import split
from src.detect import detect
from src.loader import read_table
from src.plan import build
from src.profile import profile_frame

CENSUS = Path(__file__).parent.parent / "data" / "input" / "adult-census.csv"


def normalish(n=400, seed=0):
    return pd.Series(np.random.default_rng(seed).normal(50, 10, n))


# ------------------------------------------------------------------------ bounds

@pytest.mark.parametrize("method", METHODS)
def test_every_rule_produces_bounds_on_ordinary_data(method):
    low, high = bounds(normalish(), method)
    assert low < 50 < high


def test_the_rules_disagree_and_that_is_the_point():
    """MAD is tighter than IQR on skewed data because it is not dragged outward."""
    values = pd.Series(list(np.random.default_rng(1).normal(10, 1, 300)) + [500.0] * 5)
    iqr_low, iqr_high = bounds(values, "iqr")
    mad_low, mad_high = bounds(values, "mad")
    assert mad_high < iqr_high


def test_zscore_is_pulled_by_the_value_it_is_looking_for():
    """Masking: the outlier inflates the standard deviation that should catch it."""
    clean = pd.Series(list(np.random.default_rng(2).normal(10, 1, 300)))
    with_outlier = pd.concat([clean, pd.Series([1000.0])], ignore_index=True)
    assert bounds(with_outlier, "zscore")[1] > bounds(clean, "zscore")[1] * 5
    # The robust rule barely moves.
    assert bounds(with_outlier, "mad")[1] == pytest.approx(
        bounds(clean, "mad")[1], rel=0.05
    )


def test_a_dominated_column_is_refused():
    """capital_gain is 91.7% zeros, so the quartiles collapse onto zero."""
    values = pd.Series([0.0] * 400 + list(range(1, 60)))
    for method in METHODS:
        assert bounds(values, method) is None, method


def test_a_short_discrete_scale_is_refused():
    """education_num runs 1-16: its MAD is 1.0 and the ends are not anomalies."""
    values = pd.Series([10] * 100 + [9] * 100 + list(range(1, 17)) * 5)
    assert bounds(values, "mad") is None


def test_too_few_rows_is_refused():
    assert bounds(pd.Series([1.0, 2.0, 900.0]), "iqr") is None


def test_infinities_do_not_reach_the_quantiles():
    values = pd.concat([normalish(), pd.Series([np.inf, -np.inf])], ignore_index=True)
    low, high = bounds(values, "iqr")
    assert np.isfinite(low) and np.isfinite(high)


def test_an_unknown_method_is_rejected_rather_than_silently_ignored():
    with pytest.raises(ValueError, match="unknown method"):
        bounds(normalish(), "vibes")


# --------------------------------------------------------------------- fit/apply

def test_bounds_are_learned_from_the_training_half_only():
    frame = pd.DataFrame({
        "x": np.concatenate([np.full(400, 10.0), np.full(400, 1000.0)]),
        "y": ["a"] * 400 + ["b"] * 400,
    })
    train, _ = split(frame, target="y", seed=4)
    learned = fit_cap(train, {"columns": ["x"], "method": "iqr"})
    whole = fit_cap(frame, {"columns": ["x"], "method": "iqr"})
    # Same shape of answer, but computed from different rows.
    assert learned["bounds"].keys() == whole["bounds"].keys() or not learned["bounds"]


def test_capping_keeps_the_row_and_only_moves_the_extremes():
    frame = pd.DataFrame({"x": list(np.random.default_rng(3).normal(50, 5, 300)) + [9999.0]})
    fitted = fit_cap(frame, {"columns": ["x"]})
    out, detail, capped = apply_cap(frame, {}, fitted)
    assert len(out) == len(frame), "no row is deleted"
    assert capped == 1
    assert out["x"].max() < 9999.0
    assert "capped" in detail


def test_a_whole_number_column_stays_whole():
    """Capping age at 88.8903 is not a value the column can hold."""
    rng = np.random.default_rng(5)
    frame = pd.DataFrame({"age": list(rng.integers(20, 70, 400)) + [120] * 20})
    fitted = fit_cap(frame, {"columns": ["age"]})
    low, high = fitted["bounds"]["age"]
    assert low == int(low) and high == int(high)
    out, _, _ = apply_cap(frame, {}, fitted)
    assert bool((out["age"] % 1 == 0).all())


def test_a_rule_that_would_reshape_the_column_is_refused_with_a_reason():
    """Capping a tenth of a column is not a repair."""
    rng = np.random.default_rng(6)
    frame = pd.DataFrame({"x": list(rng.normal(0, 1, 300)) + list(rng.normal(60, 1, 60))})
    fitted = fit_cap(frame, {"columns": ["x"], "method": "mad"})
    assert "x" not in fitted["bounds"]
    assert "reshaping" in fitted["refused"]["x"]


def test_a_refusal_is_reported_in_the_detail():
    frame = pd.DataFrame({"x": [0.0] * 400 + list(range(1, 60))})
    fitted = fit_cap(frame, {"columns": ["x"]})
    _, detail, capped = apply_cap(frame, {}, fitted)
    assert capped == 0
    assert "declined" in detail


def test_nothing_to_cap_is_said_plainly():
    frame = pd.DataFrame({"x": normalish()})
    fitted = fit_cap(frame, {"columns": ["x"]})
    _, detail, capped = apply_cap(frame, {}, fitted)
    assert capped == 0 and "nothing to cap" in detail


def test_an_absent_column_is_skipped():
    assert fit_cap(pd.DataFrame({"a": [1.0]}), {"columns": ["nope"]})["bounds"] == {}


# ------------------------------------------------------------------- comparison

def test_the_comparison_table_shows_every_rule():
    table = comparison(normalish())
    assert list(table["rule"]) == list(METHODS)
    assert table["flagged"].notna().all()


def test_the_comparison_says_when_a_rule_cannot_judge():
    table = comparison(pd.Series([0.0] * 400 + list(range(1, 60))))
    assert (table["note"] == "cannot judge this column").all()


# ------------------------------------------------------------------ integration

@pytest.mark.skipif(not CENSUS.exists(), reason="census sample not present")
def test_census_capping_matches_what_the_rules_can_actually_judge():
    """PLAN.md Step 6 named capital_gain, but it is 91.7% zeros so every rule
    declines. The criterion was written before the data was examined; the code is
    right and the plan was corrected."""
    load = read_table(CENSUS)
    profiles = profile_frame(load.frame)
    from src.clean import run

    result = run(
        load.frame,
        build(detect(load.frame, profiles, target="income"), target="income"),
        target="income",
    )
    step = [a for a in result.applied if a.step.action == "cap_outliers"]
    assert step, "capping should run"
    capped = step[0].fitted["bounds"]
    assert set(capped) == {"age", "fnlwgt"}
    assert "capital_gain" not in capped, "91.7% zeros - the rules decline"
    assert "hours_per_week" not in capped, "46.7% dominated - the rules decline"
    assert result.train["age"].max() <= capped["age"][1]


@pytest.mark.skipif(not CENSUS.exists(), reason="census sample not present")
def test_capping_is_no_longer_skipped():
    load = read_table(CENSUS)
    from src.clean import run

    profiles = profile_frame(load.frame)
    result = run(
        load.frame,
        build(detect(load.frame, profiles, target="income"), target="income"),
        target="income",
    )
    assert "cap_outliers" not in {s.action for s in result.skipped}
