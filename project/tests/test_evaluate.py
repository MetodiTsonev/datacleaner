"""Tests for the evidence step.

The AUC cases are hand-computed: AUC is the probability that a random positive scores
above a random negative, which for four rows can be counted on paper.
"""

import numpy as np
import pandas as pd
import pytest

from src import clean, detect, evaluate, profile
from src import plan as planner
from src.evaluate import (
    Comparison,
    Score,
    corrupt,
    fit_logistic,
    naive_baseline,
    predict,
    roc_auc,
    score_arm,
    to_matrix,
)

# ------------------------------------------------------------------------- auc

@pytest.mark.parametrize("labels,scores,expected", [
    ([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], 1.0),    # every positive above every negative
    ([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1], 0.0),    # exactly backwards
    ([0, 0, 1, 1], [0.5, 0.5, 0.5, 0.5], 0.5),    # no information: all four pairs tie
    ([0, 0, 1, 1], [0.1, 0.7, 0.5, 0.9], 0.75),   # 3 of the 4 pairs ordered correctly
    ([0, 1], [0.3, 0.4], 1.0),
])
def test_auc_matches_the_hand_computed_value(labels, scores, expected):
    assert roc_auc(np.array(labels), np.array(scores)) == pytest.approx(expected)


def test_auc_counts_a_tie_as_half_a_pair():
    # one positive, one negative, identical scores -> the single pair is a tie -> 0.5
    assert roc_auc(np.array([0, 1]), np.array([0.5, 0.5])) == pytest.approx(0.5)


def test_auc_is_undefined_with_only_one_class():
    assert np.isnan(roc_auc(np.array([1, 1, 1]), np.array([0.1, 0.2, 0.3])))


def test_auc_agrees_with_the_definition_it_claims_to_implement():
    """The rank identity should equal a brute-force count over all pairs."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        y = rng.integers(0, 2, 40)
        if y.sum() in (0, len(y)):
            continue
        s = rng.normal(y * 0.7, 1.0)
        brute = np.mean([(a > b) + 0.5 * (a == b) for a in s[y == 1] for b in s[y == 0]])
        assert roc_auc(y, s) == pytest.approx(brute)


def test_auc_ignores_the_scale_of_the_scores():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.4, 0.6, 0.9])
    assert roc_auc(y, s) == pytest.approx(roc_auc(y, s * 1000 - 7))


# -------------------------------------------------------------------- the model

def test_gradient_descent_recovers_a_known_boundary():
    rng = np.random.default_rng(1)
    X = rng.normal(0, 1, (3000, 2))
    y = (rng.random(3000) < 1 / (1 + np.exp(-(X @ np.array([2.0, -3.0]) + 0.5)))).astype(float)
    w = fit_logistic(X, y, iterations=3000)
    assert w[0] == pytest.approx(2.0, abs=0.3)
    assert w[1] == pytest.approx(-3.0, abs=0.3)
    assert w[2] == pytest.approx(0.5, abs=0.3)


def test_predictions_stay_probabilities():
    rng = np.random.default_rng(2)
    X = rng.normal(0, 50, (200, 3))  # large values would overflow an unclipped sigmoid
    p = predict(X, np.array([9.0, -9.0, 9.0, 1.0]))
    assert ((p >= 0.0) & (p <= 1.0)).all()
    assert np.isfinite(p).all()


# -------------------------------------------------------------------- the frame

def test_text_columns_become_indicator_columns():
    frame = pd.DataFrame({"city": ["sofia", "varna", "sofia"], "y": ["a", "b", "a"]})
    X, y, columns = to_matrix(frame, "y")
    assert "city=sofia" in columns and "city=varna" in columns
    assert X.shape == (3, 2)
    assert y.tolist() == [0.0, 1.0, 0.0]


def test_the_test_frame_is_forced_onto_the_training_columns():
    """A level seen only at test time must not add a column the model has no weight for."""
    train = pd.DataFrame({"c": ["a", "b"], "y": ["n", "p"]})
    test = pd.DataFrame({"c": ["a", "zzz"], "y": ["n", "p"]})
    _, _, columns = to_matrix(train, "y")
    X_test, _, _ = to_matrix(test, "y", columns=columns)
    assert X_test.shape[1] == len(columns)


def test_infinities_and_nulls_do_not_reach_the_model():
    frame = pd.DataFrame({"v": [1.0, np.inf, np.nan, 4.0], "y": ["a", "b", "a", "b"]})
    X, _, _ = to_matrix(frame, "y")
    assert np.isfinite(X).all()


# --------------------------------------------------------------- the comparison

def _dirty_frame(n=400):
    """Carries duplicate rows on purpose.

    Without them the deduplication step never fires, and the row-alignment tests below
    pass whether or not the alignment is correct.
    """
    rng = np.random.default_rng(3)
    signal = rng.normal(0, 1, n)
    frame = pd.DataFrame({
        "signal": signal,
        "group": rng.choice(["a", "b", "c"], n),
        "y": np.where(signal + rng.normal(0, 0.5, n) > 0, "high", "low"),
    })
    # 40 exact duplicates, scattered rather than appended, so dropping them removes
    # rows from the middle and shifts every label after them.
    doubled = pd.concat([frame, frame.iloc[:40]], ignore_index=True)
    return doubled.sample(frac=1.0, random_state=4).reset_index(drop=True)


def test_the_fixture_really_contains_duplicates():
    """Otherwise the two tests below cannot fail, whatever the code does."""
    frame = _dirty_frame()
    assert frame.duplicated().sum() >= 30


def test_both_arms_are_scored_on_the_same_held_out_rows():
    """The fairness invariant. Two AUCs from two different test sets compare nothing.

    This caught a real bug: the row-dropping steps reset the index, so the row labels
    the evaluation used to line the arms up pointed at the wrong rows.
    """
    frame = _dirty_frame()
    profiles = profile.profile_frame(frame)
    findings = detect.detect(frame, profiles, target="y")
    result = clean.run(frame, planner.build(findings, target="y"), target="y", seed=7)

    raw_test = frame.loc[result.test_ids]
    assert len(raw_test) == len(result.test)
    assert (raw_test["y"].to_numpy() == result.test["y"].to_numpy()).all()


def test_the_held_out_rows_are_absent_from_the_training_half():
    frame = _dirty_frame()
    findings = detect.detect(frame, profile.profile_frame(frame), target="y")
    result = clean.run(frame, planner.build(findings, target="y"), target="y", seed=7)
    assert not set(result.test_ids) & set(frame.drop(index=result.test_ids).index)


def test_compare_returns_two_usable_numbers():
    comparison = evaluate.compare(_dirty_frame(), target="y", seed=7)
    assert 0.0 <= comparison.raw.auc <= 1.0
    assert 0.0 <= comparison.cleaned.auc <= 1.0
    assert comparison.difference == pytest.approx(
        comparison.cleaned.auc - comparison.raw.auc
    )


def test_a_model_on_signal_beats_chance():
    """Guards the whole chain: if this drops to 0.5 the harness stopped working."""
    assert evaluate.compare(_dirty_frame(), target="y", seed=7).raw.auc > 0.75


# --------------------------------------------------------------- the corruption

def test_corruption_damages_the_stated_share_and_spares_the_target():
    frame = _dirty_frame()
    damaged = corrupt(frame, 0.3, target="y")
    assert damaged["y"].equals(frame["y"])           # labels untouched
    assert (damaged["signal"] == -999.0).mean() == pytest.approx(0.3, abs=0.05)
    assert damaged["group"].isin(["?", "N/A", "", "-"]).mean() == pytest.approx(0.3, abs=0.05)


def test_zero_corruption_changes_nothing():
    frame = _dirty_frame()
    assert corrupt(frame, 0.0, target="y").equals(frame)


def test_corruption_is_reproducible():
    frame = _dirty_frame()
    assert corrupt(frame, 0.2, target="y", seed=5).equals(corrupt(frame, 0.2, target="y", seed=5))


def test_the_naive_baseline_loses_rows_as_damage_rises():
    frame = _dirty_frame()
    counts = [len(naive_baseline(corrupt(frame, s, target="y"), "y")) for s in (0.0, 0.2, 0.5)]
    assert counts == sorted(counts, reverse=True)
    assert counts[-1] < counts[0] / 2


# ------------------------------------------------------------------- edge cases

def test_a_single_class_target_reports_rather_than_crashes():
    frame = pd.DataFrame({"v": [1.0, 2.0, 3.0, 4.0], "y": ["a"] * 4})
    score = score_arm(frame, frame, "y")
    assert np.isnan(score.auc)
    assert "one class" in score.note


def test_an_empty_training_frame_reports_rather_than_crashes():
    empty = pd.DataFrame({"v": [], "y": []})
    assert np.isnan(score_arm(empty, empty, "y").auc)


def test_the_summary_rounds_to_four_places_for_display():
    comparison = Comparison(
        raw=Score(0.900012, 10, 3), cleaned=Score(0.912345, 10, 3), target="y"
    )
    assert comparison.summary()["cleaned_auc"] == 0.9123
    assert comparison.summary()["difference"] == pytest.approx(0.0123, abs=1e-4)


# ------------------------------------------------------------------ the verdict

def _comparison(raw: float, cleaned: float, **kw) -> Comparison:
    return Comparison(raw=Score(raw, 100, 5), cleaned=Score(cleaned, 100, 5),
                      target="y", **kw)


def test_a_clear_improvement_is_called_an_improvement():
    kind, text = evaluate.verdict(_comparison(0.80, 0.83))
    assert kind == "better"
    assert "+0.0300" in text


def test_a_clear_loss_is_reported_not_softened():
    kind, text = evaluate.verdict(_comparison(0.83, 0.80))
    assert kind == "worse"
    assert "-0.0300" in text
    assert "worse" in text


def test_a_difference_too_small_to_mean_anything_is_called_none():
    kind, text = evaluate.verdict(_comparison(0.8000, 0.8005))
    assert kind == "same"
    assert "No measurable difference" in text


def test_a_gain_between_two_models_that_lose_to_a_coin_toss_is_not_an_improvement():
    """The honesty case. 0.43 -> 0.44 is not progress, it is noise between two
    models that are both worse than guessing."""
    kind, text = evaluate.verdict(_comparison(0.4308, 0.4359))
    assert kind == "chance", f"reported {kind!r} for two sub-chance scores"
    assert "coin toss" in text


def test_sub_chance_outranks_improvement_even_for_a_large_gain():
    kind, _ = evaluate.verdict(_comparison(0.20, 0.45))
    assert kind == "chance"


def test_a_score_above_a_half_on_either_side_is_judged_normally():
    assert evaluate.verdict(_comparison(0.51, 0.45))[0] == "worse"
    assert evaluate.verdict(_comparison(0.45, 0.51))[0] == "better"


def test_an_unscorable_comparison_returns_its_reason():
    unscorable = Comparison(
        raw=Score(float("nan"), 0, 0, "as uploaded: the column to predict is empty"),
        cleaned=Score(float("nan"), 0, 0), target="y",
    )
    kind, text = evaluate.verdict(unscorable)
    assert kind == "unscorable"
    assert "empty" in text
