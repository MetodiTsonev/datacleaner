"""The exported record.

The cleaned CSV carries no account of how it was produced. These check that the account
is complete, honest about what it does not know, and never claims a step ran when it
did not.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src import clean, detect, evaluate, features, loader, profile, report
from src import plan as planner

SAMPLE = "data/input/messy-orders.csv"
TARGET = "returned"


@pytest.fixture(scope="module")
def pipeline():
    load = loader.read_table(SAMPLE)
    profiles = profile.profile_frame(load.frame)
    findings = detect.detect(load.frame, profiles, target=TARGET)
    repair = planner.build(findings, target=TARGET)
    result = clean.run(load.frame, repair, target=TARGET)
    _, _, freport = features.build(
        result.train, result.test, profile.profile_frame(result.train), target=TARGET
    )
    comparison = evaluate.compare(load.frame, target=TARGET)
    return load, findings, repair, result, freport, comparison


def test_the_report_covers_every_stage(pipeline):
    load, findings, repair, result, freport, comparison = pipeline
    text = report.build(load, findings, repair, result, target=TARGET,
                        features=freport, comparison=comparison)
    for heading in ("How the file was read", "What was found", "What was planned",
                    "What was done", "Validation", "Preparation for a model",
                    "Did it help?", "does not tell you"):
        assert heading in text, heading


def test_the_report_names_the_file_and_the_target(pipeline):
    load, findings, repair, result, _, _ = pipeline
    text = report.build(load, findings, repair, result, target=TARGET)
    assert "messy-orders.csv" in text
    assert TARGET in text


def test_a_step_the_pipeline_could_not_perform_is_named_not_hidden(pipeline):
    """A report that lists only successes reads as though everything succeeded."""
    load, findings, repair, result, _, _ = pipeline
    text = report.build(load, findings, repair, result, target=TARGET)
    for step in result.skipped:
        assert step.action in text
    for finding in repair.unaddressed:
        assert finding.message[:40] in text


def test_without_a_measurement_the_report_says_so_rather_than_implying_success(pipeline):
    load, findings, repair, result, _, _ = pipeline
    text = report.build(load, findings, repair, result, target=TARGET, comparison=None)
    assert "Not measured" in text
    assert "changed" in text  # the only claim it may make


def test_an_unscorable_comparison_reports_its_reason(pipeline):
    load, findings, repair, result, _, _ = pipeline
    unscorable = evaluate.compare(load.frame, target="order_id")
    text = report.build(load, findings, repair, result, target="order_id",
                        comparison=unscorable)
    assert "Could not be measured" in text


def test_the_derived_step_is_marked_as_derived(pipeline):
    """The imputation nothing asked for is the clearest sign the system reasons."""
    load, findings, repair, result, _, _ = pipeline
    text = report.build(load, findings, repair, result, target=TARGET)
    assert any(s.derived_from for s in repair.steps)
    assert "derived" in text


def test_the_report_reflects_what_actually_happened_not_what_was_planned(pipeline):
    load, findings, repair, result, _, _ = pipeline
    text = report.build(load, findings, repair, result, target=TARGET)
    done = text.split("## 4. What was done")[1].split("## 5.")[0]
    for applied in result.applied:
        assert applied.step.action in done
    # a step that was planned but skipped must not appear as done
    for step in result.skipped:
        assert step.action not in done.split("**Skipped**")[0]


def test_the_truncation_of_a_long_finding_list_is_stated(pipeline):
    """A list that stops without saying so reads as the complete list."""
    load, _, repair, result, _, _ = pipeline
    many = detect.detect(load.frame, profile.profile_frame(load.frame), target=TARGET)
    serious = [f for f in many if f.severity in ("critical", "high")]
    text = report.build(load, many, repair, result, target=TARGET)
    if len(serious) > 10:
        assert "more" in text


def test_it_works_on_a_frame_with_nothing_wrong():
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "y": ["p", "n", "p", "n"]})
    frame.to_csv("/tmp/_clean_sample.csv", index=False)
    load = loader.read_table("/tmp/_clean_sample.csv")
    profiles = profile.profile_frame(load.frame)
    findings = detect.detect(load.frame, profiles, target="y")
    repair = planner.build(findings, target="y")
    result = clean.run(load.frame, repair, target="y")
    text = report.build(load, findings, repair, result, target="y")
    assert "_clean_sample.csv" in text
    assert len(text) > 200


def test_the_report_and_the_app_cannot_disagree_about_what_the_numbers_mean():
    """Both must read the verdict from the same place.

    The report used to word its own "better/worse" from the sign of the difference, so
    it called +0.0051 an improvement while the app -- correctly -- called it two models
    that both lose to a coin toss.
    """
    from src.evaluate import Comparison, Score, verdict

    sub_chance = Comparison(raw=Score(0.4308, 100, 5), cleaned=Score(0.4359, 100, 5),
                            target="y")
    kind, sentence = verdict(sub_chance)
    assert kind == "chance"

    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "y": ["p", "n", "p", "n"]})
    frame.to_csv("/tmp/_verdict_sample.csv", index=False)
    load = loader.read_table("/tmp/_verdict_sample.csv")
    profiles = profile.profile_frame(load.frame)
    findings = detect.detect(load.frame, profiles, target="y")
    repair = planner.build(findings, target="y")
    result = clean.run(load.frame, repair, target="y")

    text = report.build(load, findings, repair, result, target="y",
                        comparison=sub_chance)
    assert sentence in text
    assert "better" not in text.split("## 7.")[1].split("## 8.")[0]
