"""Stage 3 tests - detection.

One test per check that it fires, one per check that it does *not* fire on clean
data, and named regressions for the four bugs found by running against real files.

The acceptance criteria from PLAN.md Step 2 are asserted at the bottom against the
two sample files.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.detect import (
    CHECKS,
    SEVERITIES,
    TOPICS,
    Finding,
    as_table,
    detect,
    summarise,
)
from src.loader import read_table
from src.profile import profile_frame

DATA = Path(__file__).parent.parent / "data" / "input"
CENSUS = DATA / "adult-census.csv"
MESSY = DATA / "messy-orders.csv"


def run(frame: pd.DataFrame) -> list[Finding]:
    return detect(frame, profile_frame(frame))


def checks_in(findings: list[Finding]) -> set[str]:
    return {f.check for f in findings}


def only(findings: list[Finding], check: str) -> Finding:
    matching = [f for f in findings if f.check == check]
    assert len(matching) == 1, f"expected one {check}, got {len(matching)}"
    return matching[0]


# ------------------------------------------------------------------- each check

def test_disguised_missing_fires():
    frame = pd.DataFrame({"city": ["Sofia", "?", "Varna", "?"]})
    finding = only(run(frame), "disguised_missing")
    assert finding.severity == "critical"
    assert finding.affected_rows == 2
    assert finding.suggestion.action == "replace_disguised_missing"


def test_numeric_sentinel_fires_and_proposes_a_repair_when_implausible():
    """-999 inside an otherwise non-negative quantity is unambiguous."""
    frame = pd.DataFrame({"amount": [10.0, 20.0, 30.0, -999.0, -999.0, 15.0, 25.0, 5.0]})
    finding = only(run(frame), "numeric_sentinel")
    assert finding.affected_rows == 2
    assert finding.severity == "high"
    assert finding.suggestion.action == "replace_disguised_missing"


def test_numeric_sentinel_declines_to_repair_a_probable_cap():
    """99999 at the top of a non-negative column may mean "at least this much"."""
    frame = pd.DataFrame({"gain": [0.0] * 20 + [500.0, 1200.0, 99999.0, 99999.0]})
    finding = only(run(frame), "numeric_sentinel")
    assert finding.suggestion is None
    assert "cannot be decided" in finding.message


def test_numeric_sentinel_ignores_a_lone_occurrence():
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 999.0]})
    assert "numeric_sentinel" not in checks_in(run(frame))


def test_missing_values_fires():
    frame = pd.DataFrame({"x": [1.0, 2.0, None, None, 5.0, 6.0]})
    finding = only(run(frame), "missing_values")
    assert finding.affected_rows == 2
    assert finding.suggestion.params["strategy"] == "median"


def test_missing_values_suggests_mode_for_categoricals():
    frame = pd.DataFrame({"c": ["a", "b", "a", None, "a", "b"]})
    assert only(run(frame), "missing_values").suggestion.params["strategy"] == "mode"


def test_empty_column_fires():
    frame = pd.DataFrame({"keep": [1, 2, 3], "gone": [None, None, None]})
    finding = only(run(frame), "uninformative_columns")
    assert finding.evidence["empty"] == ["gone"]
    assert finding.severity == "high", "empty is worse than constant"


def test_constant_column_fires():
    frame = pd.DataFrame({"keep": [1, 2, 3], "ccy": ["BGN"] * 3})
    finding = only(run(frame), "uninformative_columns")
    assert finding.evidence["constant"] == ["ccy"]
    assert finding.severity == "medium"


def test_empty_and_constant_are_reported_as_one_finding():
    """Same defect, same repair -- one operation, not two."""
    frame = pd.DataFrame({"k": [1, 2, 3], "e": [None] * 3, "c": ["x"] * 3})
    finding = only(run(frame), "uninformative_columns")
    assert set(finding.suggestion.params["columns"]) == {"e", "c"}


def test_identifier_column_fires():
    frame = pd.DataFrame(
        {"id": [f"ORD-{i}" for i in range(60)], "v": list(range(60))}
    )
    assert "id" in only(run(frame), "identifier_columns").columns


def test_exact_duplicates_fire():
    frame = pd.DataFrame({"a": [1, 2, 1, 2], "b": ["x", "y", "x", "y"]})
    finding = only(run(frame), "exact_duplicates")
    assert finding.affected_rows == 2
    assert finding.suggestion.action == "drop_duplicate_rows"


def test_normalised_duplicates_report_only_the_extra_rows():
    """Two exact copies plus one that differs by case only."""
    frame = pd.DataFrame({"c": ["Sofia", "Sofia", "sofia ", "Varna"]})
    finding = only(run(frame), "normalised_duplicates")
    assert finding.evidence["exact"] == 1
    assert finding.evidence["additional"] == 1


def test_inconsistent_categories_fire():
    frame = pd.DataFrame({"city": ["Sofia", "sofia", " Sofia", "Varna"] * 5})
    finding = only(run(frame), "inconsistent_categories")
    assert "sofia" in finding.evidence["groups"]


def test_mixed_types_fire_and_count_the_broken_values():
    frame = pd.DataFrame({"w": ["1.5", "2.5", "3.5", "heavy", "light"] * 6})
    finding = only(run(frame), "mixed_types")
    assert finding.affected_rows == 12, "the non-numeric values are the problem"
    assert finding.suggestion is None, "needs a human decision"


def test_outliers_fire_on_a_continuous_column():
    values = list(np.random.default_rng(0).normal(10, 1, 200)) + [500.0, 600.0]
    finding = only(run(pd.DataFrame({"x": values})), "outliers")
    assert finding.suggestion.action == "cap_outliers"


def test_redundant_columns_fire():
    frame = pd.DataFrame({"label": ["a", "b", "c"] * 10, "code": [1, 2, 3] * 10})
    finding = only(run(frame), "redundant_columns")
    assert set(finding.columns) == {"label", "code"}


def test_high_skew_fires_and_offers_a_log_only_when_applicable():
    positive = pd.DataFrame({"x": [1.0] * 100 + [500.0, 900.0, 1500.0]})
    assert only(run(positive), "high_skew").suggestion.action == "log_transform"
    negative = pd.DataFrame({"x": [-1.0] * 100 + [-500.0, -900.0, -1500.0]})
    assert only(run(negative), "high_skew").suggestion is None


# ------------------------------------------------ regressions from real data runs

def test_a_short_discrete_scale_is_not_repaired_as_outliers():
    """`education_num` runs 1-16, so its MAD is 1.0.

    A modified z-score of 3.5 then flags every value ~5 steps from the median,
    which on a 16-step scale is the ends of the scale. Preschool and Doctorate are
    not anomalies, and capping them would destroy real data.
    """
    # Shaped like the real column: mass at 9-10 with thin tails, so the MAD
    # collapses to 1.0 and the rule flags the ends of the scale. Kept under 40% at
    # the mode so the dominance guard (checked first) does not fire instead, and
    # deliberately not uniform, since a flat scale has enough spread to flag
    # nothing at all.
    frame = pd.DataFrame(
        {"level": [10] * 100 + [9] * 100 + [11] * 60 + [13] * 40
                  + [1, 2, 3, 4] * 5 + [16] * 10}
    )
    finding = only(run(frame), "outliers")
    assert finding.evidence["distinct"] == 9
    assert finding.evidence["mad"] == 1.0
    assert finding.suggestion is None, "capping an ordinal scale destroys real data"
    assert finding.severity == "info"
    assert "short discrete scale" in finding.message


def test_a_dominated_column_is_not_repaired_as_outliers():
    """`hours_per_week`: one value covers 47%, so the quartiles collapse onto it.

    The spread must stay non-zero, or neither rule flags anything and there is no
    finding to make -- which is also correct, just not what this test is about.
    """
    rng = np.random.default_rng(3)
    spread = list(rng.integers(20, 61, 500)) + [1, 2, 3, 95, 98, 99]
    frame = pd.DataFrame({"hours": [40] * 450 + spread})
    finding = only(run(frame), "outliers")
    assert finding.suggestion is None
    assert "covers" in finding.message


def test_the_proposed_cap_uses_the_rule_that_actually_fired():
    """Capping at the IQR fences when the IQR rule flagged nothing is a no-op."""
    rng = np.random.default_rng(1)
    frame = pd.DataFrame({"age": list(rng.integers(17, 80, 400)) + [90] * 30})
    findings = [f for f in run(frame) if f.check == "outliers"]
    for finding in findings:
        if finding.suggestion:
            fired = "iqr" if finding.evidence["by_iqr"] else "mad"
            assert finding.suggestion.params["method"] == fired


def test_an_identifier_stays_an_identifier_when_rows_are_duplicated():
    """Duplicated records repeat an identifier, so counting all rows is backwards."""
    ids = [f"ORD-{i}" for i in range(240)]
    frame = pd.DataFrame({"order_id": ids, "v": list(range(240))})
    frame = pd.concat([frame, frame.iloc[:20]], ignore_index=True)
    assert "identifier_columns" in checks_in(run(frame))


# ------------------------------------------------------------- clean data is quiet

def test_clean_data_produces_no_findings():
    """Nothing is invented when nothing is wrong.

    The numeric column is clipped to +-2.5 sigma so no genuine extreme value
    exists: with an unclipped normal sample the MAD rule correctly flags the odd
    tail point, which is real behaviour but makes the assertion depend on luck.
    See the test below.
    """
    rng = np.random.default_rng(7)
    amount = np.clip(rng.normal(100, 10, 300), 75, 125).round(2)
    frame = pd.DataFrame(
        {
            "amount": amount,
            "city": rng.choice(["Sofia", "Varna", "Ruse"], 300),
            "paid": rng.choice(["yes", "no"], 300),
        }
    )
    assert run(frame) == []


def test_a_single_extreme_value_in_otherwise_normal_data_is_reported():
    """Documented rather than suppressed.

    One point beyond a modified z-score of 3.5 in 300 normal observations is a
    genuine extreme value, and a lone extreme value is as likely to be a typo
    (5000 for 50) as a real tail. It is reported at low severity, and the IQR rule
    is shown alongside so the reader can see the two rules disagree.
    """
    rng = np.random.default_rng(7)
    frame = pd.DataFrame({"amount": rng.normal(100, 10, 300).round(2)})
    finding = only(run(frame), "outliers")
    assert finding.severity == "low"
    assert finding.evidence["by_modified_z"] == 1
    assert finding.evidence["by_iqr"] == 0
    assert finding.suggestion.params["method"] == "mad", "match the rule that fired"


# ------------------------------------------------------------------- the contract

def test_every_finding_has_a_known_severity_and_topic():
    for frame_path in (CENSUS, MESSY):
        if not frame_path.exists():
            continue
        for finding in run(read_table(frame_path).frame):
            assert finding.severity in SEVERITIES
            assert finding.topic in TOPICS


def test_unknown_severity_is_rejected():
    with pytest.raises(ValueError, match="unknown severity"):
        Finding(check="x", severity="catastrophic", topic="missing", message="m")


def test_unknown_topic_is_rejected():
    with pytest.raises(ValueError, match="unknown topic"):
        Finding(check="x", severity="low", topic="vibes", message="m")


def test_findings_are_sorted_worst_first():
    findings = run(read_table(MESSY).frame) if MESSY.exists() else []
    ranks = [f.rank for f in findings]
    assert ranks == sorted(ranks)


def test_summarise_counts_by_severity_and_topic():
    frame = pd.DataFrame({"city": ["Sofia", "?", "Varna", "?"]})
    summary = summarise(run(frame))
    assert summary["by_severity"]["critical"] == 1
    assert summary["by_topic"]["missing"] == 1
    assert summary["blocking"]


def test_as_table_has_one_row_per_finding():
    findings = run(pd.DataFrame({"city": ["Sofia", "?", "Varna", "?"]}))
    assert len(as_table(findings)) == len(findings)


def test_every_check_is_registered():
    """A check written but never added to CHECKS would silently never run."""
    import src.checks as module

    defined = {
        name for name in dir(module)
        if name.startswith("check_") and callable(getattr(module, name))
    }
    assert defined == {c.__name__ for c in CHECKS}


# ------------------------------------------------- PLAN.md Step 2 acceptance tests

@pytest.mark.skipif(not CENSUS.exists(), reason="census sample not present")
def test_census_disguised_missing_counts():
    by_column = {
        f.columns[0]: f.affected_rows
        for f in run(read_table(CENSUS).frame)
        if f.check == "disguised_missing"
    }
    assert by_column == {"workclass": 2799, "occupation": 2809, "native_country": 857}


@pytest.mark.skipif(not CENSUS.exists(), reason="census sample not present")
def test_census_exact_duplicates_and_redundancy():
    findings = run(read_table(CENSUS).frame)
    assert only(findings, "exact_duplicates").affected_rows == 52
    assert set(only(findings, "redundant_columns").columns) == {
        "education", "education_num"
    }


@pytest.mark.skipif(not CENSUS.exists(), reason="census sample not present")
def test_census_capital_gain_top_code_is_found_but_not_repaired():
    """99,999 is the documented cap in the Adult data, not a missing value."""
    finding = only(
        [f for f in run(read_table(CENSUS).frame) if f.columns == ["capital_gain"]
         and f.check == "numeric_sentinel"], "numeric_sentinel"
    )
    assert finding.affected_rows == 244
    assert finding.suggestion is None


@pytest.mark.skipif(not MESSY.exists(), reason="messy sample not present")
def test_messy_sample_exercises_every_check_the_census_cannot():
    fired = checks_in(run(read_table(MESSY).frame))
    for check in (
        "uninformative_columns", "identifier_columns",
        "inconsistent_categories", "mixed_types", "normalised_duplicates",
        "missing_values",
    ):
        assert check in fired, f"{check} did not fire on the messy sample"
