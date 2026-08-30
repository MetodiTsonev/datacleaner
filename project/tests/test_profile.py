"""Stage 2 tests - semantic type inference.

Type inference is where "accepts arbitrary raw files" actually lives, and every
misclassification is silent: a quantity called a category gets mode-filled and
one-hot encoded instead of median-filled and scaled. Each case below is a
misclassification that happened during development.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.loader import read_table
from src.profile import (
    as_table,
    profile_column,
    profile_frame,
    type_counts,
)
from src.text import DISGUISED_TOKENS

CENSUS = Path(__file__).parent.parent / "data" / "input" / "adult-census.csv"


def typed(values, name="col") -> str:
    series = pd.Series(values)
    series.name = name
    return profile_column(series).semantic_type


# ------------------------------------------------------------------ basic types

def test_all_missing_is_empty():
    assert typed([None, None, None]) == "empty"


def test_single_value_is_constant():
    assert typed(["EUR"] * 10) == "constant"


@pytest.mark.parametrize("values", [
    ["yes", "no", "yes"], ["true", "false", "true"],
    ["Y", "N", "Y"], ["да", "не", "да"],
])
def test_two_known_values_are_boolean(values):
    assert typed(values) == "boolean"


def test_zero_one_integers_are_boolean():
    assert typed([0, 1, 1, 0]) == "boolean"


def test_dates_are_datetime():
    assert typed(pd.date_range("2024-01-01", periods=40).astype(str)) == "datetime"


def test_few_labels_are_categorical():
    assert typed(["red", "green", "blue"] * 30) == "categorical"


def test_long_strings_are_text():
    values = [f"Article body {i} with a decent quantity of prose inside it" for i in range(50)]
    assert typed(values) == "text"


def test_unique_short_strings_are_an_identifier():
    assert typed([f"ID{i:05d}" for i in range(500)]) == "identifier"


# ------------------------------------------- regressions: real misclassifications

def test_a_quantity_with_few_distinct_values_stays_numeric():
    """`age` has 74 distinct values in 48,842 rows.

    A share-based categorical rule called it a category, which would have got it
    one-hot encoded into 74 columns instead of scaled.
    """
    assert typed(list(range(17, 91)) * 3) == "numeric"


def test_decimals_stored_as_text_are_numeric_however_few():
    """Only whole numbers can be codes. 1.5 is a quantity."""
    assert typed(["1.5", "2.5", "3.5", "4.5"] * 25) == "numeric"


def test_whole_numbers_stored_as_text_with_few_values_are_codes():
    assert typed(["1", "2", "3", "4"] * 25) == "categorical"


def test_numbers_with_a_disguised_blank_are_still_numeric():
    """The column this project exists for.

    A numeric column containing "?" must not be demoted to categorical, or it
    gets mode-filled instead of median-filled.
    """
    values = [str(x) for x in range(100, 200)] + ["?", "?", "?"]
    profile = profile_column(pd.Series(values, name="price"))
    assert profile.semantic_type == "numeric"
    assert "placeholders" in profile.note
    assert "'?'" in profile.note, "the note must name the values it means"


def test_all_disguised_blanks_is_empty():
    profile = profile_column(pd.Series(["?", "N/A", "-", "?"], name="c"))
    assert profile.semantic_type == "empty"


def test_unique_values_in_a_short_column_are_not_categorical():
    """50 distinct values in 50 rows satisfied the absolute "<= 50" rule."""
    values = [f"Sentence number {i} carrying a fair amount of text here" for i in range(50)]
    assert typed(values) == "text"


def test_dates_survive_a_disguised_blank():
    values = [f"{d:02d}/03/2024" for d in range(1, 29)] + ["-"]
    assert typed(values) == "datetime"


def test_a_numeric_column_is_not_read_as_dates():
    """pandas will happily read bare integers as dates if allowed to."""
    assert typed([str(x) for x in range(1, 300)]) == "numeric"


# --------------------------------------------------------------------- reporting

def test_missing_counts_and_shares():
    profile = profile_column(pd.Series([1.0, 2.0, None, None], name="c"))
    assert profile.n_missing == 2
    assert profile.pct_missing == 50.0


def test_numeric_stats_include_skew_and_zero_share():
    profile = profile_column(pd.Series([0, 0, 0, 0, 0, 0, 0, 0, 0, 100.0], name="c"))
    assert profile.stats["pct_zero"] == 90.0
    assert profile.stats["skew"] > 1
    assert profile.stats["median"] == 0.0


def test_top_value_reported_for_categoricals():
    profile = profile_column(pd.Series(["a", "a", "a", "b"], name="c"))
    assert profile.top_value == "a"
    assert profile.top_share == 75.0


def test_as_table_has_one_row_per_column():
    frame = pd.DataFrame({"a": [1.5, 2.5, 3.5], "b": list("xyz")})
    table = as_table(profile_frame(frame))
    assert len(table) == 2
    assert list(table["column"]) == ["a", "b"]


def test_disguised_token_set_covers_the_census_marker():
    assert "?" in DISGUISED_TOKENS
    assert "n/a" in DISGUISED_TOKENS


# ------------------------------------------------------------------- integration

@pytest.mark.skipif(not CENSUS.exists(), reason="sample census file not present")
def test_census_resolves_to_six_numeric_and_nine_categorical():
    """The plan's Step 1 acceptance criterion."""
    profiles = profile_frame(read_table(CENSUS).frame)
    assert type_counts(profiles) == {"numeric": 6, "categorical": 9}


@pytest.mark.skipif(not CENSUS.exists(), reason="sample census file not present")
def test_census_skews_match_the_documented_values():
    by_name = {p.name: p for p in profile_frame(read_table(CENSUS).frame)}
    assert round(by_name["capital_gain"].stats["skew"], 2) == 11.89
    assert round(by_name["capital_loss"].stats["skew"], 2) == 4.57


@pytest.mark.skipif(not CENSUS.exists(), reason="sample census file not present")
def test_profiling_alone_reports_no_missing_values_on_the_census():
    """The trap, asserted.

    Profiling is honest about what pandas can see, and pandas sees nothing wrong.
    Stage 3 is what finds the 3,620 rows holding "?". This test exists so the
    contrast cannot be broken by accident -- it is the project's headline figure.
    """
    profiles = profile_frame(read_table(CENSUS).frame)
    assert sum(p.n_missing for p in profiles) == 0
