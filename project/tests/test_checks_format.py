"""Stage 3 - the format-hygiene and ingestion checks.

Every case below was measured as **not caught** before these checks existed. The
list came from asking what an ordinary practitioner would expect a data-cleaning
tool to notice, which turned out to be a better source of requirements than the
defects the census file happens to contain.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.detect import detect, protect_target
from src.loader import read_table
from src.profile import profile_frame


def fired(frame: pd.DataFrame, target: str | None = None) -> set[str]:
    return {f.check for f in detect(frame, profile_frame(frame), target=target)}


def one(frame: pd.DataFrame, check: str, target: str | None = None):
    matching = [
        f for f in detect(frame, profile_frame(frame), target=target)
        if f.check == check
    ]
    assert len(matching) == 1, f"expected one {check}, got {len(matching)}"
    return matching[0]


# ------------------------------------------------- numbers written for humans

@pytest.mark.parametrize("values,mark", [
    (["1 234,56", "2 500,00", "980,10", "3 100,75"], "thousands separator"),
    (["$100", "$250", "$80", "$310"], "$"),
    (["12 lv", "40 lv", "8 lv", "25 lv"], "lv"),
    (["45%", "12%", "88%", "3%"], "%"),
])
def test_numeric_in_text_is_caught(values, mark):
    """The worst of the format defects: a quantity treated as a category."""
    frame = pd.DataFrame({"amount": values * 5})
    finding = one(frame, "numeric_in_text")
    assert finding.severity == "high"
    assert finding.suggestion.action == "parse_numeric"
    assert any(mark in m for m in finding.evidence["marks_found"])


def test_a_comma_decimal_is_recognised():
    frame = pd.DataFrame({"price": ["1,50", "2,75", "10,00", "3,20"] * 5})
    finding = one(frame, "numeric_in_text")
    assert "comma decimal separator" in finding.evidence["marks_found"]


def test_ordinary_text_is_not_reported_as_a_number():
    frame = pd.DataFrame({"city": ["Sofia", "Varna", "Ruse", "Burgas"] * 5})
    assert "numeric_in_text" not in fired(frame)


def test_an_already_numeric_column_is_left_alone():
    frame = pd.DataFrame({"n": [1.5, 2.5, 3.5, 4.5] * 5})
    assert "numeric_in_text" not in fired(frame)


# ------------------------------------------------------------------ whitespace

def test_trailing_spaces_are_caught():
    frame = pd.DataFrame({"city": ["Sofia ", "Varna", "Ruse ", "Burgas"] * 5})
    finding = one(frame, "whitespace")
    assert finding.evidence["untrimmed"] == 10
    assert finding.suggestion.action == "normalise_categories"


def test_doubled_internal_spaces_are_caught():
    frame = pd.DataFrame({"name": ["Ivan  Petrov", "Maria Ivanova"] * 6})
    assert one(frame, "whitespace").evidence["doubled_spaces"] == 6


def test_clean_text_has_no_whitespace_finding():
    frame = pd.DataFrame({"city": ["Sofia", "Varna", "Ruse"] * 6})
    assert "whitespace" not in fired(frame)


# --------------------------------------------------------------- short values

def test_single_characters_in_a_long_column_are_caught():
    values = ["Ivan Petrov", "Maria Ivanova", "Georgi Dimitrov", "x"] * 5
    finding = one(pd.DataFrame({"name": values}), "short_values")
    assert "x" in finding.evidence["values"]
    assert finding.suggestion is None, "whether a stub means missing is a judgement"


def test_short_values_are_not_reported_in_a_naturally_short_column():
    """A column of country codes is not full of stubs."""
    frame = pd.DataFrame({"code": ["BG", "DE", "FR", "IT"] * 5})
    assert "short_values" not in fired(frame)


# ------------------------------------------------------------ encoding damage

def test_mojibake_is_caught():
    values = ["Ð¡Ð¾Ñ„Ð¸Ñ", "Ð’Ð°Ñ€Ð½Ð°", "Ð ÑƒÑ Ðµ", "Ð‘ÑƒÑ€Ð³Ð°Ñ"] * 5
    finding = one(pd.DataFrame({"city": values}), "encoding_damage")
    assert finding.severity == "high"
    assert finding.suggestion is None, "fixed by re-reading, not by transforming"


def test_correct_cyrillic_is_not_flagged_as_damaged():
    frame = pd.DataFrame({"city": ["София", "Варна", "Русе", "Бургас"] * 5})
    assert "encoding_damage" not in fired(frame)


# --------------------------------------------------------------- date layouts

def test_mixed_date_layouts_are_caught():
    values = ["01/02/2024", "2024-03-15", "15.04.2024", "2024-05-06"]
    finding = one(pd.DataFrame({"d": values * 5}), "mixed_date_formats")
    assert len(finding.evidence["layouts"]) >= 2
    assert finding.suggestion is None, "which layout is intended is not in the data"


def test_one_consistent_layout_is_not_flagged():
    values = pd.date_range("2024-01-01", periods=40).strftime("%Y-%m-%d").tolist()
    assert "mixed_date_formats" not in fired(pd.DataFrame({"d": values}))


# --------------------------------------------------------- control characters

def test_a_newline_inside_a_value_is_caught():
    frame = pd.DataFrame({"addr": ["Sofia\nBulgaria", "Varna", "Ruse"] * 5})
    assert one(frame, "control_characters").affected_rows == 5


# ------------------------------------------------------------- column headers

def test_duplicate_headers_cannot_be_profiled_and_say_so():
    """`frame["Amount"]` with two such columns returns a DataFrame, not a Series.

    Every operation expecting one column then fails with pandas' "truth value of a
    Series is ambiguous", which says nothing about the real problem -- including the
    check whose job is to report the duplication. So it fails early, with a message
    naming the columns and the fix.
    """
    frame = pd.DataFrame([[1, 2], [3, 4]], columns=["Amount", "Amount"])
    with pytest.raises(ValueError, match="appear more than once"):
        profile_frame(frame)


def test_the_loader_makes_duplicate_headers_unique(tmp_path):
    path = tmp_path / "dup.csv"
    path.write_text("Amount,Amount,city\n1,2,a\n3,4,b\n5,6,c\n", encoding="utf-8")
    result = read_table(path)
    assert len(set(result.frame.columns)) == 3, "every column addressable"


def test_a_numeric_suffix_in_a_header_is_reported():
    """pandas renames a duplicate header to `Amount.1` and says nothing."""
    frame = pd.DataFrame({"Amount": [1, 2], "Amount.1": [3, 4], "city": ["a", "b"]})
    finding = one(frame, "column_names")
    assert finding.evidence["numeric_suffix"] == ["Amount.1"]
    assert finding.suggestion.action == "clean_column_names"


def test_untrimmed_and_broken_headers_are_caught():
    frame = pd.DataFrame({"city ": ["a", "b"], "region\n": ["c", "d"]})
    finding = one(frame, "column_names")
    assert "city " in finding.evidence["untrimmed"]
    assert finding.evidence["line_breaks"] == ["region\n"]


def test_clean_headers_are_not_flagged():
    frame = pd.DataFrame({"city": ["a", "b"], "amount": [1, 2]})
    assert "column_names" not in fired(frame)


# ------------------------------------------------------------ rows, not columns

def test_a_fully_empty_row_is_caught():
    frame = pd.DataFrame({"a": [1.0, None, 3.0], "b": ["x", None, "z"]})
    assert one(frame, "empty_rows").affected_rows == 1


def test_a_totals_row_is_caught():
    """A total is a summary of the data, not part of it."""
    frame = pd.DataFrame({
        "city": ["Sofia", "Varna", "Ruse", "Burgas", "Plovdiv", "TOTAL"],
        "region": ["W", "E", "N", "E", "S", None],
        "amount": [100.0, 200.0, 150.0, 300.0, 250.0, 1000.0],
    })
    finding = one(frame, "summary_rows")
    assert finding.evidence["row_positions"] == [5]
    assert finding.suggestion.action == "drop_rows"


def test_a_word_like_total_in_the_middle_is_not_a_summary_row():
    """Only rows near the foot of the file are candidates."""
    frame = pd.DataFrame({
        "label": ["TOTAL"] + [f"item {i}" for i in range(40)],
        "other": ["x"] * 41,
        "n": list(range(41)),
    })
    assert "summary_rows" not in fired(frame)


# -------------------------------------------------------- the target is sacred

def test_a_missing_target_is_dropped_not_imputed():
    """A filled-in label is a guess presented as an observation."""
    frame = pd.DataFrame({
        "age": list(range(20, 50)),
        "income": [">50K", "<=50K", None] * 10,
    })
    finding = one(frame, "missing_values", target="income")
    assert finding.suggestion.action == "drop_rows_missing_target"
    assert finding.severity == "high"
    assert "guessed label is worse" in finding.message


def test_without_a_target_the_same_column_is_imputed():
    frame = pd.DataFrame({
        "age": list(range(20, 50)),
        "income": [">50K", "<=50K", None] * 10,
    })
    assert one(frame, "missing_values").suggestion.action == "impute"


def test_the_target_is_never_dropped_as_a_column():
    frame = pd.DataFrame({"v": range(60), "label": [f"L{i}" for i in range(60)]})
    finding = one(frame, "identifier_columns", target="label")
    assert finding.suggestion is None
    assert "excluded from this repair" in finding.message


def test_other_columns_in_a_shared_repair_survive():
    """Excluding the target must not cancel the repair for everything else."""
    from src.finding import Finding, Suggestion

    finding = Finding(
        check="uninformative_columns", severity="medium", topic="structure",
        message="m", columns=["a", "target"],
        suggestion=Suggestion(action="drop_columns",
                              params={"columns": ["a", "target"]}),
    )
    [protected] = protect_target([finding], "target")
    assert protected.suggestion.params["columns"] == ["a"]
