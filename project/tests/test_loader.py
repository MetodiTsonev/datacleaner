"""Stage 1 tests - reading files.

The cases that matter are the silent ones: a wrong delimiter or a wrong header
assumption produces a table rather than an error, and every later stage then
describes something that is not the data.
"""

from __future__ import annotations

from io import BytesIO

import pandas as pd
import pytest

from src.loader import ENCODINGS, read_table, sniff_delimiter


def write(tmp_path, name: str, text: str, encoding: str = "utf-8"):
    path = tmp_path / name
    path.write_bytes(text.encode(encoding))
    return path


def test_reads_a_comma_csv(tmp_path):
    path = write(tmp_path, "a.csv", "x,y\n1,2\n3,4\n")
    result = read_table(path)
    assert result.shape == (2, 2)
    assert list(result.frame.columns) == ["x", "y"]
    assert result.delimiter == ","
    assert result.encoding == "utf-8"


def test_sniffs_a_semicolon_csv(tmp_path):
    """The Excel-in-a-comma-decimal-locale case. A comma read gives one column."""
    path = write(tmp_path, "a.csv", "x;y;z\n1;2;3\n4;5;6\n")
    result = read_table(path)
    assert result.delimiter == ";"
    assert result.shape == (2, 3)


def test_sniffs_a_tab_file(tmp_path):
    path = write(tmp_path, "a.csv", "x\ty\n1\t2\n")
    assert read_table(path).shape == (1, 2)


def test_falls_back_to_cp1252_when_utf8_fails(tmp_path):
    """A pound sign written by Excel is byte 0xA3, which is not valid UTF-8."""
    path = write(tmp_path, "a.csv", "item,price\nbook,£10\n", encoding="cp1252")
    with pytest.raises(UnicodeDecodeError):
        path.read_bytes().decode("utf-8")
    result = read_table(path)
    assert result.encoding == "cp1252"
    assert "£10" in result.frame["price"].to_numpy()


def test_latin1_fallback_is_flagged_rather_than_trusted(tmp_path):
    """latin-1 decodes any byte, so success proves nothing and must be reported."""
    path = tmp_path / "a.csv"
    path.write_bytes(b"x,y\n1,\x81\x9d\n")  # undefined in cp1252, decodable in latin-1
    result = read_table(path)
    assert result.encoding == "latin-1"
    assert any("latin-1" in n for n in result.notes)


def test_header_false_generates_names(tmp_path):
    path = write(tmp_path, "a.csv", "1,2\n3,4\n")
    result = read_table(path, has_header=False)
    assert list(result.frame.columns) == ["column_1", "column_2"]
    assert result.shape == (2, 2), "no row may be consumed as a header"


def test_header_true_on_a_headerless_file_loses_a_row(tmp_path):
    """Documents the damage the toggle exists to prevent."""
    path = write(tmp_path, "a.csv", "1,2\n3,4\n")
    assert read_table(path, has_header=True).shape == (1, 2)
    assert read_table(path, has_header=False).shape == (2, 2)


def test_reads_excel(tmp_path):
    path = tmp_path / "a.xlsx"
    pd.DataFrame({"x": [1, 2], "y": ["a", "b"]}).to_excel(path, index=False)
    result = read_table(path)
    assert result.shape == (2, 2)
    assert list(result.frame.columns) == ["x", "y"]


def test_reads_a_file_object_that_was_already_consumed():
    """Streamlit hands back a buffer whose cursor may not be at the start."""
    buffer = BytesIO(b"x,y\n1,2\n")
    buffer.name = "upload.csv"
    buffer.read()
    result = read_table(buffer)
    assert result.shape == (1, 2)


def test_unsupported_extension_raises(tmp_path):
    path = write(tmp_path, "a.json", "{}")
    with pytest.raises(ValueError, match="Unsupported file type"):
        read_table(path)


def test_one_column_result_is_flagged(tmp_path):
    """A forced-wrong delimiter is not an error, so it has to be surfaced."""
    path = write(tmp_path, "a.csv", "x;y\n1;2\n")
    result = read_table(path, delimiter=",")
    assert result.frame.shape[1] == 1
    assert any("delimiter" in n for n in result.notes)


def test_empty_file_is_flagged(tmp_path):
    path = write(tmp_path, "a.csv", "x,y\n")
    result = read_table(path)
    assert result.frame.empty
    assert any("zero rows" in n for n in result.notes)


@pytest.mark.parametrize("text,expected", [
    ("a,b,c\n1,2,3", ","),
    ("a;b;c\n1;2;3", ";"),
    ("a\tb\tc\n1\t2\t3", "\t"),
    ("a|b|c\n1|2|3", "|"),
    ("single\nvalue", ","),
])
def test_sniff_delimiter(text, expected):
    assert sniff_delimiter(text) == expected


def test_summary_reports_how_the_file_was_read(tmp_path):
    path = write(tmp_path, "a.csv", "x;y\n1;2\n")
    summary = read_table(path).summary()
    assert summary["delimiter"] == "semicolon"
    assert summary["encoding"] == "utf-8"
    assert summary["header_row"] == "yes"


def test_encoding_order_puts_utf8_first():
    """latin-1 must be last: it never fails, so trying it early hides real answers."""
    assert ENCODINGS[0] == "utf-8"
    assert ENCODINGS[-1] == "latin-1"
