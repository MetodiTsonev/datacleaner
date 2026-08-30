"""Helpers for reading messy text columns.

Shared by profile, checks and validate. These live here because all three need the
same notion of "a value that actually means something".
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

# Values that look like data but mean "missing".
DISGUISED_TOKENS = frozenset(
    {
        "?", "??", "n/a", "n\\a", "na", "null", "none", "nil", "nan",
        "-", "--", "---", ".", "unknown", "unspecified", "missing", "",
        "-999", "-9999", "999", "9999",
    }
)

# Thousands separators and currency/unit marks. "1 234,56" and "$100" are numbers;
# neither parses. Common in European exports, where the comma is the decimal point.
THOUSANDS_MARKS = (" ", " ", "'", "_")
CURRENCY_MARKS = ("$", "€", "£", "лв", "lv", "BGN", "EUR", "USD", "%")

_WHITESPACE = re.compile(r"\s+")


def string_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Frame whose column labels are all strings.

    Profiles carry names as strings, so a frame with integer labels - which pandas
    produces for numeric headers, a sheet with 2020, 2021, 2022 - makes every later
    `frame[profile.name]` raise KeyError. Returns the frame unchanged when it is
    already fine, so the common path copies nothing.
    """
    labels = [str(c) for c in frame.columns]
    if labels == list(frame.columns):
        return frame
    out = frame.copy()
    out.columns = labels
    return out


def as_text(series: pd.Series) -> pd.Series:
    """Non-null values as trimmed strings."""
    return series.dropna().astype("string").str.strip()


def real_values(series: pd.Series) -> pd.Series:
    """Trimmed strings, minus anything that means "missing"."""
    text = as_text(series)
    return text[~text.str.lower().isin(DISGUISED_TOKENS)]


def disguised_values(series: pd.Series) -> pd.Series:
    """The other half of :func:`real_values`."""
    text = as_text(series)
    return text[text.str.lower().isin(DISGUISED_TOKENS)]


def normalise_scalar(value: Any) -> str:
    """Casefold, trim, collapse internal runs of whitespace."""
    return _WHITESPACE.sub(" ", str(value).strip()).casefold()


def normalise(series: pd.Series) -> pd.Series:
    """Vectorised :func:`normalise_scalar`, keeping nulls null."""
    text = (
        series.astype("string")
        .str.strip()
        .str.replace(_WHITESPACE, " ", regex=True)
        .str.casefold()
    )
    return text.where(series.notna(), pd.NA)


def strip_number_marks(values: pd.Series) -> tuple[pd.Series, list[str]]:
    """Remove human number formatting; report which marks were there.

    Order matters: thousands separators, then currency and units, then a comma
    decimal point. Doing the comma first would turn "1,234" into "1.234".
    """
    found: list[str] = []
    text = values.astype("string")

    for mark in THOUSANDS_MARKS:
        if text.str.contains(re.escape(mark), regex=True, na=False).any():
            found.append("thousands separator" if mark != "'" else "apostrophe")
            text = text.str.replace(mark, "", regex=False)
    for mark in CURRENCY_MARKS:
        if text.str.contains(re.escape(mark), case=False, regex=True, na=False).any():
            found.append(mark)
            text = text.str.replace(mark, "", case=False, regex=False)
    text = text.str.strip()
    if text.str.match(r"^-?\d+,\d{1,3}$", na=False).any():
        found.append("comma decimal separator")
        text = text.str.replace(",", ".", regex=False)
    return text, sorted(set(found))


def numeric_share_after_cleaning(values: pd.Series) -> float:
    """Share that parses as a number once formatting marks are removed."""
    cleaned, _ = strip_number_marks(values)
    return float(pd.to_numeric(cleaned, errors="coerce").notna().mean())


def plain(value: Any) -> Any:
    """numpy scalar -> plain Python, so JSON and Streamlit can hold it."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return value.item() if hasattr(value, "item") else value
