"""What's actually in each column.

The dtype isn't enough to pick a repair: a column of digits is int64 whether it holds
ages or postcodes, and you mustn't take the median of a postcode. So each column gets
a *semantic* type on top of the dtype, and that's what later stages branch on.

Thresholds are named constants rather than buried in conditionals - each one is a
judgement call. Reasoning in writing/04-implementation/01-loader-profile.md.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.text import as_text, plain, real_values

# Below 1.0 so a mostly-numeric column with a few strays is still numeric - the
# strays are a finding, not a reason to give up on the column.
NUMERIC_THRESHOLD = 0.95
# Higher: date parsing accepts far more nonsense.
DATETIME_THRESHOLD = 0.98

# Categorical if few distinct values, OR few relative to the rows. The share rule is
# what makes the absolute one scale; the 0.5 guard stops 50-in-50-rows counting.
CATEGORICAL_MAX_DISTINCT = 50
CATEGORICAL_MAX_DISTINCT_SHARE = 0.05
CATEGORICAL_MAX_SHARE_OF_ROWS = 0.5

# Annotate only, never reclassify: age has 74 distinct values in 48,842 rows and is
# obviously a quantity. Real code columns (postcodes) have HIGH cardinality anyway.
NUMERIC_CODE_HINT_DISTINCT = 10

# Measured against distinct ROWS, not all rows: an id is unique per record, and a
# duplicated record repeats it, so counting all rows gets it backwards.
IDENTIFIER_MIN_DISTINCT_SHARE = 0.95

TEXT_MIN_MEAN_LENGTH = 40  # mean chars above which a label becomes prose

# to_datetime accepts "1,50", so a European price column read as dates. Require a
# date shape first: two matching separators, or eight digits.
_DATE_SHAPED = re.compile(
    r"^\s*(?:\d{8}|\d{1,4}([-/.])\d{1,2}\1\d{1,4}"
    r"|\d{1,4}([-/.])\d{1,2}\2\d{1,4}[ T].*)\s*$"
)

BOOLEAN_SETS = (
    {"true", "false"}, {"yes", "no"}, {"y", "n"},
    {"1", "0"}, {"t", "f"}, {"да", "не"},
)

SEMANTIC_TYPES = (
    "empty", "constant", "boolean", "numeric", "datetime",
    "categorical", "identifier", "text",
)


@dataclass
class ColumnProfile:
    """What one column contains."""

    name: str
    semantic_type: str
    dtype: str
    n_rows: int
    n_missing: int
    n_distinct: int
    #: Set when the semantic type was inferred against the dtype, e.g. digits
    #: stored as text. Carries the share that parsed, so the decision is auditable.
    note: str = ""
    top_value: Any = None
    top_share: float = 0.0
    stats: dict[str, float] = field(default_factory=dict)
    examples: list[Any] = field(default_factory=list)

    @property
    def pct_missing(self) -> float:
        return 100.0 * self.n_missing / self.n_rows if self.n_rows else 0.0

    @property
    def pct_distinct(self) -> float:
        return 100.0 * self.n_distinct / self.n_rows if self.n_rows else 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def profile_column(
    series: pd.Series, *, n_distinct_rows: int | None = None
) -> ColumnProfile:
    """Describe one column and give it a semantic type.

    n_distinct_rows is only for identifier detection; defaults to the column length.
    """
    n_rows = len(series)
    present = series.dropna()
    n_missing = n_rows - len(present)
    distinct = present.nunique()

    semantic, note = _infer_type(
        series, present, distinct, n_rows, n_distinct_rows or n_rows
    )

    top_value, top_share = None, 0.0
    if len(present) and semantic in {"boolean", "categorical", "constant", "text"}:
        counts = present.value_counts()
        top_value = counts.index[0]
        top_share = 100.0 * float(counts.iloc[0]) / len(present)

    return ColumnProfile(
        name=str(series.name),
        semantic_type=semantic,
        dtype=str(series.dtype),
        n_rows=n_rows,
        n_missing=int(n_missing),
        n_distinct=int(distinct),
        note=note,
        top_value=top_value,
        top_share=round(top_share, 2),
        stats=_stats(present, semantic),
        examples=[plain(v) for v in present.unique()[:5]],
    )


def _infer_type(
    series: pd.Series,
    present: pd.Series,
    distinct: int,
    n_rows: int,
    n_distinct_rows: int,
) -> tuple[str, str]:
    """Order matters: cheapest and most certain first."""
    if len(present) == 0:
        return "empty", "every value is missing"
    if distinct == 1:
        return "constant", f"one value throughout: {present.iloc[0]!r}"

    if pd.api.types.is_bool_dtype(series):
        return "boolean", ""
    if pd.api.types.is_numeric_dtype(series):
        # A 0/1 column is a flag, not a quantity.
        if distinct == 2 and _is_integral(present):
            values = set(pd.to_numeric(present, errors="coerce").dropna().unique())
            if values <= {0, 1}:
                return "boolean", "numeric 0/1 flag"
        # Everything else numeric stays numeric. Low cardinality is *reported*,
        # not acted on: `age` has 74 distinct values in 48,842 rows and is
        # obviously a quantity, so a share-based rule would misclassify it. A
        # genuine code column (a postcode) usually has *high* cardinality, so low
        # cardinality is not the signal it appears to be.
        if distinct <= NUMERIC_CODE_HINT_DISTINCT and _is_integral(present):
            return "numeric", (
                f"only {distinct} distinct whole numbers - check whether this is a "
                "code rather than a quantity before averaging it"
            )
        return "numeric", ""
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime", ""

    text = as_text(present)

    if distinct == 2 and set(text.str.lower().unique()) in BOOLEAN_SETS:
        return "boolean", f"two values: {sorted(text.unique())}"

    # Judge the real values only. A numeric column with a few "?" in it is numeric
    # with blanks, not categorical -- otherwise it gets mode-filled, not median-filled.
    real = real_values(present)
    n_disguised = len(text) - len(real)
    disguised_note = (
        f"{n_disguised} value(s) look like disguised blanks and were ignored when "
        "deciding the type"
        if n_disguised
        else ""
    )
    if real.empty:
        return "empty", "every value is a disguised blank"
    real_distinct = int(real.nunique())

    numeric_share = pd.to_numeric(real, errors="coerce").notna().mean()
    if numeric_share >= NUMERIC_THRESHOLD:
        note = disguised_note or (
            "" if numeric_share == 1.0 else
            f"{100 * numeric_share:.1f}% of values parse as numbers - the rest "
            "are non-numeric and will be reported"
        )
        # Only whole numbers can be codes. 1.5 and 2.5 are quantities however few
        # distinct values there are.
        if real_distinct <= NUMERIC_CODE_HINT_DISTINCT and _is_integral(real):
            return "categorical", note or (
                f"numeric text with only {real_distinct} distinct whole numbers - "
                "treated as codes rather than quantities"
            )
        return "numeric", note or "stored as text, parses as numbers"

    date_share = _datetime_share(real)
    if date_share >= DATETIME_THRESHOLD:
        return "datetime", disguised_note or (
            "" if date_share == 1.0 else f"{100 * date_share:.1f}% parse as dates"
        )

    if _looks_categorical(distinct, n_rows):
        return "categorical", disguised_note
    # Order matters: a unique column of long strings is prose, not an identifier.
    if real.str.len().mean() >= TEXT_MIN_MEAN_LENGTH:
        return "text", disguised_note
    per_record = distinct / n_distinct_rows if n_distinct_rows else 0.0
    if per_record >= IDENTIFIER_MIN_DISTINCT_SHARE:
        return "identifier", (
            f"{100 * per_record:.1f}% unique per distinct row - identifies rows "
            "rather than describing them, so it must not become a feature"
        )
    return "categorical", f"high cardinality ({distinct} distinct values)"


def _looks_categorical(distinct: int, n_rows: int) -> bool:
    """Do the distinct values look like a set of labels?"""
    if n_rows == 0:
        return False
    share = distinct / n_rows
    if share >= CATEGORICAL_MAX_SHARE_OF_ROWS:
        return False
    return (
        distinct <= CATEGORICAL_MAX_DISTINCT
        or share <= CATEGORICAL_MAX_DISTINCT_SHARE
    )


def _is_integral(present: pd.Series) -> bool:
    """Every value a whole number, floats included."""
    values = pd.to_numeric(present, errors="coerce").dropna()
    return bool(len(values)) and bool(np.all(np.mod(values.to_numpy(float), 1) == 0))


def _datetime_share(as_text: pd.Series) -> float:
    """Share pandas can read as dates. Both guards are earned - see _DATE_SHAPED."""
    if pd.to_numeric(as_text, errors="coerce").notna().mean() > 0.5:
        return 0.0
    shaped = as_text.str.match(_DATE_SHAPED, na=False)
    if float(shaped.mean()) < DATETIME_THRESHOLD:
        return 0.0
    try:
        parsed = pd.to_datetime(as_text, errors="coerce", format="mixed")
    except (ValueError, TypeError):
        return 0.0
    return float(parsed.notna().mean())


def _stats(present: pd.Series, semantic: str) -> dict[str, float]:
    """Stats, only where they mean something."""
    if semantic == "numeric":
        values = pd.to_numeric(present, errors="coerce").dropna()
        if values.empty:
            return {}
        return {
            "min": float(values.min()),
            "max": float(values.max()),
            "mean": float(values.mean()),
            "median": float(values.median()),
            "std": float(values.std()),
            # Skew drives the log-transform decision in stage 7.
            "skew": float(values.skew()) if len(values) > 2 else 0.0,
            "pct_zero": round(100.0 * float((values == 0).mean()), 2),
        }
    if semantic in {"text", "categorical"}:
        lengths = as_text(present).str.len()
        return {
            "mean_length": round(float(lengths.mean()), 1),
            "max_length": float(lengths.max()),
        }
    return {}


def profile_frame(frame: pd.DataFrame) -> list[ColumnProfile]:
    """Every column, in the frame's own order.

    Raises ValueError on duplicate names: frame["Amount"] would return a DataFrame and
    everything downstream would fail with an error that says nothing useful.
    read_table makes names unique.
    """
    labels = [str(c) for c in frame.columns]
    duplicated = sorted({c for c in labels if labels.count(c) > 1})
    if duplicated:
        raise ValueError(
            f"These column names appear more than once: {duplicated}. Read the file "
            "through src.loader.read_table, which makes names unique and reports "
            "what it changed."
        )

    n_distinct_rows = len(frame.drop_duplicates()) if len(frame) else 0
    return [
        profile_column(frame[col], n_distinct_rows=n_distinct_rows)
        for col in frame.columns
    ]


def as_table(profiles: list[ColumnProfile]) -> pd.DataFrame:
    """Display table for the UI and the report."""
    rows = []
    for p in profiles:
        rows.append(
            {
                "column": p.name,
                "type": p.semantic_type,
                "dtype": p.dtype,
                "missing": p.n_missing,
                "missing %": round(p.pct_missing, 2),
                "distinct": p.n_distinct,
                "top value": plain(p.top_value),
                "top %": p.top_share or None,
                "skew": round(p.stats["skew"], 2) if "skew" in p.stats else None,
                "note": p.note,
            }
        )
    return pd.DataFrame(rows)


def type_counts(profiles: list[ColumnProfile]) -> dict[str, int]:
    """Column counts per semantic type."""
    counts: dict[str, int] = {}
    for p in profiles:
        counts[p.semantic_type] = counts.get(p.semantic_type, 0) + 1
    return {t: counts[t] for t in SEMANTIC_TYPES if t in counts}
