"""Stage 2 - profiling: what is actually in each column.

Everything downstream depends on this stage, because the задание requires the
system to accept *arbitrary* raw files. Nothing may be hardcoded about a
particular dataset, so which repair applies to a column has to be decided from
what the column contains.

pandas' own dtype is not enough for that decision. A CSV column of digits arrives
as `int64`, but so does a column of postcodes, and a postcode is not a quantity --
you must not take its median. A column of `"yes"`/`"no"` arrives as `object`, the
same dtype as free text. So this module assigns each column a **semantic type**
above the dtype, and that is what later stages branch on.

The thresholds below are deliberately conservative and stated as constants rather
than buried in conditionals, because every one of them is a judgement call that
has to be defensible. They are explained in
writing/04-implementation/01-loader-profile.md.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

#: Share of non-null values that must parse as numbers before a text column is
#: treated as numeric. Below 1.0 so that a mostly-numeric column with a few stray
#: entries is still recognised -- those strays are a finding, not a reason to give
#: up on the column.
NUMERIC_THRESHOLD = 0.95

#: Same idea for dates. Higher, because date parsing is far more willing to accept
#: nonsense: pandas will read "12" as a date if allowed to.
DATETIME_THRESHOLD = 0.98

#: A column is categorical if it has at most this many distinct values...
CATEGORICAL_MAX_DISTINCT = 50
#: However many distinct values it has, a column whose values are this unique
#: cannot be a set of labels -- it is text or an identifier.
CATEGORICAL_MAX_SHARE_OF_ROWS = 0.5

#: ...or if its distinct values are at most this share of its rows. The second
#: rule is what makes the first scale: 50 distinct values in 100 rows is not a
#: category, 50 in 50,000 rows plainly is.
CATEGORICAL_MAX_DISTINCT_SHARE = 0.05

#: At or below this many distinct whole numbers, a numeric column *might* be a
#: code rather than a quantity. Deliberately used only to annotate, never to
#: reclassify -- see `_infer_type`. Kept small: 16 distinct values of
#: `education_num` are years of schooling, a real quantity.
NUMERIC_CODE_HINT_DISTINCT = 10

#: Above this share of distinct values, a column identifies rows rather than
#: describing them. Such columns must never become model features: an identifier
#: correlated with the target by accident of collection order is a classic
#: leakage route.
IDENTIFIER_MIN_DISTINCT_SHARE = 0.99

#: Mean character length above which a non-categorical text column is free text
#: rather than a short label.
TEXT_MIN_MEAN_LENGTH = 40

#: Values that *look* like data but mean "missing". Defined here, in the first
#: stage that needs them, and imported by `detect` and `clean`. Profiling has to
#: know about them: a numeric column containing one "N/A" would otherwise profile
#: as categorical, and would then be filled with a mode instead of a median.
DISGUISED_TOKENS = frozenset(
    {
        "?", "??", "n/a", "n\\a", "na", "null", "none", "nil", "nan",
        "-", "--", "---", ".", "unknown", "unspecified", "missing", "",
        "-999", "-9999", "999", "9999",
    }
)

#: Recognised as booleans regardless of case.
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


def profile_column(series: pd.Series) -> ColumnProfile:
    """Describe one column and assign it a semantic type."""
    n_rows = len(series)
    present = series.dropna()
    n_missing = n_rows - len(present)
    distinct = present.nunique()

    semantic, note = _infer_type(series, present, distinct, n_rows)

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
        examples=[_plain(v) for v in present.unique()[:5]],
    )


def _infer_type(
    series: pd.Series, present: pd.Series, distinct: int, n_rows: int
) -> tuple[str, str]:
    """Assign a semantic type. Order matters: cheapest and most certain first."""
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

    as_text = present.astype("string").str.strip()

    if distinct == 2 and set(as_text.str.lower().unique()) in BOOLEAN_SETS:
        return "boolean", f"two values: {sorted(as_text.unique())}"

    # Judge the *real* values. A column of numbers with a few "?" in it is a
    # numeric column with disguised blanks, not a categorical column, and calling
    # it categorical here would get it mode-filled later instead of median-filled.
    real = as_text[~as_text.str.lower().isin(DISGUISED_TOKENS)]
    n_disguised = len(as_text) - len(real)
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
    if distinct / n_rows >= IDENTIFIER_MIN_DISTINCT_SHARE:
        return "identifier", (
            f"{100 * distinct / n_rows:.1f}% of values are unique - identifies "
            "rows rather than describing them, so it must not become a feature"
        )
    return "categorical", f"high cardinality ({distinct} distinct values)"


def _looks_categorical(distinct: int, n_rows: int) -> bool:
    """Whether the distinct values look like a set of labels.

    Two rules, and a guard. The absolute rule catches ordinary categories; the
    share rule lets it scale to large tables. The guard matters: without it, 50
    distinct values in 50 rows satisfies the absolute rule, so a small column of
    free text would be called categorical.
    """
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
    """True when every value is a whole number, floats included."""
    values = pd.to_numeric(present, errors="coerce").dropna()
    return bool(len(values)) and bool(np.all(np.mod(values.to_numpy(float), 1) == 0))


def _datetime_share(as_text: pd.Series) -> float:
    """Share of values pandas can read as dates.

    Guarded: pandas warns (and historically guessed per-element) on ambiguous
    input, and a purely numeric column would otherwise be read as dates.
    """
    if pd.to_numeric(as_text, errors="coerce").notna().mean() > 0.5:
        return 0.0
    try:
        parsed = pd.to_datetime(as_text, errors="coerce", format="mixed")
    except (ValueError, TypeError):
        return 0.0
    return float(parsed.notna().mean())


def _stats(present: pd.Series, semantic: str) -> dict[str, float]:
    """Numeric summary, only where it means something."""
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
        lengths = present.astype("string").str.len()
        return {
            "mean_length": round(float(lengths.mean()), 1),
            "max_length": float(lengths.max()),
        }
    return {}


def profile_frame(frame: pd.DataFrame) -> list[ColumnProfile]:
    """Profile every column, in the frame's own column order."""
    return [profile_column(frame[col]) for col in frame.columns]


def as_table(profiles: list[ColumnProfile]) -> pd.DataFrame:
    """Profiles as a display table for the UI and the report."""
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
                "top value": _plain(p.top_value),
                "top %": p.top_share or None,
                "skew": round(p.stats["skew"], 2) if "skew" in p.stats else None,
                "note": p.note,
            }
        )
    return pd.DataFrame(rows)


def type_counts(profiles: list[ColumnProfile]) -> dict[str, int]:
    """How many columns of each semantic type, for the overview."""
    counts: dict[str, int] = {}
    for p in profiles:
        counts[p.semantic_type] = counts.get(p.semantic_type, 0) + 1
    return {t: counts[t] for t in SEMANTIC_TYPES if t in counts}


def _plain(value: Any) -> Any:
    """Convert numpy scalars to plain Python so Streamlit and JSON can hold them."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return value.item() if hasattr(value, "item") else value
