"""Capping extreme values.

Detection lives in `checks.check_outliers`; this is the repair.

**Capped, never deleted.** An extreme value is as often real as wrong - a director's
salary, a genuinely long delivery - and deleting the row discards everything else in
it. Capping keeps the row, keeps the ordering, and limits the influence the value has
on a mean or a scaler. The cost is that a real extreme is understated, which is
recorded rather than hidden.

**Bounds come from the training half only**, like every other fitted step. A bound
computed from the whole file would let the test rows decide what counts as normal.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.checks import (
    DOMINANT_VALUE_SHARE,
    IQR_MULTIPLIER,
    MAD_SCALE,
    MODIFIED_Z_THRESHOLD,
    OUTLIER_MIN_DISTINCT,
)

METHODS = ("iqr", "mad", "zscore")

# Beyond this share, capping stops being a repair and becomes a reshaping of the
# column. Reported and refused rather than applied.
MAX_CAPPED_SHARE = 0.10


def bounds(values: pd.Series, method: str = "iqr") -> tuple[float, float] | None:
    """Lower and upper limit for a column, or None when the rule cannot judge it.

    Three rules, and each declines in a different situation:

    `iqr`     Q1/Q3 +- k*IQR. Quartiles are robust, but collapse when one value
              dominates the column.
    `mad`     median +- t*MAD/0.6745. Robust to masking, because the median absolute
              deviation is not dragged outward by the values it is looking for.
              Collapses on a short discrete scale, where the MAD can be 1.
    `zscore`  mean +- t*sd. Included for the comparison in chapter 2 and not
              recommended: both terms are pulled by the very values being sought, so
              one large outlier hides itself.
    """
    finite = values[np.isfinite(values)].dropna()
    if len(finite) < 20 or finite.nunique() < OUTLIER_MIN_DISTINCT:
        return None
    if float(finite.value_counts(normalize=True).iloc[0]) >= DOMINANT_VALUE_SHARE:
        return None

    if method == "iqr":
        q1, q3 = finite.quantile([0.25, 0.75])
        spread = float(q3 - q1)
        if spread <= 0:
            return None
        return float(q1 - IQR_MULTIPLIER * spread), float(q3 + IQR_MULTIPLIER * spread)

    if method == "mad":
        median = float(finite.median())
        mad = float((finite - median).abs().median())
        if mad <= 0:
            return None
        reach = MODIFIED_Z_THRESHOLD * mad / MAD_SCALE
        return median - reach, median + reach

    if method == "zscore":
        mean, sd = float(finite.mean()), float(finite.std())
        if not sd or not np.isfinite(sd):
            return None
        return mean - 3.0 * sd, mean + 3.0 * sd

    raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")


def fit_cap(train: pd.DataFrame, params: dict) -> dict[str, Any]:
    """Learn a bound per column, from the training half only."""
    method = params.get("method", "iqr")
    learned: dict[str, list[float]] = {}
    refused: dict[str, str] = {}

    for column in params.get("columns", []):
        if column not in train.columns:
            continue
        values = pd.to_numeric(train[column], errors="coerce")
        limits = bounds(values, method)
        if limits is None:
            refused[column] = (
                "the spread-based rules cannot judge this column - too few distinct "
                "values, or one value dominates it"
            )
            continue
        low, high = limits
        finite = values[np.isfinite(values)].dropna()
        # A whole-number column should stay whole. Capping age at 88.8903 is a
        # fractional age, which is not a value the column can hold.
        if len(finite) and bool(np.all(np.mod(finite.to_numpy(float), 1) == 0)):
            low, high = float(round(low)), float(round(high))
        share = float(((finite < low) | (finite > high)).mean()) if len(finite) else 0.0
        if share > MAX_CAPPED_SHARE:
            refused[column] = (
                f"the rule would cap {share:.1%} of the column. Above "
                f"{MAX_CAPPED_SHARE:.0%} this is reshaping the distribution, not "
                "repairing it"
            )
            continue
        learned[column] = [low, high]

    return {"bounds": learned, "refused": refused, "method": method}


def apply_cap(
    frame: pd.DataFrame, params: dict, fitted: dict
) -> tuple[pd.DataFrame, str, int]:
    """Clip to the learned bounds. Values inside them are untouched."""
    out = frame.copy()
    capped_total = 0
    per_column = []

    for column, (low, high) in fitted.get("bounds", {}).items():
        if column not in out.columns:
            continue
        values = pd.to_numeric(out[column], errors="coerce")
        below = int((values < low).sum())
        above = int((values > high).sum())
        if not (below or above):
            continue
        out[column] = values.clip(lower=low, upper=high)
        capped_total += below + above
        parts = []
        if below:
            parts.append(f"{below} up to {low:g}")
        if above:
            parts.append(f"{above} down to {high:g}")
        per_column.append(f"{column} ({', '.join(parts)})")

    detail = f"capped {'; '.join(per_column)}" if per_column else "nothing to cap"
    refused = fitted.get("refused") or {}
    if refused:
        detail += f"; declined {', '.join(refused)}"
    return out, detail, capped_total


def comparison(values: pd.Series) -> pd.DataFrame:
    """All three rules side by side, for the chapter-2 comparison.

    They are shown together rather than combined: when they disagree, the
    disagreement is the information.
    """
    finite = pd.to_numeric(values, errors="coerce")
    finite = finite[np.isfinite(finite)].dropna()
    rows = []
    for method in METHODS:
        limits = bounds(finite, method)
        if limits is None:
            rows.append({"rule": method, "lower": None, "upper": None,
                         "flagged": None, "note": "cannot judge this column"})
            continue
        low, high = limits
        flagged = int(((finite < low) | (finite > high)).sum())
        rows.append({
            "rule": method,
            "lower": round(low, 4),
            "upper": round(high, 4),
            "flagged": flagged,
            "note": f"{flagged / len(finite):.2%} of values" if len(finite) else "",
        })
    return pd.DataFrame(rows)
