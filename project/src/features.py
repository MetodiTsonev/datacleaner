"""Turning a clean table into a matrix a model can consume.

Cleaning fixes what is wrong. This transforms what is right but unusable: a model
takes a rectangle of numbers, and a clean table is full of dates, labels and skewed
quantities.

Kept separate from cleaning on purpose. Cleaning is a *repair* traceable to a
finding; this is *preparation*, and it happens whether or not anything was wrong.
The задание names both - "почистване, трансформиране".

Everything here is fitted on the training half and applied to both, for the same
reason imputation is: a category list or a mean learned from the whole file lets the
held-back rows decide how the training rows are encoded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.profile import ColumnProfile

# Above this absolute skew, a log transform is worth trying.
SKEW_THRESHOLD = 1.0

# ...and it is kept only if it actually helped: the skew has to at least halve, or
# land inside the threshold. A log compresses a long tail; it cannot move a spike at
# zero, because log1p(0) is 0. capital_loss is 95.3% zeros and goes 4.57 -> 4.30.
SKEW_IMPROVEMENT = 0.5

# One-hot below this many categories, frequency encoding above it. One-hot on a
# hundred categories is a hundred columns of almost all zeros.
ONE_HOT_MAX_CATEGORIES = 15

# Categories rarer than this are pooled into "other" before encoding: a category
# seen three times in 40,000 rows cannot support an estimate of anything.
RARE_CATEGORY_SHARE = 0.01

# Two features correlated above this carry the same information twice.
CORRELATION_LIMIT = 0.95


@dataclass
class FeatureReport:
    """What was built, and from what."""

    steps: list[str] = field(default_factory=list)
    columns_in: int = 0
    columns_out: int = 0
    skew_before: dict[str, float] = field(default_factory=dict)
    skew_after: dict[str, float] = field(default_factory=dict)
    dropped_correlated: list[tuple[str, str, float]] = field(default_factory=list)
    encoded: dict[str, str] = field(default_factory=dict)
    # Columns where the log was tried, measured and reverted.
    log_declined: dict[str, str] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {
            "columns_in": self.columns_in,
            "columns_out": self.columns_out,
            "steps": len(self.steps),
            "logged": len(self.skew_before),
            "log_declined": len(self.log_declined),
            "dropped_correlated": len(self.dropped_correlated),
        }


def _numeric(frame: pd.DataFrame, name: str) -> pd.Series:
    return pd.to_numeric(frame[name], errors="coerce")


# ---------------------------------------------------------------- the transforms


def log_skewed(
    train: pd.DataFrame, others: list[pd.DataFrame], profiles: list[ColumnProfile],
    *, target: str | None = None,
) -> tuple[dict[str, float], dict[str, float], dict[str, str]]:
    """log1p the strongly skewed non-negative columns, in place on every frame.

    A long right tail means most of a column's range is occupied by a handful of
    rows. log1p compresses it while leaving the ordering intact, so a linear model
    stops being dominated by the largest values. Only for non-negative columns: the
    log of a negative number does not exist.

    Applied, then measured, then kept only if it helped. A column that is mostly
    zeros keeps its skew whatever you do to the non-zero part, and transforming it
    anyway would be a step that reports success without producing any.

    Returns the skew before and after, because "we transformed it" is a claim and
    "11.89 -> 3.12" is a measurement.
    """
    before: dict[str, float] = {}
    after: dict[str, float] = {}
    declined: dict[str, str] = {}
    for profile in profiles:
        name = profile.name
        if profile.semantic_type != "numeric" or name == target:
            continue
        if name not in train.columns:
            continue
        values = _numeric(train, name).replace([np.inf, -np.inf], np.nan).dropna()
        if values.empty or float(values.min()) < 0:
            continue
        skew = float(values.skew()) if len(values) > 2 else 0.0
        if not np.isfinite(skew) or abs(skew) < SKEW_THRESHOLD:
            continue
        original = {id(f): f[name].copy() for f in [train, *others] if name in f.columns}
        for frame in [train, *others]:
            if name in frame.columns:
                frame[name] = np.log1p(_numeric(frame, name))
        result = float(_numeric(train, name).dropna().skew())

        helped = abs(result) <= abs(skew) * SKEW_IMPROVEMENT or abs(result) < SKEW_THRESHOLD
        if not helped:
            for frame in [train, *others]:
                if id(frame) in original:
                    frame[name] = original[id(frame)]
            zeros = float((values == 0).mean())
            declined[name] = (
                f"skew {skew:+.2f} would only become {result:+.2f}"
                + (
                    f" - {zeros:.0%} of this column is zero, and a log cannot move a "
                    "spike at zero"
                    if zeros > 0.5
                    else ", which is not worth the transform"
                )
            )
            continue
        before[name] = round(skew, 4)
        after[name] = round(result, 4)
    return before, after, declined


def expand_dates(
    train: pd.DataFrame, others: list[pd.DataFrame], profiles: list[ColumnProfile],
    *, target: str | None = None,
) -> list[str]:
    """Replace a date with the parts a model can use.

    A date as a number is the count of seconds since 1970, which no model can do
    anything sensible with. Its useful content is the parts: which year, which month,
    which day of the week, and whether it is a weekend - seasonality and weekly
    rhythm are real effects that the raw timestamp hides.
    """
    made: list[str] = []
    for profile in profiles:
        name = profile.name
        if profile.semantic_type != "datetime" or name == target:
            continue
        if name not in train.columns:
            continue
        for frame in [train, *others]:
            if name not in frame.columns:
                continue
            parsed = pd.to_datetime(frame[name], errors="coerce", format="mixed")
            frame[f"{name}_year"] = parsed.dt.year
            frame[f"{name}_month"] = parsed.dt.month
            frame[f"{name}_weekday"] = parsed.dt.dayofweek
            frame[f"{name}_is_weekend"] = (parsed.dt.dayofweek >= 5).astype("int8")
            frame.drop(columns=[name], inplace=True)
        made.extend(
            [f"{name}_year", f"{name}_month", f"{name}_weekday", f"{name}_is_weekend"]
        )
    return made


def encode_categories(
    train: pd.DataFrame, others: list[pd.DataFrame], profiles: list[ColumnProfile],
    *, target: str | None = None,
) -> dict[str, str]:
    """Turn labels into numbers, choosing the method by how many labels there are.

    Few categories: one column per category, holding 0 or 1. Honest, and a model can
    give each category its own weight.

    Many categories: replace the label with how often it occurs. One column instead
    of a hundred, at the cost of saying that two equally common categories are the
    same number - which is a real loss, and the reason the threshold is low.

    Rare categories are pooled into "other" first. A category seen three times cannot
    support an estimate, and one-hot encoding it adds a column that is almost all
    zeros.
    """
    chosen: dict[str, str] = {}
    for profile in profiles:
        name = profile.name
        if profile.semantic_type not in {"categorical", "boolean"} or name == target:
            continue
        if name not in train.columns:
            continue

        counts = train[name].astype("string").value_counts(normalize=True)
        keep = set(counts[counts >= RARE_CATEGORY_SHARE].index)
        pooled = len(counts) - len(keep)

        if len(keep) <= ONE_HOT_MAX_CATEGORIES:
            method = "one-hot"
            levels = sorted(keep)
            for frame in [train, *others]:
                if name not in frame.columns:
                    continue
                text = frame[name].astype("string")
                for level in levels:
                    frame[f"{name}={level}"] = (text == level).astype("int8")
                if pooled:
                    frame[f"{name}=other"] = (~text.isin(levels)).astype("int8")
                frame.drop(columns=[name], inplace=True)
        else:
            method = "frequency"
            # Frequencies from the training half only.
            frequency = train[name].astype("string").value_counts(normalize=True)
            for frame in [train, *others]:
                if name not in frame.columns:
                    continue
                mapped = frame[name].astype("string").map(frequency)
                frame[name] = mapped.fillna(0.0).astype(float)

        chosen[name] = f"{method} ({len(keep)} kept, {pooled} pooled)"
    return chosen


def scale(
    train: pd.DataFrame, others: list[pd.DataFrame], *, target: str | None = None
) -> dict[str, list[float]]:
    """Centre and rescale, using the training half's mean and spread.

    Without it, a column measured in hundreds of thousands dominates one measured in
    years purely because of its units. Columns with no spread are left alone rather
    than divided by zero.
    """
    learned: dict[str, list[float]] = {}
    for name in train.columns:
        if name == target or not pd.api.types.is_numeric_dtype(train[name]):
            continue
        values = _numeric(train, name)
        mean, sd = float(values.mean()), float(values.std())
        if not np.isfinite(mean) or not np.isfinite(sd) or sd == 0:
            continue
        learned[name] = [mean, sd]
        for frame in [train, *others]:
            if name in frame.columns:
                frame[name] = (_numeric(frame, name) - mean) / sd
    return learned


def prune_correlated(
    train: pd.DataFrame, others: list[pd.DataFrame], *, target: str | None = None
) -> list[tuple[str, str, float]]:
    """Drop one of any pair of features carrying the same information.

    Keeps the first of each pair by column order - an arbitrary rule, and stated as
    such. Which of two identical features to keep is not answerable from the data.
    """
    numeric = [
        c for c in train.columns
        if c != target and pd.api.types.is_numeric_dtype(train[c])
    ]
    if len(numeric) < 2:
        return []
    matrix = train[numeric].corr().abs()
    dropped: list[tuple[str, str, float]] = []
    removed: set[str] = set()
    for i, left in enumerate(numeric):
        if left in removed:
            continue
        for right in numeric[i + 1 :]:
            if right in removed:
                continue
            value = matrix.loc[left, right]
            if pd.notna(value) and value >= CORRELATION_LIMIT:
                removed.add(right)
                dropped.append((left, right, round(float(value), 4)))
    for frame in [train, *others]:
        present = [c for c in removed if c in frame.columns]
        if present:
            frame.drop(columns=present, inplace=True)
    return dropped


# --------------------------------------------------------------------- the stage


def build(
    train: pd.DataFrame,
    test: pd.DataFrame | None,
    profiles: list[ColumnProfile],
    *,
    target: str | None = None,
    do_log: bool = True,
    do_dates: bool = True,
    do_encode: bool = True,
    do_scale: bool = True,
    do_prune: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame | None, FeatureReport]:
    """Run the preparation steps, fitted on train and applied to both."""
    out_train = train.copy()
    out_test = test.copy() if test is not None else None
    others = [out_test] if out_test is not None else []
    report = FeatureReport(columns_in=out_train.shape[1])

    if do_log:
        before, after, declined = log_skewed(out_train, others, profiles, target=target)
        report.skew_before, report.skew_after = before, after
        report.log_declined = declined
        if before:
            report.steps.append(
                f"log-transformed {len(before)} skewed column(s): "
                + ", ".join(f"{c} {before[c]:+.2f} -> {after[c]:+.2f}" for c in before)
            )
    if do_dates:
        made = expand_dates(out_train, others, profiles, target=target)
        if made:
            report.steps.append(
                f"expanded {len(made) // 4} date column(s) into {len(made)} parts"
            )
    if do_encode:
        report.encoded = encode_categories(out_train, others, profiles, target=target)
        if report.encoded:
            report.steps.append(f"encoded {len(report.encoded)} label column(s)")
    if do_scale:
        scaled = scale(out_train, others, target=target)
        if scaled:
            report.steps.append(f"centred and rescaled {len(scaled)} column(s)")
    if do_prune:
        report.dropped_correlated = prune_correlated(out_train, others, target=target)
        if report.dropped_correlated:
            report.steps.append(
                f"dropped {len(report.dropped_correlated)} column(s) that duplicated "
                "another"
            )

    report.columns_out = out_train.shape[1]
    return out_train, out_test, report


def skew_table(report: FeatureReport) -> pd.DataFrame:
    """Skew before and after, for the report and the figure."""
    return pd.DataFrame(
        [
            {
                "column": name,
                "skew before": report.skew_before[name],
                "skew after": report.skew_after.get(name),
            }
            for name in report.skew_before
        ]
    )
