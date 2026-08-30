"""Running the plan.

Two halves, and the line between them is the point of the module.

Everything before the split is structural: it removes or rewrites values without
computing anything from the data, so it cannot leak. Everything after is *fitted* -
a median, a mode, a bound - and is computed from the training half only, then applied
to both. If a fill value came from the whole file, the test half has influenced the
training half's cleaning, and the evaluation is optimistic in a way that never shows
up on new data.

Each operation reports what it actually changed. "Applied 9 steps" is a claim; "filled
6,465 cells in 3 columns, dropped 52 rows" is an account.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd

from src.anomalies import apply_cap, fit_cap
from src.loader import make_names_unique
from src.plan import POST_SPLIT, PRE_SPLIT, Plan, Step
from src.text import (
    DISGUISED_TOKENS,
    normalise,
    plain,
    string_columns,
    strip_number_marks,
)

SEED = 20260827
DEFAULT_TEST_SIZE = 0.2


@dataclass
class OpResult:
    """What a structural operation produced."""

    frame: pd.DataFrame
    detail: str
    cells_changed: int = 0
    # Set by clean_column_names. Later steps hold the *old* names in their params, so
    # the runner has to rewrite them or every one of them silently does nothing.
    renames: dict[str, str] = field(default_factory=dict)


@dataclass
class Applied:
    """What one step did."""

    step: Step
    detail: str
    rows_before: int
    rows_after: int
    columns_before: int
    columns_after: int
    cells_changed: int = 0
    # What was learned from the training half, for steps that learn anything.
    fitted: dict[str, Any] = field(default_factory=dict)

    @property
    def rows_removed(self) -> int:
        return self.rows_before - self.rows_after

    @property
    def columns_removed(self) -> int:
        return self.columns_before - self.columns_after


@dataclass
class CleanResult:
    train: pd.DataFrame
    test: pd.DataFrame | None
    applied: list[Applied]
    # Steps in the plan this module cannot perform yet. Named, not silently dropped.
    skipped: list[Step] = field(default_factory=list)
    # Row labels from the *input* frame that landed in the test half. Kept so the
    # evaluation can line the raw arm up against the same held-back rows - comparing
    # two arms across two different test sets measures nothing.
    test_ids: pd.Index = field(default_factory=lambda: pd.Index([]))

    @property
    def frame(self) -> pd.DataFrame:
        """Train and test recombined, for a run with no target."""
        if self.test is None or self.test.empty:
            return self.train
        return pd.concat([self.train, self.test], ignore_index=True)

    def summary(self) -> dict[str, Any]:
        rows_out = len(self.train) + (len(self.test) if self.test is not None else 0)
        return {
            "steps_applied": len(self.applied),
            "steps_skipped": len(self.skipped),
            "rows_out": rows_out,
            "rows_removed": sum(a.rows_removed for a in self.applied),
            "columns_out": self.train.shape[1],
            "cells_changed": sum(a.cells_changed for a in self.applied),
            "train_rows": len(self.train),
            "test_rows": len(self.test) if self.test is not None else 0,
            "nulls_remaining": int(self.train.isna().to_numpy().sum()),
            "values_capped": sum(
                a.cells_changed for a in self.applied
                if a.step.action == "cap_outliers"
            ),
        }


# ------------------------------------------------------------------------- split


def holdout_mask(
    frame: pd.DataFrame,
    *,
    target: str | None = None,
    test_size: float = DEFAULT_TEST_SIZE,
    seed: int = SEED,
) -> np.ndarray:
    """Which rows go to the test half, as a boolean mask over positions.

    Separate from `split` because the evaluation needs to know *which* rows were held
    back, so it can score the raw and the cleaned arm on the same ones. A comparison
    across two different test sets measures nothing.
    """
    rng = np.random.default_rng(seed)
    n = len(frame)
    mask = np.zeros(n, dtype=bool)
    if n < 2:
        return mask

    if target and target in frame.columns:
        # Stratify: keep each class in the same proportion on both sides. Matters when
        # a class is rare - an unstratified split of a 5% class can put almost all of
        # it on one side.
        labels = frame[target].astype("string").fillna("__missing__")
        for _, positions in labels.groupby(labels).indices.items():
            group = np.asarray(positions)
            rng.shuffle(group)
            mask[group[: int(round(len(group) * test_size))]] = True
    else:
        order = rng.permutation(n)
        mask[order[: int(round(n * test_size))]] = True
    return mask


def split(
    frame: pd.DataFrame,
    *,
    target: str | None = None,
    test_size: float = DEFAULT_TEST_SIZE,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into train and test, stratified on the target when there is one."""
    if len(frame) < 2:
        return frame.copy(), frame.iloc[0:0].copy()
    mask = holdout_mask(frame, target=target, test_size=test_size, seed=seed)
    train = frame.iloc[np.flatnonzero(~mask)].reset_index(drop=True)
    test = frame.iloc[np.flatnonzero(mask)].reset_index(drop=True)
    return train, test


# ------------------------------------------------------- structural (pre-split)


def clean_column_names(frame: pd.DataFrame, params: dict) -> OpResult:
    cleaned = [
        " ".join(str(c).replace("\n", " ").replace("\r", " ").split())
        for c in frame.columns
    ]
    unique, renamed = make_names_unique(cleaned)
    out = frame.copy()
    out.columns = unique
    mapping = {
        str(old): new
        for old, new in zip(frame.columns, unique, strict=True)
        if str(old) != new
    }
    detail = f"tidied {len(mapping)} column name(s)"
    if renamed:
        detail += f"; made {len(renamed)} duplicate name(s) distinct"
    return OpResult(out, detail, renames=mapping)


def drop_empty_rows(frame: pd.DataFrame, params: dict) -> OpResult:
    keep = ~frame.isna().all(axis=1)
    return OpResult(
        frame[keep], f"dropped {int((~keep).sum())} blank row(s)"
    )


def drop_rows(frame: pd.DataFrame, params: dict) -> OpResult:
    """Drop by position. Used for totals rows, which are identified positionally."""
    positions = [p for p in params.get("positions", []) if 0 <= p < len(frame)]
    if not positions:
        return OpResult(frame, "no matching rows")
    out = frame.drop(index=frame.index[positions])
    return OpResult(out, f"dropped {len(positions)} row(s) at {positions}")


def parse_numeric(frame: pd.DataFrame, params: dict) -> OpResult:
    """Turn "1 234,56" into a number. Values that still don't parse become null."""
    out = frame.copy()
    notes = []
    for column in params.get("columns", []):
        if column not in out.columns:
            continue
        text = out[column].astype("string")
        cleaned, _ = strip_number_marks(text)
        numbers = pd.to_numeric(cleaned, errors="coerce")
        lost = int(numbers.isna().sum() - text.isna().sum())
        out[column] = numbers
        notes.append(f"{column} ({lost} value(s) did not parse)" if lost else column)
    return OpResult(out, "parsed " + ", ".join(notes) if notes else "nothing to parse")


def drop_columns(frame: pd.DataFrame, params: dict) -> OpResult:
    present = [c for c in params.get("columns", []) if c in frame.columns]
    if not present:
        return OpResult(frame, "no matching columns")
    return OpResult(frame.drop(columns=present), f"dropped {', '.join(present)}")


def normalise_categories(frame: pd.DataFrame, params: dict) -> OpResult:
    """Collapse spellings onto the commonest form.

    Not lowercasing: the exported file is for a person to read, so "Sofia", "sofia"
    and " Sofia " become whichever spelling is commonest rather than all becoming
    lowercase.

    Spacing is tidied *before* the vote, so " Varna " and "Varna " pool their counts
    behind "Varna" and a trailing space can never win. Case is left to the vote,
    because case can be meaningful ("US" is not "us").
    """
    out = frame.copy()
    changed_cells = 0
    touched = []
    for column in params.get("columns", []):
        if column not in out.columns or pd.api.types.is_numeric_dtype(out[column]):
            continue
        original = out[column].astype("string")
        key = normalise(original)
        # Tidy the spacing first: " Varna " and "Varna " should pool their votes
        # behind "Varna" rather than competing as separate spellings.
        tidy = original.str.strip().str.replace(r"\s+", " ", regex=True)
        best = (
            pd.DataFrame({"key": key, "value": tidy})
            .dropna()
            .groupby(["key", "value"], observed=True)
            .size()
            .reset_index(name="n")
            .sort_values(["key", "n"], ascending=[True, False])
            .drop_duplicates("key")
            .set_index("key")["value"]
        )
        replaced = key.map(best).astype("string")
        diff = int((replaced.fillna("") != original.fillna("")).sum())
        if diff:
            touched.append(f"{column} ({diff})")
            changed_cells += diff
        out[column] = replaced.where(original.notna(), pd.NA)
    detail = (
        f"unified spellings in {', '.join(touched)}" if touched else "nothing to unify"
    )
    return OpResult(out, detail, changed_cells)


def replace_disguised_missing(frame: pd.DataFrame, params: dict) -> OpResult:
    """Tokens and numeric sentinels become real nulls."""
    out = frame.copy()
    tokens = {t.casefold() for t in params.get("tokens", DISGUISED_TOKENS)}
    sentinels = set(params.get("numeric_values", []))
    columns = params.get("columns") or list(out.columns)
    total = 0
    touched = []
    for column in columns:
        if column not in out.columns:
            continue
        series = out[column]
        if pd.api.types.is_numeric_dtype(series):
            mask = series.isin(sentinels) if sentinels else pd.Series(
                False, index=series.index
            )
            if params.get("replace_infinite"):
                mask = mask | np.isinf(series.fillna(0))
            if not mask.any():
                continue
        else:
            text = series.astype("string").str.strip().str.casefold()
            mask = text.isin(tokens) & series.notna()
        count = int(mask.sum())
        if count:
            out.loc[mask, column] = pd.NA if not pd.api.types.is_numeric_dtype(
                series
            ) else np.nan
            touched.append(f"{column} ({count})")
            total += count
    detail = f"nulled {', '.join(touched)}" if touched else "nothing to convert"
    return OpResult(out, detail, total)


def drop_duplicate_rows(frame: pd.DataFrame, params: dict) -> OpResult:
    before = len(frame)
    out = frame.drop_duplicates(keep="first")
    return OpResult(out, f"removed {before - len(out)} duplicate row(s)")


def drop_rows_missing_target(frame: pd.DataFrame, params: dict) -> OpResult:
    """A guessed label is worse than a missing row, so these go."""
    target = params.get("target")
    if not target or target not in frame.columns:
        return OpResult(frame, "no target column")
    keep = frame[target].notna()
    out = frame[keep]
    return OpResult(out, f"dropped {int((~keep).sum())} row(s) with no label")


# ---------------------------------------------------------- fitted (post-split)


def fit_impute(train: pd.DataFrame, params: dict) -> dict[str, Any]:
    """Learn a fill value per column, from the training half only.

    This is the whole leakage guarantee in one function: it only ever sees `train`.
    """
    strategy = params.get("strategy", "auto")
    group_by = params.get("group_by")
    fills: dict[str, Any] = {}
    groups: dict[str, dict[str, Any]] = {}

    for column in params.get("columns", []):
        if column not in train.columns:
            continue
        series = train[column]
        numeric = pd.api.types.is_numeric_dtype(series)
        use = strategy if strategy != "auto" else ("median" if numeric else "mode")

        if use == "median" and numeric:
            fills[column] = plain(series.median())
        elif use == "mean" and numeric:
            fills[column] = plain(series.mean())
        else:
            counts = series.dropna().value_counts()
            fills[column] = plain(counts.index[0]) if not counts.empty else None

        # A median per group is closer to the truth than one global median, when the
        # grouping column actually predicts the value.
        if group_by and group_by in train.columns and numeric and use == "median":
            by_group = series.groupby(train[group_by], observed=True).median()
            groups[column] = {str(k): plain(v) for k, v in by_group.dropna().items()}

    return {"fill_values": fills, "group_medians": groups, "group_by": group_by}


def apply_impute(
    frame: pd.DataFrame, params: dict, fitted: dict
) -> tuple[pd.DataFrame, str, int]:
    out = frame.copy()
    fills = fitted.get("fill_values", {})
    groups = fitted.get("group_medians", {})
    group_by = fitted.get("group_by")
    filled = 0
    per_column = []

    for column, value in fills.items():
        if column not in out.columns:
            continue
        missing = out[column].isna()
        n = int(missing.sum())
        if not n:
            continue
        if params.get("add_indicator"):
            out[f"{column}_was_missing"] = missing.astype("int8")
        if column in groups and group_by in out.columns:
            mapped = out.loc[missing, group_by].astype("string").map(groups[column])
            out.loc[missing, column] = mapped.fillna(value)
        elif value is not None:
            out.loc[missing, column] = value
        filled += n
        per_column.append(f"{column} ({n})")

    detail = f"filled {', '.join(per_column)}" if per_column else "no gaps to fill"
    return out, detail, filled


# ------------------------------------------------------------------- the runner

# Structural operations. Some report a cell count as a third return value.
PRE_OPS = {
    "clean_column_names": clean_column_names,
    "drop_empty_rows": drop_empty_rows,
    "drop_rows": drop_rows,
    "parse_numeric": parse_numeric,
    "drop_columns": drop_columns,
    "normalise_categories": normalise_categories,
    "replace_disguised_missing": replace_disguised_missing,
    "drop_duplicate_rows": drop_duplicate_rows,
    "drop_rows_missing_target": drop_rows_missing_target,
}

# Fitted operations: (fit on train, apply to both).
POST_OPS = {
    "impute": (fit_impute, apply_impute),
    "cap_outliers": (fit_cap, apply_cap),
}


def run(
    frame: pd.DataFrame,
    plan: Plan,
    *,
    target: str | None = None,
    test_size: float = DEFAULT_TEST_SIZE,
    seed: int = SEED,
    add_indicator: bool = False,
    group_by: str | None = None,
) -> CleanResult:
    """Execute a plan: structural steps, then the split, then fitted steps."""
    applied: list[Applied] = []
    skipped: list[Step] = []
    current = string_columns(frame).copy()

    # Every step's params were recorded against the column names as read. Renaming
    # them in step 1 would otherwise leave every later step looking for a column that
    # no longer exists - and reporting success while doing nothing.
    pre_steps = [replace(s, params=dict(s.params)) for s in plan.stage(PRE_SPLIT)]
    post_steps = [replace(s, params=dict(s.params)) for s in plan.stage(POST_SPLIT)]
    remaining = pre_steps + post_steps

    for step in pre_steps:
        remaining = remaining[1:]
        op = PRE_OPS.get(step.action)
        if op is None:
            skipped.append(step)
            continue
        params = dict(step.params)
        if step.action == "drop_rows_missing_target":
            params["target"] = target
        before_rows, before_cols = current.shape
        outcome = op(current, params)
        current = outcome.frame
        if outcome.renames:
            _rename_in_plan(remaining, outcome.renames)
            if target in outcome.renames:
                target = outcome.renames[target]
        applied.append(
            Applied(step, outcome.detail, before_rows, len(current),
                    before_cols, current.shape[1], outcome.cells_changed)
        )

    # Split by hand rather than through `split`, so the source row labels of the test
    # half survive for the evaluation.
    if len(current) < 2:
        train, test = current.copy(), current.iloc[0:0].copy()
        test_ids = current.index[:0]
    else:
        mask = holdout_mask(current, target=target, test_size=test_size, seed=seed)
        test_ids = current.index[mask]
        train = current.iloc[np.flatnonzero(~mask)].reset_index(drop=True)
        test = current.iloc[np.flatnonzero(mask)].reset_index(drop=True)

    for step in post_steps:
        pair = POST_OPS.get(step.action)
        if pair is None:
            skipped.append(step)
            continue
        fit, apply = pair
        params = dict(step.params)
        params.setdefault("add_indicator", add_indicator)
        params.setdefault("group_by", group_by)

        fitted = fit(train, params)  # train only. This is the guarantee.
        before_rows, before_cols = train.shape
        train, detail, cells = apply(train, params, fitted)
        if len(test):
            test, _, test_cells = apply(test, params, fitted)
            cells += test_cells
        applied.append(
            Applied(step, detail, before_rows, len(train),
                    before_cols, train.shape[1], cells, fitted)
        )

    return CleanResult(
        train=train, test=test, applied=applied, skipped=skipped, test_ids=test_ids
    )


def _rename_in_plan(steps: list[Step], mapping: dict[str, str]) -> None:
    """Point later steps at the new column names."""
    for step in steps:
        columns = step.params.get("columns")
        if columns:
            step.params["columns"] = [mapping.get(c, c) for c in columns]
        for key in ("group_by", "other", "target"):
            if step.params.get(key) in mapping:
                step.params[key] = mapping[step.params[key]]


def as_table(result: CleanResult) -> pd.DataFrame:
    """What each step did, in order."""
    return pd.DataFrame(
        [
            {
                "#": i,
                "stage": "before split" if a.step.stage == PRE_SPLIT else "after split",
                "step": a.step.action,
                "what happened": a.detail,
                "rows -": a.rows_removed or None,
                "cols -": a.columns_removed or None,
                "cells changed": a.cells_changed or None,
            }
            for i, a in enumerate(result.applied, start=1)
        ]
    )
