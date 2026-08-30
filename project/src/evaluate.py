"""Did the cleaning help?

Everything else in this system judges itself. The checks measure the defects the
pipeline repairs, so "fewer findings afterwards" is a tautology, not evidence. The
only non-circular question is whether the data became more *useful* - so: one model,
one held-back set, two different treatments of the training data.

Logistic regression and ROC AUC by hand in NumPy, roughly sixty lines. Not because a
library would be wrong, but because the задание names NumPy for the numerical work and
sixty lines of gradient descent can be explained line by line, which a library call
cannot.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

LEARNING_RATE = 0.5
ITERATIONS = 400
# Ridge penalty. Small, and there to stop a perfectly separable column sending a
# weight to infinity rather than to tune anything.
L2 = 1e-4
# Most one-hot levels a single column may contribute.
LEVEL_CAP = 20


@dataclass
class Score:
    auc: float
    rows: int
    features: int
    note: str = ""


@dataclass
class Comparison:
    """Two treatments, one held-back set."""

    raw: Score
    cleaned: Score
    target: str
    corruption: float = 0.0
    #: Rows left if you instead just delete every incomplete row - the usual
    #: alternative, and the number that makes the case.
    naive_rows: int = 0

    @property
    def difference(self) -> float:
        return self.cleaned.auc - self.raw.auc

    def summary(self) -> dict[str, Any]:
        return {
            "raw_auc": round(self.raw.auc, 4),
            "cleaned_auc": round(self.cleaned.auc, 4),
            "difference": round(self.difference, 4),
            "raw_rows": self.raw.rows,
            "cleaned_rows": self.cleaned.rows,
            "corruption": self.corruption,
            "note": self.raw.note if self.scored else
                    "; ".join(sorted({self.raw.note, self.cleaned.note})),
        }

    @property
    def scored(self) -> bool:
        """False when either arm could not produce a number. The UI must not read a
        blank as "no difference"."""
        return bool(np.isfinite(self.raw.auc) and np.isfinite(self.cleaned.auc))


# --------------------------------------------------------------------- the model


def _sigmoid(z: np.ndarray) -> np.ndarray:
    # Clipped because exp(800) overflows and the result is 0 or 1 either way.
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def fit_logistic(
    X: np.ndarray, y: np.ndarray, *, iterations: int = ITERATIONS, rate: float = LEARNING_RATE
) -> np.ndarray:
    """Logistic regression by gradient descent. Returns weights, bias last.

    Minimises log-loss. The gradient of log-loss with respect to the weights is
    `X.T @ (predicted - actual) / n`, which is why the loop is three lines: predict,
    take the error, step against it.
    """
    n, k = X.shape
    weights = np.zeros(k + 1)
    design = np.column_stack([X, np.ones(n)])
    for _ in range(iterations):
        predicted = _sigmoid(design @ weights)
        gradient = design.T @ (predicted - y) / n
        gradient[:-1] += L2 * weights[:-1]  # no penalty on the bias
        weights -= rate * gradient
    return weights


def predict(X: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return _sigmoid(np.column_stack([X, np.ones(len(X))]) @ weights)


def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Area under the ROC curve, via the rank identity.

    AUC is the probability that a randomly chosen positive scores above a randomly
    chosen negative. That equals (mean rank of the positives - offset) / n_negatives,
    which needs a sort rather than a sweep over thresholds. Ties get the average rank,
    which is what makes a constant prediction score exactly 0.5 rather than 1.0.
    """
    y_true = np.asarray(y_true).astype(int)
    positives = int(y_true.sum())
    negatives = len(y_true) - positives
    if positives == 0 or negatives == 0:
        return float("nan")  # undefined with only one class present
    ranks = pd.Series(scores).rank(method="average").to_numpy()
    return float(
        (ranks[y_true == 1].sum() - positives * (positives + 1) / 2)
        / (positives * negatives)
    )


# ---------------------------------------------------------------------- the frame


def positive_class(labels: pd.Series) -> str | None:
    """Which label counts as 1. None when the column holds no usable value."""
    classes = sorted(labels.dropna().astype(str).unique())
    return classes[-1] if classes else None


def to_matrix(
    frame: pd.DataFrame,
    target: str,
    *,
    columns: list[str] | None = None,
    positive: str | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Numbers only, standardised, with the label as 0/1.

    Non-numeric columns are one-hot encoded here at their crudest, because the *raw*
    arm has to be runnable at all. That is the point of the comparison: the raw arm
    gets the minimum that lets a model start, and the cleaned arm gets the pipeline.

    `positive` must be passed for the test half. Letting each half pick its own
    positive class from its own values is a bug that hides: a test half containing one
    class would choose the other label and silently invert every score.
    """
    labels = frame[target]
    if positive is None:
        positive = positive_class(labels)
    y = (labels.astype(str) == positive).to_numpy().astype(float)

    features = frame.drop(columns=[target])
    numeric = features.select_dtypes(include=[np.number])
    made = {}
    for name in features.columns:
        if name in numeric.columns:
            continue
        text = features[name].astype("string")
        # Capped at 20 levels: a free-text column would otherwise contribute thousands
        # of columns that are each true once, which no model can use.
        for level in sorted(text.dropna().unique())[:LEVEL_CAP]:
            made[f"{name}={level}"] = (text == level).astype(float)
    # Built in one concat rather than column by column - repeated insertion fragments
    # the frame and pandas warns about it.
    numeric = pd.concat([numeric, pd.DataFrame(made, index=frame.index)], axis=1)

    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    if columns is not None:
        numeric = numeric.reindex(columns=columns, fill_value=0.0)
    numeric = numeric.fillna(numeric.median(numeric_only=True)).fillna(0.0)

    X = numeric.to_numpy(dtype=float)
    return X, y, list(numeric.columns)


def score_arm(
    train: pd.DataFrame, test: pd.DataFrame, target: str, *, note: str = ""
) -> Score:
    """Train on one frame, score on another. The test frame is never fitted on."""
    def unscorable(reason: str, columns: int = 0) -> Score:
        # The reason is kept, not replaced by the arm's name - a blank score in the UI
        # has to be able to say why it is blank.
        return Score(float("nan"), len(train), columns,
                     f"{note}: {reason}" if note else reason)

    if train.empty or target not in train.columns:
        return unscorable("no usable training rows")
    positive = positive_class(train[target])
    if positive is None:
        return unscorable("the column to predict is empty")
    X, y, columns = to_matrix(train, target, positive=positive)
    if len(np.unique(y)) < 2:
        return unscorable("only one class in the training rows", len(columns))
    if test.empty or target not in test.columns:
        return unscorable("no held-out rows to score on", len(columns))

    mean, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd == 0] = 1.0
    weights = fit_logistic((X - mean) / sd, y)

    X_test, y_test, _ = to_matrix(test, target, columns=columns, positive=positive)
    scores = predict((X_test - mean) / sd, weights)
    return Score(
        auc=roc_auc(y_test, scores),
        rows=len(train),
        features=len(columns),
        note=note,
    )


# ------------------------------------------------------------------- the harness


def corrupt(
    frame: pd.DataFrame, share: float, *, target: str, seed: int = 20260830
) -> pd.DataFrame:
    """Damage a share of the cells, in the ways this system claims to repair.

    Deliberately *our own* corruption, and that is a stated limitation: the numbers
    below measure recovery from our simulator, not from the world.

    The target is never touched - corrupting the labels would measure something else
    entirely.
    """
    if share <= 0:
        return frame.copy()
    rng = np.random.default_rng(seed)
    out = frame.copy()
    columns = [c for c in out.columns if c != target]
    for name in columns:
        n = len(out)
        chosen = rng.random(n) < share
        if not chosen.any():
            continue
        if pd.api.types.is_numeric_dtype(out[name]):
            out[name] = out[name].astype(float)
            out.loc[chosen, name] = -999.0  # the sentinel a real export would use
        else:
            out[name] = out[name].astype("string")
            out.loc[chosen, name] = rng.choice(["?", "N/A", "", "-"], chosen.sum())
    return out


def naive_baseline(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    """What most people do: drop every row with a blank, and hope.

    Included because it is the honest comparator. The interesting question is not
    whether the pipeline beats doing nothing, but whether it beats the reflex.
    """
    out = frame.replace(["?", "N/A", "", "-", -999.0, -999], np.nan)
    return out.dropna()


def compare(
    frame: pd.DataFrame,
    *,
    target: str,
    corruption: float = 0.0,
    seed: int = 20260830,
) -> Comparison:
    """Run the whole pipeline and score raw against cleaned on the same held-out rows.

    Imported here rather than at module top: the other direction would be a cycle, and
    this module is the only consumer.
    """
    from src import clean, detect, features, profile
    from src import plan as planner

    working = corrupt(frame, corruption, target=target, seed=seed) if corruption else frame

    profiles = profile.profile_frame(working)
    findings = detect.detect(working, profiles, target=target)
    steps = planner.build(findings, target=target)
    result = clean.run(working, steps, target=target, seed=seed)

    train, test, _ = features.build(
        result.train, result.test, profile.profile_frame(result.train), target=target
    )

    # The same source rows the pipeline held back - not a fresh split, or the two arms
    # would be scored on different data and the difference would mean nothing.
    held = result.test_ids
    raw_test = working.loc[held]
    raw_train = working.drop(index=held)

    cleaned = score_arm(train, test, target, note="pipeline")
    raw = score_arm(raw_train, raw_test, target, note="as uploaded")

    return Comparison(
        raw=raw,
        cleaned=cleaned,
        target=target,
        corruption=corruption,
        naive_rows=len(naive_baseline(raw_train, target)),
    )


def sweep(
    frame: pd.DataFrame,
    *,
    target: str,
    shares: tuple[float, ...] = (0.0, 0.1, 0.2, 0.4),
    seed: int = 20260830,
) -> pd.DataFrame:
    """The same comparison at rising damage. One row per level.

    This is the experiment that decides the claim: if the difference is flat, the
    honest conclusion is that preparation did not measurably help; if it grows with the
    damage, the claim is conditional and we can say what it is conditional on.
    """
    rows = []
    for share in shares:
        comparison = compare(frame, target=target, corruption=share, seed=seed)
        rows.append({"corruption": share, **comparison.summary(),
                     "scored": comparison.scored,
                     "naive_rows": comparison.naive_rows})
    return pd.DataFrame(rows)
