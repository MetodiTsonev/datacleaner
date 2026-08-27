"""Stage 3 - the checks themselves.

One function per defect. Each takes the frame and its profiles and returns zero or
more :class:`~src.finding.Finding` objects, so a check is entirely independent of
every other check and of the runner.

Adding a check is one function plus one line in :data:`CHECKS`. There is a test that
fails if a check is written but not registered, because an unregistered check would
simply never run and nobody would notice.

Two design notes worth defending:

**Severity is not a score.** It orders the list and decides what blocks. It is never
summed into an index -- see `writing/decisions.md` Р5 for why the first attempt's
composite quality score was withdrawn.

**A check may disclaim its own result.** The outlier rules on a column where one
value dominates, or on a short discrete scale, report their counts *and* say the
counts must not be read as error counts. Silently "fixing" values a rule cannot judge
is worse than declining.
"""

from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd

from src.finding import Finding, Suggestion
from src.profile import DISGUISED_TOKENS, ColumnProfile

#: Multiplier on the interquartile range. 1.5 is the textbook "outlier" fence and
#: 3.0 the "far out" fence; 3.0 is used because at 1.5 an ordinary skewed column
#: flags a tenth of its rows, which is noise rather than a finding.
IQR_MULTIPLIER = 3.0

#: Cut-off for the MAD-based modified z-score. 3.5 is Iglewicz & Hoaglin's
#: recommendation.
MODIFIED_Z_THRESHOLD = 3.5

#: 0.6745 is the 0.75 quantile of the standard normal, which rescales the median
#: absolute deviation so that it estimates the standard deviation for normal data.
MAD_SCALE = 0.6745

#: A column where one value covers at least this share cannot be judged by a
#: spread-based rule: the quartiles collapse onto that value.
DOMINANT_VALUE_SHARE = 0.40

#: Below this many distinct values, a numeric column is a short discrete scale and
#: spread-based outlier rules stop working. `education_num` runs 1-16, so its
#: median absolute deviation is 1.0, and a modified z-score of 3.5 then flags
#: every value more than ~5 steps from the median -- which on a 16-step scale means
#: the ends of the scale. Preschool and Doctorate are not anomalies.
OUTLIER_MIN_DISTINCT = 20

#: Absolute skew above which a log transform is worth proposing.
SKEW_THRESHOLD = 1.0

#: Redundancy checking is quadratic in columns, so it is capped.
MAX_COLUMNS_FOR_PAIRS = 40

#: Numbers conventionally used to mean "no value". `check_disguised_missing` cannot
#: find these: once a CSV column of numbers containing -999 is read, the column is
#: float64 and -999 is just a number, so a token search over text finds nothing.
#: This is one of the commonest forms the defect takes, so it gets its own check.
NUMERIC_SENTINELS = (-999.0, -9999.0, -99999.0, 999.0, 9999.0, 99999.0)

#: A sentinel must repeat at least this often to be distinguishable from a value
#: that merely happens to be 999.
SENTINEL_MIN_COUNT = 2

#: Characters that turn a number written for humans into text a parser rejects.
#: Bulgarian and wider European data routinely uses a comma as the decimal separator
#: and a space or full stop as the thousands separator, so "1 234,56" is one and a
#: half thousand -- but `pd.to_numeric` refuses it, the column stays text, and the
#: system then treats a quantity as a category. Currency symbols and unit suffixes do
#: the same damage.
THOUSANDS_MARKS = (" ", "\u00a0", "'", "_")
CURRENCY_MARKS = ("$", "€", "£", "лв", "lv", "BGN", "EUR", "USD", "%")

#: Share of values that must become numeric after stripping the marks above before a
#: text column is reported as a disguised quantity.
NUMERIC_IN_TEXT_THRESHOLD = 0.90

#: A value this short in a column whose typical value is much longer is a stub rather
#: than data -- an "x" in a name column, a "1" in a description.
SHORT_VALUE_MAX_LENGTH = 2

#: The typical value must be at least this many times longer for the comparison above
#: to mean anything.
SHORT_VALUE_LENGTH_RATIO = 3.0

#: Signatures of double-decoded UTF-8. Text encoded as UTF-8 and then read as cp1252
#: or latin-1 produces these sequences: "София" becomes "Ð¡Ð¾Ñ„Ð¸Ñ". Once written back
#: out the damage is permanent, so it must be caught at ingestion.
MOJIBAKE_MARKERS = ("Ð", "Ñ", "Ã", "â€", "Â", "ï»¿")

#: How many mojibake markers per thousand characters before a column is flagged.
MOJIBAKE_RATE = 0.02

#: Date layouts tried when looking for more than one in a single column.
DATE_LAYOUTS = (
    ("%Y-%m-%d", "YYYY-MM-DD"),
    ("%d/%m/%Y", "DD/MM/YYYY"),
    ("%m/%d/%Y", "MM/DD/YYYY"),
    ("%d.%m.%Y", "DD.MM.YYYY"),
    ("%d-%m-%Y", "DD-MM-YYYY"),
    ("%Y/%m/%d", "YYYY/MM/DD"),
)

_WHITESPACE = re.compile(r"\s+")


# ------------------------------------------------------------------- the checks


def check_disguised_missing(
    frame: pd.DataFrame, profiles: list[ColumnProfile]
) -> list[Finding]:
    """Values that look like data but mean "missing".

    The defect this project exists for. `pandas` counts `"?"` as a present value,
    so completeness looks perfect and every later missing-data step is blind.
    """
    findings = []
    for column in frame.columns:
        series = frame[column]
        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            continue
        text = series.dropna().astype("string").str.strip()
        hits = text[text.str.lower().isin(DISGUISED_TOKENS)]
        if hits.empty:
            continue
        tokens = sorted(hits.unique())
        share = len(hits) / len(frame)
        findings.append(
            Finding(
                check="disguised_missing",
                severity="critical",
                topic="missing",
                columns=[str(column)],
                affected_rows=len(hits),
                affected_share=share,
                message=(
                    f"'{column}' holds {len(hits):,} disguised missing values "
                    f"({share:.2%}) stored as {tokens}. pandas counts these as "
                    "present, so completeness looks perfect and every later "
                    "missing-value step ignores them."
                ),
                evidence={"tokens": tokens, "count": len(hits)},
                suggestion=Suggestion(
                    action="replace_disguised_missing",
                    params={"columns": [str(column)], "tokens": tokens},
                    rationale=(
                        "Convert them to real nulls so the missingness becomes "
                        "visible and can be treated."
                    ),
                ),
            )
        )
    return findings


def check_numeric_sentinels(
    frame: pd.DataFrame, profiles: list[ColumnProfile]
) -> list[Finding]:
    """Numbers standing in for "missing" inside numeric columns.

    Only values that are the column's own minimum or maximum are considered, and
    only when they repeat: a sentinel sits outside the real range by design, and a
    single occurrence cannot be told apart from a genuine measurement.

    Whether such a value *is* missing depends on the column, and the check does not
    pretend otherwise. A negative sentinel in a column that is otherwise
    non-negative is almost certainly a stand-in for "no value", and a repair is
    proposed. A sentinel at the top of a non-negative column is more likely a cap
    -- `capital_gain` in the census file is recorded up to 99,999 and no further --
    which is censoring, not missingness, and calls for a human decision. Those are
    reported without a repair.
    """
    findings = []
    for profile in profiles:
        if profile.semantic_type != "numeric":
            continue
        values = pd.to_numeric(frame[profile.name], errors="coerce").dropna()
        if len(values) < SENTINEL_MIN_COUNT * 2:
            continue
        low, high = float(values.min()), float(values.max())
        others = values[(values != low) & (values != high)]
        for candidate in (low, high):
            if candidate not in NUMERIC_SENTINELS:
                continue
            count = int((values == candidate).sum())
            if count < SENTINEL_MIN_COUNT:
                continue
            # A negative stand-in inside an otherwise non-negative quantity is the
            # unambiguous case.
            implausible = candidate < 0 and not others.empty and float(others.min()) >= 0
            share = count / len(frame)
            findings.append(
                Finding(
                    check="numeric_sentinel",
                    severity="high" if implausible else "medium",
                    topic="missing",
                    columns=[profile.name],
                    affected_rows=count,
                    affected_share=share,
                    message=(
                        f"'{profile.name}' holds the value {candidate:g} "
                        f"{count:,} times ({share:.2%}), and it is the column's "
                        f"{'minimum' if candidate == low else 'maximum'}. "
                        + (
                            "Every other value is non-negative, so this is a "
                            "stand-in for \"no value\" rather than a measurement."
                            if implausible
                            else "This is a conventional stand-in for a missing or "
                            "capped value. Whether it means \"unknown\" or \"at "
                            "least this much\" cannot be decided from the data, so "
                            "no repair is proposed."
                        )
                    ),
                    evidence={
                        "value": candidate,
                        "count": count,
                        "is_minimum": candidate == low,
                        "other_values_non_negative": bool(
                            not others.empty and float(others.min()) >= 0
                        ),
                    },
                    suggestion=Suggestion(
                        action="replace_disguised_missing",
                        params={
                            "columns": [profile.name],
                            "numeric_values": [candidate],
                        },
                        rationale=(
                            "Convert to a real null so it is imputed rather than "
                            "averaged into the column."
                        ),
                    )
                    if implausible
                    else None,
                )
            )
    return findings


def check_missing_values(
    frame: pd.DataFrame, profiles: list[ColumnProfile]
) -> list[Finding]:
    """Genuine nulls, per column."""
    findings = []
    for profile in profiles:
        if profile.n_missing == 0 or profile.semantic_type == "empty":
            continue
        share = profile.n_missing / profile.n_rows
        severity = "high" if share > 0.5 else "medium" if share > 0.05 else "low"
        strategy = "median" if profile.semantic_type == "numeric" else "mode"
        findings.append(
            Finding(
                check="missing_values",
                severity=severity,
                topic="missing",
                columns=[profile.name],
                affected_rows=profile.n_missing,
                affected_share=share,
                message=(
                    f"'{profile.name}' is missing {profile.n_missing:,} values "
                    f"({share:.2%}). Dropping those rows discards everything else "
                    "they contain."
                ),
                evidence={"semantic_type": profile.semantic_type},
                suggestion=Suggestion(
                    action="impute",
                    params={"columns": [profile.name], "strategy": strategy},
                    rationale=(
                        f"Fill with the {strategy}, computed on the training half "
                        "only, and record which values were filled."
                    ),
                ),
            )
        )
    return findings


def check_uninformative_columns(
    frame: pd.DataFrame, profiles: list[ColumnProfile]
) -> list[Finding]:
    """Columns that cannot contribute anything: no values, or only one value.

    Reported together because they are the same defect -- no variation -- and take
    the same repair. Separated in the message, because "empty" and "always the same"
    mean different things to whoever produced the file.
    """
    empty = [p.name for p in profiles if p.semantic_type == "empty"]
    constant = [p.name for p in profiles if p.semantic_type == "constant"]
    if not empty and not constant:
        return []

    parts = []
    if empty:
        parts.append(f"{len(empty)} column(s) hold no usable value at all: {empty}")
    if constant:
        parts.append(
            f"{len(constant)} column(s) hold the same value in every row: {constant}"
        )
    return [
        Finding(
            check="uninformative_columns",
            # Empty is worse: a constant column is at least a fact about the data.
            severity="high" if empty else "medium",
            topic="structure",
            columns=empty + constant,
            message=(
                "; ".join(parts)
                + ". A column with no variation cannot help any model, and it "
                "distorts column counts and encoded width."
            ),
            evidence={"empty": empty, "constant": constant},
            suggestion=Suggestion(
                action="drop_columns",
                params={"columns": empty + constant},
                rationale="No variance means no information.",
            ),
        )
    ]


def check_identifier_columns(
    frame: pd.DataFrame, profiles: list[ColumnProfile]
) -> list[Finding]:
    """Columns that identify rows rather than describing them."""
    names = [p.name for p in profiles if p.semantic_type == "identifier"]
    if not names:
        return []
    return [
        Finding(
            check="identifier_columns",
            severity="medium",
            topic="structure",
            columns=names,
            message=(
                f"{len(names)} column(s) are effectively unique per row: {names}. "
                "These identify rows rather than describing them. Used as a "
                "feature, an identifier that happens to correlate with the target "
                "through collection order is a leakage route."
            ),
            suggestion=Suggestion(
                action="drop_columns",
                params={"columns": names},
                rationale="Keep them out of the feature matrix.",
            ),
        )
    ]


def check_exact_duplicates(
    frame: pd.DataFrame, profiles: list[ColumnProfile]
) -> list[Finding]:
    """Rows identical across every column."""
    mask = frame.duplicated(keep="first")
    count = int(mask.sum())
    if count == 0:
        return []
    share = count / len(frame)
    return [
        Finding(
            check="exact_duplicates",
            severity="medium",
            topic="duplicates",
            affected_rows=count,
            affected_share=share,
            message=(
                f"{count:,} rows ({share:.2%}) are exact copies of an earlier row. "
                "Left in place they inflate the apparent sample size, and copies "
                "landing on both sides of the train/test split contaminate the "
                "evaluation."
            ),
            evidence={"count": count},
            suggestion=Suggestion(
                action="drop_duplicate_rows",
                params={"normalise": False},
                rationale=(
                    "An exact match needs no threshold, so this is settled before "
                    "the split -- afterwards, copies straddling the boundary can no "
                    "longer be removed."
                ),
            ),
        )
    ]


def check_normalised_duplicates(
    frame: pd.DataFrame, profiles: list[ColumnProfile]
) -> list[Finding]:
    """Rows identical once case and whitespace are ignored.

    Reported separately from exact duplicates, and only for the *additional* rows
    it finds, so the two counts can be compared. The gap between them measures how
    much of the duplication is formatting rather than repetition.
    """
    text_columns = [
        c for c in frame.columns if not pd.api.types.is_numeric_dtype(frame[c])
    ]
    if not text_columns:
        return []
    normalised = frame.copy()
    for column in text_columns:
        normalised[column] = _normalise_text(normalised[column])
    total = int(normalised.duplicated(keep="first").sum())
    exact = int(frame.duplicated(keep="first").sum())
    extra = total - exact
    if extra <= 0:
        return []
    share = extra / len(frame)
    return [
        Finding(
            check="normalised_duplicates",
            severity="medium",
            topic="duplicates",
            columns=[str(c) for c in text_columns],
            affected_rows=extra,
            affected_share=share,
            message=(
                f"A further {extra:,} rows ({share:.2%}) become duplicates once "
                f"case and spacing are ignored ({total:,} in total against "
                f"{exact:,} exact). That difference is duplication hidden by "
                "formatting alone."
            ),
            evidence={"exact": exact, "normalised_total": total, "additional": extra},
            suggestion=Suggestion(
                action="normalise_categories",
                params={"columns": [str(c) for c in text_columns]},
                rationale=(
                    "Normalise the text first, then remove duplicates, so that "
                    "'Sofia' and 'sofia ' are recognised as the same value."
                ),
            ),
        )
    ]


def check_inconsistent_categories(
    frame: pd.DataFrame, profiles: list[ColumnProfile]
) -> list[Finding]:
    """One category spelled several ways in the same column."""
    findings = []
    for profile in profiles:
        if profile.semantic_type not in {"categorical", "boolean"}:
            continue
        values = frame[profile.name].dropna().astype("string")
        groups: dict[str, set[str]] = {}
        for value in values.unique():
            groups.setdefault(_normalise_scalar(value), set()).add(value)
        clashes = {k: sorted(v) for k, v in groups.items() if len(v) > 1}
        if not clashes:
            continue
        affected = int(values.isin([v for vs in clashes.values() for v in vs]).sum())
        examples = list(clashes.values())[:3]
        findings.append(
            Finding(
                check="inconsistent_categories",
                severity="medium",
                topic="structure",
                columns=[profile.name],
                affected_rows=affected,
                affected_share=affected / len(frame),
                message=(
                    f"'{profile.name}' spells {len(clashes)} categor"
                    f"{'y' if len(clashes) == 1 else 'ies'} more than one way, e.g. "
                    f"{examples}. Each spelling is counted as a separate category, "
                    "which splits the data and inflates the encoded width."
                ),
                evidence={"groups": clashes},
                suggestion=Suggestion(
                    action="normalise_categories",
                    params={"columns": [profile.name]},
                    rationale="Trim, collapse spacing and unify case.",
                ),
            )
        )
    return findings


def check_mixed_types(
    frame: pd.DataFrame, profiles: list[ColumnProfile]
) -> list[Finding]:
    """Columns holding both numbers and non-numeric text.

    Disguised blanks are excluded first: a numeric column with `"?"` in it is a
    missing-value problem, already reported, not a mixed-type problem.
    """
    findings = []
    for profile in profiles:
        if profile.semantic_type in {"numeric", "empty", "constant", "datetime"}:
            continue
        text = frame[profile.name].dropna().astype("string").str.strip()
        real = text[~text.str.lower().isin(DISGUISED_TOKENS)]
        if real.empty:
            continue
        numeric_share = float(pd.to_numeric(real, errors="coerce").notna().mean())
        if not 0.05 <= numeric_share <= 0.95:
            continue
        parsed = pd.to_numeric(real, errors="coerce")
        # Count the values that break the column, not the ones that work. Reporting
        # the numeric share under a "rows affected" heading read as though 84% of
        # the column were the problem.
        n_non_numeric = int(parsed.isna().sum())
        findings.append(
            Finding(
                check="mixed_types",
                severity="high",
                topic="structure",
                columns=[profile.name],
                affected_rows=n_non_numeric,
                affected_share=n_non_numeric / len(frame),
                message=(
                    f"'{profile.name}' mixes numbers and text: {numeric_share:.1%} "
                    "of its values parse as numbers and the rest do not. Neither a "
                    "numeric nor a categorical treatment is right until it is split "
                    "or corrected."
                ),
                evidence={
                    "numeric_share": round(numeric_share, 4),
                    "non_numeric_count": n_non_numeric,
                    "non_numeric_examples": sorted(real[parsed.isna()].unique())[:5],
                },
                suggestion=None,  # Needs a human decision; see the message.
            )
        )
    return findings


def check_outliers(frame: pd.DataFrame, profiles: list[ColumnProfile]) -> list[Finding]:
    """Extreme numeric values, by two rules that are reported side by side.

    The IQR fence is the familiar rule. The MAD-based modified z-score is more
    robust on skewed data, because the median absolute deviation is not itself
    pulled outward by the extremes it is meant to find.

    Where one value dominates the column, both rules are unreliable and the
    finding says so rather than proposing a repair.
    """
    findings = []
    for profile in profiles:
        if profile.semantic_type != "numeric":
            continue
        values = pd.to_numeric(frame[profile.name], errors="coerce").dropna()
        if len(values) < 20:
            continue

        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        low, high = q1 - IQR_MULTIPLIER * iqr, q3 + IQR_MULTIPLIER * iqr
        by_iqr = int(((values < low) | (values > high)).sum()) if iqr > 0 else 0

        by_mad, mad = 0, 0.0
        median = float(values.median())
        mad = float((values - median).abs().median())
        if mad > 0:
            modified_z = MAD_SCALE * (values - median) / mad
            by_mad = int((modified_z.abs() > MODIFIED_Z_THRESHOLD).sum())

        if by_iqr == 0 and by_mad == 0:
            continue

        dominant = float(values.value_counts(normalize=True).iloc[0])
        n_distinct = int(values.nunique())
        short_scale = n_distinct < OUTLIER_MIN_DISTINCT
        unreliable = (
            dominant >= DOMINANT_VALUE_SHARE or iqr == 0 or mad == 0 or short_scale
        )
        share = max(by_iqr, by_mad) / len(frame)

        if short_scale:
            message = (
                f"'{profile.name}': the IQR rule flags {by_iqr:,} values and the "
                f"MAD rule {by_mad:,}, but this column takes only {n_distinct} "
                "distinct values. On a short discrete scale the median absolute "
                f"deviation collapses (here {mad:g}), so what the rules flag is the "
                "ends of the scale rather than anomalies. Reported, not repaired."
            )
        elif unreliable:
            message = (
                f"'{profile.name}': the IQR rule flags {by_iqr:,} values and the "
                f"MAD rule {by_mad:,}, but one value covers {dominant:.0%} of this "
                "column, so the quartiles collapse onto it and neither rule can be "
                "trusted here. Reported as a measurement, not as an error count."
            )
        else:
            iqr_part = (
                f"{by_iqr:,} values fall outside [{low:,.2f}, {high:,.2f}] "
                f"(Q1/Q3 ± {IQR_MULTIPLIER}·IQR)"
                if by_iqr
                else f"nothing falls outside [{low:,.2f}, {high:,.2f}] "
                     f"(Q1/Q3 ± {IQR_MULTIPLIER}·IQR)"
            )
            mad_part = (
                f"{by_mad:,} exceed a modified z-score of {MODIFIED_Z_THRESHOLD}"
                if by_mad
                else f"none exceeds a modified z-score of {MODIFIED_Z_THRESHOLD}"
            )
            message = (
                f"'{profile.name}': {iqr_part}, and {mad_part}. The two rules "
                "disagree because the MAD rule is not pulled outward by the values "
                "it is looking for. Extreme values drag means and scalers toward "
                "themselves."
                if bool(by_iqr) != bool(by_mad)
                else f"'{profile.name}': {iqr_part}, and {mad_part}. Extreme values "
                     "drag means and scalers toward themselves."
            )

        findings.append(
            Finding(
                check="outliers",
                severity="info" if unreliable else "low",
                topic="anomalies",
                columns=[profile.name],
                affected_rows=max(by_iqr, by_mad),
                affected_share=share,
                message=message,
                evidence={
                    "by_iqr": by_iqr,
                    "by_modified_z": by_mad,
                    "iqr_bounds": [round(float(low), 4), round(float(high), 4)],
                    "median": round(median, 4),
                    "mad": round(mad, 4),
                    "dominant_value_share": round(dominant, 4),
                    "distinct": n_distinct,
                    "rule_unreliable": unreliable,
                },
                suggestion=None
                if unreliable
                else Suggestion(
                    action="cap_outliers",
                    params={
                        "columns": [profile.name],
                        # Match the rule that actually fired. Capping at the IQR
                        # fences when the IQR rule flagged nothing is a no-op.
                        "method": "iqr" if by_iqr else "mad",
                    },
                    rationale=(
                        "Cap at the "
                        + ("IQR fences" if by_iqr else "modified z-score bound")
                        + " rather than deleting rows: an extreme value is often "
                        "real, and deleting it discards everything else in the row."
                    ),
                ),
            )
        )
    return findings


def check_high_skew(frame: pd.DataFrame, profiles: list[ColumnProfile]) -> list[Finding]:
    """Strongly asymmetric numeric columns, which a log transform can straighten."""
    findings = []
    for profile in profiles:
        skew = profile.stats.get("skew")
        if profile.semantic_type != "numeric" or skew is None:
            continue
        if abs(skew) < SKEW_THRESHOLD:
            continue
        values = pd.to_numeric(frame[profile.name], errors="coerce").dropna()
        can_log = bool(len(values)) and float(values.min()) >= 0
        findings.append(
            Finding(
                check="high_skew",
                severity="info",
                topic="features",
                columns=[profile.name],
                affected_share=0.0,
                message=(
                    f"'{profile.name}' is strongly skewed ({skew:+.2f}). Linear "
                    "models and distance-based methods assume something closer to "
                    "symmetry."
                    + ("" if can_log else " It holds negative values, so a log "
                       "transform does not apply directly.")
                ),
                evidence={"skew": round(float(skew), 4), "log_applicable": can_log},
                suggestion=Suggestion(
                    action="log_transform",
                    params={"columns": [profile.name]},
                    rationale=(
                        "log(1+x) compresses the long tail while leaving the "
                        "ordering intact. Reported with the skew before and after."
                    ),
                )
                if can_log
                else None,
            )
        )
    return findings


def check_redundant_columns(
    frame: pd.DataFrame, profiles: list[ColumnProfile]
) -> list[Finding]:
    """Pairs of columns encoding exactly the same information.

    Tested as a one-to-one mapping in both directions: `education` and
    `education_num` in the census file are the same 16 levels, once as labels and
    once as codes. Correlation is not used -- it answers a different question and
    needs a threshold this project has no basis to choose.
    """
    candidates = [
        p.name
        for p in profiles
        if p.semantic_type in {"categorical", "numeric", "boolean"}
        and 1 < p.n_distinct <= 1000
    ]
    if len(candidates) > MAX_COLUMNS_FOR_PAIRS:
        candidates = candidates[:MAX_COLUMNS_FOR_PAIRS]

    findings = []
    seen: set[str] = set()
    for i, left in enumerate(candidates):
        if left in seen:
            continue
        for right in candidates[i + 1 :]:
            if right in seen:
                continue
            pair = frame[[left, right]].dropna()
            if pair.empty:
                continue
            n_left, n_right = pair[left].nunique(), pair[right].nunique()
            if n_left != n_right:
                continue
            if len(pair.drop_duplicates()) != n_left:
                continue
            seen.add(right)
            findings.append(
                Finding(
                    check="redundant_columns",
                    severity="medium",
                    topic="structure",
                    columns=[left, right],
                    message=(
                        f"'{left}' and '{right}' are a perfect one-to-one mapping "
                        f"over {n_left} values -- the same information encoded "
                        "twice. Keeping both doubles the work and can double a "
                        "column's weight in a model."
                    ),
                    evidence={"levels": int(n_left)},
                    suggestion=Suggestion(
                        action="drop_columns",
                        params={"columns": [right]},
                        rationale=f"Redundant given '{left}'.",
                    ),
                )
            )
    return findings


def check_numeric_in_text(
    frame: pd.DataFrame, profiles: list[ColumnProfile]
) -> list[Finding]:
    """Quantities written for humans and therefore stored as text.

    `"1 234,56"`, `"$100"`, `"12 lv"`, `"45%"`. Every one is a number, and none of
    them parses. The column stays text, so the system treats a quantity as a
    category: it fills gaps with the most common string instead of the median, and
    one-hot encodes it into as many columns as it has distinct values.

    This is the most damaging of the format defects and the most common, because a
    comma decimal separator is the norm across Europe including Bulgaria.
    """
    findings = []
    for profile in profiles:
        if profile.semantic_type not in {"categorical", "text", "identifier"}:
            continue
        text = frame[profile.name].dropna().astype("string").str.strip()
        real = text[~text.str.lower().isin(DISGUISED_TOKENS)]
        if real.empty:
            continue
        if pd.to_numeric(real, errors="coerce").notna().mean() >= 0.5:
            continue  # already handled by the type inference in stage 2

        cleaned, marks = _strip_number_marks(real)
        share = float(pd.to_numeric(cleaned, errors="coerce").notna().mean())
        if share < NUMERIC_IN_TEXT_THRESHOLD or not marks:
            continue
        findings.append(
            Finding(
                check="numeric_in_text",
                severity="high",
                topic="structure",
                columns=[profile.name],
                affected_rows=len(real),
                affected_share=len(real) / len(frame),
                message=(
                    f"'{profile.name}' looks like a number written for people, not "
                    f"for a parser: {share:.0%} of its values become numeric once "
                    f"{marks} are removed. Left as text it is treated as a category "
                    "- filled with the commonest string rather than the median, and "
                    f"encoded into {profile.n_distinct} columns."
                ),
                evidence={
                    "marks_found": marks,
                    "numeric_share_after_cleaning": round(share, 4),
                    "examples": sorted(real.unique())[:5],
                },
                suggestion=Suggestion(
                    action="parse_numeric",
                    params={"columns": [profile.name], "marks": marks},
                    rationale=(
                        "Strip the separators and symbols, then convert to a number, "
                        "so the column can be averaged and scaled."
                    ),
                ),
            )
        )
    return findings


def check_whitespace(frame: pd.DataFrame, profiles: list[ColumnProfile]) -> list[Finding]:
    """Values differing from their neighbours only by spaces.

    `"Sofia "` and `"Sofia"` are two categories to every encoder and two keys to
    every join, and the difference is invisible on screen.
    """
    findings = []
    for profile in profiles:
        if profile.semantic_type in {"numeric", "empty", "datetime", "boolean"}:
            continue
        text = frame[profile.name].dropna().astype("string")
        if text.empty:
            continue
        untrimmed = int((text != text.str.strip()).sum())
        doubled = int(text.str.contains(r"\s\s", regex=True, na=False).sum())
        affected = int(
            ((text != text.str.strip()) | text.str.contains(r"\s\s", regex=True, na=False)).sum()
        )
        if affected == 0:
            continue
        parts = []
        if untrimmed:
            parts.append(f"{untrimmed:,} have leading or trailing spaces")
        if doubled:
            parts.append(f"{doubled:,} contain a doubled internal space")
        findings.append(
            Finding(
                check="whitespace",
                severity="medium",
                topic="structure",
                columns=[profile.name],
                affected_rows=affected,
                affected_share=affected / len(frame),
                message=(
                    f"'{profile.name}': " + " and ".join(parts) + ". Spacing is "
                    "invisible on screen but not to a computer, so these count as "
                    "separate values from their trimmed twins."
                ),
                evidence={"untrimmed": untrimmed, "doubled_spaces": doubled},
                suggestion=Suggestion(
                    action="normalise_categories",
                    params={"columns": [profile.name]},
                    rationale="Trim the ends and collapse runs of spaces to one.",
                ),
            )
        )
    return findings


def check_short_values(
    frame: pd.DataFrame, profiles: list[ColumnProfile]
) -> list[Finding]:
    """One- or two-character values in a column whose values are normally long.

    A single letter where a name belongs is a placeholder someone typed to get past
    a required field. It is not a missing-value token, so nothing else catches it.
    """
    findings = []
    for profile in profiles:
        if profile.semantic_type not in {"categorical", "text", "identifier"}:
            continue
        text = frame[profile.name].dropna().astype("string").str.strip()
        real = text[~text.str.lower().isin(DISGUISED_TOKENS)]
        if len(real) < 10:
            continue
        lengths = real.str.len()
        typical = float(lengths.median())
        short = real[lengths <= SHORT_VALUE_MAX_LENGTH]
        if short.empty or typical < SHORT_VALUE_MAX_LENGTH * SHORT_VALUE_LENGTH_RATIO:
            continue
        findings.append(
            Finding(
                check="short_values",
                severity="medium",
                topic="missing",
                columns=[profile.name],
                affected_rows=len(short),
                affected_share=len(short) / len(frame),
                message=(
                    f"'{profile.name}' has {len(short):,} value(s) of "
                    f"{SHORT_VALUE_MAX_LENGTH} characters or fewer, in a column "
                    f"whose typical value is {typical:.0f} characters long: "
                    f"{sorted(short.unique())[:5]}. These are placeholders rather "
                    "than data, but they are not recognised missing-value tokens, so "
                    "nothing else reports them."
                ),
                evidence={
                    "median_length": typical,
                    "values": sorted(short.unique())[:10],
                },
                suggestion=None,  # Whether a stub means "missing" is a judgement.
            )
        )
    return findings


def check_encoding_damage(
    frame: pd.DataFrame, profiles: list[ColumnProfile]
) -> list[Finding]:
    """Mojibake - text encoded as UTF-8 and then read as something else.

    "София" read as cp1252 becomes "Ð¡Ð¾Ñ„Ð¸Ñ". The damage happens at ingestion and
    becomes permanent the moment the file is written back out, so it has to be caught
    on the way in rather than diagnosed later.
    """
    findings = []
    for profile in profiles:
        if profile.semantic_type in {"numeric", "empty", "datetime", "boolean"}:
            continue
        text = frame[profile.name].dropna().astype("string")
        if text.empty:
            continue
        total_chars = int(text.str.len().sum())
        if total_chars == 0:
            continue
        hits = sum(int(text.str.count(re.escape(m)).sum()) for m in MOJIBAKE_MARKERS)
        rate = hits / total_chars
        if rate < MOJIBAKE_RATE:
            continue
        affected = int(
            text.str.contains("|".join(re.escape(m) for m in MOJIBAKE_MARKERS),
                              regex=True, na=False).sum()
        )
        findings.append(
            Finding(
                check="encoding_damage",
                severity="high",
                topic="structure",
                columns=[profile.name],
                affected_rows=affected,
                affected_share=affected / len(frame),
                message=(
                    f"'{profile.name}' shows signs of encoding damage in "
                    f"{affected:,} value(s), e.g. {sorted(text.unique())[:2]}. This "
                    "pattern comes from text stored as UTF-8 and then read as cp1252 "
                    "or latin-1. It cannot be repaired from the loaded data - the "
                    "file must be re-read with the right encoding."
                ),
                evidence={
                    "marker_rate_per_char": round(rate, 5),
                    "examples": sorted(text.unique())[:3],
                },
                suggestion=None,  # Fixed by re-reading, not by transforming.
            )
        )
    return findings


def check_mixed_date_formats(
    frame: pd.DataFrame, profiles: list[ColumnProfile]
) -> list[Finding]:
    """More than one date layout in a single column.

    `01/02/2024` and `2024-03-15` in the same column means at least one of them will
    be misread, and `01/02` is ambiguous between January the second and the first of
    February. Parsing such a column silently produces plausible wrong dates, which is
    worse than failing.
    """
    findings = []
    for profile in profiles:
        if profile.semantic_type not in {"categorical", "text", "datetime", "identifier"}:
            continue
        text = frame[profile.name].dropna().astype("string").str.strip()
        real = text[~text.str.lower().isin(DISGUISED_TOKENS)]
        if len(real) < 4:
            continue
        matched: dict[str, int] = {}
        unmatched = set(real.unique())
        for fmt, label in DATE_LAYOUTS:
            parsed = pd.to_datetime(pd.Series(list(unmatched)), format=fmt, errors="coerce")
            ok = {v for v, good in zip(unmatched, parsed.notna(), strict=True) if good}
            if ok:
                matched[label] = len(ok)
                unmatched -= ok
        if len(matched) < 2:
            continue
        covered = sum(matched.values())
        if covered / real.nunique() < 0.8:
            continue
        findings.append(
            Finding(
                check="mixed_date_formats",
                severity="high",
                topic="structure",
                columns=[profile.name],
                affected_rows=len(real),
                affected_share=len(real) / len(frame),
                message=(
                    f"'{profile.name}' uses {len(matched)} different date layouts: "
                    f"{matched}. At least one will be misread, and a day/month "
                    "ambiguity produces a plausible wrong date rather than an error."
                ),
                evidence={"layouts": matched, "unrecognised": sorted(unmatched)[:5]},
                suggestion=None,  # Which layout is intended is not in the data.
            )
        )
    return findings


def check_control_characters(
    frame: pd.DataFrame, profiles: list[ColumnProfile]
) -> list[Finding]:
    """Non-printable characters inside values.

    Newlines and tabs pasted into a cell break CSV round-trips, shift columns and
    are invisible in a spreadsheet.
    """
    findings = []
    for profile in profiles:
        if profile.semantic_type in {"numeric", "empty", "datetime", "boolean"}:
            continue
        text = frame[profile.name].dropna().astype("string")
        if text.empty:
            continue
        mask = text.map(
            lambda v: any(
                unicodedata.category(ch) == "Cc" and ch not in "\t"
                for ch in str(v)
            )
            or "\n" in str(v)
            or "\r" in str(v)
            or "\t" in str(v)
        )
        affected = int(mask.sum())
        if affected == 0:
            continue
        findings.append(
            Finding(
                check="control_characters",
                severity="medium",
                topic="structure",
                columns=[profile.name],
                affected_rows=affected,
                affected_share=affected / len(frame),
                message=(
                    f"'{profile.name}' has {affected:,} value(s) containing a "
                    "newline, tab or other non-printable character. These are "
                    "invisible in a spreadsheet and break CSV round-trips by shifting "
                    "columns."
                ),
                evidence={"count": affected},
                suggestion=Suggestion(
                    action="normalise_categories",
                    params={"columns": [profile.name], "strip_control": True},
                    rationale="Replace them with a single space and trim.",
                ),
            )
        )
    return findings


def check_column_names(
    frame: pd.DataFrame, profiles: list[ColumnProfile]
) -> list[Finding]:
    """Header problems: duplicates, stray spacing, embedded newlines.

    Two columns both called `Amount` do not survive a CSV read -- pandas silently
    renames the second to `Amount.1` and nothing tells you. A header with a trailing
    space fails every lookup by its visible name.
    """
    names = [str(c) for c in frame.columns]
    duplicated = sorted({n.strip().casefold() for n in names
                         if names.count(n) > 1}) or []
    suffixed = [n for n in names if re.search(r"\.\d+$", n)]
    untrimmed = [n for n in names if n != n.strip()]
    newlines = [n for n in names if "\n" in n or "\r" in n]
    if not (duplicated or suffixed or untrimmed or newlines):
        return []

    parts = []
    if duplicated:
        parts.append(f"{len(duplicated)} name(s) appear more than once: {duplicated}")
    if suffixed:
        parts.append(
            f"{len(suffixed)} name(s) end in a numeric suffix ({suffixed[:3]}), which "
            "is what pandas adds when it finds a duplicate header"
        )
    if untrimmed:
        parts.append(f"{len(untrimmed)} name(s) have stray spacing: {untrimmed[:3]}")
    if newlines:
        parts.append(f"{len(newlines)} name(s) contain a line break: {newlines[:3]}")
    return [
        Finding(
            check="column_names",
            severity="medium",
            topic="structure",
            columns=sorted(set(suffixed + untrimmed + newlines)),
            message=(
                "; ".join(parts) + ". A header that does not match its visible name "
                "fails every lookup, and a duplicate header means one of the two "
                "columns is not the one you think it is."
            ),
            evidence={
                "duplicated": duplicated,
                "numeric_suffix": suffixed,
                "untrimmed": untrimmed,
                "line_breaks": newlines,
            },
            suggestion=Suggestion(
                action="clean_column_names",
                params={},
                rationale=(
                    "Trim, collapse spacing, remove line breaks, and make duplicates "
                    "distinct in a way that says which is which."
                ),
            ),
        )
    ]


def check_empty_rows(frame: pd.DataFrame, profiles: list[ColumnProfile]) -> list[Finding]:
    """Rows with no value in any column.

    Blank separator rows are ordinary in spreadsheet exports. They count toward every
    row total and dilute every share.
    """
    if frame.empty:
        return []
    blank = frame.isna().all(axis=1)
    count = int(blank.sum())
    if count == 0:
        return []
    return [
        Finding(
            check="empty_rows",
            severity="medium",
            topic="structure",
            affected_rows=count,
            affected_share=count / len(frame),
            message=(
                f"{count:,} row(s) are empty in every column. Blank separator rows "
                "are common in spreadsheet exports; they count toward the row total "
                "and dilute every percentage computed from it."
            ),
            evidence={"count": count},
            suggestion=Suggestion(
                action="drop_empty_rows",
                params={},
                rationale="They carry nothing.",
            ),
        )
    ]


def check_summary_rows(
    frame: pd.DataFrame, profiles: list[ColumnProfile]
) -> list[Finding]:
    """Totals rows at the foot of an export.

    A `TOTAL` row is not an observation. Left in place it becomes the maximum of
    every numeric column, so it sets the outlier bounds, shifts the mean and
    survives as a training row. It is usually recognisable because its label column
    says so while its other label columns are empty.
    """
    if len(frame) < 3:
        return []
    label_columns = [
        p.name for p in profiles
        if p.semantic_type in {"categorical", "text", "identifier"}
    ]
    if not label_columns:
        return []

    markers = ("total", "totals", "sum", "subtotal", "grand total",
               "общо", "всичко", "сума", "итого")
    hits: set[int] = set()
    for column in label_columns:
        text = frame[column].astype("string").str.strip().str.casefold()
        hits |= set(frame.index[text.isin(markers)])

    # A summary row also tends to leave most of its label columns blank.
    if len(label_columns) > 1:
        blank_labels = frame[label_columns].isna().sum(axis=1)
        mostly_blank = blank_labels >= len(label_columns) - 1
        numeric_columns = [p.name for p in profiles if p.semantic_type == "numeric"]
        if numeric_columns:
            has_numbers = frame[numeric_columns].notna().all(axis=1)
            hits |= set(frame.index[mostly_blank & has_numbers])

    hits &= set(frame.index[-max(3, len(frame) // 100):])  # only near the foot
    if not hits:
        return []
    return [
        Finding(
            check="summary_rows",
            severity="high",
            topic="structure",
            affected_rows=len(hits),
            affected_share=len(hits) / len(frame),
            message=(
                f"{len(hits)} row(s) near the end of the file look like totals rather "
                f"than observations (rows {sorted(hits)}). A totals row becomes the "
                "maximum of every numeric column, so it sets the outlier bounds, "
                "shifts the mean, and is trained on as though it were a record."
            ),
            evidence={"row_positions": sorted(hits)},
            suggestion=Suggestion(
                action="drop_rows",
                params={"positions": sorted(hits)},
                rationale="A total is a summary of the data, not part of it.",
            ),
        )
    ]


def _strip_number_marks(values: pd.Series) -> tuple[pd.Series, list[str]]:
    """Remove human number formatting, reporting which marks were present.

    Order matters. Thousands separators go first, then currency and units, then a
    comma decimal separator becomes a full stop -- doing that last avoids turning
    "1,234" into "1.234" when the comma was a thousands mark.
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
    # A single comma with two or three digits after it is a decimal separator.
    if text.str.match(r"^-?\d+,\d{1,3}$", na=False).any():
        found.append("comma decimal separator")
        text = text.str.replace(",", ".", regex=False)
    return text, sorted(set(found))



#: Every check, in the order their findings read best.
CHECKS = (
    check_column_names,
    check_disguised_missing,
    check_numeric_in_text,
    check_encoding_damage,
    check_mixed_date_formats,
    check_numeric_sentinels,
    check_missing_values,
    check_mixed_types,
    check_uninformative_columns,
    check_identifier_columns,
    check_redundant_columns,
    check_exact_duplicates,
    check_normalised_duplicates,
    check_inconsistent_categories,
    check_whitespace,
    check_control_characters,
    check_short_values,
    check_empty_rows,
    check_summary_rows,
    check_outliers,
    check_high_skew,
)


# ------------------------------------------------------------------- text helpers


def _normalise_scalar(value: Any) -> str:
    """Casefold, trim, collapse internal whitespace. Used to group spellings."""
    return _WHITESPACE.sub(" ", str(value).strip()).casefold()


def _normalise_text(series: pd.Series) -> pd.Series:
    """Vectorised form of :func:`_normalise_scalar`, preserving nulls."""
    text = series.astype("string").str.strip().str.replace(
        _WHITESPACE, " ", regex=True
    ).str.casefold()
    return text.where(series.notna(), np.nan)
