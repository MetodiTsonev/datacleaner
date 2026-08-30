"""DataCleaner - automated data preparation and processing.

Streamlit entry point. Master's thesis, UNWE (Изкуственият интелект в икономиката).

The interface is a thin client: it collects a file and some choices, calls into
`src/`, and displays what comes back. No data-processing logic lives here -- that
keeps every stage testable without a browser, and it means the pipeline can be
explained independently of the interface.

Run:  streamlit run app.py
"""

from __future__ import annotations

import re
from pathlib import Path

import streamlit as st

from src.clean import DEFAULT_TEST_SIZE
from src.clean import as_table as clean_table
from src.clean import run as run_plan
from src.detect import as_table as findings_table
from src.detect import detect, summarise
from src.loader import probe_header, read_table
from src.plan import PRE_SPLIT
from src.plan import as_table as plan_table
from src.plan import build as build_plan
from src.profile import as_table, profile_frame, type_counts
from src.validate import (
    Rule,
    infer_rules,
    is_applicable,
    rules_table,
    validate,
    violations_table,
)

INPUT_DIR = Path(__file__).parent / "data" / "input"

#: Offered in the sample picker. `.txt` is deliberately absent: `data/input/` holds
#: input data, and a `.names.txt` describing a dataset was being offered as one --
#: selecting it produced a 106x2 table whose first column name was a sentence from a
#: licence notice. Upload still accepts `.txt`, where the user knows what they have.
READABLE = {".csv", ".tsv", ".xlsx", ".xls"}

st.set_page_config(page_title="DataCleaner", layout="wide")


# --------------------------------------------------------------------- sidebar

st.sidebar.title("DataCleaner")
st.sidebar.caption("Automated data preparation for AI systems")

st.sidebar.subheader("Data")

source = st.sidebar.radio(
    "Source",
    ["Sample file", "Upload"],
    help="Samples live in data/input/. Upload accepts any CSV or Excel file.",
)

handle = None
if source == "Sample file":
    samples = sorted(p.name for p in INPUT_DIR.glob("*") if p.suffix.lower() in READABLE)
    if samples:
        handle = INPUT_DIR / st.sidebar.selectbox("File", samples, key="sample_file")
    else:
        st.sidebar.warning(f"No readable files in {INPUT_DIR}")
else:
    handle = st.sidebar.file_uploader("CSV or Excel", type=["csv", "tsv", "txt", "xlsx", "xls"])

if handle is None:
    st.title("DataCleaner")
    st.info("Choose a sample file or upload one to begin.", icon="⬅")
    st.stop()

# Detected per file, but left as a checkbox the user can override. Getting it wrong
# is silent: a headerless file read as though it had a header turns its own first row
# into column names, so every later stage describes a table one row short and the
# column list is full of values. Detection sets the default; the user decides.
detected_header = probe_header(handle)
# A stable widget key, reset when the file changes. A key that embeds the filename
# would give each file its own widget, which is invisible to anything addressing the
# checkbox by name -- including the tests.
file_id = str(getattr(handle, "name", handle))
if st.session_state.get("_header_detected_for") != file_id:
    st.session_state["has_header"] = detected_header
    st.session_state["_header_detected_for"] = file_id

has_header = st.sidebar.checkbox(
    "First row contains column names",
    key="has_header",
    help=(
        "Detected automatically from whether the first row's values look like data. "
        "Override if the guess is wrong."
    ),
)
if not detected_header:
    st.sidebar.caption(
        "Detected **no header row** — the first row's values look like data."
    )

with st.sidebar.expander("Advanced"):
    forced_delimiter = st.text_input(
        "Force delimiter", value="", max_chars=1,
        help="Leave empty to detect it. Set this if the table came out as one column.",
    )
    forced_encoding = st.selectbox(
        "Force encoding", ["detect", "utf-8", "cp1252", "latin-1"],
        help="Leave on detect unless characters look damaged.",
    )
    decimal_mark = st.selectbox(
        "Decimal separator", [". (point)", ", (comma)"],
        help="Comma is the convention across most of Europe: 1,5 means one and a half.",
    )
    skip_rows = st.number_input(
        "Rows to skip above the header", min_value=-1, value=-1, step=1,
        help=(
            "-1 detects junk rows automatically. Spreadsheet exports often begin "
            "with a title row and a blank row. Set 0 to read them as data."
        ),
    )


# ------------------------------------------------------------------- main pane

st.title("DataCleaner")


@st.cache_data(show_spinner="Reading and checking the file...")
def _prepare(source: bytes | str, name: str, **options):
    """load -> profile -> detect, cached on the file and the options.

    Keyed on the file's bytes for an upload and its path for a sample, so switching
    files or changing an option recomputes and nothing else does.
    """
    from io import BytesIO

    handle = BytesIO(source) if isinstance(source, bytes) else source
    if isinstance(handle, BytesIO):
        handle.name = name
    target = options.pop("target", None)
    loaded = read_table(handle, **options)
    column_profiles = profile_frame(loaded.frame)
    return loaded, column_profiles, detect(loaded.frame, column_profiles, target=target)


read_options = {
    "has_header": has_header,
    "encoding": None if forced_encoding == "detect" else forced_encoding,
    "delimiter": forced_delimiter or None,
    "decimal": "," if decimal_mark.startswith(",") else ".",
    "skip_rows": None if skip_rows < 0 else int(skip_rows),
}

st.sidebar.subheader("Target")

# Read once cheaply just to list the columns for the target selector.
try:
    _peek, _, _ = _prepare(
        handle.getvalue() if hasattr(handle, "getvalue") else str(handle),
        getattr(handle, "name", str(handle)),
        **read_options,
        target=None,
    )
except Exception as exc:  # noqa: BLE001 - shown, not swallowed
    st.title("DataCleaner")
    st.error(f"Could not read this file. {type(exc).__name__}: {exc}")
    st.stop()

target = st.sidebar.selectbox(
    "Column to predict (optional)",
    ["— none —", *[str(c) for c in _peek.frame.columns]],
    key="target_column",
    help=(
        "Naming it protects it: the target is never filled in, transformed or "
        "dropped, and rows with no label are removed rather than guessed."
    ),
)
target = None if target == "— none —" else target

load, profiles, findings = _prepare(
    handle.getvalue() if hasattr(handle, "getvalue") else str(handle),
    getattr(handle, "name", str(handle)),
    **read_options,
    target=target,
)
frame = load.frame
findings_summary = summarise(findings)

(
    tab_data, tab_profile, tab_findings, tab_validate, tab_plan, tab_run
) = st.tabs(["Data", "Profile", "Findings", "Validate", "Plan", "Run"])


def stage_header(number: int, title: str, plain: str, detail: str = "") -> None:
    """One line saying what this tab does, before any of the detail.

    The tabs are stages of one pipeline, and reading them out of order makes no
    sense, so each says where it sits and what it is for in ordinary words.
    """
    st.subheader(f"Stage {number} of 6 — {title}")
    st.markdown(f"**{plain}**")
    if detail:
        st.caption(detail)
repair_plan = build_plan(findings, target=target)


with tab_data:
    stage_header(
        1, "Reading the file",
        "Before anything can be judged, the file has to be read correctly.",
        "A wrong delimiter, a missed header row or the wrong decimal mark turns good "
        "data into nonsense without raising a single error. What was detected is "
        "shown below so you can correct it.",
    )
    rows, cols = frame.shape
    a, b, c = st.columns(3)
    a.metric("Rows", f"{rows:,}")
    b.metric("Columns", cols)
    c.metric("Missing cells pandas reports", f"{int(frame.isna().sum().sum()):,}")

    load_summary = load.summary()
    st.caption(
        f"`{load_summary['file']}` · encoding **{load_summary['encoding']}** · "
        f"delimiter **{load_summary['delimiter']}** · decimal "
        f"**{load_summary['decimal']}** · header row "
        f"**{load_summary['header_row']}** · rows skipped "
        f"**{load_summary['skipped_rows']}**"
    )
    for note in load.notes:
        st.warning(note, icon="⚠")

    st.subheader("First rows")
    st.dataframe(frame.head(20), width="stretch")


with tab_profile:
    stage_header(
        2, "Working out what each column is",
        "The system decides what kind of thing every column holds.",
        "Not the storage format — the *meaning*. A quantity, a label, a date, an "
        "identifier. This decides how the column gets repaired later: a quantity is "
        "filled with a median and scaled, a label is filled with its commonest value "
        "and encoded. Get this wrong and everything after it is wrong.",
    )

    counts = type_counts(profiles)
    if counts:
        cs = st.columns(len(counts))
        for col, (name, n) in zip(cs, counts.items(), strict=True):
            col.metric(name, n)

    if target:
        st.caption(f"Target column: **{target}** — protected from every repair.")
    st.dataframe(as_table(profiles), width="stretch", hide_index=True)

    flagged = [p for p in profiles if p.note]
    if flagged:
        st.subheader("Columns worth a second look")
        for p in flagged:
            examples = ", ".join(f"`{v}`" for v in p.examples[:4] if v is not None)
            st.markdown(
                f"**{p.name}** ({p.semantic_type}) — {p.note}"
                + (f"  \n<small>examples: {examples}</small>" if examples else ""),
                unsafe_allow_html=True,
            )

    total_missing = sum(p.n_missing for p in profiles)
    if total_missing == 0:
        st.info(
            "No missing values found — but that is what pandas can see, not what "
            "is true. Values such as `?`, `N/A` or `-999` are stored as ordinary "
            "data and count as present. Finding those is the next stage.",
            icon="🔎",
        )


with tab_findings:
    stage_header(
        3, "Finding what looks wrong",
        f"The system checked this file against 22 common problems and found "
        f"{findings_summary['total']}.",
        "This is the system's own opinion, from rules that apply to any table. Each "
        "finding comes with a suggested repair where one is known. Where it is not, "
        "the finding says so rather than hiding — some defects need a person to "
        "decide, and some rules cannot judge the column they ran on.",
    )

    if not findings:
        st.success("No problems detected in this file.", icon="✅")
    else:
        cols = st.columns(max(len(findings_summary["by_severity"]), 1))
        for col, (sev, n) in zip(cols, findings_summary["by_severity"].items(), strict=False):
            col.metric(sev, n)

        st.dataframe(findings_table(findings), width="stretch", hide_index=True)

        if findings_summary["blocking"]:
            st.error(
                f"{len(findings_summary['blocking'])} critical finding(s). These are not a "
                "matter of degree: until they are fixed, every completeness figure "
                "for this file is wrong.",
                icon="🚫",
            )

        st.subheader("In detail")
        icon = {"critical": "🚫", "high": "🔴", "medium": "🟠", "low": "🟡", "info": "ℹ️"}
        for f in findings:
            with st.expander(
                f"{icon.get(f.severity, '·')} {f.severity} · {f.check} · "
                f"{', '.join(f.columns) or 'whole table'}"
            ):
                st.write(f.message)
                if f.suggestion:
                    st.markdown(
                        f"**Proposed repair:** `{f.suggestion.action}` — "
                        f"{f.suggestion.rationale}"
                    )
                else:
                    st.markdown(
                        "**No repair proposed.** See the message above for why."
                    )
                if f.evidence:
                    st.json(f.evidence, expanded=False)

        st.subheader("Coverage of the four mandated topics")
        st.caption(
            "The задание names four theoretical topics. Every finding is tagged "
            "with the one it belongs to, so coverage can be reported rather than "
            "asserted. `structure` covers defects belonging to none of them."
        )
        st.dataframe(
            {
                "topic": list(findings_summary["by_topic"]),
                "findings": list(findings_summary["by_topic"].values()),
            },
            width="stretch",
            hide_index=True,
        )


with tab_validate:
    stage_header(
        4, "Checking it against your rules",
        "The previous tab is the system's opinion. This one is yours.",
        "Here you declare what *must* be true — an age cannot be negative, a city has "
        "to be one of five, an order number cannot repeat. The data cannot tell you "
        "any of that; you know it. Rows that break a rule are set aside with the "
        "reason, not deleted.",
    )

    # Draft rules are cached per file so editing them survives a rerun. Streamlit
    # re-executes this script on every interaction, so anything the user changes has
    # to live in session_state or it is lost the moment they touch a widget.
    rules_key = f"rules::{load.name}::{target}"
    if rules_key not in st.session_state:
        st.session_state[rules_key] = infer_rules(frame, profiles, target=target)
    rules: list[Rule] = st.session_state[rules_key]

    left, right = st.columns([3, 2])

    with left:
        st.markdown("**Draft contract**")
        st.caption(
            "Inferred from this one file, so every rule is a guess about intent. "
            "Inference cannot tell an intentional constraint from an accident of "
            "this batch: if every city here is Sofia, *city must be Sofia* is a true "
            "description and a useless rule. Review before trusting."
        )
        if rules:
            st.dataframe(rules_table(rules), width="stretch", hide_index=True)
        else:
            st.info("Nothing could be inferred from this file.")
        if st.button("Reset to inferred", width="stretch"):
            st.session_state[rules_key] = infer_rules(frame, profiles, target=target)
            st.rerun()

    with right:
        st.markdown("**Add a rule**")
        st.caption(
            "This is the answer to *could you add a constraint?* — the rule applies "
            "immediately and the rows that fail it appear below."
        )

        # Deliberately NOT inside `st.form`. A form batches its inputs and does not
        # rerun when one of them changes, so the value fields below -- which depend on
        # the chosen constraint -- were never created. Choosing "range" showed no
        # minimum or maximum box, and submitting then produced a rule with no bounds.
        # Plain widgets rerun on every change, so the right fields always appear.
        column = st.selectbox(
            "Column", [str(c) for c in frame.columns], key="new_rule_column"
        )
        kind = st.selectbox(
            "Constraint",
            ["not_null", "unique", "range", "allowed_values", "pattern", "type",
             "compare"],
            key="new_rule_kind",
        )

        params: dict = {}
        ready = True
        if kind == "range":
            lo = st.text_input("Minimum (blank for none)", key="new_rule_min")
            hi = st.text_input("Maximum (blank for none)", key="new_rule_max")
            for label, raw in (("min", lo), ("max", hi)):
                if raw.strip():
                    try:
                        params[label] = float(raw.replace(",", "."))
                    except ValueError:
                        st.error(f"{raw!r} is not a number.")
                        ready = False
            if ready and not params:
                st.warning("Give at least one bound, or the rule does nothing.")
                ready = False
        elif kind == "allowed_values":
            raw = st.text_area("Allowed values, one per line", key="new_rule_values")
            params["values"] = [v.strip() for v in raw.splitlines() if v.strip()]
            if not params["values"]:
                st.warning(
                    "List at least one value. An empty list would mean *no value is "
                    "allowed*, which rejects every row."
                )
                ready = False
        elif kind == "pattern":
            params["regex"] = st.text_input(
                "Regular expression", value=r"^.+$", key="new_rule_regex"
            )
            try:
                re.compile(params["regex"])
            except re.error as exc:
                st.error(f"Not a valid regular expression: {exc}")
                ready = False
        elif kind == "type":
            params["expected"] = st.selectbox(
                "Expected", ["numeric", "datetime"], key="new_rule_expected"
            )
        elif kind == "compare":
            params["op"] = st.selectbox(
                "Operator", ["<=", "<", ">=", ">", "=="], key="new_rule_op"
            )
            others = [str(c) for c in frame.columns if str(c) != column]
            if not others:
                st.warning("Needs a second column to compare against.")
                ready = False
            else:
                params["other"] = st.selectbox(
                    "Compared with", others, key="new_rule_other"
                )

        candidate = Rule(column, kind, params, inferred=False, note="declared by you")
        if ready:
            st.caption(f"Will add: **{candidate.label}**")
            if not is_applicable(frame, candidate):
                st.warning(
                    "This rule cannot be evaluated against this data — it would be "
                    "added but would do nothing."
                )

        if st.button("Add rule", width="stretch", disabled=not ready):
            st.session_state[rules_key] = [*rules, candidate]
            st.rerun()

    st.divider()

    result = validate(frame, rules)
    vsummary = result.summary()

    a, b, c, d = st.columns(4)
    a.metric("Rules", vsummary["rules_checked"])
    b.metric("Broken", vsummary["rules_broken"])
    c.metric("Rows valid", f"{vsummary['rows_valid']:,}")
    d.metric("Rows rejected", f"{vsummary['rows_rejected']:,}")

    if vsummary["caution"]:
        st.warning(vsummary["caution"], icon="⚠")

    if result.inapplicable:
        st.info(
            f"{len(result.inapplicable)} rule(s) could not be evaluated and did "
            "nothing: "
            + "; ".join(f"*{r.label}*" for r in result.inapplicable)
            + ". A rule that never runs looks exactly like a rule that passed, so it "
            "is reported rather than counted as a success.",
            icon="🔇",
        )

    if result.passed:
        st.success("Every rule holds for every row.", icon="✅")
    else:
        st.markdown("**Rules that failed**")
        st.dataframe(
            violations_table(result.violations), width="stretch", hide_index=True
        )

        st.markdown("**Quarantine** — rejected rows, each with its reason")
        st.caption(
            "Rejected rows are kept and shown, not dropped. Discarding data because "
            "it failed a rule written a minute ago is how datasets get quietly "
            "destroyed."
        )
        st.dataframe(result.rejected.head(50), width="stretch", hide_index=True)
        st.download_button(
            "Download quarantine as CSV",
            result.rejected.to_csv(index=False).encode("utf-8"),
            file_name=f"rejected-{load.name}",
            mime="text/csv",
        )


with tab_plan:
    psummary = repair_plan.summary()
    stage_header(
        5, "Deciding what to do, and in what order",
        f"The findings become {psummary['steps']} repair(s), to run in this order.",
        "Not a list of suggestions — a sequence. Order matters: tidy the column names "
        "before anything refers to them, convert '?' to a real blank before trying to "
        "fill blanks. Steps marked *before split* cannot learn anything from the data; "
        "steps marked *after split* are calculated from the training half only, so the "
        "held-back half never influences them.",
    )

    a, b, c, d = st.columns(4)
    a.metric("Steps", psummary["steps"])
    b.metric("Before split", psummary["pre_split"])
    c.metric("After split", psummary["post_split"])
    d.metric("Derived", psummary["derived"])

    if not repair_plan.steps:
        st.success("Nothing to repair in this file.", icon="✅")
    else:
        st.dataframe(plan_table(repair_plan), width="stretch", hide_index=True)

        st.subheader("Why each step, in order")
        for i, step in enumerate(repair_plan.steps, start=1):
            where = "before split" if step.stage == PRE_SPLIT else "after split"
            with st.expander(f"{i}. {step.action} — {where}"):
                st.write(step.why)
                st.markdown(
                    "**Columns:** "
                    + (", ".join(f"`{c}`" for c in step.columns) or "the whole table")
                )
                if step.from_checks:
                    st.markdown(
                        "**Asked for by:** "
                        + ", ".join(f"`{c}`" for c in step.from_checks)
                    )
                if step.derived_from:
                    st.info(
                        f"Nothing reported this. It follows from "
                        f"**{step.derived_from}**: converting a token to a null "
                        "creates missing values that did not exist when the checks "
                        "ran, and nothing downstream handles a null.",
                        icon="🧩",
                    )
                if step.params.get("strategy"):
                    st.caption(f"strategy: `{step.params['strategy']}`")

    if repair_plan.unaddressed:
        st.subheader("Reported but not repaired")
        st.caption(
            "Kept in view rather than dropped — a plan that quietly omits them looks "
            "complete when it is not."
        )
        for f in repair_plan.unaddressed:
            st.markdown(
                f"**{f.check}** ({', '.join(f.columns) or 'whole table'}) — {f.message}"
            )


with tab_run:
    stage_header(
        6, "Doing it, and showing the receipts",
        "The plan runs, and every step says exactly what it changed.",
        "'Applied 6 steps' is a claim. 'Nulled 6,465 values in 3 columns, removed 52 "
        "duplicate rows, filled 6,456' is an account you can check. The cleaned data "
        "and the held-back test half can be downloaded at the bottom.",
    )

    left, right = st.columns([2, 1])
    with right:
        test_size = st.slider(
            "Test share", 0.1, 0.5, DEFAULT_TEST_SIZE, 0.05,
            help="Held back and never used to compute a fill value.",
        )
        add_indicator = st.checkbox(
            "Add was-missing columns",
            help=(
                "A 0/1 column recording which values were filled. Worth having when "
                "the fact that something was missing may itself carry information."
            ),
        )
        group_options = ["— none —"] + [
            p.name for p in profiles if p.semantic_type in {"categorical", "boolean"}
        ]
        group_by = st.selectbox(
            "Fill numbers by group", group_options,
            help=(
                "A median per group is closer to the truth than one global median, "
                "when the grouping column actually predicts the value."
            ),
        )
        group_by = None if group_by == "— none —" else group_by

    with left:
        if not repair_plan.steps:
            st.info("Nothing to run — the plan is empty for this file.")
        else:
            st.caption(
                f"Running {len(repair_plan.steps)} step(s). Changing an option above "
                "re-runs them."
            )

    if repair_plan.steps:
        result = run_plan(
            frame, repair_plan, target=target,
            test_size=test_size, add_indicator=add_indicator, group_by=group_by,
        )
        rsummary = result.summary()

        a, b, c, d = st.columns(4)
        a.metric("Rows out", f"{rsummary['rows_out']:,}",
                 delta=f"-{rsummary['rows_removed']:,}" if rsummary["rows_removed"] else None)
        b.metric("Columns out", rsummary["columns_out"],
                 delta=frame.shape[1] and rsummary["columns_out"] - frame.shape[1] or None)
        c.metric("Cells changed", f"{rsummary['cells_changed']:,}")
        d.metric(
            "Nulls remaining", rsummary["nulls_remaining"],
            help=(
                f"{rsummary['values_capped']:,} extreme value(s) were capped rather "
                "than deleted."
            ),
        )

        st.markdown("**What each step did**")
        st.dataframe(clean_table(result), width="stretch", hide_index=True)

        if result.skipped:
            st.info(
                "Not run yet: "
                + ", ".join(f"`{s.action}`" for s in result.skipped)
                + ". These arrive in the anomaly and feature stages.",
                icon="🚧",
            )

        st.markdown("**What was learned, and from which half**")
        st.caption(
            "Every value below was computed from the training rows only. If it had "
            "come from the whole file, the test half would have influenced the "
            "training half's cleaning, and the evaluation would be optimistic in a "
            "way that never shows up on new data."
        )
        learned_any = False
        for a_ in result.applied:
            limits = a_.fitted.get("bounds") if a_.fitted else None
            if limits:
                learned_any = True
                st.dataframe(
                    {
                        "column": list(limits),
                        "kept between": [
                            f"{low:g} and {high:g}" for low, high in limits.values()
                        ],
                    },
                    width="stretch", hide_index=True,
                )
            refused = (a_.fitted or {}).get("refused") or {}
            for column, why in refused.items():
                st.info(
                    f"**{column}** was not capped — {why}.",
                    icon="🤚",
                )

            fills = a_.fitted.get("fill_values") if a_.fitted else None
            if not fills:
                continue
            learned_any = True
            st.dataframe(
                {
                    "column": list(fills),
                    "fill value": [
                        # Not str(): a median of 463.65999999999997 is float noise,
                        # not a value anyone chose.
                        f"{v:g}" if isinstance(v, float) else str(v)
                        for v in fills.values()
                    ],
                },
                width="stretch", hide_index=True,
            )
        if not learned_any:
            st.caption("No step in this plan learns anything from the data.")

        st.divider()
        st.markdown("**Before and after**")
        cols = st.columns(2)
        cols[0].caption(f"Before — {frame.shape[0]:,} x {frame.shape[1]}")
        cols[0].dataframe(frame.head(12), width="stretch")
        cols[1].caption(
            f"After — train {len(result.train):,} x {result.train.shape[1]}"
            + (f", test {len(result.test):,}" if result.test is not None else "")
        )
        cols[1].dataframe(result.train.head(12), width="stretch")

        st.divider()
        st.markdown("**Export**")
        e1, e2 = st.columns(2)
        e1.download_button(
            "Cleaned training data (CSV)",
            result.train.to_csv(index=False).encode("utf-8"),
            file_name=f"clean-train-{load.name}", mime="text/csv", width="stretch",
        )
        if result.test is not None and len(result.test):
            e2.download_button(
                "Held-out test data (CSV)",
                result.test.to_csv(index=False).encode("utf-8"),
                file_name=f"clean-test-{load.name}", mime="text/csv", width="stretch",
            )


st.divider()
st.caption(
    "Stages 1–6 of 9 (loading, profiling, detection, validation, planning, "
    "cleaning). Anomaly capping, feature engineering and the before/after model "
    "check follow — see PLAN.md."
)
