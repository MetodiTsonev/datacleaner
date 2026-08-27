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

from src.detect import as_table as findings_table
from src.detect import detect, summarise
from src.loader import probe_header, read_table
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


try:
    load = read_table(
        handle,
        has_header=has_header,
        encoding=None if forced_encoding == "detect" else forced_encoding,
        delimiter=forced_delimiter or None,
        decimal="," if decimal_mark.startswith(",") else ".",
        skip_rows=None if skip_rows < 0 else int(skip_rows),
    )
except Exception as exc:  # noqa: BLE001 - shown to the user rather than swallowed
    st.error(f"Could not read this file. {type(exc).__name__}: {exc}")
    st.stop()

frame = load.frame

st.sidebar.subheader("Target")
target = st.sidebar.selectbox(
    "Column to predict (optional)",
    ["— none —", *[str(c) for c in frame.columns]],
    key="target_column",
    help=(
        "Naming it protects it: the target is never filled in, transformed or "
        "dropped, and rows with no label are removed rather than guessed."
    ),
)
target = None if target == "— none —" else target

try:
    profiles = profile_frame(frame)
except ValueError as exc:
    st.error(str(exc))
    st.stop()
findings = detect(frame, profiles, target=target)
findings_summary = summarise(findings)

tab_data, tab_profile, tab_findings, tab_validate = st.tabs(
    ["Data", "Profile", "Findings", "Validate"]
)


with tab_data:
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
    st.subheader("What is in each column")
    st.caption(
        "The **type** column is a semantic type inferred from the values, not the "
        "storage dtype. It is what later stages branch on: a quantity gets a "
        "median and a scale, a category gets a mode and an encoding."
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
    st.subheader(f"{findings_summary['total']} problem(s) found")
    st.caption(
        "Each finding carries a suggested repair where the system knows one. A "
        "finding with no repair is shown as such rather than hidden — some defects "
        "need a human decision, and some rules cannot judge the column they ran on."
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
    st.subheader("Validation against declared rules")
    st.caption(
        "The **Findings** tab is discovery — the system deciding for itself what "
        "looks wrong, from rules that apply to any table. This tab is validation: "
        "checking the data against rules *you* declare. A rule carries information "
        "the data does not — that an age cannot be negative, or that a city must be "
        "one of five — so this is where your knowledge of the domain enters."
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


st.divider()
st.caption(
    "Stages 1–4 of 9 (loading, profiling, detection, validation). The repair plan, "
    "cleaning, anomalies, features and the before/after check follow — see PLAN.md."
)
