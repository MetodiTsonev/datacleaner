"""DataCleaner - automated data preparation and processing.

Streamlit entry point. Master's thesis, UNWE (Изкуственият интелект в икономиката).

The interface is a thin client: it collects a file and some choices, calls into
`src/`, and displays what comes back. No data-processing logic lives here -- that
keeps every stage testable without a browser, and it means the pipeline can be
explained independently of the interface.

Run:  streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.detect import as_table as findings_table
from src.detect import detect, summarise
from src.loader import read_table
from src.profile import as_table, profile_frame, type_counts

INPUT_DIR = Path(__file__).parent / "data" / "input"
READABLE = {".csv", ".tsv", ".txt", ".xlsx", ".xls"}

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
        handle = INPUT_DIR / st.sidebar.selectbox("File", samples)
    else:
        st.sidebar.warning(f"No readable files in {INPUT_DIR}")
else:
    handle = st.sidebar.file_uploader("CSV or Excel", type=["csv", "tsv", "txt", "xlsx", "xls"])

# Header detection is a user choice, not a guess. Getting it wrong is silent: a
# headerless file read with the default turns its own first data row into column
# names, and every later stage then describes a table one row short.
has_header = st.sidebar.checkbox(
    "First row contains column names",
    value=True,
    help="Uncheck for files that start straight into data.",
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

if handle is None:
    st.info("Choose a sample file or upload one to begin.", icon="⬅")
    st.stop()

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

tab_data, tab_profile, tab_findings = st.tabs(["Data", "Profile", "Findings"])


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


st.divider()
st.caption(
    "Stages 1–3 of 9 (loading, profiling, detection). Validation against declared "
    "constraints, the repair plan, cleaning, anomalies, features and the "
    "before/after check follow — see PLAN.md."
)
