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
    )
except Exception as exc:  # noqa: BLE001 - shown to the user rather than swallowed
    st.error(f"Could not read this file. {type(exc).__name__}: {exc}")
    st.stop()

frame = load.frame
profiles = profile_frame(frame)

tab_data, tab_profile = st.tabs(["Data", "Profile"])


with tab_data:
    rows, cols = frame.shape
    a, b, c = st.columns(3)
    a.metric("Rows", f"{rows:,}")
    b.metric("Columns", cols)
    c.metric("Missing cells pandas reports", f"{int(frame.isna().sum().sum()):,}")

    summary = load.summary()
    st.caption(
        f"`{summary['file']}` · encoding **{summary['encoding']}** · delimiter "
        f"**{summary['delimiter']}** · header row **{summary['header_row']}**"
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


st.divider()
st.caption(
    "Stages 1–2 of 8 (loading, profiling). Problem detection, the repair plan, "
    "cleaning, anomalies, features and the before/after check follow — see PLAN.md."
)
