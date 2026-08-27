"""DataCleaner - automated data preparation and processing.

Streamlit entry point. Master's thesis, UNWE (Изкуственият интелект в икономиката).

The interface is a thin client: it collects a file and some choices, calls into
`src/`, and displays what comes back. No data-processing logic lives here -- that
keeps every step testable without a browser, and it means the pipeline can be
explained (and defended) independently of the UI.

Run:  streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

INPUT_DIR = Path(__file__).parent / "data" / "input"

st.set_page_config(page_title="DataCleaner", layout="wide")


# --------------------------------------------------------------------- sidebar

st.sidebar.title("DataCleaner")
st.sidebar.caption("Automated data preparation for AI systems")

st.sidebar.subheader("1. Choose data")

source = st.sidebar.radio(
    "Source",
    ["Sample file", "Upload"],
    help="Sample files live in data/input/. Upload accepts any CSV or Excel file.",
)

handle: Path | object | None = None

if source == "Sample file":
    samples = sorted(
        p.name
        for p in INPUT_DIR.glob("*")
        if p.suffix.lower() in {".csv", ".xlsx", ".xls"}
    )
    if samples:
        chosen = st.sidebar.selectbox("File", samples)
        handle = INPUT_DIR / chosen
    else:
        st.sidebar.warning(f"No CSV or Excel files in {INPUT_DIR}")
else:
    handle = st.sidebar.file_uploader("CSV or Excel", type=["csv", "xlsx", "xls"])

# Header detection is deliberately a user choice rather than a guess. Getting it
# wrong is silent and destructive: a headerless file read with the default
# consumes its own first data row as column names, and every later step then
# describes a table with one row missing and nonsense column labels. The census
# sample ships in both forms so the difference is visible.
has_header = st.sidebar.checkbox(
    "First row contains column names",
    value=True,
    help="Uncheck for files that start straight into data.",
)


# ------------------------------------------------------------------- main pane

st.title("DataCleaner")

if handle is None:
    st.info("Choose a sample file or upload one to begin.", icon="←")
    st.stop()

read_kwargs = {} if has_header else {"header": None}
# Both a Path and Streamlit's UploadedFile expose `.name`.
name = handle.name

try:
    if name.lower().endswith((".xlsx", ".xls")):
        frame = pd.read_excel(handle, **read_kwargs)
    else:
        frame = pd.read_csv(handle, **read_kwargs)
except Exception as exc:  # noqa: BLE001 - surfaced to the user, not swallowed
    st.error(f"Could not read `{name}`: {type(exc).__name__}: {exc}")
    st.stop()

if not has_header:
    frame.columns = [f"column_{i + 1}" for i in range(frame.shape[1])]

rows, cols = frame.shape
a, b, c = st.columns(3)
a.metric("Rows", f"{rows:,}")
b.metric("Columns", cols)
c.metric("Missing cells pandas reports", f"{int(frame.isna().sum().sum()):,}")

st.caption(f"`{name}`")

st.subheader("First rows")
st.dataframe(frame.head(20), width="stretch")

st.divider()
st.caption(
    "Step 0 of the plan: the file loads and can be seen. Profiling, problem "
    "detection, the repair plan and the cleaning steps are added next -- see "
    "PLAN.md."
)
