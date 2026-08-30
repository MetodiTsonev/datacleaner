"""The app renders without raising.

Written after a bug that HTTP checks could not catch. `app.py` bound `summary`
twice -- once to the detection summary at module level, then again to the loader
summary inside the Data tab -- and because Streamlit runs the whole script top to
bottom on every interaction, the Findings tab read the wrong dict and raised
`KeyError: 'total'` on every render.

The server still answered 200 and `/_stcore/health` still said `ok`, because
Streamlit renders exceptions *inside* the page. So a reachability check can never
detect a broken page, and every UI change needs this instead.

`AppTest` runs the script in-process and exposes whatever it raised.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = Path(__file__).parent.parent / "app.py"
DATA = Path(__file__).parent.parent / "data" / "input"
SAMPLES = [p.name for p in sorted(DATA.glob("*.csv"))]

#: The census file is 48,842 rows and every tab profiles and detects on it, so a
#: generous budget. Still far below a hung run.
TIMEOUT = 180


# Widgets are addressed by key, never by position. Position broke the moment a
# selectbox was added above the file picker, and the failure said "not in list"
# rather than anything about the UI.
def run_app(**session_state) -> AppTest:
    app = AppTest.from_file(str(APP), default_timeout=TIMEOUT)
    for key, value in session_state.items():
        app.session_state[key] = value
    return app.run()


def test_app_renders_with_no_file_chosen():
    app = run_app()
    assert not app.exception, app.exception


@pytest.mark.parametrize("sample", SAMPLES)
def test_app_renders_for_every_sample_file(sample):
    """Every shipped sample must render every tab. This is the regression."""
    app = AppTest.from_file(str(APP), default_timeout=TIMEOUT).run()
    assert not app.exception, app.exception
    app.selectbox(key="sample_file").select(sample).run()
    assert not app.exception, f"{sample}: {app.exception}"


def test_the_findings_tab_reports_a_count_for_the_messy_sample():
    """Guards the exact expression that broke: findings_summary['total']."""
    if "messy-orders.csv" not in SAMPLES:
        pytest.skip("messy sample not present")
    app = AppTest.from_file(str(APP), default_timeout=TIMEOUT).run()
    app.selectbox(key="sample_file").select("messy-orders.csv").run()
    assert not app.exception, app.exception
    headers = [h.value for h in app.subheader]
    assert any(h.startswith("Stage 3 of") for h in headers), headers


def test_unticking_the_header_box_still_renders():
    if not SAMPLES:
        pytest.skip("no samples present")
    app = AppTest.from_file(str(APP), default_timeout=TIMEOUT).run()
    app.checkbox(key="has_header").uncheck().run()
    assert not app.exception, app.exception


def test_no_name_crosses_a_tab_boundary():
    """The shape of the bug, caught statically as well as at runtime.

    `with` does not create a scope in Python, so a name assigned at the top of the
    script and then reassigned inside `with tab_data:` is one name with two
    meanings -- and the second meaning leaks into every later tab. That is exactly
    how `summary` broke the Findings tab.

    Branch assignment is not the hazard: `handle = None` followed by `handle = ...`
    in each arm of an if/else is one meaning, and is allowed. Only crossing into or
    out of a `with` block is flagged.
    """
    import ast

    tree = ast.parse(APP.read_text())

    def names_assigned(body, *, descend_into_with: bool):
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            if isinstance(node, ast.With):
                if descend_into_with:
                    yield from names_assigned(node.body, descend_into_with=True)
                continue
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        yield target.id
            for attr in ("body", "orelse", "finalbody"):
                yield from names_assigned(
                    getattr(node, attr, []) or [], descend_into_with=descend_into_with
                )

    outside = set(names_assigned(tree.body, descend_into_with=False))
    inside = set()
    for node in tree.body:
        if isinstance(node, ast.With):
            inside |= set(names_assigned(node.body, descend_into_with=True))

    crossing = outside & inside
    assert not crossing, (
        f"assigned both outside and inside a `with` block: {sorted(crossing)}. "
        "Streamlit shares one module scope across every tab, so this name carries "
        "two meanings and the second wins for all later tabs."
    )


def test_the_validate_tab_renders_a_draft_contract():
    """Stage 4's tab must render for a file that breaks its own inferred rules."""
    if "messy-orders.csv" not in SAMPLES:
        pytest.skip("messy sample not present")
    app = AppTest.from_file(str(APP), default_timeout=TIMEOUT).run()
    app.selectbox(key="sample_file").select("messy-orders.csv").run()
    assert not app.exception, app.exception
    headers = [h.value for h in app.subheader]
    assert any(h.startswith("Stage 4 of") for h in headers), headers


def test_selecting_a_target_still_renders():
    """The target selector feeds detection and validation, so it must be exercised."""
    if "messy-orders.csv" not in SAMPLES:
        pytest.skip("messy sample not present")
    app = AppTest.from_file(str(APP), default_timeout=TIMEOUT).run()
    app.selectbox(key="sample_file").select("messy-orders.csv").run()
    app.selectbox(key="target_column").select("returned").run()
    assert not app.exception, app.exception


def test_a_headerless_file_does_not_fill_the_target_list_with_values():
    """The 'column to predict' list showed 39, State-gov, 77516.

    A headerless file read as though it had a header turns its own first row into
    column names. Detection now sets the checkbox default per file.
    """
    if "adult-census-noheader-sample.csv" not in SAMPLES:
        pytest.skip("headerless sample not present")
    app = AppTest.from_file(str(APP), default_timeout=TIMEOUT).run()
    app.selectbox(key="sample_file").select("adult-census-noheader-sample.csv").run()
    assert not app.exception, app.exception
    assert app.checkbox(key="has_header").value is False, "detected as headerless"
    options = list(app.selectbox(key="target_column").options)[1:]
    assert all(o.startswith("column_") for o in options), options


def test_a_file_with_a_real_header_is_detected_as_having_one():
    if "adult-census.csv" not in SAMPLES:
        pytest.skip("census sample not present")
    app = AppTest.from_file(str(APP), default_timeout=TIMEOUT).run()
    app.selectbox(key="sample_file").select("adult-census.csv").run()
    assert app.checkbox(key="has_header").value is True
    assert "age" in app.selectbox(key="target_column").options


@pytest.mark.parametrize("kind", [
    "not_null", "unique", "range", "allowed_values", "pattern", "type", "compare",
])
def test_every_rule_kind_can_be_submitted_from_the_form(kind):
    """Submitted with whatever the form defaults to, which is what a user does first.

    `range` with both bounds blank crashed the page; `allowed_values` with a blank
    textarea quarantined the entire file.
    """
    if "messy-orders.csv" not in SAMPLES:
        pytest.skip("messy sample not present")
    app = AppTest.from_file(str(APP), default_timeout=TIMEOUT).run()
    app.selectbox(key="sample_file").select("messy-orders.csv").run()
    boxes = {s.label: s for s in app.selectbox}
    boxes["Column"].select("amount").run()
    boxes = {s.label: s for s in app.selectbox}
    boxes["Constraint"].select(kind).run()
    assert not app.exception, app.exception
    app.button[-1].click().run()
    assert not app.exception, f"{kind}: {app.exception}"
    rejected = int({m.label: m.value for m in app.metric}["Rows rejected"])
    assert rejected < 200, f"{kind} quarantined {rejected} rows on form defaults"


def test_the_sample_picker_offers_only_data_files():
    """`data/input/adult-census.names.txt` documents the dataset; it is not data.

    Selecting it produced a 106x2 table whose first column name was a sentence from
    a licence notice.
    """
    app = AppTest.from_file(str(APP), default_timeout=TIMEOUT).run()
    offered = list(app.selectbox(key="sample_file").options)
    assert offered, "at least one sample must be offered"
    assert not [f for f in offered if f.endswith(".txt")], offered


def test_choosing_a_constraint_immediately_shows_its_value_fields():
    """The add-rule controls were inside `st.form`, which does not rerun when one of
    its own widgets changes -- so the minimum and maximum boxes for `range` were
    never created. Choosing `range` showed no bounds, and submitting produced a rule
    with none, which then crashed rendering the contract table.
    """
    if "messy-orders.csv" not in SAMPLES:
        pytest.skip("messy sample not present")
    app = AppTest.from_file(str(APP), default_timeout=TIMEOUT).run()
    app.selectbox(key="sample_file").select("messy-orders.csv").run()
    assert not [t for t in app.text_input if "Minimum" in t.label]
    app.selectbox(key="new_rule_kind").select("range").run()
    assert not app.exception, app.exception
    labels = [t.label for t in app.text_input]
    assert any("Minimum" in label for label in labels), labels
    assert any("Maximum" in label for label in labels), labels


def test_the_add_button_is_disabled_until_the_rule_is_complete():
    """A range with no bounds cannot be added at all now."""
    if "messy-orders.csv" not in SAMPLES:
        pytest.skip("messy sample not present")
    app = AppTest.from_file(str(APP), default_timeout=TIMEOUT).run()
    app.selectbox(key="sample_file").select("messy-orders.csv").run()
    app.selectbox(key="new_rule_kind").select("range").run()
    add = [b for b in app.button if b.label == "Add rule"]
    assert add and add[0].disabled, "no bounds given, so nothing to add"
    app.text_input(key="new_rule_min").set_value("0.01").run()
    add = [b for b in app.button if b.label == "Add rule"]
    assert add and not add[0].disabled


def test_the_run_tab_executes_the_plan_and_reports_each_step():
    if "adult-census.csv" not in SAMPLES:
        pytest.skip("census sample not present")
    app = AppTest.from_file(str(APP), default_timeout=TIMEOUT).run()
    app.selectbox(key="sample_file").select("adult-census.csv").run()
    app.selectbox(key="target_column").select("income").run()
    assert not app.exception, app.exception

    assert not [b for b in app.button if b.label == "Run"], (
        "no Run button: it reran the script and st.tabs loses the active tab, so the "
        "result was never visible"
    )
    metrics = {m.label: m.value for m in app.metric}
    assert metrics["Nulls remaining"] == "0"
    assert int(metrics["Cells changed"].replace(",", "")) > 6000


def test_every_tab_says_which_stage_it_is_and_what_it_does():
    """The tabs are stages of one pipeline and only make sense in order.

    Added after the interface was reported as hard to follow: the tab labels alone
    ("Profile", "Plan") say nothing about what happens or why it comes where it does.
    """
    if "adult-census.csv" not in SAMPLES:
        pytest.skip("census sample not present")
    app = AppTest.from_file(str(APP), default_timeout=TIMEOUT).run()
    app.selectbox(key="sample_file").select("adult-census.csv").run()
    assert not app.exception, app.exception
    headers = [h.value for h in app.subheader]
    total = max(
        int(h.split(" of ")[1].split(" ")[0]) for h in headers if h.startswith("Stage ")
    )
    for stage in range(1, total + 1):
        assert any(h.startswith(f"Stage {stage} of") for h in headers), (stage, headers)


def test_the_features_tab_needs_the_run_tab_first():
    """It works on the cleaned data, so it says so rather than rendering empty."""
    if "adult-census.csv" not in SAMPLES:
        pytest.skip("census sample not present")
    app = AppTest.from_file(str(APP), default_timeout=TIMEOUT).run()
    app.selectbox(key="sample_file").select("adult-census.csv").run()
    assert not app.exception, app.exception
    headers = [h.value for h in app.subheader]
    assert any(h.startswith("Stage 7 of") for h in headers), headers
