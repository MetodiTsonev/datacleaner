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
    app.selectbox[0].select(sample).run()
    assert not app.exception, f"{sample}: {app.exception}"


def test_the_findings_tab_reports_a_count_for_the_messy_sample():
    """Guards the exact expression that broke: findings_summary['total']."""
    if "messy-orders.csv" not in SAMPLES:
        pytest.skip("messy sample not present")
    app = AppTest.from_file(str(APP), default_timeout=TIMEOUT).run()
    app.selectbox[0].select("messy-orders.csv").run()
    assert not app.exception, app.exception
    headers = [h.value for h in app.subheader]
    assert any("problem(s) found" in h for h in headers), headers


def test_unticking_the_header_box_still_renders():
    if not SAMPLES:
        pytest.skip("no samples present")
    app = AppTest.from_file(str(APP), default_timeout=TIMEOUT).run()
    app.checkbox[0].uncheck().run()
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
