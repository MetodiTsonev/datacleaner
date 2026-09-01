"""The dependency list is closed, and this is what closes it.

Anti-drift rule 4 in PLAN.md: pandas, numpy, streamlit, openpyxl, pytest. Anything else
needs a decisions.md entry first. A rule nothing checks is a comment, so this walks the
imports of every module in the system and fails on a stranger.

matplotlib is deliberately absent from the allowed set: it is an authoring tool for the
thesis figures (Р13) and must never be reachable from src/ or app.py.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

PROJECT = Path(__file__).parent.parent
SYSTEM_FILES = sorted(PROJECT.glob("src/*.py")) + [PROJECT / "app.py"]

#: Third-party packages the system itself may import. openpyxl is never imported by
#: name - pandas reaches for it to read .xlsx.
ALLOWED_THIRD_PARTY = {"pandas", "numpy", "streamlit"}

#: Named so the failure message can say *why* something is refused rather than just
#: that it is not on a list.
REFUSED = {
    "sklearn": "scikit-learn - the задание names Pandas and NumPy for the data logic",
    "scipy": "SciPy - same reason; the stats we need are short enough to write out",
    "matplotlib": "matplotlib is for the thesis figures only (Р13), not the system",
    "pydantic": "explicitly not building schema validation this way (rule 7)",
    "pandera": "explicitly not building data contracts (rule 7)",
    "yaml": "no config files; the first project's undeclared import broke a clean install",
}


def _imported_roots(path: Path) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            roots.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


@pytest.mark.parametrize("path", SYSTEM_FILES, ids=lambda p: p.name)
def test_no_module_imports_outside_the_closed_list(path):
    stdlib = sys.stdlib_module_names
    for root in _imported_roots(path):
        if root in stdlib or root == "src":
            continue
        assert root not in REFUSED, f"{path.name} imports {root}: {REFUSED[root]}"
        assert root in ALLOWED_THIRD_PARTY, (
            f"{path.name} imports {root!r}, which is not on the closed dependency list "
            f"{sorted(ALLOWED_THIRD_PARTY)}. Adding it needs a writing/decisions.md "
            "entry first — see anti-drift rule 4 in PLAN.md."
        )


def test_the_figures_script_is_not_reachable_from_the_system():
    """scripts/ may use matplotlib; src/ may not import scripts/."""
    for path in SYSTEM_FILES:
        assert "scripts" not in _imported_roots(path), f"{path.name} imports scripts/"


#: Declared, but for development rather than for the system: test runner, linter, and
#: the figure-drawing library. Each has a decisions.md entry (Р11, Р13).
AUTHORING_ONLY = {"pytest", "ruff", "matplotlib"}


def test_the_declared_requirements_match_the_closed_list():
    """requirements.txt may not quietly grow past what the decisions log accounts for."""
    declared = {
        line.split(">=")[0].split("==")[0].strip()
        for line in (PROJECT / "requirements.txt").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    }
    expected = ALLOWED_THIRD_PARTY | {"openpyxl"} | AUTHORING_ONLY
    assert declared == expected, (
        f"requirements.txt is {sorted(declared)}; expected {sorted(expected)}. "
        "Adding a dependency needs a writing/decisions.md entry first (rule 4)."
    )


def test_the_package_itself_installs_only_what_the_system_needs():
    """`pip install -e .` must not drag in an authoring tool.

    The figure library lives in an optional group precisely so a person who only wants
    to run the app never installs it.
    """
    text = (PROJECT / "pyproject.toml").read_text()
    runtime = text.split("dependencies = [")[1].split("]")[0]
    for tool in AUTHORING_ONLY:
        assert tool not in runtime, f"{tool} is in the package's runtime dependencies"
