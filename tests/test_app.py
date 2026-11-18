"""Checks on app.py that do not require running Streamlit.

Nothing imports app.py -- it is a script that executes on import -- so it has
no tests, and a bug in it is found by opening the page. These are the few
things worth catching without that: read as source, not imported.
"""

import ast
from pathlib import Path

import pytest

APP = Path(__file__).resolve().parent.parent / "app.py"


@pytest.fixture(scope="module")
def tree():
    return ast.parse(APP.read_text(encoding="utf-8"))


def function(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"app.py has no function {name!r}")


@pytest.mark.parametrize(
    "name", ["grid_database", "heatmap_inputs", "site_source", "data_stamp"]
)
def test_every_branch_returns_the_same_shape(tree, name):
    """A function whose callers unpack it must return one arity everywhere.

    grid_database once returned two values on the happy path and three when
    the database was missing, so a checkout without data/grid raised a
    ValueError on unpack instead of quietly losing the overlay -- the one
    case the branch exists to handle, and the one nobody runs.
    """
    widths = {
        len(node.value.elts)
        for node in ast.walk(function(tree, name))
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Tuple)
    }
    assert len(widths) <= 1, f"{name} returns tuples of differing width: {widths}"


def test_the_database_caches_are_keyed_on_the_files(tree):
    """Everything derived from the grid takes data_stamp as its first
    argument, so replacing the database under a running server invalidates
    the caches instead of being served the old one until a restart."""
    for name in ("render_overlay", "contour_paths", "sample_surface"):
        node = function(tree, name)
        assert node.args.args, f"{name} takes no arguments"
        assert node.args.args[0].arg == "stamp", (
            f"{name} must take the data stamp first, or its cache outlives "
            "a change to the database"
        )
        assert any(
            isinstance(d, ast.Call) and getattr(d.func, "attr", "") == "cache_data"
            for d in node.decorator_list
        ), f"{name} is not cached, so the stamp argument is pointless"
