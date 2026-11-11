"""Tests for ``scripts/validate_grid.py``.

A validator nobody has watched fail is just a script that prints PASS. These
build a small but genuine database offline -- real cached hazard responses
replayed through the real builder -- run every check against it, and then
damage it in each of the ways that matter and confirm each one is caught.

The grid is four nodes around San Francisco so that a committed reference site
falls on one of them; that is what lets the golden check run without the
network. See docs/phase-2-gridded-heatmap.md, checkpoint 2.
"""

from pathlib import Path

import pandas as pd
import pytest

import scripts.validate_grid as validator
from frer.cache import load_entry
from frer.grid import (
    GRID_COLUMNS,
    VALUE_COLUMNS,
    GridSpec,
    build_grid,
    build_manifest,
)
from frer.params import DEFAULT_SITE_CLASSES
from frer.single import SD1_COLUMN, SDS_COLUMN

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO_ROOT / "tests" / "usgs_cache"

VERSION = "ASCE 7-28"
NSHM = "2023"
SITE_CLASSES = list(DEFAULT_SITE_CLASSES)

# San Francisco CA: a usgs_160 site with committed cache for every class, whose
# coordinates land exactly on a 0.05-degree node.
SF_SITE_ID = 41
SF_LAT, SF_LON = 37.75, -122.40

# Written out rather than computed: SF_LON + 0.05 is -122.35000000000001,
# which is not the same number as the node named -122.35.
SPEC = GridSpec(
    lat_min=37.75,
    lat_max=37.80,
    lon_min=-122.40,
    lon_max=-122.35,
    step=0.05,
    name="sf_tiny",
)


@pytest.fixture(scope="module")
def database() -> pd.DataFrame:
    """A real four-node database, built offline from committed responses.

    The canned fetcher answers by site class, so the node at San Francisco's
    own coordinates gets San Francisco's own curves for every class and must
    reproduce its committed reference values. The other three nodes carry the
    same numbers, which is geographically meaningless but structurally sound --
    the geometry and the checks are what is under test.
    """
    by_vs30 = {
        float(vs30): load_entry(CACHE_DIR, SF_SITE_ID, site_class)["hazard_curves"][
            NSHM
        ]
        for site_class, vs30 in DEFAULT_SITE_CLASSES.items()
    }

    def fetch_hazard(nshm, lat, lon, vs30):
        return {"response": {"hazardCurves": by_vs30[float(vs30)]}}

    import tempfile

    with tempfile.TemporaryDirectory() as work:
        return build_grid(
            SPEC,
            SITE_CLASSES,
            VERSION,
            work_dir=Path(work),
            pause=0,
            fetch_hazard=fetch_hazard,
        )


@pytest.fixture
def manifest(database) -> dict:
    return build_manifest(SPEC, SITE_CLASSES, VERSION, database, built_at="now")


# --- the happy path --------------------------------------------------------


def test_a_sound_database_passes_every_check(database, manifest, capsys):
    """All three offline checks pass on a database built the normal way."""
    assert len(database) == len(SPEC) * len(SITE_CLASSES) == 16
    validator.check_database(database, SPEC)
    validator.check_manifest(manifest, database, SPEC)
    validator.check_golden(database, SPEC)
    assert "FAIL" not in capsys.readouterr().out


def test_the_golden_check_reaches_the_reference_site(database, capsys):
    """The node at San Francisco is compared, not silently skipped."""
    validator.check_golden(database, SPEC)
    out = capsys.readouterr().out
    assert "San Francisco CA" in out
    assert "database vs committed reference" in out


# --- each way it should fail -----------------------------------------------


def test_a_tampered_value_is_caught(database):
    """The headline check: change a number, validation fails."""
    tampered = database.copy()
    tampered.loc[tampered.index[0], SDS_COLUMN] *= 1.10
    with pytest.raises(validator.ValidationError, match="tolerance"):
        validator.check_golden(tampered, SPEC)


def test_a_missing_row_is_caught(database):
    with pytest.raises(validator.ValidationError, match="rows, expected"):
        validator.check_database(database.iloc[:-1], SPEC)


def test_a_duplicated_node_is_caught(database):
    doubled = pd.concat([database, database.iloc[[0]]], ignore_index=True)
    with pytest.raises(validator.ValidationError, match="rows, expected"):
        validator.check_database(doubled, SPEC)


def test_a_nan_is_caught(database):
    broken = database.copy()
    broken.loc[broken.index[0], SD1_COLUMN] = float("nan")
    with pytest.raises(validator.ValidationError, match="NaN"):
        validator.check_database(broken, SPEC)


def test_a_non_positive_value_is_caught(database):
    broken = database.copy()
    broken.loc[broken.index[0], SDS_COLUMN] = 0.0
    with pytest.raises(validator.ValidationError, match="non-positive"):
        validator.check_database(broken, SPEC)


def test_a_node_off_the_grid_is_caught(database):
    """A row must sit on the graticule, not merely inside the bounding box."""
    broken = database.copy()
    broken.loc[broken.index[0], "lat"] = 37.77  # inside the box, off the grid
    with pytest.raises(validator.ValidationError, match="off the grid"):
        validator.check_database(broken, SPEC)


def test_an_implausible_ratio_is_caught(database):
    """FREr above DE is not necessarily wrong, but it must not pass silently."""
    broken = database.copy()
    broken[VALUE_COLUMNS[2]] = broken[VALUE_COLUMNS[0]] * 2.0
    with pytest.raises(validator.ValidationError, match="outside the expected"):
        validator.check_database(broken, SPEC)


def test_a_stale_calculator_version_is_caught(database, manifest):
    """A database built by older arithmetic must not pass as current."""
    stale = {**manifest, "calculator_version": "0.0.1-old"}
    with pytest.raises(validator.ValidationError, match="rebuild before trusting"):
        validator.check_manifest(stale, database, SPEC)


def test_a_manifest_describing_a_different_grid_is_caught(database, manifest):
    wrong = {**manifest, "grid": {**manifest["grid"], "step": 0.10}}
    with pytest.raises(validator.ValidationError, match="manifest grid step"):
        validator.check_manifest(wrong, database, SPEC)


def test_a_wrong_vs30_is_caught(database, manifest):
    """Class D at the retired 265 rather than the 260 the code now sends."""
    wrong = {**manifest, "site_classes": {**manifest["site_classes"], "D": 265}}
    with pytest.raises(validator.ValidationError, match="vs30 265"):
        validator.check_manifest(wrong, database, SPEC)


# --- the committed cache tripwire ------------------------------------------


def test_cache_check_passes_on_a_clean_tree(capsys):
    """The real check, against the real repository."""
    validator.check_cache_untouched()
    assert "PASS" in capsys.readouterr().out


def test_cache_check_fails_when_git_reports_a_change(monkeypatch):
    """A modified fixture must stop validation, not just colour the output."""

    class Dirty:
        stdout = " M tests/usgs_cache/site0041_BC.json\n"

    monkeypatch.setattr(validator.subprocess, "run", lambda *a, **k: Dirty())
    with pytest.raises(validator.ValidationError, match="never be written to"):
        validator.check_cache_untouched()


def test_grid_columns_are_what_the_validator_expects():
    """Guards against the schema drifting away from the checks."""
    assert validator.GRID_COLUMNS == GRID_COLUMNS
