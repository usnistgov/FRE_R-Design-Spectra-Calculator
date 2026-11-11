"""Tests for the gridded design-value database in ``frer.grid``.

The arithmetic behind each node is already covered: ``build_grid`` drives
``frer.single.calc_site``, which test_benchmark.py and test_single.py validate
against the committed reference values. What is new is the geometry, the resume
ledger, the file format and the guards around the committed cache, so that is
what these cover.

The golden test below is the one that matters. Six usgs_160 sites fall inside
the Bay Area grid and land exactly on its nodes, and their frozen responses are
already committed, so the builder can be checked against reference values
without a single API call -- and without interpolation error muddying the
comparison.

Nothing here touches the network.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from frer.cache import load_entry
from frer.grid import (
    BAY_AREA,
    COORD_DECIMALS,
    GRID_COLUMNS,
    RATIO_COLUMNS,
    VALUE_COLUMNS,
    GridSpec,
    add_ratios,
    build_grid,
    build_manifest,
    guard_not_in_tests,
    load_grid,
    read_ledger,
    relative_differences,
    rows_from_cache,
    sort_grid,
    write_grid,
    write_manifest,
)
from frer.params import DEFAULT_SITE_CLASSES
from frer.single import SDS_COLUMN, rename_legacy_spectral_columns

REPO_ROOT = Path(__file__).resolve().parent.parent
TESTS_DIR = REPO_ROOT / "tests"
CACHE_DIR = TESTS_DIR / "usgs_cache"
SITES_CSV = REPO_ROOT / "sites.csv"

VERSION = "ASCE 7-28"
NSHM = "2023"
SITE_CLASSES = list(DEFAULT_SITE_CLASSES)
RTOL = 0.02  # the benchmark tolerance

# One cached entry, used as a canned hazard response so a build can be
# exercised end to end without the network.
SAMPLE_SITE_ID = 41  # San Francisco CA, on a grid node
SAMPLE_CLASS = "BC"


# --- geometry --------------------------------------------------------------


def test_bay_area_grid_has_the_expected_nodes():
    """The shipped grid is 27 x 23 over the nine-county box."""
    assert len(BAY_AREA.lats) == 27
    assert len(BAY_AREA.lons) == 23
    assert len(BAY_AREA) == 621

    nodes = BAY_AREA.nodes()
    assert len(nodes) == 621
    assert nodes.iloc[0].tolist() == [BAY_AREA.lat_min, BAY_AREA.lon_min]
    assert nodes.iloc[-1].tolist() == [BAY_AREA.lat_max, BAY_AREA.lon_max]


def test_nodes_sit_on_the_graticule():
    """Every node is a multiple of the step, to the rounding the files use."""
    nodes = BAY_AREA.nodes()
    for axis in ("lat", "lon"):
        remainder = (nodes[axis] / BAY_AREA.step).round() * BAY_AREA.step
        assert (nodes[axis] - remainder).abs().max() < 10**-COORD_DECIMALS


def test_node_order_is_stable():
    """Two calls give the same rows in the same order."""
    assert BAY_AREA.nodes().equals(BAY_AREA.nodes())


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"step": 0}, "step must be positive"),
        ({"lat_min": 38.4, "lat_max": 37.1}, "empty or inverted"),
        ({"lat_min": 37.11}, "not a multiple of step"),
        ({"lat_max": 52.0}, "outside the conterminous"),
    ],
)
def test_validate_rejects_unusable_grids(kwargs, message):
    """A grid that cannot be sampled fails when it is defined, not mid-build."""
    spec = GridSpec(
        **{
            "lat_min": 37.10,
            "lat_max": 38.40,
            "lon_min": -122.80,
            "lon_max": -121.70,
            "step": 0.05,
            **kwargs,
        }
    )
    with pytest.raises(ValueError, match=message):
        spec.validate()


# --- the committed cache is never written to -------------------------------


@pytest.mark.parametrize(
    "path", [CACHE_DIR, TESTS_DIR, TESTS_DIR / "usgs_cache" / "nested"]
)
def test_guard_refuses_paths_inside_tests(path):
    """A build cannot be pointed at the committed fixtures, even by accident."""
    with pytest.raises(ValueError, match="committed fixtures"):
        guard_not_in_tests(path, "work_dir")


def test_guard_allows_a_normal_work_directory(tmp_path):
    assert guard_not_in_tests(tmp_path, "work_dir") == tmp_path.resolve()


def test_build_refuses_to_run_inside_tests():
    """The guard is wired into build_grid, not merely available to it."""
    with pytest.raises(ValueError, match="committed fixtures"):
        build_grid(BAY_AREA, ["BC"], VERSION, work_dir=CACHE_DIR / "scratch")


# --- the golden check ------------------------------------------------------


def golden_sites() -> pd.DataFrame:
    """usgs_160 sites inside the grid that land exactly on its nodes."""
    from frer.sites import load_sites

    sites = load_sites(SITES_CSV, "usgs_160")
    inside = sites[
        sites["lat"].between(BAY_AREA.lat_min, BAY_AREA.lat_max)
        & sites["lon"].between(BAY_AREA.lon_min, BAY_AREA.lon_max)
    ]
    on_node = inside[
        inside["lat"].apply(lambda v: any(abs(v - x) < 1e-9 for x in BAY_AREA.lats))
        & inside["lon"].apply(lambda v: any(abs(v - x) < 1e-9 for x in BAY_AREA.lons))
    ]
    return on_node.reset_index(drop=True)


def golden_comparison() -> pd.DataFrame:
    """Relative differences between rebuilt rows and the committed references."""
    sites = golden_sites()
    built = rows_from_cache(CACHE_DIR, sites, SITE_CLASSES, VERSION)
    named = built.merge(sites[["name", "lat", "lon"]], on=["lat", "lon"], how="left")

    reference = rename_legacy_spectral_columns(
        pd.read_csv(TESTS_DIR / "usgs_160" / VERSION / "frer_design_vals.csv")
    )[["name", "site_class", *VALUE_COLUMNS]]
    return relative_differences(named, reference)


def test_there_are_golden_sites_on_exact_nodes():
    """The grid would be unvalidatable without them, so assert they exist."""
    sites = golden_sites()
    assert len(sites) == 6
    assert set(sites["name"]) == {
        "Concord CA",
        "Oakland CA",
        "San Francisco CA",
        "San Jose CA",
        "San Mateo CA",
        "Vallejo CA",
    }


def test_rebuilt_nodes_match_the_committed_references():
    """The builder reproduces the benchmark values at every golden node."""
    differences = golden_comparison()
    assert len(differences) == 6 * len(SITE_CLASSES)
    for column in VALUE_COLUMNS:
        assert differences[column].max() <= RTOL


def test_the_golden_check_can_fail():
    """A perturbed value must be caught, or the check above proves nothing."""
    sites = golden_sites()
    built = rows_from_cache(CACHE_DIR, sites, SITE_CLASSES, VERSION)
    named = built.merge(sites[["name", "lat", "lon"]], on=["lat", "lon"], how="left")
    named.loc[0, SDS_COLUMN] *= 1.10  # 10%, well outside the 2% tolerance

    reference = rename_legacy_spectral_columns(
        pd.read_csv(TESTS_DIR / "usgs_160" / VERSION / "frer_design_vals.csv")
    )[["name", "site_class", *VALUE_COLUMNS]]
    differences = relative_differences(named, reference)
    assert differences[SDS_COLUMN].max() > RTOL


def test_comparison_refuses_a_partial_join():
    """Joining on something that does not line up fails loudly.

    The reference CSVs number sites per site set, so joining two sets on ``id``
    silently compares different cities. Anything that fails to match every row
    has to raise rather than quietly report on the subset it did match.
    """
    sites = golden_sites()
    built = rows_from_cache(CACHE_DIR, sites, SITE_CLASSES, VERSION)
    named = built.merge(sites[["name", "lat", "lon"]], on=["lat", "lon"], how="left")

    reference = rename_legacy_spectral_columns(
        pd.read_csv(TESTS_DIR / "usgs_160" / VERSION / "frer_design_vals.csv")
    )[["name", "site_class", *VALUE_COLUMNS]]
    with pytest.raises(ValueError, match="do not describe the same points"):
        relative_differences(named, reference.head(3))


# --- ratios ----------------------------------------------------------------


def test_ratios_are_derived_not_stored():
    """Ratio columns are absent from the stored schema and added on read."""
    assert not set(RATIO_COLUMNS) & set(GRID_COLUMNS)

    table = pd.DataFrame(
        {
            "lat": [37.75],
            "lon": [-122.40],
            "site_class": ["BC"],
            VALUE_COLUMNS[0]: [1.0],
            VALUE_COLUMNS[1]: [0.5],
            VALUE_COLUMNS[2]: [0.6],
            VALUE_COLUMNS[3]: [0.2],
        }
    )
    out = add_ratios(table)
    assert out["Sdsfr_over_Sds"].iloc[0] == pytest.approx(0.6)
    assert out["Sd1fr_over_Sd1"].iloc[0] == pytest.approx(0.4)


def test_zero_denominator_gives_nan_not_infinity():
    """A degenerate DE value must not read as an enormous ratio."""
    table = pd.DataFrame(
        {
            "lat": [37.75],
            "lon": [-122.40],
            "site_class": ["BC"],
            VALUE_COLUMNS[0]: [0.0],
            VALUE_COLUMNS[1]: [0.0],
            VALUE_COLUMNS[2]: [0.6],
            VALUE_COLUMNS[3]: [0.2],
        }
    )
    out = add_ratios(table)
    assert out["Sdsfr_over_Sds"].isna().all()


# --- files -----------------------------------------------------------------


def sample_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            [37.80, -122.25, "BC", 1.0, 0.5, 0.6, 0.2],
            [37.75, -122.40, "BC", 1.1, 0.6, 0.7, 0.3],
            [37.75, -122.40, "D", 1.2, 0.7, 0.8, 0.4],
        ],
        columns=GRID_COLUMNS,
    )


def test_grid_file_round_trips(tmp_path):
    path = write_grid(sample_table(), tmp_path / "g.csv.gz")
    back = load_grid(path)
    assert list(back.columns) == GRID_COLUMNS
    pd.testing.assert_frame_equal(back, sort_grid(sample_table()), check_dtype=False)


def test_grid_file_is_byte_identical_between_writes(tmp_path):
    """Rebuilding to the same path from the same data gives the same bytes.

    Two things would break this. gzip stamps a modification time by default,
    which write_grid pins, and the row order out of a build depends on the
    order nodes happened to complete in, which sort_grid settles. Without both,
    a diff on the committed database would be noise.

    Same path on purpose: gzip also stores the file name in its header, so
    reproducibility is a property of rebuilding in place.
    """
    path = tmp_path / "g.csv.gz"
    first = write_grid(sample_table(), path).read_bytes()
    second = write_grid(sample_table().iloc[::-1], path).read_bytes()
    assert first == second


def test_load_grid_rejects_a_file_missing_columns(tmp_path):
    path = tmp_path / "bad.csv.gz"
    sample_table().drop(columns=[SDS_COLUMN]).to_csv(
        path, index=False, compression="gzip"
    )
    with pytest.raises(ValueError, match=SDS_COLUMN):
        load_grid(path)


def test_manifest_records_what_the_database_was_built_from(tmp_path):
    table = sample_table()
    manifest = build_manifest(BAY_AREA, SITE_CLASSES, VERSION, table, built_at="now")
    assert manifest["grid"]["nodes"] == 621
    assert manifest["version"] == VERSION
    assert manifest["nshm"] == 2023
    assert manifest["rows"] == len(table)
    assert manifest["calculator_version"]
    # The vs30 the fixtures depend on, not the design-maps value for class D.
    assert manifest["site_classes"]["D"] == 265

    path = write_manifest(manifest, tmp_path / "m.json")
    assert json.loads(path.read_text(encoding="utf-8")) == manifest


# --- building and resuming -------------------------------------------------


@pytest.fixture
def canned_hazard():
    """A real cached hazard response, replayed for every node.

    Exercises the build end to end -- pull_site, calc_site, the ledger, the
    ordering -- without the network. Every node gets the same curves, so the
    values are not geographically meaningful; the geometry and bookkeeping are
    what is under test.
    """
    entry = load_entry(CACHE_DIR, SAMPLE_SITE_ID, SAMPLE_CLASS)
    assert entry is not None, "sample cache entry is missing"
    curves = entry["hazard_curves"][NSHM]

    def fetch_hazard(nshm, lat, lon, vs30):
        return {"response": {"hazardCurves": curves}}

    return fetch_hazard


@pytest.fixture
def tiny_grid():
    """Four nodes, on the graticule, around San Francisco."""
    return GridSpec(
        lat_min=37.75,
        lat_max=37.80,
        lon_min=-122.40,
        lon_max=-122.35,
        step=0.05,
        name="tiny",
    )


def test_build_calculates_every_node(tmp_path, tiny_grid, canned_hazard):
    table = build_grid(
        tiny_grid,
        ["BC", "D"],
        VERSION,
        work_dir=tmp_path,
        pause=0,
        fetch_hazard=canned_hazard,
    )
    assert len(table) == len(tiny_grid) * 2 == 8
    assert set(table["site_class"]) == {"BC", "D"}
    assert table[VALUE_COLUMNS].notna().all().all()
    assert (table[VALUE_COLUMNS] > 0).all().all()


def test_build_resumes_without_duplicating_or_dropping(
    tmp_path, tiny_grid, canned_hazard
):
    """An interrupted run continues from the ledger and lands in the same place."""
    partial = build_grid(
        tiny_grid,
        ["BC", "D"],
        VERSION,
        work_dir=tmp_path,
        pause=0,
        limit=3,
        fetch_hazard=canned_hazard,
    )
    assert len(partial) == 3

    complete = build_grid(
        tiny_grid,
        ["BC", "D"],
        VERSION,
        work_dir=tmp_path,
        pause=0,
        fetch_hazard=canned_hazard,
    )
    assert len(complete) == 8
    assert not complete.duplicated(subset=["lat", "lon", "site_class"]).any()

    one_shot = build_grid(
        tiny_grid,
        ["BC", "D"],
        VERSION,
        work_dir=tmp_path / "fresh",
        pause=0,
        fetch_hazard=canned_hazard,
    )
    pd.testing.assert_frame_equal(complete, one_shot)


def test_a_failing_node_is_left_for_a_later_run(tmp_path, tiny_grid):
    """A node that fails is not written, so a re-run retries it."""
    calls = {"n": 0}

    def flaky(nshm, lat, lon, vs30):
        calls["n"] += 1
        raise RuntimeError("service unavailable")

    table = build_grid(
        tiny_grid, ["BC"], VERSION, work_dir=tmp_path, pause=0, fetch_hazard=flaky
    )
    assert calls["n"] == len(tiny_grid)
    assert table.empty
    assert read_ledger(tmp_path / "tiny_asce7-28.partial.csv").empty
