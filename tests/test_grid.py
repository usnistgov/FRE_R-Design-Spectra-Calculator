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
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from frer import landmask, usgs_api
from frer.cache import load_entry
from frer.grid import (
    BAY_AREA,
    COORD_DECIMALS,
    GRID_COLUMNS,
    INTERPOLATION_TOLERANCE,
    RATIO_COLUMNS,
    VALUE_COLUMNS,
    GridSpec,
    _preflight_candidates,
    add_ratios,
    build_grid,
    build_manifest,
    guard_not_in_tests,
    load_grid,
    node_values,
    preflight,
    read_ledger,
    relative_differences,
    rows_from_cache,
    sort_grid,
    write_grid,
    write_manifest,
)
from frer.params import DEFAULT_SITE_CLASSES
from frer.single import SDS_COLUMN, rename_legacy_spectral_columns
from scripts.build_grid import refused_rows

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
    """The shipped grid is 35 x 29 at 0.025 deg over the trimmed box.

    Every node is calculated, water included -- see the coastline tests below
    for why -- so len() is the full product.
    """
    assert BAY_AREA.step == 0.025
    assert len(BAY_AREA.lats) == 35
    assert len(BAY_AREA.lons) == 29
    assert len(BAY_AREA) == 35 * 29 == 1015

    nodes = BAY_AREA.nodes()
    assert len(nodes) == 1015
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

    sites = load_sites(SITES_CSV)
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
    assert manifest["grid"]["nodes"] == 1015
    assert manifest["version"] == VERSION
    assert manifest["nshm"] == 2023
    assert manifest["rows"] == len(table)
    assert manifest["calculator_version"]
    # The vs30 actually sent to the hazard service for class D.
    assert manifest["site_classes"]["D"] == 260

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


# --- coordinates the service refuses ---------------------------------------
#
# The hazard service answers a small number of individual coordinates with
# HTTP 200 and `status: error`, reproducibly, in about a tenth of a second,
# while all eight of that point's neighbours compute normally. (37.75, -122.80)
# is one, and it sits inside the Bay Area grid. Two things followed from that:
# the preflight used to sample exactly one such point and die before the build
# began, and a build that lost a single node would refuse to write anything
# after a day of work. Both are covered here.


def _refusing(curves, refuse_at):
    """A canned fetcher that refuses the given (lat, lon) pairs."""
    refused = {(round(lat, 4), round(lon, 4)) for lat, lon in refuse_at}

    def fetch_hazard(nshm, lat, lon, vs30):
        if (round(lat, 4), round(lon, 4)) in refused:
            raise usgs_api.USGSApiError("hazard call status: error")
        return {"response": {"hazardCurves": curves}}

    return fetch_hazard


def test_preflight_retries_on_another_row_when_a_point_is_refused(tiny_grid):
    """One refused coordinate must not decide whether a day-long build starts."""
    entry = load_entry(CACHE_DIR, SAMPLE_SITE_ID, SAMPLE_CLASS)
    curves = entry["hazard_curves"][NSHM]

    first_lat, first_lon = _preflight_candidates(tiny_grid)[0]
    result = preflight(
        tiny_grid,
        version_id=VERSION,
        fetch_hazard=_refusing(curves, [(first_lat, first_lon)]),
    )

    assert result["checked"] is True
    assert result["lat"] != first_lat, "should have moved to a different row"
    assert len(result["attempts"]) == 1
    assert "status: error" in result["attempts"][0]


def test_preflight_reports_when_every_candidate_is_refused(tiny_grid):
    """Exhausting the candidates is reported, not raised, and is not 'adequate'."""

    def always_refuses(nshm, lat, lon, vs30):
        raise usgs_api.USGSApiError("hazard call status: error")

    result = preflight(tiny_grid, version_id=VERSION, fetch_hazard=always_refuses)

    assert result["checked"] is False
    assert result["adequate"] is False
    assert result["attempts"], "should record what it tried"
    assert "could not evaluate" in result["verdict"]


def test_preflight_candidates_stay_on_the_grid():
    """Every candidate must be a real node with a real neighbour to its east."""
    lats = {round(float(value), COORD_DECIMALS) for value in BAY_AREA.lats}
    lons = {round(float(value), COORD_DECIMALS) for value in BAY_AREA.lons}
    for lat, lon in _preflight_candidates(BAY_AREA):
        assert round(lat, COORD_DECIMALS) in lats
        assert round(lon, COORD_DECIMALS) in lons
        assert round(lon + BAY_AREA.step, COORD_DECIMALS) in lons


def test_preflight_avoids_the_known_bad_point():
    """Regression: the old code sampled (37.75, -122.80), which is refused."""
    assert (37.75, -122.80) not in [
        (round(lat, 2), round(lon, 2)) for lat, lon in _preflight_candidates(BAY_AREA)
    ]


def test_failures_are_handed_back_to_the_caller(tmp_path, tiny_grid):
    """failures_out is what distinguishes a refusal from a node not yet reached."""
    entry = load_entry(CACHE_DIR, SAMPLE_SITE_ID, SAMPLE_CLASS)
    curves = entry["hazard_curves"][NSHM]
    bad = (float(tiny_grid.lats[0]), float(tiny_grid.lons[0]))

    failures = []
    table = build_grid(
        tiny_grid,
        ["BC"],
        VERSION,
        work_dir=tmp_path,
        pause=0,
        failures_out=failures,
        fetch_hazard=_refusing(curves, [bad]),
    )

    assert len(table) == len(tiny_grid) - 1
    assert len(failures) == 1
    assert (failures[0]["lat"], failures[0]["lon"]) == bad
    assert failures[0]["site_class"] == "BC"
    assert "status: error" in failures[0]["error"]


def test_refused_rows_counts_only_pairs_the_service_refused(tmp_path, tiny_grid):
    """A gap nobody refused is the build's problem, not the service's."""
    entry = load_entry(CACHE_DIR, SAMPLE_SITE_ID, SAMPLE_CLASS)
    curves = entry["hazard_curves"][NSHM]
    bad = (float(tiny_grid.lats[0]), float(tiny_grid.lons[0]))

    failures = []
    table = build_grid(
        tiny_grid,
        ["BC"],
        VERSION,
        work_dir=tmp_path,
        pause=0,
        failures_out=failures,
        fetch_hazard=_refusing(curves, [bad]),
    )

    accounted = refused_rows(tiny_grid, ["BC"], table, failures)
    assert len(accounted) == 1
    assert (accounted[0]["lat"], accounted[0]["lon"]) == bad

    # Drop another row without recording a refusal: an interrupted run. The
    # shortfall is now 2 while only 1 is explained, which is what stops
    # run_build from writing a partial database.
    interrupted = table.iloc[:-1]
    assert len(refused_rows(tiny_grid, ["BC"], interrupted, failures)) == 1
    assert len(tiny_grid) - len(interrupted) == 2


def test_manifest_declares_refused_pairs(tiny_grid, canned_hazard, tmp_path):
    """The declaration has to reach the manifest, sorted, or validation can't use it."""
    table = build_grid(
        tiny_grid,
        ["BC"],
        VERSION,
        work_dir=tmp_path,
        pause=0,
        fetch_hazard=canned_hazard,
    )
    refused = [
        {"lat": 37.80, "lon": -122.40, "site_class": "D", "error": "x"},
        {"lat": 37.75, "lon": -122.40, "site_class": "BC", "error": "y"},
    ]
    manifest = build_manifest(
        tiny_grid, ["BC"], VERSION, table, refused=refused, built_at="now"
    )
    assert [entry["site_class"] for entry in manifest["refused"]] == ["BC", "D"]


def test_manifest_declares_nothing_when_a_build_is_complete(
    tiny_grid, canned_hazard, tmp_path
):
    table = build_grid(
        tiny_grid,
        ["BC"],
        VERSION,
        work_dir=tmp_path,
        pause=0,
        fetch_hazard=canned_hazard,
    )
    manifest = build_manifest(tiny_grid, ["BC"], VERSION, table, built_at="now")
    assert manifest["refused"] == []


# --- the coastline ----------------------------------------------------------
#
# 262 of the grid's 1,015 nodes are in water. They are calculated anyway, and
# the coastline's job is not to skip them but to say which cells a finished map
# shades. Interpolating a surface from land nodes alone leaves it stopping up
# to 1.36 miles short of the shore, because a cell the coastline passes through
# has no seaward corner to interpolate from.
#
# Water rows are interpolation support, never design values. These cover the
# classification, the clip box, and the provenance that keeps a database honest
# about which shoreline it was drawn against.


def test_the_old_coarse_nodes_are_still_nodes():
    """Halving the step must not strand values already pulled at 0.05 deg.

    0.05 is exactly twice 0.025 and the bounds are multiples of both, so every
    node of the coarser grid survives. That is what makes a refinement additive
    rather than a re-pull.
    """
    nodes = {
        (round(row.lat, 4), round(row.lon, 4)) for row in BAY_AREA.nodes().itertuples()
    }
    coarse = GridSpec(
        lat_min=BAY_AREA.lat_min,
        lat_max=BAY_AREA.lat_max,
        lon_min=BAY_AREA.lon_min,
        lon_max=BAY_AREA.lon_max,
        step=0.05,
        name="coarse",
        coastline=BAY_AREA.coastline,
    )
    for row in coarse.nodes().itertuples():
        assert (round(row.lat, 4), round(row.lon, 4)) in nodes


def test_the_coastline_does_not_remove_any_node():
    """Regression: it is metadata now, not a filter. Water is calculated too.

    Skipping water cost the map its coastline, which is the one thing the
    heatmap has to get right at the edge of the land.
    """
    without = GridSpec(
        lat_min=BAY_AREA.lat_min,
        lat_max=BAY_AREA.lat_max,
        lon_min=BAY_AREA.lon_min,
        lon_max=BAY_AREA.lon_max,
        step=BAY_AREA.step,
        name="no_coastline",
    )
    assert BAY_AREA.coastline == landmask.DEFAULT_MASK
    assert len(BAY_AREA) == len(without) == 1015
    assert BAY_AREA.nodes().equals(without.nodes())


def test_the_documented_land_and_water_split():
    """The shape of the split, so a re-vendored coastline that moves it shows."""
    nodes = BAY_AREA.nodes()
    land = sum(
        landmask.is_land(float(row.lon), float(row.lat), BAY_AREA.coastline)
        for row in nodes.itertuples()
    )
    assert land == 753
    assert len(nodes) - land == 262


def test_every_water_node_supports_a_land_cell():
    """Water is carried to interpolate from, so each should be near land.

    At this extent every water node is within one cell of land -- which is why
    calculating all of them costs four hours over a collar and is simpler.
    """
    step = BAY_AREA.step
    nodes = {
        (round(row.lat, 4), round(row.lon, 4)) for row in BAY_AREA.nodes().itertuples()
    }
    land = {
        node for node in nodes if landmask.is_land(node[1], node[0], BAY_AREA.coastline)
    }
    far = []
    for lat, lon in nodes - land:
        ring = [
            (round(lat + dlat, 4), round(lon + dlon, 4))
            for dlat in (-step, 0, step)
            for dlon in (-step, 0, step)
            if dlat or dlon
        ]
        if not any(n in land for n in ring):
            far.append((lat, lon))
    assert len(far) <= 100, f"{len(far)} water nodes are not adjacent to land"


def test_every_validation_site_is_on_land_and_on_a_node():
    """The six golden sites are real cities; losing one would gut validation."""
    nodes = {
        (round(row.lat, 4), round(row.lon, 4)) for row in BAY_AREA.nodes().itertuples()
    }
    from frer.sites import load_sites

    sites = load_sites(SITES_CSV)
    inside = sites[
        sites["lat"].between(BAY_AREA.lat_min, BAY_AREA.lat_max)
        & sites["lon"].between(BAY_AREA.lon_min, BAY_AREA.lon_max)
    ]
    assert len(inside) == 6
    for _, site in inside.iterrows():
        assert (round(site["lat"], 4), round(site["lon"], 4)) in nodes, site["name"]


def test_the_refused_point_is_no_longer_in_the_grid():
    """(37.75, -122.80) is in the Gulf of the Farallones and the service refuses it.

    The trimmed extent excludes it. That is the extent doing the work, not the
    coastline, which no longer removes anything.
    """
    assert not landmask.is_land(-122.80, 37.75)
    assert not BAY_AREA.lon_min <= -122.80
    nodes = {
        (round(row.lat, 2), round(row.lon, 2)) for row in BAY_AREA.nodes().itertuples()
    }
    assert (37.75, -122.80) not in nodes


@pytest.mark.parametrize(
    ("lon", "lat", "expected", "what"),
    [
        (-122.40, 37.75, True, "San Francisco"),
        (-121.90, 37.35, True, "San Jose"),
        (-122.80, 37.75, False, "Gulf of the Farallones"),
        (-122.55, 37.75, False, "just off Ocean Beach"),
        (-122.25, 37.60, False, "middle of San Francisco Bay"),
    ],
)
def test_land_and_water_are_told_apart(lon, lat, expected, what):
    assert landmask.is_land(lon, lat) is expected, what


def test_the_bay_reads_as_water_because_holes_are_honoured():
    """Without hole handling the Bay would read as land and 46 nodes would return."""
    assert not landmask.is_land(-122.25, 37.60)
    assert landmask.is_land(-122.40, 37.75)  # the peninsula, just west of it


def test_asking_outside_the_clip_box_raises_rather_than_guessing():
    """The vendored file is clipped; beyond it there is no coastline to answer with."""
    with pytest.raises(landmask.MaskError, match="outside the"):
        landmask.is_land(-100.0, 40.0)


def test_an_unknown_coastline_names_the_script_that_makes_it():
    with pytest.raises(landmask.MaskError, match="vendor_landmask"):
        landmask.is_land(-122.40, 37.75, "no_such_mask")


def test_the_manifest_pins_the_coastline_by_digest():
    """A database must record which shoreline a map of it should be clipped to."""
    manifest = build_manifest(
        BAY_AREA, SITE_CLASSES, VERSION, sample_table(), built_at="now"
    )
    assert manifest["coastline"] == "bay_area_land"
    assert manifest["coastline_sha256"] == landmask.digest("bay_area_land")
    assert len(manifest["coastline_sha256"]) == 64


def test_the_digest_does_not_depend_on_how_the_file_was_checked_out():
    """The same coastline must hash the same on every platform.

    git treats .geojson as text, so with core.autocrlf a Windows working
    copy carries CRLF where the committed blob carries LF. Hashing the raw
    bytes made the digest a property of the checkout: a database built on
    Windows recorded one value and a Linux deployment computed another,
    over a single trailing byte, and refused to draw the map.
    """
    source = landmask.DATA_DIR / "bay_area_land.geojson"
    unix = source.read_bytes().replace(b"\r\n", b"\n")
    windows = unix.replace(b"\n", b"\r\n")
    assert unix != windows, "fixture is not exercising the difference"

    digests = set()
    for variant in (unix, windows):
        with tempfile.TemporaryDirectory() as folder:
            copy = Path(folder) / "checkout.geojson"
            copy.write_bytes(variant)
            original, landmask.DATA_DIR = landmask.DATA_DIR, Path(folder)
            try:
                landmask._load.cache_clear()
                digests.add(landmask.digest("checkout"))
            finally:
                landmask.DATA_DIR = original
                landmask._load.cache_clear()
    assert len(digests) == 1, f"line endings changed the digest: {digests}"
    assert digests == {landmask.digest("bay_area_land")}


def test_a_grid_with_no_coastline_records_none():
    spec = GridSpec(
        lat_min=37.75, lat_max=37.80, lon_min=-122.40, lon_max=-122.35, name="tiny"
    )
    manifest = build_manifest(
        spec, SITE_CLASSES, VERSION, sample_table(), built_at="now"
    )
    assert manifest["coastline"] is None
    assert manifest["coastline_sha256"] is None


def test_preflight_candidates_are_land_nodes_with_an_eastern_neighbour():
    """The last column has no eastern neighbour to compare against."""
    nodes = {
        (round(row.lat, 4), round(row.lon, 4)) for row in BAY_AREA.nodes().itertuples()
    }
    candidates = _preflight_candidates(BAY_AREA)
    assert candidates
    for lat, lon in candidates:
        assert (round(lat, 4), round(lon, 4)) in nodes
        assert (round(lat, 4), round(lon + BAY_AREA.step, 4)) in nodes


# --- what the preflight measures --------------------------------------------
#
# It used to ask whether the midpoint between two nodes came back *identical*
# to one of them, which would mean the service snapped to a grid no finer than
# our step. It does not snap: it interpolates, so that test could never pass at
# any step, and the "reduce --step" it advised on failure could not have helped.
# What it reported as failure was the real change in hazard across half a cell,
# which the two stored endpoints carry and the database does not lose.
#
# The question that does bear on a heatmap is how far interpolating between
# stored nodes lands from the service. That is what is measured now.


def _by_longitude(curves_by_lon):
    """A fetcher returning different committed curves at different longitudes."""

    def fetch_hazard(nshm, lat, lon, vs30):
        return {"response": {"hazardCurves": curves_by_lon[round(lon, 6)]}}

    return fetch_hazard


def _varying(tiny_grid):
    """Three distinct value fields across one cell, from three cached classes."""
    curves = {
        site_class: load_entry(CACHE_DIR, SAMPLE_SITE_ID, site_class)["hazard_curves"][
            NSHM
        ]
        for site_class in ("BC", "C", "D")
    }
    lat, lon = _preflight_candidates(tiny_grid)[0]
    points = {
        "node": round(lon, 6),
        "midpoint": round(lon + tiny_grid.step / 2, 6),
        "neighbour": round(lon + tiny_grid.step, 6),
    }
    fetcher = _by_longitude(
        {
            points["node"]: curves["BC"],
            points["midpoint"]: curves["C"],
            points["neighbour"]: curves["D"],
        }
    )
    return lat, points, fetcher


def test_a_flat_field_interpolates_exactly(tiny_grid, canned_hazard):
    """Identical values either side: nothing for interpolation to get wrong."""
    result = preflight(tiny_grid, version_id=VERSION, fetch_hazard=canned_hazard)
    assert result["checked"] is True
    assert result["interpolation_error"] == 0.0
    assert result["adequate"] is True
    assert result["tolerance"] == INTERPOLATION_TOLERANCE


def test_the_error_is_the_midpoint_against_the_straight_line(tiny_grid):
    """Computed independently here, so the formula cannot drift unnoticed."""
    lat, points, fetcher = _varying(tiny_grid)
    values = {
        label: node_values(lat, lon, "BC", VERSION, fetch_hazard=fetcher)
        for label, lon in points.items()
    }
    expected = max(
        abs(
            values["midpoint"][column]
            - (values["node"][column] + values["neighbour"][column]) / 2
        )
        / ((values["node"][column] + values["neighbour"][column]) / 2)
        for column in VALUE_COLUMNS
    )

    result = preflight(tiny_grid, version_id=VERSION, fetch_hazard=fetcher)
    assert result["interpolation_error"] == pytest.approx(expected, rel=1e-12)


def test_variation_across_a_cell_is_reported_but_does_not_fail_the_check(tiny_grid):
    """Hazard moving across a cell is signal the endpoints carry, not an error."""
    _, _, fetcher = _varying(tiny_grid)
    result = preflight(
        tiny_grid, version_id=VERSION, tolerance=1.0, fetch_hazard=fetcher
    )
    # The field genuinely varies between the three points ...
    assert result["midpoint_vs_node"] > 0.01
    assert result["midpoint_vs_neighbour"] > 0.01
    # ... yet the grid is judged on interpolation error alone.
    assert result["adequate"] is True
    assert result["interpolation_error"] < result["midpoint_vs_node"]


def test_the_tolerance_is_what_decides(tiny_grid):
    """Same measurement, opposite verdicts: the threshold is doing the deciding."""
    _, _, fetcher = _varying(tiny_grid)
    strict = preflight(
        tiny_grid, version_id=VERSION, tolerance=0.0, fetch_hazard=fetcher
    )
    loose = preflight(
        tiny_grid, version_id=VERSION, tolerance=1.0, fetch_hazard=fetcher
    )
    assert strict["adequate"] is False
    assert loose["adequate"] is True
    assert strict["interpolation_error"] == loose["interpolation_error"]
    assert "over the" in strict["verdict"]
    assert "within" in loose["verdict"]


def test_the_old_snapping_criterion_no_longer_decides(tiny_grid):
    """Regression: matching neither neighbour exactly is no longer a failure.

    The old rule failed any field whose midpoint differed from both neighbours,
    which is true of every real location at every step -- it could not be
    passed. Adequacy now turns on interpolation error against the tolerance, so
    the same field passes or fails depending on that and nothing else.

    (The field here is built from three different site classes, which is a far
    bigger jump than real neighbouring nodes ever show, so it needs a loose
    tolerance to pass. That is the point: the threshold decides, not the
    midpoint's failure to coincide with a neighbour.)
    """
    _, _, fetcher = _varying(tiny_grid)
    result = preflight(
        tiny_grid, version_id=VERSION, tolerance=1.0, fetch_hazard=fetcher
    )
    assert result["midpoint_vs_node"] > 1e-9  # the old rule would have failed it
    assert result["midpoint_vs_neighbour"] > 1e-9
    assert result["adequate"] is True  # the new one does not
