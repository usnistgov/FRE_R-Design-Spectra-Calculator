"""Tests for the single-site entry point in ``frer.single``.

``frer.calc.calc_design_values`` delegates to ``calc_site``, so the benchmark in
test_benchmark.py already covers the math through the batch path. These tests
cover the single-site API on its own terms: that it reproduces the committed
reference values for a site, that its input guards reject out-of-model requests
before any network call, and that ``pull_site`` asks for exactly the data a
given code version needs.
"""

import math
from pathlib import Path

import pandas as pd
import pytest

from frer.cache import load_entry
from frer.params import NSHM_BOUNDS, VERSION_NSHM
from frer.single import (
    RP_PERIODS,
    calc_site,
    check_location,
    check_vs30,
    frer_mps_columns,
    new_entry,
    pull_site,
    return_period_columns,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
TESTS_DIR = REPO_ROOT / "tests"
CACHE_DIR = TESTS_DIR / "usgs_cache"

RTOL = 0.02  # matches the benchmark tolerance

# A frtc_testbed site whose frozen cache and reference row are both committed.
# id 168 is "Irvine CA - CFS UCSD", the one testbed site with site class C.
SITE_ID = 168
SITE_CLASS = "C"


def reference_row(version: str) -> pd.Series:
    """The committed reference row for the site under test."""
    csv = TESTS_DIR / "frtc_testbed" / version / "frer_design_vals.csv"
    table = pd.read_csv(csv)
    match = table[table["site_class"] == SITE_CLASS]
    assert len(match) == 1, f"expected exactly one class {SITE_CLASS} testbed row"
    return match.iloc[0]


@pytest.mark.parametrize("version", sorted(VERSION_NSHM))
def test_calc_site_matches_reference(version):
    """calc_site on a frozen cache entry reproduces the committed FREr values."""
    entry = load_entry(CACHE_DIR, SITE_ID, SITE_CLASS)
    if entry is None:
        pytest.skip(f"no frozen cache for site {SITE_ID} class {SITE_CLASS}")
    expected = reference_row(version)

    result = calc_site(entry, version)

    assert result.sdsfr == pytest.approx(expected["Sdsfr"], rel=RTOL)
    assert result.sd1fr == pytest.approx(expected["Sd1fr"], rel=RTOL)

    # The full multi-period spectrum, via the same helper the CSV writer uses.
    for column, value in frer_mps_columns(result.fr_spectrum).items():
        assert value == pytest.approx(expected[column], rel=RTOL), column


def test_calc_site_carries_site_metadata():
    """The result describes the site it came from, for labelling and export."""
    entry = load_entry(CACHE_DIR, SITE_ID, SITE_CLASS)
    if entry is None:
        pytest.skip(f"no frozen cache for site {SITE_ID} class {SITE_CLASS}")

    result = calc_site(entry, "ASCE 7-22")

    assert result.site_class == SITE_CLASS
    assert result.version_id == "ASCE 7-22"
    assert result.nshm == VERSION_NSHM["ASCE 7-22"]
    assert result.lat == entry["lat"]
    assert result.lon == entry["lon"]
    # DE and FREr spectra are both usable for plotting.
    assert len(result.de_period) == len(result.de_sa)
    assert len(result.fr_spectrum.period) == len(result.fr_spectrum.sa)


def test_return_period_columns_asce728():
    """ASCE 7-28 exposes FRE_R/MCE_R/DE return periods at 0.2 s and 1.0 s."""
    entry = load_entry(CACHE_DIR, SITE_ID, SITE_CLASS)
    if entry is None:
        pytest.skip(f"no frozen cache for site {SITE_ID} class {SITE_CLASS}")

    result = calc_site(entry, "ASCE 7-28")
    cols = return_period_columns(result)

    expected_keys = {
        f"{gm}_rp_{f'{t:.1f}'.replace('.', 'p')}"
        for gm in ("frer", "mcer", "de")
        for t in RP_PERIODS
    }
    assert set(cols) == expected_keys
    assert all(math.isfinite(v) and v > 0 for v in cols.values())
    # FRE_R targets a more frequent (shorter return period) motion than MCE_R,
    # and DE is 2/3 of MCE_R, so at each period frer < de < mcer.
    for t in RP_PERIODS:
        s = f"{t:.1f}".replace(".", "p")
        assert cols[f"frer_rp_{s}"] < cols[f"de_rp_{s}"] < cols[f"mcer_rp_{s}"]


def test_return_period_columns_absent_for_asce722():
    """ASCE 7-22 has no hazard-curve return period, so no columns are added."""
    entry = load_entry(CACHE_DIR, SITE_ID, SITE_CLASS)
    if entry is None:
        pytest.skip(f"no frozen cache for site {SITE_ID} class {SITE_CLASS}")

    result = calc_site(entry, "ASCE 7-22")

    assert result.ls_spectrum is None
    assert return_period_columns(result) == {}


def test_calc_site_rejects_unknown_version():
    entry = new_entry(34.05, -118.25, "D", 260)
    with pytest.raises(ValueError, match="not set up for code version"):
        calc_site(entry, "ASCE 7-16")


@pytest.mark.parametrize(
    "lat,lon",
    [
        (21.30, -157.85),  # Honolulu: outside the conterminous models
        (61.22, -149.90),  # Anchorage: ditto
        (34.05, -130.00),  # west of the model's western edge
    ],
)
def test_check_location_rejects_outside_conus(lat, lon):
    with pytest.raises(ValueError, match="outside the conterminous"):
        check_location(lat, lon)


def test_check_location_accepts_inside_conus():
    lon_min, lat_min, lon_max, lat_max = NSHM_BOUNDS
    check_location(34.05, -118.25)  # Los Angeles
    check_location(lat_min, lon_min)  # corners are inclusive
    check_location(lat_max, lon_max)


@pytest.mark.parametrize("vs30", [0, 149, 1501, 3000])
def test_check_vs30_rejects_out_of_range(vs30):
    with pytest.raises(ValueError, match="vs30 must be between"):
        check_vs30(vs30)


def test_pull_site_validates_before_fetching():
    """An out-of-model location fails without making any web-service call."""
    calls = []

    def spy(*args, **kwargs):
        calls.append(args)
        raise AssertionError("should not have been called")

    with pytest.raises(ValueError, match="outside the conterminous"):
        pull_site(
            21.30,
            -157.85,
            "D",
            260,
            "ASCE 7-22",
            fetch_design=spy,
            fetch_hazard=spy,
        )
    assert calls == []


@pytest.mark.parametrize(
    "version,wants_design",
    [("ASCE 7-22", True), ("ASCE 7-28", False)],
)
def test_pull_site_requests_only_what_the_version_needs(version, wants_design):
    """7-22 needs the design-value service; 7-28 is fully probabilistic."""
    entry_curves = [{"imt": {"value": "PGA", "display": "PGA"}}]
    design_calls, hazard_calls = [], []

    def fetch_design(lat, lon, site_class, risk_category):
        design_calls.append((lat, lon, site_class, risk_category))
        return {"response": {"data": {"sds": 1.0, "sd1": 0.5}}}

    def fetch_hazard(nshm, lat, lon, vs30):
        hazard_calls.append((nshm, lat, lon, vs30))
        return {"response": {"hazardCurves": entry_curves}}

    entry = pull_site(
        34.05,
        -118.25,
        "D",
        260,
        version,
        risk_category="III",
        fetch_design=fetch_design,
        fetch_hazard=fetch_hazard,
    )

    nshm = VERSION_NSHM[version]
    assert hazard_calls == [(nshm, 34.05, -118.25, 260)]
    assert entry["hazard_curves"][str(nshm)] == entry_curves
    if wants_design:
        assert design_calls == [(34.05, -118.25, "D", "III")]
        assert entry["design_values"] is not None
    else:
        assert design_calls == []
        assert entry["design_values"] is None
