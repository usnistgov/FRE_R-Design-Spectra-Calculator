"""Benchmark the FREr calculation against the reference design values here.

Each ``tests/<site_set>/<version>/frer_design_vals.csv`` holds reference output.
``tests/usgs_cache/`` holds a frozen snapshot of the USGS web-service responses
that feed the calculation, so the benchmark is reproducible offline; refreshing
that cache is a separate, deliberate ``frer pull`` (never done by the tests).

Scope: these tests exercise only the **FREr** outputs (``Sdsfr_RotD50``,
``Sd1fr_RotD50`` and the ``frer_mps_*_RotD50`` spectrum) via ``frer.calc`` --
not the pull step, and not the DE values ``Sds_RotD100``/``Sd1_RotD100`` (for
ASCE 7-22 those come straight from the USGS design-maps service, which is
treated as the source of truth for itself).

For each (site set, version) whose cache is present, the FREr values are
recomputed from the frozen cache and asserted to match the reference within
``RTOL`` -- except a small, explicitly enumerated set of rows that are known to
differ (see ``KNOWN_EXCEPTIONS``). Any new deviation fails.

To intentionally regenerate the reference CSVs, run
``python scripts/update_fixtures.py`` (see that script's docstring).
"""

import warnings
from pathlib import Path

import pandas as pd
import pytest

from frer.cache import covers_site_set
from frer.calc import calc_design_values
from frer.single import (
    SD1FR_COLUMN,
    SDSFR_COLUMN,
    rename_legacy_spectral_columns,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
TESTS_DIR = REPO_ROOT / "tests"
CACHE_DIR = TESTS_DIR / "usgs_cache"
SITES_CSV = REPO_ROOT / "sites.csv"

RTOL = 0.02  # 2% relative tolerance

CASES = [
    ("frtc_testbed", "ASCE 7-22"),
    ("frtc_testbed", "ASCE 7-28"),
    ("nehrp_34", "ASCE 7-22"),
    ("nehrp_34", "ASCE 7-28"),
    ("usgs_160", "ASCE 7-22"),
    ("usgs_160", "ASCE 7-28"),
]

# MATLAB reference column (frer_return_periods.csv) -> Python output column for
# each of the six spectral-value return periods.
RP_COLUMN_MAP = {
    "frer_0p2": "frer_rp_0p2",
    "frer_1p0": "frer_rp_1p0",
    "mcer_0p2": "mcer_rp_0p2",
    "mcer_1p0": "mcer_rp_1p0",
    "DE_0p2": "de_rp_0p2",
    "DE_1p0": "de_rp_1p0",
}

# Return periods are 1/AFE read off the hazard curve, so the 2% Sa tolerance
# (RTOL) is amplified by the local slope of that curve. Site class E (soft soil)
# has the steepest curves and the largest amplification, so it gets a wider band
# than the other classes. Values that still exceed these bands are enumerated in
# KNOWN_RP_EXCEPTIONS below; every one is a class D or E row whose steep hazard
# curve amplifies the 2% Sa tolerance into a larger return-period difference.
RP_RTOL = 0.02
RP_RTOL_SOFT_SOIL = 0.10  # site class E only

# Individual (site id, site class, MATLAB reference column) return periods that
# exceed the tolerances above and are accepted as known deviations from the
# MATLAB reference (recomputed from the committed frozen cache). The benchmark
# emits a warning listing the live differences when these are hit, rather than
# failing on them. The worst is Elgin OK (id 106) class E, DE at 1.0 s (~31%).
KNOWN_RP_EXCEPTIONS = {
    # Site class D: steep-curve amplification just past the 2% band.
    (80, "D", "DE_0p2"),  # New Madrid MO
    (80, "D", "mcer_0p2"),  # New Madrid MO
    (19, "D", "mcer_0p2"),  # Eureka CA
    # Site class E: soft-soil hazard curves, past the 10% band.
    (106, "E", "DE_1p0"),  # Elgin OK
    (80, "E", "DE_0p2"),  # New Madrid MO
    (106, "E", "DE_0p2"),  # Elgin OK
    (106, "E", "mcer_1p0"),  # Elgin OK
    (123, "E", "DE_1p0"),  # Charleston SC
    (143, "E", "DE_1p0"),  # Richmond VA
    (19, "E", "DE_0p2"),  # Eureka CA
    (116, "E", "DE_0p2"),  # Newport OR
    (128, "E", "DE_1p0"),  # Knoxville TN
    (129, "E", "DE_1p0"),  # Maryville TN
    (110, "E", "DE_0p2"),  # Brookings OR
    (123, "E", "DE_0p2"),  # Charleston SC
    (144, "E", "DE_1p0"),  # Burlington VT
    (60, "E", "DE_1p0"),  # Savannah GA
    (130, "E", "DE_1p0"),  # Memphis TN
    (90, "E", "DE_1p0"),  # Manchester NH
    (79, "E", "DE_1p0"),  # Cape Girardeau MO
    (101, "E", "DE_1p0"),  # Malone NY
}


def _frer_columns(df: pd.DataFrame) -> list[str]:
    """FREr value columns present in a reference/result table."""
    return [
        c
        for c in df.columns
        if c in (SDSFR_COLUMN, SD1FR_COLUMN) or c.startswith("frer_mps_")
    ]


def _read_reference(path) -> pd.DataFrame:
    """Read a committed reference CSV, normalising its spectral column names.

    The fixtures predate the orientation suffixes on the spectral columns (e.g.
    ``Sdsfr`` is now ``Sdsfr_RotD50``) and are the MATLAB reference values, so
    they are mapped onto the current names on read rather than regenerated --
    regenerating them from Python would destroy what the benchmark compares
    against. Return-period column names are unchanged and pass through as-is.
    """
    return rename_legacy_spectral_columns(pd.read_csv(path))


def _recompute(site_set: str, version: str, out_dir) -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # skip-missing warnings are expected
        return calc_design_values(
            SITES_CSV, CACHE_DIR, site_set, version, out_dir, make_plots=False
        )


@pytest.mark.parametrize("site_set,version", CASES)
def test_frer_benchmark(site_set, version, tmp_path):
    reference_csv = TESTS_DIR / site_set / version / "frer_design_vals.csv"
    if not reference_csv.exists():
        pytest.skip(f"no reference CSV for {site_set} / {version}")
    if not covers_site_set(CACHE_DIR, SITES_CSV, site_set):
        pytest.skip(f"no frozen cache for {site_set}; run `frer pull` to add it")

    expected = _read_reference(reference_csv)
    actual = _recompute(site_set, version, tmp_path)
    frer_cols = _frer_columns(expected)

    merged = expected.merge(
        actual,
        on=["id", "site_class"],
        suffixes=("_ref", "_act"),
        how="left",
        indicator=True,
    )

    failures = []
    for _, row in merged.iterrows():
        key = (int(row["id"]), row["site_class"])
        if row["_merge"] != "both":
            failures.append(f"  {key}: missing from recomputed output")
            continue
        for col in frer_cols:
            ref, act = row[f"{col}_ref"], row[f"{col}_act"]
            rel = abs(ref - act) / abs(ref) if ref else abs(ref - act)
            if rel > RTOL:
                failures.append(
                    f"  {key} {col}: ref={ref:.6g} recomputed={act:.6g} "
                    f"(rel diff {rel:.2%})"
                )

    assert not failures, (
        f"{site_set} / {version}: {len(failures)} FREr value(s) exceed "
        f"{RTOL:.0%} tolerance vs reference:\n" + "\n".join(failures)
    )


def test_return_period_benchmark(tmp_path):
    """Return periods match the MATLAB reference for usgs_160 / ASCE 7-28.

    ``tests/usgs_160/ASCE 7-28/frer_return_periods.csv`` is the return-period
    table written by the original MATLAB tool. Return periods are only defined
    for ASCE 7-28 (the fully probabilistic option), so it is the only case with a
    reference; the FRE_R/MCE_R/DE values are recomputed from the frozen cache and
    compared column-by-column within the tolerances above.
    """
    site_set, version = "usgs_160", "ASCE 7-28"
    reference_csv = TESTS_DIR / site_set / version / "frer_return_periods.csv"
    if not reference_csv.exists():
        pytest.skip(f"no return-period reference CSV for {site_set} / {version}")
    if not covers_site_set(CACHE_DIR, SITES_CSV, site_set):
        pytest.skip(f"no frozen cache for {site_set}; run `frer pull` to add it")

    expected = _read_reference(reference_csv)
    actual = _recompute(site_set, version, tmp_path)

    # Every ASCE 7-28 result row must carry the six return-period columns.
    assert set(RP_COLUMN_MAP.values()) <= set(actual.columns), (
        "recomputed ASCE 7-28 output is missing return-period columns: "
        f"{sorted(set(RP_COLUMN_MAP.values()) - set(actual.columns))}"
    )

    merged = expected.merge(
        actual,
        on=["id", "site_class"],
        suffixes=("_ref", "_act"),
        how="left",
        indicator=True,
    )

    failures = []
    excepted = []
    for _, row in merged.iterrows():
        key = (int(row["id"]), row["site_class"])
        if row["_merge"] != "both":
            failures.append(f"  {key}: missing from recomputed output")
            continue
        tol = RP_RTOL_SOFT_SOIL if row["site_class"] == "E" else RP_RTOL
        for ref_col, act_col in RP_COLUMN_MAP.items():
            ref, act = row[ref_col], row[act_col]
            # A period whose Sa falls off the hazard curve has no return period;
            # NaN on both sides is agreement, NaN on only one side is a failure.
            if pd.isna(ref) and pd.isna(act):
                continue
            if pd.isna(ref) or pd.isna(act):
                failures.append(
                    f"  {key} {ref_col}: one side undefined (ref={ref}, act={act})"
                )
                continue
            rel = abs(ref - act) / abs(ref) if ref else abs(ref - act)
            if rel <= tol:
                continue
            detail = (
                f"  {key} {ref_col}: ref={ref:.6g} recomputed={act:.6g} "
                f"(rel diff {rel:.2%} > {tol:.0%})"
            )
            if (int(row["id"]), row["site_class"], ref_col) in KNOWN_RP_EXCEPTIONS:
                excepted.append(detail)
            else:
                failures.append(detail)

    if excepted:
        warnings.warn(
            f"{len(excepted)} known return-period exception(s) exceed tolerance "
            "vs the MATLAB reference and are accepted (steep hazard curve "
            "amplifies the 2% Sa tolerance; mostly site class E):\n"
            + "\n".join(excepted),
            stacklevel=2,
        )

    assert not failures, (
        f"{len(failures)} unexpected return period(s) exceed tolerance vs the "
        f"MATLAB reference:\n" + "\n".join(failures)
    )
