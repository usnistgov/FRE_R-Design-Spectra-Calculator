"""Benchmark the FREr calculation against the reference design values here.

Each ``tests/<site_set>/<version>/frer_design_vals.csv`` holds reference output.
``tests/usgs_cache/`` holds a frozen snapshot of the USGS web-service responses
that feed the calculation, so the benchmark is reproducible offline; refreshing
that cache is a separate, deliberate ``frer pull`` (never done by the tests).

Scope: these tests exercise only the **FREr** outputs (``Sdsfr``, ``Sd1fr`` and
the ``frer_mps_*`` spectrum) via ``frer.calc`` -- not the pull step, and not the
DE values ``Sds``/``Sd1`` (for ASCE 7-22 those come straight from the USGS
design-maps service, which is treated as the source of truth for itself).

For each (site set, version) whose cache is present, the FREr values are
recomputed from the frozen cache and asserted to match the reference within
``RTOL`` -- except a small, explicitly enumerated set of rows that are known to
differ (see ``KNOWN_EXCEPTIONS`` and VALIDATION.md). Any new deviation fails.

To intentionally regenerate the reference CSVs, run
``python scripts/update_fixtures.py`` (see that script's docstring).
"""

import warnings
from pathlib import Path

import pandas as pd
import pytest

from frer.cache import covers_site_set
from frer.calc import calc_design_values

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

# (site_set, version) -> {(id, site_class): reason}. These reference rows differ
# from a recompute of the FREr values and are excluded from the tolerance gate;
# every reason is a USGS-input change since the reference, not a calc bug (see
# VALIDATION.md). Everything else must match within RTOL.
KNOWN_EXCEPTIONS = {
    ("nehrp_34", "ASCE 7-22"): {
        (28, "D"): "USGS conus-2018 hazard service errors for Reno NV vs30=265, "
        "so no 2018 curve to compute FREr from",
    },
    ("nehrp_34", "ASCE 7-28"): {
        (sid, "E"): "NSHM 2023 provisional soft-soil (class E) CEUS amplification "
        "revised since reference"
        for sid in (30, 31, 32, 33, 34)
    },
}


def _frer_columns(df: pd.DataFrame) -> list[str]:
    """FREr value columns present in a reference/result table."""
    return [
        c for c in df.columns if c in ("Sdsfr", "Sd1fr") or c.startswith("frer_mps_")
    ]


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

    expected = pd.read_csv(reference_csv)
    actual = _recompute(site_set, version, tmp_path)
    frer_cols = _frer_columns(expected)

    merged = expected.merge(
        actual,
        on=["id", "site_class"],
        suffixes=("_ref", "_act"),
        how="left",
        indicator=True,
    )
    exceptions = KNOWN_EXCEPTIONS.get((site_set, version), {})

    failures = []
    for _, row in merged.iterrows():
        key = (int(row["id"]), row["site_class"])
        if key in exceptions:
            continue
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


def test_known_exceptions_are_real():
    """Each enumerated exception must actually still be an exception.

    Guards against the allowlist going stale: if a USGS input changes back, the
    corresponding row would silently pass and should be removed from
    KNOWN_EXCEPTIONS. Verified for site sets whose cache is committed.
    """
    import tempfile

    for (site_set, version), rows in KNOWN_EXCEPTIONS.items():
        if not covers_site_set(CACHE_DIR, SITES_CSV, site_set):
            continue
        reference = pd.read_csv(TESTS_DIR / site_set / version / "frer_design_vals.csv")
        frer_cols = _frer_columns(reference)
        with tempfile.TemporaryDirectory() as td:
            actual = _recompute(site_set, version, td)
        merged = reference.merge(
            actual,
            on=["id", "site_class"],
            suffixes=("_ref", "_act"),
            how="left",
            indicator=True,
        )
        for (sid, sc), reason in rows.items():
            match = merged[(merged["id"] == sid) & (merged["site_class"] == sc)]
            assert len(match), f"exception {(sid, sc)} not found in reference"
            row = match.iloc[0]
            if row["_merge"] != "both":
                continue  # a legitimately-missing row (e.g. Reno D)
            exceeds = any(
                abs(row[f"{c}_ref"] - row[f"{c}_act"]) / abs(row[f"{c}_ref"]) > RTOL
                for c in frer_cols
            )
            assert exceeds, (
                f"{(site_set, version, sid, sc)} is listed in KNOWN_EXCEPTIONS "
                f"but its FREr values now match within {RTOL:.0%} -- remove it. "
                f"({reason})"
            )
