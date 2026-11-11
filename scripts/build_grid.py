"""Build the gridded design-value database (deliberate, like update_fixtures).

Calculates FREr and DE design values on a regular grid and writes
``data/grid/<grid>_<version>.csv.gz`` plus a manifest. Every node goes through
the same ``frer.single`` pair the batch workflow and the benchmark use, so this
adds data to a validated pipeline rather than a second implementation of it.

The full Bay Area build is roughly 2,500 hazard calls at 12-15 s each -- about
eleven hours. It appends each node to a ledger before making the next call, so
an interrupted run resumes where it stopped.

    python scripts/build_grid.py --dry-run-from-cache  # no network; prove it first
    python scripts/build_grid.py                       # preflight, then the build
    python scripts/build_grid.py --resume              # continue an interrupted run
    python scripts/build_grid.py --limit 20            # a short live sample

``--dry-run-from-cache`` is the one to run first. It rebuilds the validation
sites from the committed ``tests/usgs_cache`` and compares against the
committed reference values, with no API call, so a wiring error shows up in
seconds instead of eleven hours.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from frer.grid import (
    BAY_AREA,
    VALUE_COLUMNS,
    build_grid,
    build_manifest,
    preflight,
    relative_differences,
    rows_from_cache,
    sort_grid,
    worst_differences,
    write_grid,
    write_manifest,
)
from frer.params import DEFAULT_SITE_CLASSES
from frer.single import rename_legacy_spectral_columns
from frer.sites import load_sites

REPO_ROOT = Path(__file__).resolve().parent.parent
SITES_CSV = REPO_ROOT / "sites.csv"
FROZEN_CACHE = REPO_ROOT / "tests" / "usgs_cache"
REFERENCE_SET = "usgs_160"
DATA_DIR = REPO_ROOT / "data" / "grid"
WORK_DIR = REPO_ROOT / "results" / "grid_cache"

VERSION = "ASCE 7-28"
SITE_CLASSES = list(DEFAULT_SITE_CLASSES)

# The benchmark's tolerance (see tests/test_single.py). The dry run should come
# in far under it -- it recomputes from the same cache the references were made
# from -- so anything approaching this is a wiring error, not rounding.
RTOL = 0.02


def validation_sites(spec) -> pd.DataFrame:
    """Reference-set sites that fall inside the grid and land on its nodes.

    Only exact nodes are useful here: a site between nodes would fold
    interpolation error into a comparison meant to isolate the arithmetic.
    """
    sites = load_sites(SITES_CSV, REFERENCE_SET)
    inside = sites[
        sites["lat"].between(spec.lat_min, spec.lat_max)
        & sites["lon"].between(spec.lon_min, spec.lon_max)
    ]
    on_node = inside[
        inside["lat"].apply(lambda v: any(abs(v - lat) < 1e-9 for lat in spec.lats))
        & inside["lon"].apply(lambda v: any(abs(v - lon) < 1e-9 for lon in spec.lons))
    ]
    return on_node.reset_index(drop=True)


def reference_values(version: str) -> pd.DataFrame:
    """The committed reference rows, with legacy column names normalised."""
    path = REPO_ROOT / "tests" / REFERENCE_SET / version / "frer_design_vals.csv"
    return rename_legacy_spectral_columns(pd.read_csv(path))


def report(differences: pd.DataFrame) -> bool:
    """Print the per-metric difference table; True if everything is in tolerance."""
    summary = worst_differences(differences)
    print(f"\n  {'metric':<18}{'max':>12}{'median':>12}")
    worst = 0.0
    for column, stats in summary.items():
        print(f"  {column:<18}{stats['max']:>11.2e}{stats['median']:>12.2e}")
        worst = max(worst, stats["max"])
    ok = worst <= RTOL
    print(f"\n  worst relative difference {worst:.2e} (tolerance {RTOL})")
    print("  PASS" if ok else "  FAIL")
    return ok


def dry_run(spec) -> int:
    """Rebuild the validation sites from committed cache and check them."""
    sites = validation_sites(spec)
    print(f"Dry run from {FROZEN_CACHE} -- no network.")
    print(f"{len(sites)} validation site(s) on exact grid nodes:")
    for _, site in sites.iterrows():
        print(f"  {site['name']:<20} {site['lat']:>7.2f},{site['lon']:>9.2f}")

    built = rows_from_cache(FROZEN_CACHE, sites, SITE_CLASSES, VERSION)
    if built.empty:
        print("\nNo cached entries found for those sites; nothing to compare.")
        return 1

    # Carry the site name so the comparison joins on name and class, never on
    # the reference CSV's set-local id.
    named = built.merge(sites[["name", "lat", "lon"]], on=["lat", "lon"], how="left")
    expected = reference_values(VERSION)[["name", "site_class", *VALUE_COLUMNS]]

    print(f"\nComparing {len(named)} (site, class) rows against {REFERENCE_SET}.")
    return 0 if report(relative_differences(named, expected)) else 1


def run_build(spec, args) -> int:
    """Preflight, then calculate every node, then write the database."""
    result = None
    if args.no_preflight:
        print("Preflight skipped (--no-preflight).")
    else:
        print(f"Preflight: is a {spec.step} deg step fine enough for the model?")
        result = preflight(spec, version_id=VERSION)
        print(f"  midpoint vs node      {result['midpoint_vs_node']:.3e}")
        print(f"  midpoint vs neighbour {result['midpoint_vs_neighbour']:.3e}")
        print(f"  {result['verdict']}")
        if not result["adequate"]:
            print(
                "\nStopping before the long build. The grid would undersample the "
                "model;\nreduce --step, or re-run with --no-preflight to build "
                "anyway."
            )
            return 1

    table = build_grid(
        spec,
        SITE_CLASSES,
        VERSION,
        work_dir=WORK_DIR,
        pause=args.pause,
        limit=args.limit,
        progress=print,
    )

    expected_rows = len(spec) * len(SITE_CLASSES)
    if args.limit is None and len(table) != expected_rows:
        print(
            f"\n{len(table):,} of {expected_rows:,} rows present; "
            "re-run with --resume to retry the rest. Not writing a partial database."
        )
        return 1

    slug = VERSION.lower().replace(" ", "")
    csv_path = write_grid(table, DATA_DIR / f"{spec.name}_{slug}.csv.gz")
    manifest = build_manifest(
        spec, SITE_CLASSES, VERSION, table, result, repo_root=REPO_ROOT
    )
    manifest_path = write_manifest(manifest, DATA_DIR / f"{spec.name}_{slug}.json")

    print(f"\nWrote {csv_path} ({csv_path.stat().st_size / 1024:.0f} kB)")
    print(f"Wrote {manifest_path}")
    print("\nNow run: python scripts/validate_grid.py")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--dry-run-from-cache",
        action="store_true",
        help="check the builder against committed values, with no network",
    )
    ap.add_argument(
        "--resume", action="store_true", help="continue from the existing ledger"
    )
    ap.add_argument("--limit", type=int, help="stop after this many nodes (sampling)")
    ap.add_argument(
        "--no-preflight", action="store_true", help="skip the grid-spacing check"
    )
    ap.add_argument("--step", type=float, help="override the grid step, in degrees")
    ap.add_argument(
        "--pause", type=float, default=1.0, help="seconds between calls (default 1)"
    )
    args = ap.parse_args(argv)

    spec = (
        BAY_AREA
        if args.step is None
        else type(BAY_AREA)(
            lat_min=BAY_AREA.lat_min,
            lat_max=BAY_AREA.lat_max,
            lon_min=BAY_AREA.lon_min,
            lon_max=BAY_AREA.lon_max,
            step=args.step,
            name=BAY_AREA.name,
        )
    )
    spec.validate()

    if args.dry_run_from_cache:
        return dry_run(spec)

    print(
        f"{spec.name}: {len(spec):,} nodes x {len(SITE_CLASSES)} classes = "
        f"{len(spec) * len(SITE_CLASSES):,} calls for {VERSION}"
    )
    if not args.resume:
        print("(--resume continues an interrupted run from the ledger)")
    return run_build(spec, args)


if __name__ == "__main__":
    sys.exit(main())


# sort_grid is re-exported for the validator, which checks the stored file is
# in the one order write_grid ever produces.
__all__ = ["main", "sort_grid", "validation_sites", "reference_values", "RTOL"]
