"""Validate the gridded design-value database.

Run this after a build, and any time you want to satisfy yourself the database
still means what it claims. It exits non-zero on the first failing check, so it
is usable from a shell or a hook.

    python scripts/validate_grid.py            # offline
    python scripts/validate_grid.py --network  # also re-pull a few nodes live

It checks five things:

1. The committed input cache under ``tests/usgs_cache`` is unmodified. That
   tree is the benchmark's frozen input; nothing in this project should ever
   write to it, and git is the tripwire.
2. The database is internally sound -- expected rows, no gaps, no NaNs, no
   negative values, every node inside the declared bounds, ratios in a
   plausible range.
3. Its manifest matches the code and the grid that produced it, so a database
   built by older arithmetic is detectable rather than quietly wrong.
4. The six validation sites, recomputed from the committed cache, agree with
   both the committed reference values and the database itself. They land on
   exact grid nodes, so this comparison carries no interpolation error.
5. With --network, a sample of nodes re-pulled from USGS still matches.

To convince yourself it can fail, change a value in the database and run it
again, then restore it with git checkout. A validator nobody has watched
fail is just a script that prints PASS.
"""

import argparse
import random
import subprocess
import sys
from pathlib import Path

import pandas as pd

from frer.grid import (
    BAY_AREA,
    COORD_DECIMALS,
    GRID_COLUMNS,
    RATIO_COLUMNS,
    VALUE_COLUMNS,
    add_ratios,
    load_grid,
    load_manifest,
    node_values,
    relative_differences,
    rows_from_cache,
    worst_differences,
)
from frer.params import DEFAULT_SITE_CLASSES
from frer.single import rename_legacy_spectral_columns
from frer.sites import load_sites

REPO_ROOT = Path(__file__).resolve().parent.parent
TESTS_DIR = REPO_ROOT / "tests"
CACHE_DIR = TESTS_DIR / "usgs_cache"
SITES_CSV = REPO_ROOT / "sites.csv"
DATA_DIR = REPO_ROOT / "data" / "grid"
REFERENCE_SET = "usgs_160"

VERSION = "ASCE 7-28"
SITE_CLASSES = list(DEFAULT_SITE_CLASSES)
RTOL = 0.02

# FREr is below DE at every site in the committed references; the app's own
# deltas run about -25% to -66%. A ratio at or above 1 is not necessarily
# wrong, but it is not something to discover from a map legend.
RATIO_RANGE = (0.1, 1.0)

SPOT_CHECK_NODES = 5


class ValidationError(Exception):
    """A check that did not pass."""


def ok(message: str) -> None:
    print(f"  PASS  {message}")


def section(title: str) -> None:
    print(f"\n{title}")


def check_cache_untouched() -> None:
    """The frozen USGS responses must be exactly as committed."""
    section("1. Committed input cache")
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(REPO_ROOT),
                "status",
                "--porcelain",
                "--",
                str(CACHE_DIR),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValidationError(
            f"could not ask git about {CACHE_DIR}: {error}"
        ) from error

    changed = [line for line in result.stdout.splitlines() if line.strip()]
    if changed:
        listing = "\n    ".join(changed[:10])
        more = f"\n    ... and {len(changed) - 10} more" if len(changed) > 10 else ""
        raise ValidationError(
            f"{len(changed)} file(s) under {CACHE_DIR.name} differ from the "
            f"commit:\n    {listing}{more}\n  That tree is committed input and "
            "should never be written to."
        )
    ok(f"{CACHE_DIR.relative_to(REPO_ROOT)} matches the commit")


def check_database(table: pd.DataFrame, spec) -> None:
    """Shape, coverage and plausibility of the stored values."""
    section("2. Database integrity")

    if list(table.columns) != GRID_COLUMNS:
        raise ValidationError(
            f"columns are {list(table.columns)}, expected {GRID_COLUMNS}"
        )

    expected = len(spec) * len(SITE_CLASSES)
    if len(table) != expected:
        raise ValidationError(f"{len(table):,} rows, expected {expected:,}")
    ok(f"{len(table):,} rows = {len(spec):,} nodes x {len(SITE_CLASSES)} classes")

    duplicates = table.duplicated(subset=["lat", "lon", "site_class"]).sum()
    if duplicates:
        raise ValidationError(f"{duplicates} duplicated (node, class) row(s)")

    # Compared against the grid's own nodes rather than its bounds. That is the
    # stronger statement -- a row has to sit *on* the graticule, not merely
    # inside the box -- and it sidesteps the float edge where a bound computed
    # as lon_min + step lands a hair beyond the node of the same name.
    def node_set(frame):
        return {
            (round(float(lat), COORD_DECIMALS), round(float(lon), COORD_DECIMALS))
            for lat, lon in frame[["lat", "lon"]].drop_duplicates().to_numpy()
        }

    expected_nodes = node_set(spec.nodes())
    actual_nodes = node_set(table)
    missing = expected_nodes - actual_nodes
    extra = actual_nodes - expected_nodes
    if missing or extra:
        detail = []
        if missing:
            detail.append(f"{len(missing)} node(s) absent, e.g. {sorted(missing)[0]}")
        if extra:
            detail.append(f"{len(extra)} row(s) off the grid, e.g. {sorted(extra)[0]}")
        raise ValidationError("; ".join(detail))
    ok(f"{len(actual_nodes):,} nodes, each on the grid, none missing")

    if table[VALUE_COLUMNS].isna().any().any():
        raise ValidationError("NaN values present")
    if (table[VALUE_COLUMNS] <= 0).any().any():
        raise ValidationError("non-positive design values present")
    ok("no NaNs, no non-positive values")

    ratios = add_ratios(table)
    low, high = RATIO_RANGE
    for name in RATIO_COLUMNS:
        series = ratios[name]
        if series.isna().any():
            raise ValidationError(f"{name} has NaN values")
        if not series.between(low, high).all():
            worst = series.min() if series.min() < low else series.max()
            raise ValidationError(
                f"{name} ranges {series.min():.3f}..{series.max():.3f}, "
                f"outside the expected {low}..{high} (worst {worst:.3f})"
            )
        ok(f"{name} ranges {series.min():.3f}..{series.max():.3f}")


def check_manifest(manifest: dict, table: pd.DataFrame, spec) -> None:
    """The database still matches the code and grid it claims to come from."""
    section("3. Provenance")
    from frer import __version__

    if manifest["version"] != VERSION:
        raise ValidationError(
            f"manifest says {manifest['version']}, expected {VERSION}"
        )
    if manifest["rows"] != len(table):
        raise ValidationError(
            f"manifest says {manifest['rows']} rows, file has {len(table)}"
        )

    grid = manifest["grid"]
    for field in ("lat_min", "lat_max", "lon_min", "lon_max", "step"):
        if abs(grid[field] - getattr(spec, field)) > 1e-9:
            raise ValidationError(
                f"manifest grid {field}={grid[field]} but the code says "
                f"{getattr(spec, field)}"
            )
    ok(f"grid, version and row count agree ({grid['nodes']:,} nodes, {VERSION})")

    if manifest["calculator_version"] != __version__:
        raise ValidationError(
            f"built by calculator {manifest['calculator_version']}, "
            f"this is {__version__}; rebuild before trusting it"
        )
    ok(f"built by calculator {manifest['calculator_version']} (current)")

    for site_class, vs30 in manifest["site_classes"].items():
        if DEFAULT_SITE_CLASSES[site_class] != vs30:
            raise ValidationError(
                f"class {site_class} built at vs30 {vs30}, "
                f"code uses {DEFAULT_SITE_CLASSES[site_class]}"
            )
    ok("site classes built at the vs30 values the fixtures depend on")

    preflight = manifest.get("preflight", {})
    if not preflight.get("checked"):
        print("  NOTE  built with --no-preflight; grid spacing was not verified")
    elif not preflight.get("adequate"):
        print(f"  NOTE  preflight flagged the spacing: {preflight.get('verdict')}")
    else:
        ok(f"preflight: {preflight.get('verdict')}")


def golden_sites(spec) -> pd.DataFrame:
    sites = load_sites(SITES_CSV, REFERENCE_SET)
    inside = sites[
        sites["lat"].between(spec.lat_min, spec.lat_max)
        & sites["lon"].between(spec.lon_min, spec.lon_max)
    ]
    return inside[
        inside["lat"].apply(lambda v: any(abs(v - x) < 1e-9 for x in spec.lats))
        & inside["lon"].apply(lambda v: any(abs(v - x) < 1e-9 for x in spec.lons))
    ].reset_index(drop=True)


def report(differences: pd.DataFrame, label: str) -> None:
    """Print max and median per metric, and fail if anything is out of tolerance."""
    summary = worst_differences(differences)
    print(f"  {label}")
    print(f"    {'metric':<18}{'max':>12}{'median':>12}")
    worst, worst_column = 0.0, ""
    for column, stats in summary.items():
        print(f"    {column:<18}{stats['max']:>11.2e}{stats['median']:>12.2e}")
        if stats["max"] > worst:
            worst, worst_column = stats["max"], column
    if worst > RTOL:
        raise ValidationError(
            f"{label}: {worst_column} differs by {worst:.2%}, over the {RTOL:.0%} "
            "tolerance"
        )


def check_golden(table: pd.DataFrame, spec) -> None:
    """Recomputed sites must match both the references and the database."""
    section("4. Validation sites (exact grid nodes)")
    sites = golden_sites(spec)
    if sites.empty:
        raise ValidationError("no reference sites land on grid nodes; nothing to check")
    print(f"  {len(sites)} site(s): {', '.join(sites['name'])}")

    rebuilt = rows_from_cache(CACHE_DIR, sites, SITE_CLASSES, VERSION)
    named = rebuilt.merge(sites[["name", "lat", "lon"]], on=["lat", "lon"], how="left")

    # Joined on name and class, never on the reference CSV's set-local id.
    reference = rename_legacy_spectral_columns(
        pd.read_csv(TESTS_DIR / REFERENCE_SET / VERSION / "frer_design_vals.csv")
    )[["name", "site_class", *VALUE_COLUMNS]]
    report(relative_differences(named, reference), "recomputed vs committed reference")

    from_db = table.merge(sites[["name", "lat", "lon"]], on=["lat", "lon"], how="inner")
    if len(from_db) != len(named):
        raise ValidationError(
            f"database holds {len(from_db)} of the {len(named)} validation rows"
        )
    report(relative_differences(from_db, reference), "database vs committed reference")


def check_live(table: pd.DataFrame, count: int, seed: int | None) -> None:
    """Re-pull a sample of nodes and compare to what is stored."""
    section(f"5. Live spot check ({count} nodes)")
    sample = table.sample(count, random_state=seed)
    rows = []
    for row in sample.itertuples():
        print(f"  pulling {row.lat:.4f},{row.lon:.4f} [{row.site_class}] ...")
        rows.append(node_values(row.lat, row.lon, row.site_class, VERSION))
    fresh = pd.DataFrame(rows, columns=GRID_COLUMNS)
    report(
        relative_differences(fresh, sample, on=("lat", "lon", "site_class")),
        "freshly pulled vs stored",
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--network", action="store_true", help="also re-pull a sample of nodes"
    )
    ap.add_argument(
        "--nodes", type=int, default=SPOT_CHECK_NODES, help="how many to re-pull"
    )
    ap.add_argument("--seed", type=int, help="make the sample reproducible")
    args = ap.parse_args(argv)

    spec = BAY_AREA
    slug = VERSION.lower().replace(" ", "")
    csv_path = DATA_DIR / f"{spec.name}_{slug}.csv.gz"
    manifest_path = DATA_DIR / f"{spec.name}_{slug}.json"

    print(f"Validating {csv_path.relative_to(REPO_ROOT)}")
    try:
        check_cache_untouched()

        if not csv_path.exists():
            raise ValidationError(
                f"{csv_path} does not exist yet. Build it with "
                "`python scripts/build_grid.py`, or check the cache alone "
                "with the check above."
            )
        table = load_grid(csv_path)
        manifest = load_manifest(manifest_path)

        check_database(table, spec)
        check_manifest(manifest, table, spec)
        check_golden(table, spec)
        if args.network:
            random.seed(args.seed)
            check_live(table, args.nodes, args.seed)
    except ValidationError as error:
        print(f"\n  FAIL  {error}")
        print("\nValidation failed.")
        return 1

    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
