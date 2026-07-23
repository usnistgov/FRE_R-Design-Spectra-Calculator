"""Regenerate the benchmark reference (golden) design-value CSVs.

Run this deliberately, only when you intend to update the fixtures -- for
example after a manual, intentional refresh of the frozen input cache
(``tests/usgs_cache/``), or when switching the source of truth from the MATLAB
reference outputs to current Python outputs. For each site set whose frozen
cache is complete, it recomputes ``frer_design_vals.csv`` and writes it into
``tests/<site_set>/<version>/``, then prints a summary of what changed on the
FREr columns so you can review the diff before committing.

It never calls the network and never modifies ``tests/usgs_cache/``; refreshing
that cache is a separate, manual ``frer pull``.

    python scripts/update_fixtures.py                     # all cached sets
    python scripts/update_fixtures.py --site-set nehrp_34 # one set
    python scripts/update_fixtures.py --plots             # also redraw plots
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from frer.cache import covers_site_set
from frer.calc import calc_design_values

REPO_ROOT = Path(__file__).resolve().parent.parent
TESTS_DIR = REPO_ROOT / "tests"
CACHE_DIR = TESTS_DIR / "usgs_cache"
SITES_CSV = REPO_ROOT / "sites.csv"

SITE_SETS = ["frtc_testbed", "nehrp_34", "usgs_160"]
VERSIONS = ["ASCE 7-22", "ASCE 7-28"]

FRER_COLS = ("Sdsfr", "Sd1fr")  # plus any frer_mps_* columns


def _frer_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c in FRER_COLS or c.startswith("frer_mps_")]


def _summarize(
    site_set: str, version: str, old: pd.DataFrame | None, new: pd.DataFrame
):
    """Print a short review of how the FREr values changed."""
    label = f"{site_set} / {version}"
    if old is None:
        print(f"  {label}: created ({len(new)} rows)")
        return

    keys = ["id", "site_class"]
    merged = old.merge(
        new, on=keys, suffixes=("_old", "_new"), how="outer", indicator=True
    )
    added = (merged["_merge"] == "right_only").sum()
    removed = (merged["_merge"] == "left_only").sum()

    both = merged[merged["_merge"] == "both"]
    frer_cols = _frer_columns(new)
    worst_rel, worst_where, changed = 0.0, "", 0
    for _, row in both.iterrows():
        for col in frer_cols:
            o, n = row[f"{col}_old"], row[f"{col}_new"]
            if pd.isna(o) or pd.isna(n):
                continue
            rel = abs(o - n) / abs(o) if o else abs(o - n)
            if rel > 1e-9:
                changed += 1
            if rel > worst_rel:
                where = f"{(int(row['id']), row['site_class'])} {col}"
                worst_rel, worst_where = rel, where

    print(
        f"  {label}: {len(new)} rows | +{added} -{removed} | "
        f"{changed} FREr cell(s) changed | "
        f"max FREr change {worst_rel:.2%}"
        + (f" at {worst_where}" if worst_where else "")
    )


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--site-set", choices=SITE_SETS, help="only this site set")
    ap.add_argument("--version", choices=VERSIONS, help="only this code version")
    ap.add_argument(
        "--plots", action="store_true", help="also regenerate plots (default: CSV only)"
    )
    args = ap.parse_args(argv)

    site_sets = [args.site_set] if args.site_set else SITE_SETS
    versions = [args.version] if args.version else VERSIONS

    wrote_any = False
    print(f"Regenerating fixtures from frozen cache: {CACHE_DIR}")
    for site_set in site_sets:
        if not covers_site_set(CACHE_DIR, SITES_CSV, site_set):
            print(
                f"  {site_set}: SKIP (frozen cache incomplete; "
                f"`frer pull --site-set {site_set} --cache-dir {CACHE_DIR}`)"
            )
            continue
        for version in versions:
            target = TESTS_DIR / site_set / version / "frer_design_vals.csv"
            old = pd.read_csv(target) if target.exists() else None
            # calc writes to out_dir/<site_set>/<version>/frer_design_vals.csv,
            # i.e. straight into the fixture location when out_dir == tests/.
            new = calc_design_values(
                SITES_CSV,
                CACHE_DIR,
                site_set,
                version,
                TESTS_DIR,
                make_plots=args.plots,
            )
            _summarize(site_set, version, old, new)
            wrote_any = True

    if not wrote_any:
        print("Nothing regenerated (no complete frozen cache for the selection).")
        return 1
    print("\nDone. Review the changes with `git diff -- tests/` before committing.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
