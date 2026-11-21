"""Build the frozen test cache from a local USGS hazard-curve data release.

The cache under ``tests/`` is the benchmark's frozen input. It used to be
assembled by ``frer pull``, one HTTP call per (site, site class) -- 640 calls
and most of a day -- and what came back depended on whichever model revision
the service's bare-year path happened to point at that week.

The published data release is a better source for it. It is a fixed revision
by construction (CONUS.2023.R1 is what ``conus-2023.R1`` serves), it carries
every usgs_160 site because all 160 are NSHM test sites, and it covers eight
Vs30 values rather than the four a pull bothered with. Building from it takes
seconds, is exactly reproducible, and produces a cache around 8 MB per four
site classes instead of 344 MB, because the release holds the total curve
rather than the service's full per-magnitude breakdown.

    python scripts/build_cache_from_release.py
    python scripts/build_cache_from_release.py --classes BC C D E

Entries are written in the same shape :func:`frer.cache.new_entry` produces, so
nothing downstream can tell a release-built entry from a pulled one. That is
the point: the API path stays the reference, and
``tests/test_api_spot_check.py`` re-pulls a sample to prove the two agree.

Writes into ``results/`` by default. Adopting a built cache as the frozen one
is a deliberate, separate step -- see ``--out``.
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

from frer import local_hazard
from frer.cache import cache_path
from frer.params import DEFAULT_SITE_CLASSES, NSHM_EDITION, SITE_CLASS_VS30
from frer.sites import SITE_SET, load_sites

REPO_ROOT = Path(__file__).resolve().parent.parent
RELEASE_DIR = REPO_ROOT / "data" / "usgs hazard curve" / "conus_2023.R1-MPRS-test_sites"
SITES_CSV = REPO_ROOT / "sites.csv"
OUT_DIR = REPO_ROOT / "results" / "usgs_cache_r1_local"

NSHM = 2023

# Site classes to build: the ones the batch workflow calculates. The release
# covers every NEHRP class, so this costs only disk -- unlike a pull, where
# each class was 160 more calls.
CLASSES = list(DEFAULT_SITE_CLASSES)


def vs30_for(site_class: str) -> int:
    """The Vs30 a site class is built at, preferring the batch pull's table."""
    if site_class in DEFAULT_SITE_CLASSES:
        return int(DEFAULT_SITE_CLASSES[site_class])
    return int(SITE_CLASS_VS30[site_class])


def build(
    release_dir: Path,
    sites_csv: Path,
    out_dir: Path,
    classes=CLASSES,
    curves_file: str = local_hazard.CURVES_FILE,
    verbose: bool = True,
) -> dict:
    """Write one cache entry per (site, site class); return a summary."""
    sites = load_sites(sites_csv)
    out_dir.mkdir(parents=True, exist_ok=True)

    written, missing = 0, []
    for site_class in classes:
        vs30 = vs30_for(site_class)
        started = time.time()
        curves = local_hazard.load_curves(release_dir, vs30, curves_file)
        have = local_hazard.available_points(curves)
        if verbose:
            print(
                f"  vs30-{vs30} (class {site_class}): 22 curve files in "
                f"{time.time() - started:4.1f}s, {len(have)} sites available"
            )

        for _, site in sites.iterrows():
            if site["name"] not in have:
                missing.append({"site": site["name"], "site_class": site_class})
                continue
            entry = local_hazard.entry_from_curves(
                curves,
                site["name"],
                site["lat"],
                site["lon"],
                site_class,
                vs30,
                nshm=NSHM,
                name=site["name"],
                site_id=int(site["id"]),
                region=site["region"],
            )
            path = cache_path(out_dir, entry["site_id"], site_class)
            # Written compactly and with sorted keys so that rebuilding an
            # unchanged release reproduces identical bytes and a diff on the
            # frozen cache means something.
            tmp = path.with_suffix(".json.tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(entry, f, sort_keys=True, separators=(",", ":"))
            tmp.replace(path)
            written += 1

    manifest = {
        "source": "USGS static hazard-curve data release",
        "release_dir": str(release_dir),
        "curves_file": curves_file,
        "nshm": NSHM,
        "edition": NSHM_EDITION[NSHM],
        "site_set": SITE_SET,
        "sites": int(len(sites)),
        "site_classes": {c: vs30_for(c) for c in classes},
        "entries": written,
        "missing": missing,
    }
    (out_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="",
    )
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--release-dir", type=Path, default=RELEASE_DIR)
    parser.add_argument("--sites", type=Path, default=SITES_CSV)
    parser.add_argument(
        "--out",
        type=Path,
        default=OUT_DIR,
        help=f"destination cache directory (default: {OUT_DIR}). Pointing this "
        "at tests/ replaces the frozen cache and needs --adopt as well.",
    )
    parser.add_argument(
        "--adopt",
        action="store_true",
        help="permit writing into tests/, replacing the frozen cache",
    )
    parser.add_argument("--classes", nargs="+", default=CLASSES)
    parser.add_argument(
        "--curves",
        choices=("curves.csv", "curves-truncated.csv"),
        default=local_hazard.CURVES_FILE,
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="delete the destination's existing entries first, so a rebuild "
        "with fewer classes does not leave the old ones behind",
    )
    args = parser.parse_args(argv)

    resolved = args.out.resolve()
    in_tests = any(p.name == "tests" for p in (resolved, *resolved.parents))
    if in_tests and not args.adopt:
        print(
            f"{resolved} is inside tests/, which holds the frozen benchmark "
            f"input. Pass --adopt if replacing it is what you mean.",
            file=sys.stderr,
        )
        return 1
    if not args.release_dir.is_dir():
        print(f"no data release at {args.release_dir}", file=sys.stderr)
        return 1

    if args.clean and args.out.is_dir():
        removed = len(list(args.out.glob("*.json")))
        shutil.rmtree(args.out)
        print(f"removed {removed} existing entries from {args.out}")

    print(f"building from {args.release_dir}")
    manifest = build(args.release_dir, args.sites, args.out, args.classes, args.curves)
    size = sum(p.stat().st_size for p in args.out.glob("*.json"))
    print(
        f"\nwrote {manifest['entries']} entries "
        f"({manifest['sites']} sites x {len(args.classes)} classes) "
        f"to {args.out}"
    )
    print(f"cache size: {size / 1e6:.1f} MB")
    if manifest["missing"]:
        print(f"{len(manifest['missing'])} (site, class) pairs had no curves:")
        for item in manifest["missing"][:10]:
            print(f"  {item['site']} [{item['site_class']}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
