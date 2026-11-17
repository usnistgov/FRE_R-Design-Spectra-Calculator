"""Archive frozen cache entries and reference CSVs before changing them.

Refreshing ``tests/usgs_cache`` is deliberate and hard to undo: the responses
are large, the pull is slow, and the values that come back depend on what the
USGS services return that day. This takes a copy first, so a change can be
quantified afterwards rather than asserted.

The copy goes under ``results/``, which is git-ignored, because the cache is
hundreds of megabytes. Only the small report produced later is worth
committing.

    python scripts/archive_cache.py --site-class D --label vs30-265
    python scripts/archive_cache.py --label pre-refresh      # every class

Alongside the files it records a manifest of SHA-256 digests, including a
separate digest of each entry's ``design_values`` block. That is what makes it
possible to show afterwards that a hazard-curve refresh left the design-value
half of the cache untouched, instead of taking it on trust.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO_ROOT / "tests" / "usgs_cache"
TESTS_DIR = REPO_ROOT / "tests"
ARCHIVE_ROOT = REPO_ROOT / "results" / "archive"

SITE_SETS = ["frtc_testbed", "nehrp_34", "usgs_160"]
VERSIONS = ["ASCE 7-22", "ASCE 7-28"]


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def entry_digests(path: Path) -> dict:
    """Digests of a cache entry as a whole and of its two halves separately.

    The halves matter: a refresh aimed at the hazard curves should leave
    ``design_values`` byte for byte identical, and that is only checkable if
    both were recorded beforehand.
    """
    raw = path.read_bytes()
    entry = json.loads(raw)
    design = entry.get("design_values")
    curves = entry.get("hazard_curves") or {}
    return {
        "file": digest(raw),
        "design_values": digest(json.dumps(design, sort_keys=True).encode("utf-8")),
        "hazard_curves": {
            year: digest(json.dumps(curve, sort_keys=True).encode("utf-8"))
            for year, curve in sorted(curves.items())
        },
        "site_class": entry.get("site_class"),
        "vs30": entry.get("vs30"),
        "bytes": len(raw),
    }


def git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip() or None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--label", required=True, help="name for this archive")
    ap.add_argument("--site-class", help="only entries for this class (e.g. D)")
    ap.add_argument(
        "--out", type=Path, default=ARCHIVE_ROOT, help="archive root directory"
    )
    ap.add_argument(
        "--force", action="store_true", help="overwrite an existing archive"
    )
    args = ap.parse_args(argv)

    destination = Path(args.out) / args.label
    if destination.exists() and not args.force:
        print(f"{destination} already exists; pass --force to replace it.")
        return 1
    if destination.exists():
        shutil.rmtree(destination)

    pattern = f"site*_{args.site_class}.json" if args.site_class else "site*.json"
    entries = sorted(CACHE_DIR.glob(pattern))
    if not entries:
        print(f"No cache entries match {pattern} in {CACHE_DIR}.")
        return 1

    cache_out = destination / "usgs_cache"
    cache_out.mkdir(parents=True)
    manifest_entries, total = {}, 0
    for path in entries:
        shutil.copy2(path, cache_out / path.name)
        manifest_entries[path.name] = entry_digests(path)
        total += manifest_entries[path.name]["bytes"]
    print(f"Archived {len(entries)} cache entr(ies), {total / 1e6:.0f} MB")

    # The reference CSVs are small and change alongside the cache, so they are
    # always taken whatever the class filter -- a diff is only meaningful with
    # both halves of the before picture.
    references = {}
    for site_set in SITE_SETS:
        for version in VERSIONS:
            source = TESTS_DIR / site_set / version
            if not source.is_dir():
                continue
            for csv in sorted(source.glob("*.csv")):
                target = destination / "references" / site_set / version / csv.name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(csv, target)
                references[f"{site_set}/{version}/{csv.name}"] = digest(
                    csv.read_bytes()
                )
    print(f"Archived {len(references)} reference CSV(s)")

    manifest = {
        "label": args.label,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_sha": git_sha(),
        "source_cache": str(CACHE_DIR.relative_to(REPO_ROOT)),
        "site_class": args.site_class,
        "entry_count": len(entries),
        "entries": manifest_entries,
        "references": references,
    }
    manifest_path = destination / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(f"\nArchive: {destination}")
    print(f"Manifest: {manifest_path}")
    print("\nThis lives under results/, which is git-ignored; nothing is committed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
