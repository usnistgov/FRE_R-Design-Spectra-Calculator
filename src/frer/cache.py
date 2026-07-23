"""JSON-per-site cache of USGS web-service responses.

Each (site, site class) pair gets one JSON file in the cache directory:
``site{id:04d}_{site_class}.json``, containing site metadata plus the raw
design-value and hazard-curve responses. This naturally supports
resume-on-failure: a pull re-run only fetches what is missing.
"""

import json
from pathlib import Path

DEFAULT_CACHE_DIR = Path("results") / "usgs_cache"


def cache_path(cache_dir: Path, site_id: int, site_class: str) -> Path:
    return Path(cache_dir) / f"site{int(site_id):04d}_{site_class}.json"


def covers_site_set(cache_dir: Path, sites_csv: Path, site_set: str) -> bool:
    """True if the cache has an entry file for every (site, class) in the set.

    Sites can belong to multiple sets, so a partially-cached set (e.g. usgs_160
    sharing a few sites with nehrp_34) leaves stray files; require full coverage
    so such a set is treated as "not cached" rather than partially processed.
    """
    from .sites import load_sites, site_class_list

    sites = load_sites(sites_csv, site_set)
    for _, row in sites.iterrows():
        for site_class, _vs30 in site_class_list(row):
            if not cache_path(cache_dir, row["id"], site_class).exists():
                return False
    return True


def load_entry(cache_dir: Path, site_id: int, site_class: str) -> dict | None:
    """Load a cached site entry, or None if it does not exist."""
    path = cache_path(cache_dir, site_id, site_class)
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_entry(cache_dir: Path, entry: dict) -> Path:
    """Write a site entry to the cache (atomically via a temp file)."""
    path = cache_path(cache_dir, entry["site_id"], entry["site_class"])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(entry, f)
    tmp.replace(path)
    return path


def new_entry(site_row: dict, site_class: str, vs30: float) -> dict:
    """Create a fresh cache entry from a sites.csv row."""
    return {
        "site_id": int(site_row["id"]),
        "name": site_row["name"],
        "lat": float(site_row["lat"]),
        "lon": float(site_row["lon"]),
        "region": site_row["region"],
        "site_class": site_class,
        "vs30": float(vs30),
        "design_values": None,  # raw asce7-22 design maps response
        "hazard_curves": {},  # keyed by NSHM year, e.g. "2018", "2023"
    }
