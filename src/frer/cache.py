"""JSON-per-site cache of USGS web-service responses.

Each (site, site class) pair gets one JSON file in the cache directory:
``site{id:04d}_{site_class}.json``, containing site metadata plus the hazard
curves, in the shape the hazard web service returns them. Entries are built
either by ``frer pull`` or, for the frozen benchmark input, by
``scripts/build_cache_from_release.py`` from a published data release; the two
produce the same structure on purpose, so neither path is privileged.
"""

import json
from pathlib import Path

DEFAULT_CACHE_DIR = Path("results") / "usgs_cache"


def cache_path(cache_dir: Path, site_id: int, site_class: str) -> Path:
    return Path(cache_dir) / f"site{int(site_id):04d}_{site_class}.json"


def covers_sites(cache_dir: Path, sites_csv: Path) -> bool:
    """True if the cache has an entry file for every (site, site class) pair.

    All or nothing: a partially populated cache is treated as not cached
    rather than as something to compute a partial answer from, because a
    reference table missing rows is harder to notice than one that was never
    produced.
    """
    from .sites import load_sites, site_class_list

    sites = load_sites(sites_csv)
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
        "hazard_curves": {},  # keyed by NSHM year, e.g. "2023"
    }
