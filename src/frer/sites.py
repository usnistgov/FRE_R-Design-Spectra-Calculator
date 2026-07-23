"""Site table loading and site-set expansion."""

from pathlib import Path

import pandas as pd

from .params import DEFAULT_SITE_CLASSES

SITE_SETS = ("usgs_160", "nehrp_34", "frtc_testbed")


def load_sites(sites_csv: Path, site_set: str | None = None) -> pd.DataFrame:
    """Load sites.csv, optionally filtered to one site set.

    Rows are returned in file order (ascending id), matching the order the
    original MATLAB tool processed them.
    """
    sites = pd.read_csv(sites_csv)
    if site_set is not None:
        if site_set not in SITE_SETS:
            raise ValueError(f"unknown site set {site_set!r}; choose from {SITE_SETS}")
        sites = sites[sites[f"{site_set}_id"].notna()]
    return sites.reset_index(drop=True)


def site_class_list(site_row) -> list[tuple[str, float]]:
    """Return [(site_class, vs30), ...] to pull/calculate for one site.

    Sites with an explicit site class in sites.csv use it (with the site's
    vs30); others default to the standard four classes BC/C/D/E.
    """
    site_class = site_row["site_class"]
    if isinstance(site_class, str) and site_class.strip():
        return [(site_class, float(site_row["vs30"]))]
    return list(DEFAULT_SITE_CLASSES.items())
