"""Develop FR and LS seismic design values from the cached hazard curves.

Port of MATLAB ``calc_FREr.m``: develops 2-point and multi-period design
spectra for every (site, site class) pair in the site table. FREr values are
calculated using a risk-targeted approach.
"""

import warnings
from pathlib import Path

import pandas as pd

from . import cache, plots
from .params import VERSION_ID, VERSION_NSHM
from .single import (
    calc_site,
    design_value_columns,
    frer_mps_columns,
    return_period_columns,
)
from .sites import SITE_SET, load_sites, site_class_list


def calc_design_values(
    sites_csv: Path,
    cache_dir: Path,
    out_dir: Path,
    version_id: str = VERSION_ID,
    make_plots: bool = True,
    skip_missing: bool = True,
) -> pd.DataFrame:
    """Calculate FREr and DE design values for every site and site class.

    Writes ``frer_design_vals.csv`` (plus plots) under
    ``out_dir/<site set>/<version>/`` and returns the results table.

    If ``skip_missing`` is True, (site, class) pairs whose required cached data
    is absent are warned about and skipped; otherwise a missing entry raises.
    """
    if version_id not in VERSION_NSHM:
        raise ValueError(f"not set up for code version {version_id!r}")
    nshm = VERSION_NSHM[version_id]

    save_dir = Path(out_dir) / SITE_SET / version_id
    save_dir.mkdir(parents=True, exist_ok=True)

    sites = load_sites(sites_csv)
    rows = []
    for _, site in sites.iterrows():
        for site_class, _vs30 in site_class_list(site):
            entry = cache.load_entry(cache_dir, site["id"], site_class)
            msg = None
            if entry is None:
                msg = (
                    f"Missing cached USGS data for site {site['id']} "
                    f"({site['name']}) class {site_class}"
                )
            elif entry["hazard_curves"].get(str(nshm)) is None:
                msg = (
                    f"Site {site['id']} ({site['name']}) class {site_class} "
                    f"has no NSHM {nshm} hazard curves"
                )
            if msg is not None:
                if skip_missing:
                    warnings.warn(
                        f"{msg}: skipping (run `frer pull` to fetch)", stacklevel=2
                    )
                    continue
                raise FileNotFoundError(f"{msg}: run `frer pull` first")
            result = calc_site(entry, version_id)

            row = {
                "id": int(site["id"]),
                "name": site["name"],
                "lat": site["lat"],
                "lon": site["lon"],
                "region": site["region"],
                "site_class": site_class,
            }
            # 2-point design values, headers annotated with the ground-motion
            # orientation each is reported in (DE is RotD100, FREr RotD50)
            row.update(design_value_columns(result))

            # The full multi-period FREr spectrum. This used to be written
            # only for the 9 frtc_testbed rows; it is the actual output of the
            # calculation, of which the 2-point values above are a
            # condensation, so it is now pinned for every row.
            row.update(frer_mps_columns(result.fr_spectrum))
            # Return periods of the FRE_R/MCE_R/DE ordinates
            row.update(return_period_columns(result))
            rows.append(row)

    table = pd.DataFrame(rows)
    table.to_csv(save_dir / "frer_design_vals.csv", index=False)

    # Scatter plots comparing FREr with DE across the whole site set
    if make_plots:
        plots.plot_scatter_comparisons(save_dir / "Scatter Plots", table)

    return table
