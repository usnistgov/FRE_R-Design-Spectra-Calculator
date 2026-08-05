"""Develop FR and LS seismic design values from cached USGS data.

Port of MATLAB ``calc_FREr.m``: uses cached USGS web-service data to develop
2-point and multi-point design spectra. FREr values are calculated using a
risk-targeted approach.
"""

import warnings
from pathlib import Path

import pandas as pd

from . import cache, plots
from .params import VERSION_NSHM
from .single import calc_site, frer_mps_columns, return_period_columns
from .sites import load_sites, site_class_list


def calc_design_values(
    sites_csv: Path,
    cache_dir: Path,
    site_set: str,
    version_id: str,
    out_dir: Path,
    make_plots: bool = True,
    skip_missing: bool = True,
) -> pd.DataFrame:
    """Calculate FREr and DE design values for one site set and code version.

    Writes ``frer_design_vals.csv`` (plus plots) under
    ``out_dir/site_set/version_id/`` and returns the results table.

    If ``skip_missing`` is True, (site, class) pairs whose required cached data
    is absent are warned about and skipped; otherwise a missing entry raises.
    """
    if version_id not in VERSION_NSHM:
        raise ValueError(f"not set up for code version {version_id!r}")
    nshm = VERSION_NSHM[version_id]

    save_dir = Path(out_dir) / site_set / version_id
    save_dir.mkdir(parents=True, exist_ok=True)

    sites = load_sites(sites_csv, site_set)
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
            elif version_id == "ASCE 7-22" and entry["design_values"] is None:
                msg = (
                    f"Site {site['id']} ({site['name']}) class {site_class} "
                    f"has no ASCE 7-22 design values"
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
                "id": int(site[f"{site_set}_id"]),
                "name": site["name"],
                "lat": site["lat"],
                "lon": site["lon"],
                "region": site["region"],
                "site_class": site_class,
                "Sds": result.sds,
                "Sd1": result.sd1,
                "Sdsfr": result.sdsfr,
                "Sd1fr": result.sd1fr,
            }

            # Add multi-period FREr spectrum values for the testbed cases
            if site_set == "frtc_testbed":
                row.update(frer_mps_columns(result.fr_spectrum))
            # Return periods of the FRE_R/MCE_R/DE ordinates (ASCE 7-28 only)
            row.update(return_period_columns(result))
            rows.append(row)

            # Spectra plots for the NEHRP and testbed site sets
            if make_plots and site_set in ("nehrp_34", "frtc_testbed"):
                plots.plot_spectra(
                    save_dir / "Spectra_Plots - loglog", site["name"], result
                )

    table = pd.DataFrame(rows)
    table.to_csv(save_dir / "frer_design_vals.csv", index=False)

    # Scatter plots comparing FREr with DE for the full usgs_160 set
    if make_plots and site_set == "usgs_160":
        plots.plot_scatter_comparisons(save_dir / "Scatter Plots", table)

    return table
