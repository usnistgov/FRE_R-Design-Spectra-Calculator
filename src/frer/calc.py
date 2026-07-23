"""Develop FR and LS seismic design values from cached USGS data.

Port of MATLAB ``calc_FREr.m``: uses cached USGS web-service data to develop
2-point and multi-point design spectra. FREr values are calculated using a
risk-targeted approach.
"""

import warnings
from pathlib import Path

import pandas as pd

from . import cache, plots
from .params import FR_PARAMS, LS_PARAMS, VERSION_NSHM
from .sites import load_sites, site_class_list
from .spectrum import develop_2point_design_values, develop_rt_spectrum


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
            hazard_curves = entry["hazard_curves"][str(nshm)]

            # Develop design spectrum based on FR risk-target values
            rt_spectrum_fr = develop_rt_spectrum(hazard_curves, FR_PARAMS)

            # Develop the DE (LS) spectrum
            if version_id == "ASCE 7-28":
                # Fully probabilistic MPS from the hazard curves
                rt_spectrum_ls = develop_rt_spectrum(hazard_curves, LS_PARAMS)
                ls_period = rt_spectrum_ls.period
                ls_sa_de = (2 / 3) * rt_spectrum_ls.sa
                sds, sd1 = develop_2point_design_values(ls_sa_de, ls_period)
            else:  # ASCE 7-22: use the design-value web service results directly
                dv = entry["design_values"]["response"]["data"]
                ls_period = dv["multiPeriodDesignSpectrum"]["periods"]
                ls_sa_de = dv["multiPeriodDesignSpectrum"]["ordinates"]
                sds = dv["sds"]
                sd1 = dv["sd1"]

            # 2-point design values for FREr
            sdsfr, sd1fr = develop_2point_design_values(
                rt_spectrum_fr.sa, rt_spectrum_fr.period
            )

            row = {
                "id": int(site[f"{site_set}_id"]),
                "name": site["name"],
                "lat": site["lat"],
                "lon": site["lon"],
                "region": site["region"],
                "site_class": site_class,
                "Sds": sds,
                "Sd1": sd1,
                "Sdsfr": sdsfr,
                "Sd1fr": sd1fr,
            }

            # Add multi-period FREr spectrum values for the testbed cases
            if site_set == "frtc_testbed":
                for t, sa in zip(rt_spectrum_fr.period, rt_spectrum_fr.sa, strict=True):
                    t_str = f"{t:4.2f}".replace(".", "p")
                    row[f"frer_mps_{t_str}"] = sa
            rows.append(row)

            # Spectra plots for the NEHRP and testbed site sets
            if make_plots and site_set in ("nehrp_34", "frtc_testbed"):
                plots.plot_spectra(
                    save_dir / "Spectra_Plots - loglog",
                    site["name"],
                    site_class,
                    version_id,
                    rt_spectrum_fr,
                    ls_period,
                    ls_sa_de,
                    sds,
                    sd1,
                    sdsfr,
                    sd1fr,
                )

    table = pd.DataFrame(rows)
    table.to_csv(save_dir / "frer_design_vals.csv", index=False)

    # Scatter plots comparing FREr with DE for the full usgs_160 set
    if make_plots and site_set == "usgs_160":
        plots.plot_scatter_comparisons(save_dir / "Scatter Plots", table)

    return table
