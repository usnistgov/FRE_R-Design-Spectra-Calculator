"""Single-site FREr workflow, independent of ``sites.csv`` and the disk cache.

The batch pipeline in :mod:`frer.pull` and :mod:`frer.calc` is driven by a site
table and caches every response as JSON on disk. That is the wrong shape for an
interactive caller that wants one arbitrary location, so this module exposes the
same two stages for a single site:

``pull_site``
    Fetch the USGS responses one site needs and return them as an in-memory
    entry (no disk I/O). The entry uses the same keys as
    :func:`frer.cache.new_entry`, so cached entries work here interchangeably.
``calc_site``
    Develop the FREr and DE design values for one entry.

:func:`calc_site` holds the only implementation of the per-site calculation;
:func:`frer.calc.calc_design_values` calls it for every row of a site set, so
the batch benchmark and any interactive caller exercise identical code.
"""

from dataclasses import dataclass

import numpy as np

from . import usgs_api
from .params import (
    DE_ORIENTATION,
    FR_ORIENTATION,
    FR_PARAMS,
    LS_PARAMS,
    NSHM_BOUNDS,
    VERSION_NSHM,
    VS30_RANGE,
)
from .spectrum import RTSpectrum, develop_2point_design_values, develop_rt_spectrum


@dataclass
class SiteResult:
    """FREr and DE design values for one (site, site class, code version)."""

    version_id: str
    site_class: str
    vs30: float
    lat: float
    lon: float
    name: str
    risk_category: str
    sds: float  # DE short-period design value
    sd1: float  # DE 1-second design value
    sdsfr: float  # FREr short-period design value
    sd1fr: float  # FREr 1-second design value
    fr_spectrum: RTSpectrum  # FREr multi-period spectrum
    de_period: np.ndarray  # DE multi-period spectrum periods (s)
    de_sa: np.ndarray  # DE multi-period spectrum ordinates (g)
    ls_spectrum: RTSpectrum | None = None  # MCEr spectrum (ASCE 7-28 only)

    @property
    def nshm(self) -> int:
        """NSHM release year the FREr values were developed from."""
        return VERSION_NSHM[self.version_id]


def check_location(lat: float, lon: float) -> None:
    """Raise :class:`ValueError` if a location is outside the NSHM models.

    The hazard service answers out-of-model coordinates with an HTTP 200 and
    ``status: success`` but unusable curves, so callers have to bounds-check
    themselves rather than relying on the response.
    """
    lon_min, lat_min, lon_max, lat_max = NSHM_BOUNDS
    if not (lat_min <= lat <= lat_max and lon_min <= lon <= lon_max):
        raise ValueError(
            f"({lat:.4f}, {lon:.4f}) is outside the conterminous-U.S. hazard "
            f"models, which cover latitude {lat_min} to {lat_max} and longitude "
            f"{lon_min} to {lon_max}"
        )


def check_vs30(vs30: float) -> None:
    """Raise :class:`ValueError` if vs30 is outside the accepted range."""
    low, high = VS30_RANGE
    if not low <= vs30 <= high:
        raise ValueError(f"vs30 must be between {low} and {high} m/s, got {vs30}")


def new_entry(
    lat: float,
    lon: float,
    site_class: str,
    vs30: float,
    name: str = "",
    risk_category: str = "II",
) -> dict:
    """Build an empty entry for an arbitrary location.

    Mirrors :func:`frer.cache.new_entry`, which requires a ``sites.csv`` row and
    so cannot describe a site that is not in the table. ``site_id`` and
    ``region`` are None because an ad-hoc location has neither.
    """
    return {
        "site_id": None,
        "name": name or f"{lat:.4f}, {lon:.4f}",
        "lat": float(lat),
        "lon": float(lon),
        "region": None,
        "site_class": site_class,
        "vs30": float(vs30),
        "risk_category": risk_category,
        "design_values": None,
        "hazard_curves": {},
    }


def pull_site(
    lat: float,
    lon: float,
    site_class: str,
    vs30: float,
    version_id: str,
    name: str = "",
    risk_category: str = "II",
    timeout: float = usgs_api.TIMEOUT,
    session=None,
    fetch_design=None,
    fetch_hazard=None,
) -> dict:
    """Fetch the USGS data one site needs for ``version_id``; return an entry.

    Only what the version actually uses is requested: ASCE 7-22 takes its DE
    values from the design-value service and needs NSHM 2018 hazard curves,
    while ASCE 7-28 is fully probabilistic and needs only NSHM 2023 curves.
    By default both calls go through the retrying session built by
    :func:`frer.usgs_api.build_session`.

    ``fetch_design(lat, lon, site_class, risk_category)`` and
    ``fetch_hazard(nshm, lat, lon, vs30)`` override how the raw responses are
    obtained -- a caller with its own cache (such as the Streamlit app) wraps
    the :mod:`frer.usgs_api` functions and passes them in here rather than
    reimplementing this function.

    Raises :class:`ValueError` for an out-of-model location and
    :class:`frer.usgs_api.USGSApiError` if a call fails after its retries.
    """
    if version_id not in VERSION_NSHM:
        raise ValueError(f"not set up for code version {version_id!r}")
    check_location(lat, lon)
    check_vs30(vs30)

    if fetch_design is None:

        def fetch_design(lat, lon, site_class, risk_category):
            return usgs_api.call_design_value_api(
                f"{lat:.4f}, {lon:.4f}",
                lat,
                lon,
                risk_category=risk_category,
                site_class=site_class,
                timeout=timeout,
                session=session,
            )

    if fetch_hazard is None:

        def fetch_hazard(nshm, lat, lon, vs30):
            return usgs_api.call_hazard_api(
                nshm, lat, lon, vs30, timeout=timeout, session=session
            )

    entry = new_entry(lat, lon, site_class, vs30, name, risk_category)

    if version_id == "ASCE 7-22":
        entry["design_values"] = fetch_design(lat, lon, site_class, risk_category)

    nshm = VERSION_NSHM[version_id]
    hazard = fetch_hazard(nshm, lat, lon, vs30)
    entry["hazard_curves"][str(nshm)] = hazard["response"]["hazardCurves"]
    return entry


def calc_site(entry: dict, version_id: str) -> SiteResult:
    """Develop FREr and DE design values for one cached or freshly pulled site.

    ``entry`` is either a :func:`frer.cache.new_entry` dict loaded from the disk
    cache or one returned by :func:`pull_site`. Required data must already be
    present; callers that tolerate gaps should check before calling (see
    :func:`frer.calc.calc_design_values`).
    """
    if version_id not in VERSION_NSHM:
        raise ValueError(f"not set up for code version {version_id!r}")
    nshm = VERSION_NSHM[version_id]
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
        rt_spectrum_ls = None  # no probabilistic LS spectrum for ASCE 7-22
        dv = entry["design_values"]["response"]["data"]
        ls_period = dv["multiPeriodDesignSpectrum"]["periods"]
        ls_sa_de = dv["multiPeriodDesignSpectrum"]["ordinates"]
        sds = dv["sds"]
        sd1 = dv["sd1"]

    # 2-point design values for FREr
    sdsfr, sd1fr = develop_2point_design_values(
        rt_spectrum_fr.sa, rt_spectrum_fr.period
    )

    return SiteResult(
        version_id=version_id,
        site_class=entry["site_class"],
        vs30=entry["vs30"],
        lat=entry["lat"],
        lon=entry["lon"],
        name=entry["name"],
        risk_category=entry.get("risk_category", "II"),
        sds=sds,
        sd1=sd1,
        sdsfr=sdsfr,
        sd1fr=sd1fr,
        fr_spectrum=rt_spectrum_fr,
        de_period=ls_period,
        de_sa=ls_sa_de,
        ls_spectrum=rt_spectrum_ls,
    )


# CSV column names for the 2-point design values. Each carries the ground-motion
# orientation its value is reported in (see FR_ORIENTATION / DE_ORIENTATION), so
# a row read back out of a CSV cannot be compared across orientations by
# accident -- the FREr and DE columns are on different bases.
SDS_COLUMN = f"Sds_{DE_ORIENTATION}"
SD1_COLUMN = f"Sd1_{DE_ORIENTATION}"
SDSFR_COLUMN = f"Sdsfr_{FR_ORIENTATION}"
SD1FR_COLUMN = f"Sd1fr_{FR_ORIENTATION}"


def design_value_columns(result: SiteResult) -> dict:
    """The four 2-point design-value CSV columns for one site.

    Shared by the batch CSV writer and the single-site export so the two cannot
    drift apart in how they name or order the design values.
    """
    return {
        SDS_COLUMN: result.sds,
        SD1_COLUMN: result.sd1,
        SDSFR_COLUMN: result.sdsfr,
        SD1FR_COLUMN: result.sd1fr,
    }


def frer_mps_columns(spectrum: RTSpectrum) -> dict:
    """Return the ``frer_mps_*`` CSV columns for a FREr multi-period spectrum.

    Shared by the batch CSV writer and any single-site export so the two cannot
    drift apart in how they name periods (0.2 s becomes
    ``frer_mps_0p20_RotD50``; the suffix is the orientation, as above).
    """
    columns = {}
    for t, sa in zip(spectrum.period, spectrum.sa, strict=True):
        t_str = f"{t:4.2f}".replace(".", "p")
        columns[f"frer_mps_{t_str}_{FR_ORIENTATION}"] = sa
    return columns


# Spectral-value column names used before the orientation suffixes were added.
# The committed benchmark fixtures (and any CSV exported by an earlier version)
# still use them, so readers map old -> new rather than rewriting those files.
LEGACY_SPECTRAL_COLUMNS = {
    "Sds": SDS_COLUMN,
    "Sd1": SD1_COLUMN,
    "Sdsfr": SDSFR_COLUMN,
    "Sd1fr": SD1FR_COLUMN,
}


def rename_legacy_spectral_columns(table):
    """Return ``table`` with pre-suffix spectral columns renamed to current ones.

    Only the spectral columns move; the ``*_rp_*`` return-period columns are
    unsuffixed by design and are left exactly as they are.
    """
    suffix = f"_{FR_ORIENTATION}"
    mapping = {
        old: new for old, new in LEGACY_SPECTRAL_COLUMNS.items() if old in table.columns
    }
    mapping.update(
        {
            c: f"{c}{suffix}"
            for c in table.columns
            if c.startswith("frer_mps_") and not c.endswith(suffix)
        }
    )
    return table.rename(columns=mapping)


# Periods (s) at which return periods are reported, per the MATLAB block
# "Quantify Return Periods of Spectral Values at specific periods".
RP_PERIODS = (0.2, 1.0)


def _rp_at(values: np.ndarray, period: np.ndarray, t: float) -> float:
    """Return the array value at period ``t`` (an exact member of ``T_VEC``)."""
    return float(values[period == t][0])


def return_period_columns(result: SiteResult) -> dict:
    """Return-period CSV columns (years) at 0.2 s and 1.0 s.

    Reports the hazard-curve return period of the FRE_R, MCE_R, and DE spectral
    ordinates (DE is 2/3 of MCE_R). Empty for ASCE 7-22: return periods are only
    defined for the fully probabilistic ASCE 7-28 ground motions, whereas the
    ASCE 7-22 design values mix risk-targeted and deterministic caps that have no
    single hazard-curve return period.

    Shared by the batch CSV writer and the single-site export so the two cannot
    drift apart in how they name columns.
    """
    if result.ls_spectrum is None:
        return {}
    fr, ls = result.fr_spectrum, result.ls_spectrum
    columns = {}
    for t in RP_PERIODS:
        t_str = f"{t:.1f}".replace(".", "p")  # 0.2 -> "0p2", 1.0 -> "1p0"
        columns[f"frer_rp_{t_str}"] = _rp_at(fr.return_period, fr.period, t)
        columns[f"mcer_rp_{t_str}"] = _rp_at(ls.return_period, ls.period, t)
        columns[f"de_rp_{t_str}"] = _rp_at(ls.return_period_67, ls.period, t)
    return columns
