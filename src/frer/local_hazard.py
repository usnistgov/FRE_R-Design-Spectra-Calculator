"""Read USGS static hazard-curve data releases as if they were API responses.

:mod:`frer.usgs_api` fetches hazard curves from the nshmp dynamic hazard web
service, one HTTP call per (point, Vs30). The USGS publishes the same curves as
static data releases -- a ``curves.csv`` per Vs30 and per intensity measure,
covering either the model's named test sites or the whole conterminous-U.S.
grid at 0.05 degrees. This module reads those files and reshapes them into the
structure the web service returns, so that
:func:`frer.single.calc_site` cannot tell where its curves came from and the
FREr arithmetic is shared exactly between the two paths.

Two layouts are supported, distinguished by whether the CSV's first column is
``name``:

``test sites``
    ``name,lon,lat,<ground motions...>`` -- keyed by site name.
``gridded``
    ``lon,lat,<ground motions...>`` -- keyed by ``(lat, lon)``.

Both live under ``<release>/vs30-<n>/<IMT>/curves.csv``. Pick ``curves.csv``
and not ``curves-truncated.csv``: the truncated file drops the long
return-period tail, which moves a short-period design value by around 3%.

Mind the model revision. A release is a fixed revision of the NSHM -- the
CONUS.2023.R1 release is what ``conus-2023.R1`` serves -- whereas the
unversioned ``conus-2023`` path the pull uses follows whatever revision is
current. Curves from the two agree only where the revisions do.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from .spectrum import T_VEC

# Coordinates are rounded to this many decimals before being used as keys, the
# same rule (and the same reason) as frer.grid.COORD_DECIMALS: a node must have
# exactly one spelling, whichever file it was read from.
COORD_DECIMALS = 4

# Both the data release and the web service name an intensity measure the same
# way, which is what lets one mapping serve both. The release uses the name as
# a directory ("SA1P0"); the service reports it as ``imt.value``, with the
# period also spelled out in ``imt.display`` ("1.0 s"), and
# :func:`frer.spectrum._find_hazard_curve` reads the period back off that
# display string -- so both strings are built here from one source.
CURVES_FILE = "curves.csv"


def imt_label(period: float) -> tuple[str, str]:
    """Return the ``(IMT name, display label)`` for a spectral period."""
    if period == 0:
        return "PGA", "PGA"
    text = f"{period:g}"
    if "." not in text:  # 1 -> 1.0, matching "SA1P0" / "1.0 s"
        text += ".0"
    return f"SA{text.replace('.', 'P')}", f"{text} s"


# Period -> (IMT name, display label) for every period the multi-period
# spectrum needs. PGV is in the releases too and is never one of them.
IMT_LABELS = {float(t): imt_label(float(t)) for t in T_VEC}


def curves_path(
    release_dir: Path, vs30: float, period: float, curves_file: str
) -> Path:
    """Where one Vs30's curves for one period live inside a release."""
    imt, _display = IMT_LABELS[float(period)]
    return Path(release_dir) / f"vs30-{int(vs30)}" / imt / curves_file


def _read_one(path: Path, bbox: tuple | None):
    """Read one curves CSV; return ``(xs, {key: ys})``.

    ``bbox`` is ``(lat_min, lat_max, lon_min, lon_max)`` and is applied while
    reading so that a national file costs only its parse, not its memory. Keys
    are site names for a test-site file and ``(lat, lon)`` for a gridded one.
    """
    if not path.is_file():
        raise FileNotFoundError(f"missing hazard curves: {path}")
    table = pd.read_csv(path)

    named = table.columns[0] == "name"
    first_value = 3 if named else 2
    xs = np.array([float(c) for c in table.columns[first_value:]], dtype=float)

    if bbox is not None:
        lat_min, lat_max, lon_min, lon_max = bbox
        table = table[
            table["lat"].between(lat_min, lat_max)
            & table["lon"].between(lon_min, lon_max)
        ]

    values = table.iloc[:, first_value:].to_numpy(dtype=float)
    if named:
        keys = table["name"].tolist()
    else:
        keys = list(
            zip(
                table["lat"].round(COORD_DECIMALS),
                table["lon"].round(COORD_DECIMALS),
                strict=True,
            )
        )
    return xs, dict(zip(keys, values, strict=True))


def load_curves(
    release_dir: Path,
    vs30: float,
    curves_file: str = CURVES_FILE,
    bbox: tuple | None = None,
    progress=None,
) -> dict:
    """Load one Vs30's curves for every period the spectrum needs.

    Returns ``{period: (xs, {key: ys})}``. ``xs`` are ground motions in g, read
    from the column headers and shared by every point in the file; ``ys`` are
    that point's annual frequencies of exceedance. ``progress(period, path)``
    is called before each file, which is worth passing for a national release:
    a single gridded ``curves.csv`` is around 110 MB and there are 22 of them
    per site class.
    """
    release_dir = Path(release_dir)
    if not release_dir.is_dir():
        raise FileNotFoundError(f"no hazard release at {release_dir}")

    curves = {}
    for period in IMT_LABELS:
        path = curves_path(release_dir, vs30, period, curves_file)
        if progress is not None:
            progress(period, path)
        curves[period] = _read_one(path, bbox)
    return curves


def hazard_curves(curves: dict, key) -> list:
    """Build the service-shaped ``hazardCurves`` list for one point.

    Mirrors the part of the nshmp hazard response that
    :func:`frer.spectrum._find_hazard_curve` reads: an ``imt`` label and a
    ``data[0]`` Total component carrying ``values.xs`` and ``values.ys``.

    Non-positive and blank annual frequencies are left in place. The calculator
    already drops them, which is what the truncated curves (blank tail) and the
    full curves (zero tail) both need.
    """
    out = []
    for period, (xs, by_key) in curves.items():
        if key not in by_key:
            raise KeyError(f"{key!r} is not in the {period} s hazard curves")
        imt, display = IMT_LABELS[period]
        out.append(
            {
                "imt": {"value": imt, "display": display},
                "data": [
                    {
                        "component": "Total",
                        "values": {"xs": xs.tolist(), "ys": by_key[key].tolist()},
                    }
                ],
            }
        )
    return out


def entry_from_curves(
    curves: dict,
    key,
    lat: float,
    lon: float,
    site_class: str,
    vs30: float,
    nshm: int = 2023,
    name: str = "",
    site_id=None,
    region=None,
) -> dict:
    """Build a cache-shaped entry for one point, ready for ``calc_site``.

    Same keys as :func:`frer.cache.new_entry`, so a locally built entry is
    interchangeable with a cached web-service one.
    """
    return {
        "site_id": site_id,
        "name": name or f"{lat:.4f}, {lon:.4f}",
        "lat": float(lat),
        "lon": float(lon),
        "region": region,
        "site_class": site_class,
        "vs30": float(vs30),
        "hazard_curves": {str(nshm): hazard_curves(curves, key)},
    }


def lattice_axes(keys) -> tuple[np.ndarray, np.ndarray]:
    """Sorted unique ``(lats, lons)`` of a set of ``(lat, lon)`` keys."""
    lats = np.array(sorted({float(lat) for lat, _lon in keys}))
    lons = np.array(sorted({float(lon) for _lat, lon in keys}))
    return lats, lons


def interpolate_lattice(values: dict, targets, log: bool = False) -> dict:
    """Bilinearly interpolate lattice values onto arbitrary points.

    ``values`` maps ``(lat, lon)`` to an array; ``targets`` is a sequence of
    ``(lat, lon)``. Returns ``{target: array}``, with a target that cannot be
    surrounded by four known corners omitted rather than extrapolated -- a
    hole in the lattice or a point outside it produces no answer instead of a
    confident wrong one.

    ``log`` interpolates the logarithm, which is what annual frequencies of
    exceedance want: hazard falls roughly log-linearly, so interpolating the
    rates themselves would bias every intermediate point high. Non-positive
    and blank entries have no logarithm and come back blank, which is where a
    curve's zero tail already ends up.

    A target that coincides with a lattice node returns that node's values
    unchanged, so mixing exact and interpolated points is safe.
    """
    from scipy.interpolate import RegularGridInterpolator

    lats, lons = lattice_axes(values)
    if len(lats) < 2 or len(lons) < 2:
        raise ValueError(
            f"need at least a 2x2 lattice to interpolate, got {len(lats)}x{len(lons)}"
        )

    width = len(next(iter(values.values())))
    cube = np.full((len(lats), len(lons), width), np.nan)
    lat_at = {value: i for i, value in enumerate(lats)}
    lon_at = {value: i for i, value in enumerate(lons)}
    for (lat, lon), row in values.items():
        cube[lat_at[float(lat)], lon_at[float(lon)]] = row

    if log:
        with np.errstate(divide="ignore", invalid="ignore"):
            cube = np.where(cube > 0, np.log(cube), np.nan)

    interp = RegularGridInterpolator(
        (lats, lons), cube, method="linear", bounds_error=False, fill_value=np.nan
    )
    points = np.array([(float(lat), float(lon)) for lat, lon in targets])
    out = interp(points)
    if log:
        out = np.exp(out)

    return {
        tuple(target): row
        for target, row in zip(targets, out, strict=True)
        if not np.all(np.isnan(row))
    }


def interpolate_curves(curves: dict, targets) -> dict:
    """Interpolate a loaded release onto finer points.

    Takes and returns the ``{period: (xs, {key: ys})}`` shape
    :func:`load_curves` produces, so an interpolated set is handed to
    :func:`entry_from_curves` exactly like a read one. The ground motions
    ``xs`` are a property of the intensity measure, not the point, so they
    carry through untouched and only the rates move.

    Interpolating the curves and then computing is not the same as computing
    and then interpolating the design values: the risk integral is not linear
    in the hazard rate. Which is closer to the truth is a question for a
    comparison, not for this function.
    """
    targets = [
        (round(float(lat), COORD_DECIMALS), round(float(lon), COORD_DECIMALS))
        for lat, lon in targets
    ]
    out = {}
    for period, (xs, by_key) in curves.items():
        out[period] = (xs, interpolate_lattice(by_key, targets, log=True))
    return out


def available_points(curves: dict) -> set:
    """Keys present in every period of a loaded release.

    A point missing from one intensity measure cannot be calculated, so the
    intersection is what a caller may ask for.
    """
    sets = [set(by_key) for _xs, by_key in curves.values()]
    return set.intersection(*sets) if sets else set()
