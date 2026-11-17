"""Gridded FREr and DE design values over a rectangular region.

The single-site workflow in :mod:`frer.single` answers "what are the values
here". This builds the same values on a regular lat/lon grid so a region can be
mapped as a field rather than as a scatter of sites.

The arithmetic is not reimplemented: every node goes through
:func:`frer.single.pull_site` and :func:`frer.single.calc_site`, the same pair
``frer.calc.calc_design_values`` uses for the batch workflow and that
``tests/test_benchmark.py`` validates against the committed reference values.
What is new here is only the geometry, the resume ledger, and the file format,
which is what ``tests/test_grid.py`` covers.

Grid spacing is a sampling decision, not a display one. The NSHM is computed on
a coarse grid and the hazard service interpolates between its nodes, so
sampling far finer than the model fetches many points that carry no additional
information. :func:`preflight` measures that directly before a long build
commits to a spacing.
"""

import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__, cache, landmask, usgs_api
from .params import DEFAULT_SITE_CLASSES, NSHM_BOUNDS, VERSION_NSHM
from .single import (
    SD1_COLUMN,
    SD1FR_COLUMN,
    SDS_COLUMN,
    SDSFR_COLUMN,
    calc_site,
    design_value_columns,
    pull_site,
)

# Coordinates are rounded to this many decimals everywhere, so a node has one
# spelling in the ledger, the output and any comparison against it. Four places
# is about 11 m, far finer than any grid step worth using.
COORD_DECIMALS = 4

# Written with a fixed format so an unchanged cache reproduces an identical
# file and `git diff` on the output means something.
FLOAT_FORMAT = "%.6f"

KEY_COLUMNS = ["lat", "lon", "site_class"]
VALUE_COLUMNS = [SDS_COLUMN, SD1_COLUMN, SDSFR_COLUMN, SD1FR_COLUMN]
GRID_COLUMNS = KEY_COLUMNS + VALUE_COLUMNS

# Ratios are derived on read rather than stored: a stored derived value is a
# second place for the number to be wrong. Note that each divides a RotD50
# FREr value by a RotD100 DE value, so the ratio is not orientation-consistent
# -- the same comparison the app already presents as its "vs DE" delta, and
# one any legend drawn from it has to spell out.
RATIO_COLUMNS = {
    "Sdsfr_over_Sds": (SDSFR_COLUMN, SDS_COLUMN),
    "Sd1fr_over_Sd1": (SD1FR_COLUMN, SD1_COLUMN),
}


@dataclass(frozen=True)
class GridSpec:
    """A rectangular lat/lon grid, inclusive of both bounds.

    ``step`` is in degrees. The bounds are expected to be multiples of it, so
    that nodes land on the graticule the hazard model is most likely defined
    on; :meth:`validate` says so rather than silently shifting them.
    """

    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    step: float = 0.05
    name: str = "grid"
    # The vendored coastline this grid is drawn against, by name. It does NOT
    # decide which nodes are calculated -- every node in the rectangle is. It
    # decides which cells a map shades and which rows feed a colour scale.
    #
    # It briefly did filter nodes, to skip open water. That stopped making
    # sense once the extent was trimmed: almost all remaining water is within
    # one cell of land, so skipping it saved four hours and cost the map its
    # coastline, because a cell the shore passes through needs its seaward
    # corners to interpolate from. Water rows are interpolation support; they
    # are never presented as design values.
    coastline: str | None = None

    def validate(self) -> None:
        """Raise :class:`ValueError` if the grid is unusable."""
        if self.step <= 0:
            raise ValueError(f"step must be positive, got {self.step}")
        if self.lat_min >= self.lat_max or self.lon_min >= self.lon_max:
            raise ValueError("grid bounds are empty or inverted")

        lon_min, lat_min, lon_max, lat_max = NSHM_BOUNDS
        outside = (
            self.lat_min < lat_min
            or self.lat_max > lat_max
            or self.lon_min < lon_min
            or self.lon_max > lon_max
        )
        if outside:
            raise ValueError(
                f"{self.name} extends outside the conterminous-U.S. hazard models "
                f"(latitude {lat_min} to {lat_max}, longitude {lon_min} to {lon_max})"
            )

        for label, value in (
            ("lat_min", self.lat_min),
            ("lat_max", self.lat_max),
            ("lon_min", self.lon_min),
            ("lon_max", self.lon_max),
        ):
            offset = abs(value / self.step - round(value / self.step))
            if offset > 1e-6:
                raise ValueError(
                    f"{label}={value} is not a multiple of step={self.step}; "
                    "nodes would fall off the graticule"
                )

    def axis(self, low: float, high: float) -> np.ndarray:
        """The node coordinates along one axis, inclusive of both ends."""
        count = int(round((high - low) / self.step)) + 1
        return np.round(low + self.step * np.arange(count), COORD_DECIMALS)

    @property
    def lats(self) -> np.ndarray:
        return self.axis(self.lat_min, self.lat_max)

    @property
    def lons(self) -> np.ndarray:
        return self.axis(self.lon_min, self.lon_max)

    def nodes(self) -> pd.DataFrame:
        """Every node as a ``lat``/``lon`` frame, in a stable order."""
        self.validate()
        lat, lon = np.meshgrid(self.lats, self.lons, indexing="ij")
        return pd.DataFrame({"lat": lat.ravel(), "lon": lon.ravel()})

    def __len__(self) -> int:
        return len(self.lats) * len(self.lons)

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "lat_min": self.lat_min,
            "lat_max": self.lat_max,
            "lon_min": self.lon_min,
            "lon_max": self.lon_max,
            "step": self.step,
            "nodes": len(self),
            "coastline": self.coastline,
        }


# The built-up core of the San Francisco Bay Area, at half the spacing the grid
# first used. Its bounds are multiples of 0.025 and it contains six usgs_160
# sites whose committed cache and reference values make them exact-node
# validation points.
#
# The extent was trimmed in from the nine-county rectangle (37.10-38.40,
# -122.80 to -121.70) because most of what it gained was ocean and empty hill
# country, and the spacing was halved because 0.05 deg was measurably too
# coarse: interpolating between nodes departed from the service by about 2% at
# the median and 8.2% at worst, against the 2% the rest of the project treats as
# agreement. Measured on the two worst cells, halving the step cut that by 2.8x.
# Trimming paid for the refinement -- 4x the areal resolution for 1.45x the
# calls.
#
# Note that every node of the old 0.05 grid is still a node here, so values
# already pulled at the coarser spacing remain usable.
BAY_AREA = GridSpec(
    lat_min=37.30,
    lat_max=38.15,
    lon_min=-122.55,
    lon_max=-121.85,
    step=0.025,
    name="bay_area",
    # Every node in the rectangle is calculated, water included. 753 of the
    # 1,015 are on land; the other 262 exist so that a map can be interpolated
    # out to the shoreline instead of stopping up to 1.36 miles short of it,
    # and are clipped away when it is drawn. Measured across four coastal
    # nodes, hazard steps 6.3% across the shore against 9.3% inland, so there
    # is no discontinuity to interpolate over.
    coastline=landmask.DEFAULT_MASK,
)


def guard_not_in_tests(path: Path, what: str) -> Path:
    """Refuse a path inside ``tests/``; that tree is committed input.

    The frozen USGS responses under ``tests/usgs_cache`` are the benchmark's
    input and must never be written by a build. Checking the resolved path
    makes a stray ``--cache-dir`` argument fail loudly instead of quietly
    contaminating the fixtures.
    """
    resolved = Path(path).resolve()
    for parent in (resolved, *resolved.parents):
        if parent.name == "tests":
            raise ValueError(
                f"{what} resolves inside a tests/ directory ({resolved}); "
                "that tree holds committed fixtures and is never written to"
            )
    return resolved


def add_ratios(table: pd.DataFrame) -> pd.DataFrame:
    """Return ``table`` with the FREr/DE ratio columns added.

    A zero denominator yields NaN rather than an infinity, so a missing or
    degenerate DE value cannot masquerade as an enormous ratio.
    """
    out = table.copy()
    for name, (numerator, denominator) in RATIO_COLUMNS.items():
        bottom = out[denominator].replace(0, np.nan)
        out[name] = out[numerator] / bottom
    return out


def _row(result, lat: float, lon: float, site_class: str) -> dict:
    """One grid row from a calculated site result."""
    row = {
        "lat": round(float(lat), COORD_DECIMALS),
        "lon": round(float(lon), COORD_DECIMALS),
        "site_class": site_class,
    }
    row.update(design_value_columns(result))
    return {column: row[column] for column in GRID_COLUMNS}


def node_values(
    lat: float,
    lon: float,
    site_class: str,
    version_id: str,
    vs30: float | None = None,
    **pull_kwargs,
) -> dict:
    """Pull and calculate one node, returning a grid row.

    ``pull_kwargs`` are handed to :func:`frer.single.pull_site`, which is how a
    caller supplies its own ``session`` or substitutes ``fetch_hazard`` to run
    without the network.
    """
    if vs30 is None:
        vs30 = DEFAULT_SITE_CLASSES[site_class]
    entry = pull_site(lat, lon, site_class, vs30, version_id, **pull_kwargs)
    return _row(calc_site(entry, version_id), lat, lon, site_class)


def rows_from_cache(
    cache_dir: Path,
    sites: pd.DataFrame,
    site_classes,
    version_id: str,
    loader=None,
) -> pd.DataFrame:
    """Grid rows for sites that already have a cached entry.

    Used by the dry run: the validation sites land exactly on grid nodes and
    their responses are already committed, so the whole builder can be checked
    against the reference values without a single API call.
    """
    if loader is None:
        loader = cache.load_entry

    rows = []
    for _, site in sites.iterrows():
        for site_class in site_classes:
            entry = loader(cache_dir, int(site["id"]), site_class)
            if entry is None:
                continue
            result = calc_site(entry, version_id)
            rows.append(_row(result, site["lat"], site["lon"], site_class))
    return sort_grid(pd.DataFrame(rows, columns=GRID_COLUMNS))


def sort_grid(table: pd.DataFrame) -> pd.DataFrame:
    """Put rows in the one order the output is ever written in."""
    return table.sort_values(KEY_COLUMNS, kind="stable").reset_index(drop=True)


# --- preflight -------------------------------------------------------------


# The departure from linear interpolation that a gridded heatmap may carry.
# Matched to the tolerance the benchmark and the validator already use to call
# two results the same, because that is the standard the rest of the project
# holds design values to.
INTERPOLATION_TOLERANCE = 0.02

PREFLIGHT_CANDIDATES = 5


def _preflight_candidates(spec: GridSpec, limit: int = PREFLIGHT_CANDIDATES):
    """``(lat, lon)`` starting points for the spacing check, best first.

    Taken from the middle of the grid rather than a corner. The hazard service
    refuses the occasional individual coordinate -- answering HTTP 200 with
    ``status: error`` in a tenth of a second while all eight of that point's
    neighbours compute normally -- and one such point should not decide whether
    a day-long build starts. Walking outward in latitude retries a refused
    triple on a different row.
    """
    available = {
        (round(float(row.lat), COORD_DECIMALS), round(float(row.lon), COORD_DECIMALS))
        for row in spec.nodes().itertuples()
    }
    if not available:
        raise ValueError(f"{spec.name} has no nodes to sample")
    lats = sorted({lat for lat, _ in available})
    step = spec.step

    # A candidate needs its eastern neighbour to be a node as well, which is
    # false on the last column. Cells wholly on land are preferred: the map
    # that has to be adequate is the land one, and water nodes are only there
    # to interpolate from. A water cell is still usable if a row has no other.
    pairs = []
    middle = lats[len(lats) // 2]
    for lat in sorted(lats, key=lambda value: abs(value - middle)):
        lons = sorted(lon for other, lon in available if other == lat)
        usable = [
            lon for lon in lons if (lat, round(lon + step, COORD_DECIMALS)) in available
        ]
        if spec.coastline:
            on_land = [
                lon
                for lon in usable
                if landmask.is_land(lon, lat, spec.coastline)
                and landmask.is_land(
                    round(lon + step, COORD_DECIMALS), lat, spec.coastline
                )
            ]
            usable = on_land or usable
        if usable:
            pairs.append((lat, usable[len(usable) // 2]))
        if len(pairs) >= limit:
            break
    if not pairs:
        raise ValueError(
            f"{spec.name} has no node with an eastern neighbour to compare against"
        )
    return pairs


def preflight(
    spec: GridSpec,
    site_class: str = "BC",
    version_id: str = "ASCE 7-28",
    tolerance: float = INTERPOLATION_TOLERANCE,
    candidates: int = PREFLIGHT_CANDIDATES,
    **pull_kwargs,
) -> dict:
    """Measure what sampling at ``spec.step`` costs the finished heatmap.

    Calculates three points on one row: a node, the midpoint to its eastern
    neighbour, and that neighbour. The database stores the two endpoints and a
    map interpolates between them, so the question that matters is how far that
    interpolation lands from what the service itself returns in between.

    This used to test something else: whether the midpoint came back *identical*
    to one of its neighbours, which would mean the service snapped to a grid no
    finer than ``step``. It does not snap -- it returns continuously varying
    values -- so that test could never pass at any step, and the advice it gave
    on failure ("reduce --step") could not have helped. What it reported as a
    failure was the genuine change in hazard across half a cell, which the
    database captures at its endpoints and does not lose.

    Costs three hazard calls per attempt. A triple containing a coordinate the
    service refuses is retried on another row rather than ending the run;
    ``checked`` is False only if every candidate was refused. Returns a dict
    recorded in the manifest, so a finished database carries the evidence for
    its own resolution.
    """
    attempts: list[str] = []
    for lat, lon in _preflight_candidates(spec, candidates):
        points = {
            "node": lon,
            "midpoint": round(lon + spec.step / 2, COORD_DECIMALS + 1),
            "neighbour": round(lon + spec.step, COORD_DECIMALS),
        }
        try:
            values = {
                label: node_values(
                    lat, point_lon, site_class, version_id, **pull_kwargs
                )
                for label, point_lon in points.items()
            }
        except Exception as error:  # a coordinate the service will not compute
            attempts.append(f"({lat:.4f}, {lon:.4f}) {type(error).__name__}: {error}")
            continue
        break
    else:
        return {
            "checked": False,
            "adequate": False,
            "attempts": attempts,
            "verdict": (
                f"could not evaluate: the service refused all {len(attempts)} "
                "candidate triple(s)"
            ),
        }

    def spread(a: dict, b: dict) -> float:
        return max(
            abs(a[column] - b[column]) / abs(b[column]) if b[column] else 0.0
            for column in VALUE_COLUMNS
        )

    # How much hazard changes across half a cell. Real variation, captured by
    # the two stored endpoints -- reported for context, never a pass or fail.
    to_node = spread(values["midpoint"], values["node"])
    to_neighbour = spread(values["midpoint"], values["neighbour"])

    # What a map actually gets wrong between nodes: the midpoint against the
    # straight line joining its neighbours. The trend across the cell is
    # captured by the endpoints; this is the curvature the grid cannot carry.
    straight = {
        column: (values["node"][column] + values["neighbour"][column]) / 2
        for column in VALUE_COLUMNS
    }
    error = spread(values["midpoint"], straight)
    adequate = error <= tolerance

    return {
        "checked": True,
        "attempts": attempts,
        "lat": lat,
        "lons": points,
        "site_class": site_class,
        "version": version_id,
        "step": spec.step,
        "midpoint_vs_node": to_node,
        "midpoint_vs_neighbour": to_neighbour,
        "interpolation_error": error,
        "tolerance": tolerance,
        "adequate": bool(adequate),
        "verdict": (
            f"interpolating between {spec.step} deg nodes is within {error:.2%} "
            f"of the service (tolerance {tolerance:.0%}); hazard itself moves "
            f"{max(to_node, to_neighbour):.1%} across half a cell, which the "
            "stored endpoints carry"
            if adequate
            else (
                f"interpolating between {spec.step} deg nodes departs from the "
                f"service by {error:.2%}, over the {tolerance:.0%} tolerance; "
                "a map built from this grid would misread between nodes. "
                "Reduce --step."
            )
        ),
    }


# --- build -----------------------------------------------------------------


def _ledger_path(work_dir: Path, spec: GridSpec, version_id: str) -> Path:
    slug = version_id.lower().replace(" ", "").replace("/", "-")
    return Path(work_dir) / f"{spec.name}_{slug}.partial.csv"


def read_ledger(path: Path) -> pd.DataFrame:
    """Rows completed by an earlier run, or an empty frame."""
    if not Path(path).exists():
        return pd.DataFrame(columns=GRID_COLUMNS)
    table = pd.read_csv(path)
    table["lat"] = table["lat"].round(COORD_DECIMALS)
    table["lon"] = table["lon"].round(COORD_DECIMALS)
    return table


def build_grid(
    spec: GridSpec,
    site_classes,
    version_id: str,
    work_dir: Path,
    pause=usgs_api.PAUSE_RANGE,
    limit: int | None = None,
    progress=None,
    failures_out: list | None = None,
    **pull_kwargs,
) -> pd.DataFrame:
    """Calculate every (node, site class), resuming from the ledger.

    Each completed row is appended to a plain-CSV ledger under ``work_dir``
    before the next call is made, so an interrupted run restarts where it
    stopped. ``work_dir`` is refused if it resolves inside ``tests/``.

    A node that fails after its retries is left out of the ledger rather than
    recorded as a gap, so a later run picks it up -- the same resume-on-failure
    behaviour as :func:`frer.pull.pull_sites`. Pass a list as ``failures_out``
    to receive one dict per failure, which is how a caller tells a node the
    service permanently refuses from one it has simply not reached yet.
    """
    spec.validate()
    work_dir = guard_not_in_tests(work_dir, "work_dir")
    work_dir.mkdir(parents=True, exist_ok=True)
    ledger = _ledger_path(work_dir, spec, version_id)

    done = read_ledger(ledger)
    seen = set(map(tuple, done[KEY_COLUMNS].to_numpy())) if len(done) else set()

    nodes = spec.nodes()
    wanted = [
        (float(node.lat), float(node.lon), site_class)
        for site_class in site_classes
        for node in nodes.itertuples()
    ]
    todo = [key for key in wanted if key not in seen]
    if limit is not None:
        todo = todo[:limit]

    if progress:
        progress(
            f"{len(wanted):,} (node, class) pairs; {len(seen):,} already done; "
            f"{len(todo):,} to calculate"
        )

    failures = 0
    for index, (lat, lon, site_class) in enumerate(todo, start=1):
        label = f"{lat:.4f},{lon:.4f} [{site_class}]"
        try:
            row = node_values(lat, lon, site_class, version_id, **pull_kwargs)
        except Exception as error:  # exhausted retries, bad payload, bounds
            failures += 1
            if failures_out is not None:
                failures_out.append(
                    {
                        "lat": round(lat, COORD_DECIMALS),
                        "lon": round(lon, COORD_DECIMALS),
                        "site_class": site_class,
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
            if progress:
                progress(f"  FAIL  {label}: {error}")
        else:
            header = not ledger.exists()
            pd.DataFrame([row], columns=GRID_COLUMNS).to_csv(
                ledger, mode="a", header=header, index=False, float_format=FLOAT_FORMAT
            )
            if progress and (index % 25 == 0 or index == len(todo)):
                progress(f"  {index:,}/{len(todo):,}  {label}")
        usgs_api.pace(pause)

    if progress and failures:
        progress(f"{failures:,} node(s) failed and were left for a later run")
    return sort_grid(read_ledger(ledger))


# --- files -----------------------------------------------------------------


def write_grid(table: pd.DataFrame, path: Path) -> Path:
    """Write the grid CSV, gzipped reproducibly.

    gzip records a modification time by default, which would make two builds of
    identical data differ byte for byte and rob `git diff` of its meaning;
    ``mtime=0`` pins it.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sort_grid(table).to_csv(
        path,
        index=False,
        float_format=FLOAT_FORMAT,
        compression={"method": "gzip", "mtime": 0},
    )
    return path


def load_grid(path: Path) -> pd.DataFrame:
    """Read a grid CSV written by :func:`write_grid`."""
    table = pd.read_csv(path)
    missing = set(GRID_COLUMNS) - set(table.columns)
    if missing:
        raise ValueError(f"{path} is missing column(s) {sorted(missing)}")
    return table


def git_sha(repo_root: Path) -> str | None:
    """The current commit, or None outside a repository."""
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip() or None


def build_manifest(
    spec: GridSpec,
    site_classes,
    version_id: str,
    table: pd.DataFrame,
    preflight_result: dict | None = None,
    repo_root: Path | None = None,
    built_at: str | None = None,
    refused: list | None = None,
) -> dict:
    """Provenance for a finished grid.

    Enough to tell, later, whether a database still matches the code and the
    inputs that produced it -- which is what makes a stale one detectable
    rather than quietly wrong.
    """
    return {
        "grid": spec.as_dict(),
        "version": version_id,
        "nshm": VERSION_NSHM[version_id],
        "site_classes": {
            site_class: DEFAULT_SITE_CLASSES[site_class] for site_class in site_classes
        },
        "columns": GRID_COLUMNS,
        "rows": int(len(table)),
        "calculator_version": __version__,
        "git_sha": git_sha(repo_root) if repo_root else None,
        "built_at": built_at
        or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "preflight": preflight_result or {"checked": False},
        # The coastline the finished map is drawn against. Pinned by digest, so
        # a database built against one shoreline and drawn against another is
        # detectable rather than quietly different.
        "coastline": spec.coastline,
        "coastline_sha256": (
            landmask.digest(spec.coastline) if spec.coastline else None
        ),
        # (node, class) pairs the hazard service would not compute. Recorded so
        # that a gap which is the service's rather than the build's is declared
        # instead of inferred, and so validation can insist on exactly these.
        "refused": sorted(
            (dict(entry) for entry in (refused or [])),
            key=lambda entry: (entry["site_class"], entry["lat"], entry["lon"]),
        ),
    }


def write_manifest(manifest: dict, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", "utf-8")
    return path


def load_manifest(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# --- comparison ------------------------------------------------------------


def relative_differences(
    actual: pd.DataFrame, expected: pd.DataFrame, on=("name", "site_class")
) -> pd.DataFrame:
    """Per-row relative difference on the value columns.

    ``on`` defaults to name and site class rather than ``id`` deliberately. The
    ``id`` in a reference CSV is that site set's own index -- nehrp_34 numbers
    its sites 1 to 34 and usgs_160 numbers its own 1 to 160 -- so joining two
    sets on ``id`` lines up entirely different cities and reports differences
    of over 100%.
    """
    on = list(on)
    merged = actual.merge(expected, on=on, suffixes=("_actual", "_expected"))
    if len(merged) != len(actual):
        raise ValueError(
            f"joined {len(merged)} of {len(actual)} rows on {on}; "
            "the two tables do not describe the same points"
        )

    out = merged[on].copy()
    for column in VALUE_COLUMNS:
        a, b = merged[f"{column}_actual"], merged[f"{column}_expected"]
        out[column] = (a - b).abs() / b.abs().where(b != 0, other=np.nan)
    return out


def worst_differences(differences: pd.DataFrame) -> dict:
    """Max and median relative difference per value column."""
    return {
        column: {
            "max": float(differences[column].max()),
            "median": float(differences[column].median()),
        }
        for column in VALUE_COLUMNS
        if column in differences
    }
