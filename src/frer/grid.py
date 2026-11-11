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

See ``docs/phase-2-gridded-heatmap.md`` for the plan this implements.
"""

import json
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__, cache
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
        }


# The nine-county core of the San Francisco Bay Area. Its bounds are multiples
# of 0.05 and it contains six usgs_160 sites whose committed cache and
# reference values make them exact-node validation points.
BAY_AREA = GridSpec(
    lat_min=37.10,
    lat_max=38.40,
    lon_min=-122.80,
    lon_max=-121.70,
    step=0.05,
    name="bay_area",
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


def preflight(
    spec: GridSpec,
    site_class: str = "BC",
    version_id: str = "ASCE 7-28",
    tolerance: float = 1e-9,
    **pull_kwargs,
) -> dict:
    """Check that ``spec.step`` is fine enough for the hazard model.

    Calculates three points on one row: a node, the midpoint to its neighbour,
    and that neighbour. If the midpoint's design values match one of the two
    exactly, the service is snapping to a grid no finer than ``step`` and
    sampling at ``step`` loses nothing. If it differs from both, the model
    resolves finer and the grid is undersampling it.

    Costs three hazard calls. Returns a dict recorded in the manifest, so a
    finished database carries the evidence for its own resolution.
    """
    lat = float(spec.lats[len(spec.lats) // 2])
    lon = float(spec.lons[0])
    points = {
        "node": lon,
        "midpoint": round(lon + spec.step / 2, COORD_DECIMALS + 1),
        "neighbour": round(lon + spec.step, COORD_DECIMALS),
    }

    values = {
        label: node_values(lat, point_lon, site_class, version_id, **pull_kwargs)
        for label, point_lon in points.items()
    }

    def spread(a: dict, b: dict) -> float:
        return max(
            abs(a[column] - b[column]) / abs(b[column]) if b[column] else 0.0
            for column in VALUE_COLUMNS
        )

    to_node = spread(values["midpoint"], values["node"])
    to_neighbour = spread(values["midpoint"], values["neighbour"])
    snapping = min(to_node, to_neighbour) <= tolerance

    return {
        "checked": True,
        "lat": lat,
        "lons": points,
        "site_class": site_class,
        "version": version_id,
        "step": spec.step,
        "midpoint_vs_node": to_node,
        "midpoint_vs_neighbour": to_neighbour,
        "adequate": bool(snapping),
        "verdict": (
            f"midpoint matches a neighbouring node; step {spec.step} deg captures "
            "the model's resolution"
            if snapping
            else (
                f"midpoint differs from both neighbours (by {to_node:.3%} and "
                f"{to_neighbour:.3%}); the model resolves finer than {spec.step} deg "
                "and this grid would undersample it"
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
    pause: float = 1.0,
    limit: int | None = None,
    progress=None,
    **pull_kwargs,
) -> pd.DataFrame:
    """Calculate every (node, site class), resuming from the ledger.

    Each completed row is appended to a plain-CSV ledger under ``work_dir``
    before the next call is made, so an interrupted run restarts where it
    stopped. ``work_dir`` is refused if it resolves inside ``tests/``.

    A node that fails after its retries is left out of the ledger rather than
    recorded as a gap, so a later run picks it up -- the same resume-on-failure
    behaviour as :func:`frer.pull.pull_sites`.
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
            if progress:
                progress(f"  FAIL  {label}: {error}")
        else:
            header = not ledger.exists()
            pd.DataFrame([row], columns=GRID_COLUMNS).to_csv(
                ledger, mode="a", header=header, index=False, float_format=FLOAT_FORMAT
            )
            if progress and (index % 25 == 0 or index == len(todo)):
                progress(f"  {index:,}/{len(todo):,}  {label}")
        time.sleep(pause)

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
