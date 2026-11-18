"""Render a field of the gridded database as an image for the map overlay.

The map in :mod:`map_view` answers "what are the values *here*". This answers
"how do they vary across the region": one of six fields, interpolated between
grid nodes, clipped to the coastline and handed to deck.gl as a single image.

**An image, not drawn cells.** While a selection is live Streamlit rewrites the
colour accessor of every layer whose deck.gl type appears in a table it keeps
in its frontend bundle, fading whatever it does not consider selected to 40%
alpha. ``ColumnLayer`` is in that table; a heatmap drawn as cells would wash
out the moment someone clicked a site pin -- the bug already fixed once for the
pins. ``BitmapLayer`` is absent from the table entirely, so it cannot be
touched. Verified against Streamlit 1.60.0 by reading the table and by probing
a live map; see ``tests/test_heatmap.py``.

**Water is drawn from, not drawn.** The database carries 262 nodes in water so
that a cell the shoreline passes through has seaward corners to interpolate
from. Those rows reach the raster and then the coastline cuts them away, which
is what lets the surface run to the shore instead of stopping up to 1.36 miles
short of it. They never reach the colour scale, which is computed from land
rows alone, because the scale has to describe the values the map actually
shows and no water value is ever shown.

That is a smaller correction than it once was, and worth stating accurately:
at the original wide extent the water nodes reached the Gulf of the Farallones
and ran far below every land value, so excluding them bought real dynamic
range. At the trimmed extent the remaining water is San Francisco Bay and a
narrow coastal strip, which is seismically much like the land beside it --
class D S_DS spans 1.15-1.92 on land against 1.03-1.94 in water, and only 10
of the 262 water nodes fall below the land minimum. Excluding them now moves
the bottom of the ramp by about a tenth and the top not at all. It is a
question of the legend being honest, not of the picture being washed out.

**The hover readout is the same surface, sampled.** Streamlit's deck chart
reports which *object* was picked and throws deck.gl's cursor coordinate away,
so a continuous readout is not available to us: the value under the pointer has
to be precomputed somewhere and looked up. :func:`sample_surface` evaluates the
same bilinear surface the image is coloured from on a grid finer than the
database's own, and the deck layer that carries it is
:func:`map_view.readout_layer`. Sampling at half the node spacing tracks the
surface to about 1% at the ninetieth percentile, which is the same order as the
0.89% the preflight measured between interpolation and the service itself;
going finer would render precision the database does not have, and costs four
times the bytes per halving.

Nothing here imports Streamlit or pydeck. It turns a frame into an image and a
list of sampled points; the deck layers are in :mod:`map_view`.
"""

import base64
import functools
import io
from dataclasses import dataclass

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.image import imsave
from matplotlib.path import Path as MplPath

from frer import landmask
from frer.grid import (
    RATIO_COLUMNS,
    SD1_COLUMN,
    SD1FR_COLUMN,
    SDS_COLUMN,
    SDSFR_COLUMN,
    add_ratios,
)

# The ramp, warm, with red at the top because the brief asks for red to mean
# a higher design value.
#
# This is a deliberate departure from the project's visualisation guidance,
# which says sequential magnitude is one hue light to dark. The rule exists to
# keep a rainbow out, and what actually makes a rainbow unreadable is that its
# lightness wanders. This ramp is multi-hue but its lightness falls at every
# step -- L* 90.8 down to 25.5, a 65-point range with no step smaller than 7 --
# so it still orders correctly in greyscale and under any colour-vision
# deficiency, which is the property the rule is protecting. Checked, not
# assumed; the test that pins it is in tests/test_heatmap.py.
#
# It starts at a warm tan rather than at the near-white the published scheme
# begins with, because the fill is drawn at FILL_ALPHA and a pale yellow at
# that alpha over a pale basemap is indistinguishable from no data at all.
SEQUENTIAL_RAMP = (
    "#fee391",
    "#fec44f",
    "#fe9929",
    "#f57021",
    "#ec4f1d",
    "#d7301f",
    "#b30019",
    "#7f0014",
)

COLORMAP = LinearSegmentedColormap.from_list("frer_sequential", SEQUENTIAL_RAMP)

# Output resolution of the fill. The contour lines are vector now, so this
# only has to be fine enough that the band edges and the coastline do not
# show their pixels: at 700 px/deg a pixel is about 125 m, well under the
# 2.2 km the data itself resolves.
PIXELS_PER_DEGREE = 700

# The land/water edge is computed at this multiple of the output resolution
# and averaged down, so a pixel the shoreline crosses takes partial alpha
# instead of a hard jag. It is the whole cost of a first render -- a
# point-in-polygon test per subpixel -- which is why it is worth keeping the
# fill's resolution modest and spending the budget here instead.
EDGE_SUPERSAMPLE = 2

# Discrete bands rather than a continuous wash. A smooth ramp at an opacity
# low enough to read the streets through carries almost no contrast -- the
# complaint this answers -- because the eye reads absolute lightness badly
# and edges well. Quantising creates edges, and gives each colour a value
# interval the legend can name instead of a position on a gradient.
#
# Eight is enough steps to show the shape of the surface and few enough that
# each band is unambiguous against the legend. It also makes the image far
# smaller, since flat regions compress and a gradient does not.
BANDS = 8

# How much of the basemap the fill takes. Low, because the structure is
# carried by the contour lines and the exact value by the hover readout, so
# the wash only has to say roughly how high this is. In the image rather than
# on the deck layer: layer opacity would scale the lines with it.
FILL_ALPHA = 0.18

# The contour lines are drawn by deck.gl, not baked into the image, which is
# what makes them crisp at every zoom instead of a magnified raster.
#
# That puts them in a PathLayer, which declares getColor and so sits in the
# table Streamlit rewrites while a selection is live. The rewrite clamps an
# unselected object with `Math.min(alpha, 102)` -- so a line drawn at 102 is
# already at that floor and the fade cannot move it. Confirmed on a live map:
# at 255 the lines visibly dim when a pin is clicked, at 102 they do not.
CONTOUR_ALPHA = 102

# Each line takes the colour of the band above it, darkened, so that it still
# reads against its own fill at the pale end of the ramp.
CONTOUR_DARKEN = 0.78

# The surface the contours are traced on, as a refinement of the node grid.
# Four is where the curves stop looking faceted; past it the vertex count
# doubles for a smoothness the data does not actually carry.
CONTOUR_REFINE = 4


@dataclass(frozen=True)
class Field:
    """One selectable quantity, and how to label it."""

    key: str
    label: str
    column: str
    unit: str
    description: str

    @property
    def is_ratio(self) -> bool:
        return self.column in RATIO_COLUMNS


# The six fields the overlay offers. Keys are stable; labels are display text.
#
# The two ratios divide a RotD50 FREr value by a RotD100 DE value, so they are
# not orientation-consistent. That is the same comparison the app already
# presents as its "vs DE" delta, so the overlay is consistent with the rest of
# the tool -- but any legend drawn from one has to say so, which is why the
# mixing is spelled out in `description` rather than left to a footnote.
FIELDS: dict[str, Field] = {
    field.key: field
    for field in (
        Field(
            key="sds",
            label="S_DS",
            column=SDS_COLUMN,
            unit="g",
            description="DE short-period design value, RotD100",
        ),
        Field(
            key="sd1",
            label="S_D1",
            column=SD1_COLUMN,
            unit="g",
            description="DE 1-second design value, RotD100",
        ),
        Field(
            key="sdsfr",
            label="S_DSfr",
            column=SDSFR_COLUMN,
            unit="g",
            description="FREr short-period design value, RotD50",
        ),
        Field(
            key="sd1fr",
            label="S_D1fr",
            column=SD1FR_COLUMN,
            unit="g",
            description="FREr 1-second design value, RotD50",
        ),
        Field(
            key="sdsfr_over_sds",
            label="S_DSfr / S_DS",
            column="Sdsfr_over_Sds",
            unit="",
            description="FREr over DE, short period (RotD50 over RotD100)",
        ),
        Field(
            key="sd1fr_over_sd1",
            label="S_D1fr / S_D1",
            column="Sd1fr_over_Sd1",
            unit="",
            description="FREr over DE, 1 second (RotD50 over RotD100)",
        ),
    )
}


@dataclass(frozen=True)
class Overlay:
    """A rendered field, ready to hand to a ``BitmapLayer``."""

    png: bytes
    #: ``[west, south, east, north]`` -- the image's edges, not its pixel
    #: centres, which is what deck.gl's ``bounds`` means.
    bounds: tuple[float, float, float, float]
    vmin: float
    vmax: float
    field: Field
    site_class: str
    #: How many nodes fed the colour scale. Land only; see the module docstring.
    land_nodes: int
    #: How many discrete steps the surface was quantised into.
    bands: int = BANDS

    @property
    def bounds_of_bands(self) -> list[float]:
        """The ``bands + 1`` values the band edges fall on."""
        return legend_ticks(self.vmin, self.vmax, self.bands + 1)

    @property
    def data_uri(self) -> str:
        """The image as a ``data:`` URI.

        Note for the caller: pydeck reads a bare string layer argument as a
        deck.gl accessor expression, so this has to be passed through
        ``map_view._literal`` or the whole deck fails to parse in the browser.
        """
        return "data:image/png;base64," + base64.b64encode(self.png).decode("ascii")


class HeatmapError(Exception):
    """The overlay cannot be rendered from what was given."""


def check_coastline(manifest: dict, coastline: str | None) -> None:
    """The database must have been built against the shoreline we will clip to.

    The manifest pins the coastline by digest precisely so a map drawn against
    a re-vendored polygon is detectable rather than quietly different -- the
    clip would move while the values stayed put.
    """
    if manifest.get("coastline") != coastline:
        raise HeatmapError(
            f"database built against coastline {manifest.get('coastline')!r}, "
            f"drawing against {coastline!r}"
        )
    if coastline is None:
        return
    current = landmask.digest(coastline)
    if manifest.get("coastline_sha256") != current:
        raise HeatmapError(
            f"coastline {coastline!r} has changed since the build "
            f"({manifest.get('coastline_sha256')} then, {current} now); "
            "the map would clip somewhere the database does not know about"
        )


def field_values(table: pd.DataFrame, site_class: str, field: Field) -> pd.DataFrame:
    """The ``lat``/``lon``/value rows for one field and one site class."""
    rows = table[table["site_class"] == site_class]
    if rows.empty:
        classes = sorted(table["site_class"].unique())
        raise HeatmapError(f"no rows for site class {site_class!r}; have {classes}")
    if field.is_ratio:
        rows = add_ratios(rows)
    if field.column not in rows.columns:
        raise HeatmapError(f"no column {field.column!r} in the database")
    out = rows[["lat", "lon", field.column]].rename(columns={field.column: "value"})
    if out["value"].isna().any():
        raise HeatmapError(f"{field.column} has NaN values for class {site_class}")
    return out


def _grid_array(rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(lats, lons, values)`` with ``values`` indexed ``[lat, lon]``.

    The frame must be a complete rectangle; a hole would interpolate as if the
    neighbouring node were adjacent, which is a quiet way to be wrong.
    """
    pivot = rows.pivot(index="lat", columns="lon", values="value").sort_index()
    pivot = pivot.sort_index(axis=1)
    if pivot.isna().any().any():
        missing = int(pivot.isna().sum().sum())
        raise HeatmapError(f"{missing} node(s) missing from the rectangle")
    return (
        pivot.index.to_numpy(dtype=float),
        pivot.columns.to_numpy(dtype=float),
        pivot.to_numpy(dtype=float),
    )


def _upsample(values, src_lats, src_lons, out_lats, out_lons) -> np.ndarray:
    """Bilinear interpolation onto a finer regular grid.

    Separable, so it is two passes of :func:`numpy.interp` rather than a
    dependency. Linear rather than log-space on purpose: measured across nine
    cells the two were within a tenth of a percent of each other (log 1.97%
    median, linear 2.07%), because the midpoint sits above the straight line in
    some cells and below it in others, so no monotone transform helps. Step
    size is the lever that worked, and that was spent going to 0.025 deg.
    """
    along_lon = np.empty((len(src_lats), len(out_lons)))
    for row, source in enumerate(values):
        along_lon[row] = np.interp(out_lons, src_lons, source)
    out = np.empty((len(out_lats), len(out_lons)))
    for column in range(len(out_lons)):
        out[:, column] = np.interp(out_lats, src_lats, along_lon[:, column])
    return out


def band_index(normalised: np.ndarray, bands: int) -> np.ndarray:
    """Which band each pixel falls in, as ``0 .. bands - 1``.

    The top of the range belongs to the last band rather than starting a new
    one of its own, which is what the clip is for.
    """
    return np.clip((np.clip(normalised, 0.0, 1.0) * bands).astype(int), 0, bands - 1)


def _darken(rgb, factor: float) -> list[int]:
    return [int(round(channel * factor)) for channel in rgb]


def contours(
    table: pd.DataFrame,
    site_class: str,
    field_key: str,
    coastline: str | None = landmask.DEFAULT_MASK,
    bands: int = BANDS,
    limits: tuple[float, float] | None = None,
    refine: int = CONTOUR_REFINE,
) -> list[dict]:
    """The band boundaries as lon/lat paths, coloured and clipped to land.

    Traced rather than drawn into the image so that deck.gl can draw them at
    the viewport's resolution instead of the raster's -- a baked line is a
    magnified line as soon as anyone zooms in.

    Each path carries its own colour, which is why the layer that draws them
    has to declare an accessor and so has to live at ``CONTOUR_ALPHA``; see
    that constant.

    Segments are cut where they leave the land rather than drawn and clipped,
    because a BitmapLayer can be masked by its alpha channel and a PathLayer
    cannot. A run of one point is dropped: two make a line, one makes nothing.
    """
    if field_key not in FIELDS:
        raise HeatmapError(f"unknown field {field_key!r}; have {sorted(FIELDS)}")
    if bands < 2:
        raise HeatmapError(f"a banded scale needs at least two bands, got {bands}")
    field = FIELDS[field_key]
    rows = field_values(table, site_class, field)
    src_lats, src_lons, values = _grid_array(rows)
    vmin, vmax = limits if limits else colour_limits(rows, coastline)
    if vmax <= vmin:
        return []

    out_lats = _refine(src_lats, refine)
    out_lons = _refine(src_lons, refine)
    surface = _upsample(values, src_lats, src_lons, out_lats, out_lons)

    # The interior boundaries only: the two ends of the scale are the edges
    # of the data, not steps within it.
    levels = legend_ticks(vmin, vmax, bands + 1)[1:-1]
    traced = Figure().subplots().contour(out_lons, out_lats, surface, levels=levels)

    paths: list[dict] = []
    for index, segments in enumerate(traced.allsegs):
        red, green, blue, _ = COLORMAP((index + 1) / (bands - 1), bytes=True)
        colour = [*_darken((red, green, blue), CONTOUR_DARKEN), CONTOUR_ALPHA]
        for segment in segments:
            if coastline is None:
                runs = [segment]
            else:
                runs = _split_on_land(segment, coastline)
            for run in runs:
                paths.append(
                    {
                        "path": [
                            [
                                round(float(lon), CELL_DECIMALS),
                                round(float(lat), CELL_DECIMALS),
                            ]
                            for lon, lat in run
                        ],
                        "colour": colour,
                        "value": float(levels[index]),
                    }
                )
    return paths


def _split_on_land(segment: np.ndarray, coastline: str) -> list[np.ndarray]:
    """One traced line cut into the stretches that are over land."""
    inside = on_land(segment[:, 0], segment[:, 1], coastline)
    runs, start = [], None
    for index, land in enumerate(inside):
        if land and start is None:
            start = index
        elif not land and start is not None:
            if index - start > 1:
                runs.append(segment[start:index])
            start = None
    if start is not None and len(inside) - start > 1:
        runs.append(segment[start:])
    return runs


def band_colour(index: int, bands: int) -> str:
    """One band's fill, as hex, from the same ramp that colours the image.

    Taken through ``bytes=True``, the same call the raster makes, because
    matplotlib's float-to-hex rounds where its float-to-byte truncates and
    the two disagree in the last digit. Invisible on screen, but it would
    mean the legend swatch and the pixel it explains were different colours.
    """
    red, green, blue, _ = COLORMAP(index / (bands - 1), bytes=True)
    return f"#{red:02x}{green:02x}{blue:02x}"


def _inside(points: np.ndarray, polygons) -> np.ndarray:
    """Which of an ``(N, 2)`` array of ``(lon, lat)`` points are on land.

    A vectorised second opinion on :func:`frer.landmask.is_land`, which tests
    one point at a time in Python: that costs about a second per render here
    and far longer per raster. Keeping the fast path in this module rather
    than in ``landmask`` leaves that one free of a plotting dependency and its
    coastline reviewable as data; ``tests/test_heatmap.py`` pins the two
    against each other so the duplication cannot drift.
    """
    inside = np.zeros(len(points), dtype=bool)
    for rings in polygons:
        # A ring's holes are water -- San Francisco Bay is a hole in the
        # coastline, not a separate polygon -- so they subtract.
        this = MplPath(np.asarray(rings[0])).contains_points(points)
        for hole in rings[1:]:
            this &= ~MplPath(np.asarray(hole)).contains_points(points)
        inside |= this
    return inside


def _check_clip(lons, lats, clip, coastline) -> None:
    if (
        np.min(lons) < clip["west"]
        or np.max(lons) > clip["east"]
        or np.min(lats) < clip["south"]
        or np.max(lats) > clip["north"]
    ):
        raise landmask.MaskError(
            f"points run outside the {coastline!r} clip box {clip}; "
            "re-vendor the coastline with scripts/vendor_landmask.py"
        )


def on_land(lons, lats, coastline: str) -> np.ndarray:
    """Which of a paired sequence of points are on land, as a bool array."""
    polygons, clip, _ = landmask._load(coastline)
    lons = np.asarray(lons, dtype=float)
    lats = np.asarray(lats, dtype=float)
    _check_clip(lons, lats, clip, coastline)
    return _inside(np.column_stack([lons, lats]), polygons)


def land_fraction(lons, lats, coastline: str) -> np.ndarray:
    """How much of each pixel is land, in ``[0, 1]``, indexed ``[lat, lon]``.

    ``lons`` and ``lats`` are the pixel centres of a regular raster. Each
    pixel is sampled ``EDGE_SUPERSAMPLE`` times per axis and averaged, so a
    pixel the shoreline crosses comes out part-opaque and the coast reads as a
    line rather than a staircase.
    """
    return _land_fraction(
        float(lons[0]),
        float(lons[-1]),
        len(lons),
        float(lats[0]),
        float(lats[-1]),
        len(lats),
        coastline,
    ).copy()


@functools.lru_cache(maxsize=8)
def _land_fraction(lon0, lon1, nx, lat0, lat1, ny, coastline) -> np.ndarray:
    """The clip for one raster geometry, cached and shared.

    The coastline does not move when the field or the site class changes, so
    this is computed once per raster size and reused across all twenty-four
    combinations. It is the dominant cost of a render -- roughly a million
    point-in-polygon tests once the edge supersampling is counted -- and
    caching it is what keeps switching fields interactive.

    Keyed on the geometry rather than the arrays because arrays are not
    hashable. Callers get a copy, so a mutating caller cannot poison the
    cache.
    """
    lons = np.linspace(lon0, lon1, nx)
    lats = np.linspace(lat0, lat1, ny)
    polygons, clip, _ = landmask._load(coastline)
    _check_clip(lons, lats, clip, coastline)

    mesh_lon, mesh_lat = np.meshgrid(_supersampled(lons), _supersampled(lats))
    inside = _inside(np.column_stack([mesh_lon.ravel(), mesh_lat.ravel()]), polygons)
    coarse = inside.reshape(ny, EDGE_SUPERSAMPLE, nx, EDGE_SUPERSAMPLE)
    return coarse.mean(axis=(1, 3))


def _supersampled(centres: np.ndarray) -> np.ndarray:
    """``EDGE_SUPERSAMPLE`` sub-centres spanning each pixel of ``centres``."""
    step = centres[1] - centres[0]
    offsets = (np.arange(EDGE_SUPERSAMPLE) + 0.5) / EDGE_SUPERSAMPLE - 0.5
    return (centres[:, None] + offsets * step).ravel()


def colour_limits(rows: pd.DataFrame, coastline: str | None) -> tuple[float, float]:
    """The range the colour scale spans, from land rows only.

    Water is clipped out of the image, so a scale that included it would put
    numbers on the legend that appear nowhere on the map. See the module
    docstring for how much this actually moves the ramp at this extent --
    less than it would have at the original one.
    """
    if rows.empty:
        raise HeatmapError("no land nodes to scale the colours from")
    if coastline is None:
        values = rows["value"]
    else:
        values = rows.loc[on_land(rows["lon"], rows["lat"], coastline), "value"]
    if values.empty:
        raise HeatmapError("no land nodes to scale the colours from")
    return float(values.min()), float(values.max())


def render(
    table: pd.DataFrame,
    site_class: str,
    field_key: str,
    coastline: str | None = landmask.DEFAULT_MASK,
    pixels_per_degree: int = PIXELS_PER_DEGREE,
    limits: tuple[float, float] | None = None,
    bands: int = BANDS,
) -> Overlay:
    """Render one field of one site class to an overlay image.

    ``limits`` pins the colour scale, for holding it steady while the user
    switches site class; by default it is taken from this class's land rows.

    The image carries its own alpha -- a faint fill and near-solid contour
    lines -- and is meant to be drawn at layer opacity 1. Putting the alpha
    on the layer instead would scale the two together, which is what made a
    wash faint enough to see the streets through also too faint to read.
    """
    if bands < 2:
        raise HeatmapError(f"a banded scale needs at least two bands, got {bands}")
    if field_key not in FIELDS:
        raise HeatmapError(f"unknown field {field_key!r}; have {sorted(FIELDS)}")
    field = FIELDS[field_key]
    rows = field_values(table, site_class, field)
    src_lats, src_lons, values = _grid_array(rows)

    half = 0.5 / pixels_per_degree
    width = max(2, round((src_lons[-1] - src_lons[0]) * pixels_per_degree))
    height = max(2, round((src_lats[-1] - src_lats[0]) * pixels_per_degree))
    out_lons = np.linspace(src_lons[0], src_lons[-1], width)
    out_lats = np.linspace(src_lats[0], src_lats[-1], height)

    surface = _upsample(values, src_lats, src_lons, out_lats, out_lons)
    vmin, vmax = limits if limits else colour_limits(rows, coastline)
    span = vmax - vmin
    normalised = (surface - vmin) / span if span > 0 else np.zeros_like(surface)

    band = band_index(normalised, bands)
    rgba = COLORMAP(band / (bands - 1), bytes=True)
    ink = np.full(surface.shape, FILL_ALPHA)

    if coastline is None:
        land = np.ones_like(surface)
        land_nodes = len(rows)
    else:
        land = land_fraction(out_lons, out_lats, coastline)
        land_nodes = int(on_land(rows["lon"], rows["lat"], coastline).sum())
    rgba[..., 3] = np.round(ink * land * 255).astype(np.uint8)

    buffer = io.BytesIO()
    # deck.gl reads the first row of the image as the northern edge; the raster
    # is built south-first, so it is flipped rather than the bounds inverted.
    imsave(buffer, np.flipud(rgba), format="png")

    return Overlay(
        png=buffer.getvalue(),
        bounds=(
            float(src_lons[0] - half),
            float(src_lats[0] - half),
            float(src_lons[-1] + half),
            float(src_lats[-1] + half),
        ),
        vmin=vmin,
        vmax=vmax,
        field=field,
        site_class=site_class,
        land_nodes=land_nodes,
        bands=bands,
    )


# How finely the hover readout samples the surface, as a divisor of the
# database's own node spacing. Two gives cells of 0.0125 deg -- about 1.1 by
# 1.4 km -- which at the region-wide view is a few pixels across, so the
# number changes as the pointer moves rather than stepping visibly.
#
# This is a display choice, not a data one, and the two errors do not mix.
# Sampling buys nothing against the USGS service: the manifest's preflight
# records interpolation between 0.025 deg nodes as within 0.89% of it, and
# only more API calls could move that. It buys fidelity against our own
# interpolated surface, which is what the reader is actually being shown.
#
# Measured over 60,000 random land points, against the deck the app really
# sends (gzipped, because the websocket deflates and the repeated label text
# compresses to almost nothing):
#
#     divisor   cells    S_DS p90   worst field   deck gzipped   serialise
#           1     753       2.17%         4.36%         133 kB         1 ms
#           2   2,925       1.07%         2.09%         146 kB         4 ms
#           4  11,544       0.53%         1.03%         192 kB        23 ms
#
# Two is the default because its worst field is the same order as the data's
# own 0.89%, and because the serialise cost is paid on every rerun -- every
# slider nudge -- while the accuracy is only read on hover. Four is a
# one-line change and costs about 46 kB and 20 ms more per rerun; nothing
# below it is worth having, since 0.52% would be rendering precision the
# database does not carry.
#
# Adaptive refinement was measured and rejected: 7,896 variable-size cells
# gave a worse p90 (1.40%) than 11,544 uniform ones (1.03%), because the
# gradients here are not concentrated enough to pay for the bookkeeping.
SAMPLE_DIVISOR = 2

# Coordinates go on the wire once per cell, so they are rounded to the fewest
# decimals that are still exact for every divisor we would plausibly use: a
# divisor of 8 puts nodes on multiples of 0.003125, which needs six.
CELL_DECIMALS = 6


@dataclass(frozen=True)
class Samples:
    """One field's surface, sampled on a grid fine enough to hover over.

    ``points`` is ``(lat, lon, label)`` per sampled cell centre, land only,
    with the label already formatted -- the browser does no arithmetic, it
    looks up a string. ``half_lat``/``half_lon`` are the half-extents of one
    cell, which is what a layer needs to draw the square around each centre.
    """

    points: tuple[tuple[float, float, str], ...]
    half_lat: float
    half_lon: float
    field: Field
    site_class: str
    divisor: int

    @property
    def detail(self) -> str:
        """The tooltip's second line: what the reader is looking at."""
        return f"{self.field.label}, interpolated"


def format_value(field: Field, value: float) -> str:
    """One value as the readout shows it: three decimals, unit if it has one."""
    return f"{value:.3f} {field.unit}".strip()


def _refine(axis: np.ndarray, divisor: int) -> np.ndarray:
    """``axis`` with ``divisor`` times as many points, every original kept.

    Built by subdividing each interval rather than by ``linspace`` across the
    whole range, so the database's own nodes come back bit-for-bit. That
    matters: at an exact node ``numpy.interp`` returns the stored value
    untouched, so a cell centred on a node reads out the number the service
    returned rather than an interpolation of it that lands a digit away.
    """
    if divisor < 1:
        raise HeatmapError(f"sample divisor must be at least 1, got {divisor}")
    axis = np.asarray(axis, dtype=float)
    if divisor == 1:
        return axis
    out = np.empty((len(axis) - 1) * divisor + 1)
    out[::divisor] = axis
    for step in range(1, divisor):
        weight = step / divisor
        out[step::divisor] = axis[:-1] * (1 - weight) + axis[1:] * weight
    return out


def sample_surface(
    table: pd.DataFrame,
    site_class: str,
    field_key: str,
    coastline: str | None = landmask.DEFAULT_MASK,
    divisor: int = SAMPLE_DIVISOR,
) -> Samples:
    """Sample one field finely enough that hovering reads as continuous.

    Water is dropped here rather than drawn transparent: an invisible cell is
    still pickable, so a water cell would answer the pointer with a number the
    map does not show and the legend does not cover.
    """
    if field_key not in FIELDS:
        raise HeatmapError(f"unknown field {field_key!r}; have {sorted(FIELDS)}")
    field = FIELDS[field_key]
    src_lats, src_lons, values = _grid_array(field_values(table, site_class, field))

    out_lats = _refine(src_lats, divisor)
    out_lons = _refine(src_lons, divisor)
    surface = _upsample(values, src_lats, src_lons, out_lats, out_lons)

    mesh_lon, mesh_lat = np.meshgrid(out_lons, out_lats)
    flat_lat = mesh_lat.ravel()
    flat_lon = mesh_lon.ravel()
    flat_value = surface.ravel()
    if coastline is not None:
        keep = on_land(flat_lon, flat_lat, coastline)
        flat_lat, flat_lon, flat_value = (
            flat_lat[keep],
            flat_lon[keep],
            flat_value[keep],
        )
    if not len(flat_lat):
        raise HeatmapError("no land cells to read values from")

    return Samples(
        points=tuple(
            (
                round(float(lat), CELL_DECIMALS),
                round(float(lon), CELL_DECIMALS),
                format_value(field, value),
            )
            for lat, lon, value in zip(flat_lat, flat_lon, flat_value, strict=True)
        ),
        # Rounded because these go into a deck.gl accessor expression as
        # literal text, and linspace leaves 0.00625 as 0.006250000000001421.
        half_lat=round(float(out_lats[1] - out_lats[0]) / 2, CELL_DECIMALS + 4),
        half_lon=round(float(out_lons[1] - out_lons[0]) / 2, CELL_DECIMALS + 4),
        field=field,
        site_class=site_class,
        divisor=divisor,
    )


# Ink for the legend. The muted step is the same in light and dark, so the
# tick labels need no theme switch; everything else inherits the host's text
# colour rather than naming one, which is what keeps the legend legible when
# Streamlit's theme changes underneath it.
MUTED_INK = "#898781"
HAIRLINE = "rgba(128,128,128,0.35)"

LEGEND_TICKS = 5


def legend_ticks(vmin: float, vmax: float, count: int = LEGEND_TICKS) -> list[float]:
    """Evenly spaced tick values across the ramp, ends included."""
    if count < 2:
        raise ValueError("a legend needs at least its two ends")
    step = (vmax - vmin) / (count - 1)
    return [vmin + step * index for index in range(count)]


def legend_html(overlay: Overlay, version: str, ticks: int = LEGEND_TICKS) -> str:
    """The colour bar, its scale, and what the colours mean.

    Built here rather than in the app so it can be tested without Streamlit,
    and so the ramp in the bar is the same object that coloured the image --
    a legend drawn from a second copy of the palette is a legend that can
    disagree with its map.

    ``version`` is stated rather than inferred. The overlay is ASCE 7-28 only
    and must not appear to follow the sidebar's code-version selector, which
    governs the single-site results underneath it.
    """
    unit = f" {overlay.field.unit}" if overlay.field.unit else ""

    # One swatch per band, in the band's own colour, so the bar is a key to
    # the image rather than a picture of the ramp it came from: a reader can
    # match a patch of map to a swatch and read its interval off the scale.
    swatches = "".join(
        f'<div style="flex:1;background:{band_colour(index, overlay.bands)}'
        + ("" if index == overlay.bands - 1 else f";border-right:1px solid {HAIRLINE}")
        + '"></div>'
        for index in range(overlay.bands)
    )

    # Positioned at their true fraction of the scale rather than spaced by a
    # flex row, which put a five-label row at 10/30/50/70/90% of the bar and
    # was close enough to pass unnoticed against a gradient. Against banded
    # swatches a label that misses its boundary is visibly wrong.
    values = legend_ticks(overlay.vmin, overlay.vmax, ticks)
    labels = "".join(
        f'<span style="position:absolute;left:{index / (ticks - 1):.4%};'
        f'{_tick_shift(index, ticks)}">{value:.3f}</span>'
        for index, value in enumerate(values)
    )

    # Spelled out, not footnoted: each ratio divides a RotD50 FREr value by a
    # RotD100 DE value, so it is not orientation-consistent. The app already
    # presents that comparison in its "vs DE" deltas; a map of it has to say
    # so where the reader is looking at it.
    caveat = (
        "<div style='margin-top:4px;font-size:0.72rem;color:"
        f"{MUTED_INK}'>Mixed orientations: FRE<sub>R</sub> is RotD50 and DE is "
        "RotD100, as in the per-site deltas.</div>"
        if overlay.field.is_ratio
        else ""
    )

    return (
        "<div style=\"font-family:system-ui,-apple-system,'Segoe UI',sans-serif;"
        f"max-width:520px;padding:8px 10px;border:1px solid {HAIRLINE};"
        'border-radius:6px">'
        '<div style="font-size:0.82rem;opacity:0.85;margin-bottom:6px">'
        f"<b>{overlay.field.label}</b>{unit} &middot; {overlay.field.description}"
        "</div>"
        f'<div style="display:flex;height:12px;border-radius:3px;'
        f'overflow:hidden;border:1px solid {HAIRLINE}">{swatches}</div>'
        '<div style="position:relative;height:1.1em;margin-top:3px;'
        "font-size:0.72rem;font-variant-numeric:tabular-nums;"
        f'color:{MUTED_INK}">{labels}</div>'
        f'<div style="margin-top:4px;font-size:0.72rem;color:{MUTED_INK}">'
        f"Site class {overlay.site_class} &middot; {version} &middot; "
        f"{overlay.land_nodes:,} grid nodes on land, interpolated into "
        f"{overlay.bands} steps; the lines on the map are the step "
        "boundaries. Water is not shown.</div>"
        f"{caveat}"
        "</div>"
    )


def _tick_shift(index: int, count: int) -> str:
    """Centre a label on its mark, pulling the two ends inside the bar."""
    if index == 0:
        return ""
    if index == count - 1:
        return "transform:translateX(-100%)"
    return "transform:translateX(-50%)"
