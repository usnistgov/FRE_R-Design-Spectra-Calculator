"""The gridded heatmap overlay: raster, clip, colour scale, and the deck layer.

All offline. The renderer reads the committed database, so these also serve as
a check that the database and the renderer still agree about its shape.
"""

import io
import json
import struct
from pathlib import Path

import numpy as np
import pandas as pd
import pydeck as pdk
import pytest
from matplotlib.colors import to_hex, to_rgb
from matplotlib.image import imread

import map_view
from frer import heatmap, landmask
from frer.grid import BAY_AREA, SDS_COLUMN, load_grid, load_manifest

REPO_ROOT = Path(__file__).resolve().parent.parent
GRID_CSV = REPO_ROOT / "data" / "grid" / "bay_area_asce7-28.csv.gz"
GRID_JSON = REPO_ROOT / "data" / "grid" / "bay_area_asce7-28.json"


@pytest.fixture(scope="module")
def table():
    return load_grid(GRID_CSV)


@pytest.fixture(scope="module")
def manifest():
    return load_manifest(GRID_JSON)


def png_size(data: bytes) -> tuple[int, int]:
    """Width and height from a PNG's IHDR, without decoding the image."""
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    width, height = struct.unpack(">II", data[16:24])
    return width, height


# --- the fast point-in-polygon path ------------------------------------------
#
# heatmap carries a vectorised land test because landmask.is_land, one point at
# a time in Python, is far too slow for a few hundred thousand pixels. Two
# implementations of one piece of geometry is a thing worth pinning rather than
# trusting, so these hold them to the same answers.


def test_the_fast_land_test_agrees_with_the_reference_on_every_grid_node():
    nodes = BAY_AREA.nodes()
    fast = heatmap.on_land(nodes["lon"], nodes["lat"], BAY_AREA.coastline)
    slow = np.array(
        [
            landmask.is_land(float(row.lon), float(row.lat), BAY_AREA.coastline)
            for row in nodes.itertuples()
        ]
    )
    assert (fast == slow).all()
    assert int(fast.sum()) == 753


def test_the_fast_land_test_agrees_off_the_graticule_too():
    """Nodes are a special case; the raster samples between them."""
    rng = np.random.default_rng(20261117)
    lons = rng.uniform(BAY_AREA.lon_min, BAY_AREA.lon_max, 400)
    lats = rng.uniform(BAY_AREA.lat_min, BAY_AREA.lat_max, 400)
    fast = heatmap.on_land(lons, lats, BAY_AREA.coastline)
    slow = np.array(
        [
            landmask.is_land(float(lon), float(lat), BAY_AREA.coastline)
            for lon, lat in zip(lons, lats, strict=True)
        ]
    )
    assert (fast == slow).all()


def test_the_bay_is_water_to_the_fast_path_as_well():
    """Holes in the coastline must subtract, or the Bay reads as land."""
    assert not heatmap.on_land([-122.36], [37.82], BAY_AREA.coastline)[0]
    assert heatmap.on_land([-122.42], [37.77], BAY_AREA.coastline)[0]


def test_a_raster_outside_the_clip_box_raises_rather_than_guessing():
    with pytest.raises(landmask.MaskError, match="clip box"):
        heatmap.on_land([-130.0], [37.8], BAY_AREA.coastline)


# --- the raster --------------------------------------------------------------


def test_bilinear_upsampling_reproduces_the_source_nodes():
    """An upsample that does not pass through its own data is not interpolation."""
    src_lats = np.array([0.0, 1.0, 2.0])
    src_lons = np.array([0.0, 1.0])
    values = np.array([[0.0, 10.0], [20.0, 30.0], [40.0, 50.0]])
    out = heatmap._upsample(values, src_lats, src_lons, src_lats, src_lons)
    assert np.allclose(out, values)


def test_bilinear_upsampling_puts_the_midpoint_on_the_straight_line():
    src_lats = np.array([0.0, 1.0])
    src_lons = np.array([0.0, 1.0])
    values = np.array([[0.0, 10.0], [20.0, 30.0]])
    out = heatmap._upsample(
        values, src_lats, src_lons, np.array([0.5]), np.array([0.5])
    )
    assert out[0, 0] == pytest.approx(15.0)


def test_a_hole_in_the_rectangle_is_refused(table):
    """A missing node would interpolate as if its neighbour were adjacent."""
    rows = heatmap.field_values(table, "D", heatmap.FIELDS["sds"])
    with pytest.raises(heatmap.HeatmapError, match="missing from the rectangle"):
        heatmap._grid_array(rows.iloc[1:])


# --- the clip ----------------------------------------------------------------


def test_the_coastline_cuts_the_image_and_leaves_most_of_it(table):
    overlay = heatmap.render(table, "D", "sds")
    width, height = png_size(overlay.png)
    alpha = heatmap.land_fraction(
        np.linspace(BAY_AREA.lon_min, BAY_AREA.lon_max, width),
        np.linspace(BAY_AREA.lat_min, BAY_AREA.lat_max, height),
        BAY_AREA.coastline,
    )
    # Something is cut, and it is not most of the picture: the trimmed extent
    # is mostly land, which is why it was trimmed.
    assert 0.2 < float((alpha == 0).mean()) < 0.6
    assert float((alpha > 0).mean()) > 0.4


def test_the_edge_is_feathered_rather_than_jagged(table):
    """Supersampling should leave partly-covered pixels at the shore."""
    alpha = heatmap.land_fraction(
        np.linspace(BAY_AREA.lon_min, BAY_AREA.lon_max, 200),
        np.linspace(BAY_AREA.lat_min, BAY_AREA.lat_max, 240),
        BAY_AREA.coastline,
    )
    partial = (alpha > 0) & (alpha < 1)
    assert partial.sum() > 100, "no partial pixels: the coast will be a staircase"


def test_the_clip_is_cached_but_callers_cannot_poison_it():
    lons = np.linspace(BAY_AREA.lon_min, BAY_AREA.lon_max, 50)
    lats = np.linspace(BAY_AREA.lat_min, BAY_AREA.lat_max, 60)
    first = heatmap.land_fraction(lons, lats, BAY_AREA.coastline)
    first[:] = 7.0
    second = heatmap.land_fraction(lons, lats, BAY_AREA.coastline)
    assert not (second == 7.0).all()


# --- the colour scale --------------------------------------------------------


def test_the_colour_scale_is_taken_from_land_rows_only(table):
    """The legend must describe values the map actually shows.

    Water is clipped out of the image, so it must not set the ends of the
    ramp. Note how little this moves at the trimmed extent: water here is the
    Bay and a coastal strip, not the open Pacific, and it overlaps the land
    range almost completely. The exclusion is about the legend being honest,
    not about rescuing dynamic range -- see the module docstring.
    """
    rows = heatmap.field_values(table, "D", heatmap.FIELDS["sds"])
    land = rows.loc[heatmap.on_land(rows["lon"], rows["lat"], BAY_AREA.coastline)]

    assert heatmap.colour_limits(rows, BAY_AREA.coastline) == (
        pytest.approx(land["value"].min()),
        pytest.approx(land["value"].max()),
    )
    assert heatmap.colour_limits(rows, None) != heatmap.colour_limits(
        rows, BAY_AREA.coastline
    )


def test_water_no_longer_runs_below_the_land_at_this_extent(table):
    """Pins the fact the docstring rests on, so a re-extent shows up here.

    At the original wide extent the water nodes reached the Gulf of the
    Farallones and sat far below every land value. They do not any more.
    """
    rows = heatmap.field_values(table, "D", heatmap.FIELDS["sds"])
    land_mask = heatmap.on_land(rows["lon"], rows["lat"], BAY_AREA.coastline)
    land, water = rows.loc[land_mask, "value"], rows.loc[~land_mask, "value"]
    assert water.max() > land.max()
    assert int((water < land.min()).sum()) < 0.1 * len(water)


def test_every_field_renders_and_scales_to_its_own_land_values(table):
    for key, field in heatmap.FIELDS.items():
        overlay = heatmap.render(table, "D", key)
        assert overlay.vmin < overlay.vmax, key
        assert overlay.land_nodes == 753, key
        assert overlay.field is field
        if field.is_ratio:
            # FREr is below DE everywhere in the committed references; a ratio
            # at or above 1 is not something to meet first on a map legend.
            assert 0.0 < overlay.vmin < overlay.vmax < 1.0, key


def test_pinned_limits_override_the_data(table):
    """So a class switch can keep one scale instead of resetting it."""
    overlay = heatmap.render(table, "D", "sds", limits=(0.5, 3.0))
    assert (overlay.vmin, overlay.vmax) == (0.5, 3.0)


def test_a_field_with_no_land_nodes_cannot_be_scaled(table):
    rows = heatmap.field_values(table, "D", heatmap.FIELDS["sds"])
    water = rows[~heatmap.on_land(rows["lon"], rows["lat"], BAY_AREA.coastline)]
    with pytest.raises(heatmap.HeatmapError, match="no land nodes"):
        heatmap.colour_limits(water.iloc[:0], BAY_AREA.coastline)


# --- the image ---------------------------------------------------------------


def test_the_image_covers_the_grid_and_its_bounds_are_pixel_edges(table):
    overlay = heatmap.render(table, "D", "sds")
    west, south, east, north = overlay.bounds
    half = 0.5 / heatmap.PIXELS_PER_DEGREE
    assert west == pytest.approx(BAY_AREA.lon_min - half)
    assert south == pytest.approx(BAY_AREA.lat_min - half)
    assert east == pytest.approx(BAY_AREA.lon_max + half)
    assert north == pytest.approx(BAY_AREA.lat_max + half)


def test_the_image_is_north_up(table):
    """deck.gl reads row 0 as the northern edge; the raster is built south-first.

    Sds rises towards the north-east of this box, so the top of the image must
    be the stronger end. Read back through the alpha channel so the comparison
    only counts land.
    """
    overlay = heatmap.render(table, "D", "sds")
    pixels = imread(io.BytesIO(overlay.png))
    height = pixels.shape[0]
    # Any ink at all is land. Not "mostly opaque": the fill is FILL_ALPHA,
    # so a threshold above it matches nothing and quietly compares NaNs,
    # which is what this test did for one commit.
    inked = pixels[..., 3] > heatmap.FILL_ALPHA / 2
    assert inked.any()

    def darkness(block, mask):
        """Mean CIE lightness, inverted. The ramp is multi-hue, so no single
        channel is monotonic in value, but lightness is -- by construction,
        and test_the_ramp_gets_darker_at_every_step is what holds it there."""
        rgb = block[..., :3][mask]
        linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
        return float(1.0 - (linear * [0.2126, 0.7152, 0.0722]).sum(axis=1).mean())

    top = darkness(pixels[: height // 4], inked[: height // 4])
    bottom = darkness(pixels[-(height // 4) :], inked[-(height // 4) :])
    north = table[(table["site_class"] == "D") & (table["lat"] > 38.0)][SDS_COLUMN]
    south = table[(table["site_class"] == "D") & (table["lat"] < 37.45)][SDS_COLUMN]
    assert (north.mean() > south.mean()) == (top > bottom)


def test_the_data_uri_is_a_png(table):
    overlay = heatmap.render(table, "D", "sds")
    assert overlay.data_uri.startswith("data:image/png;base64,")
    assert len(overlay.data_uri) < 400_000, "too large to put in a deck payload"


def test_an_unknown_field_is_refused(table):
    with pytest.raises(heatmap.HeatmapError, match="unknown field"):
        heatmap.render(table, "D", "Sds")


def test_an_unknown_site_class_is_refused(table):
    with pytest.raises(heatmap.HeatmapError, match="no rows for site class"):
        heatmap.render(table, "Z", "sds")


# --- provenance --------------------------------------------------------------


def test_the_committed_database_matches_the_coastline_it_is_drawn_against(manifest):
    heatmap.check_coastline(manifest, BAY_AREA.coastline)


def test_a_database_built_against_another_coastline_is_refused(manifest):
    with pytest.raises(heatmap.HeatmapError, match="built against coastline"):
        heatmap.check_coastline({**manifest, "coastline": "elsewhere"}, "bay_area_land")


def test_a_re_vendored_coastline_is_refused(manifest):
    """The clip would move while the values stayed where they were."""
    stale = {**manifest, "coastline_sha256": "0" * 64}
    with pytest.raises(heatmap.HeatmapError, match="has changed since the build"):
        heatmap.check_coastline(stale, BAY_AREA.coastline)


# --- the deck layer ----------------------------------------------------------
#
# The overlay must not fade when a site pin is selected. Streamlit rewrites the
# colour accessor of every layer whose deck.gl type is in a table in its
# frontend bundle; BitmapLayer is absent from that table and declares no colour
# accessor either. These pin both halves, because a later move to a cell layer
# for hover-picking would reintroduce the phase 1 bug silently.


def test_the_overlay_is_a_bitmap_layer():
    layer = map_view.heatmap_layer("data:image/png;base64,AAAA", (-1.0, 2.0, 3.0, 4.0))
    assert layer.type == "BitmapLayer"
    assert layer.id == map_view.HEATMAP_LAYER_ID


def test_the_overlay_declares_no_colour_accessor():
    layer = map_view.heatmap_layer("data:image/png;base64,AAAA", (-1.0, 2.0, 3.0, 4.0))
    rendered = layer.to_json()
    for accessor in ("getFillColor", "getColor", "getLineColor"):
        assert accessor not in rendered


def test_the_overlay_does_not_answer_clicks():
    """A pickable overlay across the region would swallow every pin click."""
    layer = map_view.heatmap_layer("data:image/png;base64,AAAA", (-1.0, 2.0, 3.0, 4.0))
    assert layer.pickable is False


def test_the_image_is_sent_as_a_literal_not_an_expression():
    """An unquoted data URI reaches the browser as code and blanks the deck.

    pydeck prefixes a bare string layer argument with ``@@=``, marking it a
    deck.gl accessor expression; the browser then tries to parse the URI as
    code and fails on the colon in ``data:``, taking the whole deck with it.
    Quoting is pydeck's escape hatch -- it strips the quotes and emits the
    plain string -- so the check is on what comes out, not what went in.
    """
    uri = "data:image/png;base64,AAAA"
    emitted = json.loads(map_view.heatmap_layer(uri, (-1.0, 2.0, 3.0, 4.0)).to_json())
    assert emitted["image"] == uri

    # The failure this guards against, so the test shows its own teeth.
    unguarded = pdk.Layer(
        "BitmapLayer", id="x", data=None, image=uri, bounds=[-1, 2, 3, 4]
    )
    assert json.loads(unguarded.to_json())["image"] == f"@@={uri}"


def test_bounds_must_be_west_south_east_north():
    with pytest.raises(ValueError, match="west/south/east/north"):
        map_view.heatmap_layer("data:image/png;base64,AAAA", (3.0, 2.0, -1.0, 4.0))


def test_an_empty_image_is_refused():
    with pytest.raises(ValueError, match="non-empty string"):
        map_view.heatmap_layer("", (-1.0, 2.0, 3.0, 4.0))


# --- the deck ----------------------------------------------------------------

SITES = pd.DataFrame(
    {
        "id": [1, 2],
        "name": ["Oakland CA", "San Jose CA"],
        "lat": [37.80, 37.34],
        "lon": [-122.27, -121.89],
        "region": ["CA", "CA"],
    }
)


def test_the_map_without_an_overlay_is_unchanged():
    deck = map_view.site_map(SITES)
    assert [layer.id for layer in deck.layers] == [map_view.SITE_LAYER_ID]


def test_the_overlay_is_drawn_under_the_pins(table):
    overlay = heatmap.render(table, "D", "sds")
    deck = map_view.site_map(SITES, None, (overlay.data_uri, overlay.bounds))
    assert [layer.id for layer in deck.layers] == [
        map_view.HEATMAP_LAYER_ID,
        map_view.SITE_LAYER_ID,
    ]


def test_the_pins_still_carry_no_colour_accessor_with_an_overlay_present(table):
    """Both layers have to stay immune, not just the new one."""
    overlay = heatmap.render(table, "D", "sds")
    deck = map_view.site_map(SITES, 1, (overlay.data_uri, overlay.bounds))
    for layer in deck.layers:
        assert "getColor" not in layer.to_json()


# --- the legend --------------------------------------------------------------
#
# Built in frer.heatmap rather than in the app so it can be checked without
# Streamlit, and so the bar is filled from the same ramp that coloured the
# image. A legend drawn from a second copy of the palette is a legend that can
# disagree with its own map.


def test_the_legend_swatches_are_the_colours_the_image_was_banded_into(table):
    """One swatch per band, each in that band's own fill."""
    overlay = heatmap.render(table, "D", "sds")
    html = heatmap.legend_html(overlay, "ASCE 7-28")
    for index in range(overlay.bands):
        assert heatmap.band_colour(index, overlay.bands) in html
    assert html.count("flex:1;background:") == overlay.bands


def test_a_band_swatch_is_the_colour_that_band_is_drawn_in(table):
    """The one way the legend could lie: a second copy of the ramp."""
    overlay = heatmap.render(table, "D", "sds", pixels_per_degree=40)
    pixels = imread(io.BytesIO(overlay.png))
    fill = np.isclose(pixels[..., 3], heatmap.FILL_ALPHA, atol=0.01)
    assert fill.any()
    swatches = {heatmap.band_colour(i, overlay.bands) for i in range(overlay.bands)}
    drawn = {to_hex(pixel[:3]) for pixel in pixels[fill][:400]}
    assert drawn <= swatches


def test_the_legend_spans_the_overlays_own_limits(table):
    overlay = heatmap.render(table, "D", "sds")
    html = heatmap.legend_html(overlay, "ASCE 7-28")
    assert f"{overlay.vmin:.3f}" in html
    assert f"{overlay.vmax:.3f}" in html


def test_legend_ticks_run_end_to_end():
    ticks = heatmap.legend_ticks(1.0, 2.0, 5)
    assert ticks[0] == 1.0
    assert ticks[-1] == 2.0
    assert ticks == pytest.approx([1.0, 1.25, 1.5, 1.75, 2.0])


def test_a_legend_needs_two_ends():
    with pytest.raises(ValueError, match="at least its two ends"):
        heatmap.legend_ticks(1.0, 2.0, 1)


def test_the_legend_states_the_code_version_it_is_showing(table):
    """The overlay is 7-28 only and must not look like it follows the sidebar."""
    overlay = heatmap.render(table, "D", "sds")
    assert "ASCE 7-28" in heatmap.legend_html(overlay, "ASCE 7-28")


def test_the_legend_names_the_site_class(table):
    overlay = heatmap.render(table, "BC", "sds")
    assert "Site class BC" in heatmap.legend_html(overlay, "ASCE 7-28")


def test_the_legend_says_water_is_not_shown(table):
    """So nobody reads the clipped area as missing data."""
    overlay = heatmap.render(table, "D", "sds")
    assert "ater is not shown" in heatmap.legend_html(overlay, "ASCE 7-28")


def test_a_ratio_legend_spells_out_the_mixed_orientations(table):
    """RotD50 over RotD100 is the one thing a reader cannot infer from the map."""
    for key in ("sdsfr_over_sds", "sd1fr_over_sd1"):
        html = heatmap.legend_html(heatmap.render(table, "D", key), "ASCE 7-28")
        assert "RotD50" in html and "RotD100" in html
        assert "Mixed orientations" in html


def test_an_absolute_legend_carries_no_orientation_caveat(table):
    """It is a RotD100 value throughout; there is nothing mixed to warn about."""
    html = heatmap.legend_html(heatmap.render(table, "D", "sds"), "ASCE 7-28")
    assert "Mixed orientations" not in html


def test_the_legend_marks_units_only_where_there_are_units(table):
    assert " g " in heatmap.legend_html(heatmap.render(table, "D", "sds"), "v")
    ratio = heatmap.legend_html(heatmap.render(table, "D", "sdsfr_over_sds"), "v")
    assert "S_DSfr / S_DS</b> &middot;" in ratio


# --- opacity -----------------------------------------------------------------


def test_the_overlay_is_drawn_at_full_layer_opacity(table):
    """Its alpha is in the image; scaling the layer would dim the lines."""
    overlay = heatmap.render(table, "D", "sds")
    deck = map_view.site_map(SITES, None, (overlay.data_uri, overlay.bounds))
    assert deck.layers[0].opacity == 1.0
    assert map_view.HEATMAP_OPACITY == 1.0


# --- the sampled surface -----------------------------------------------------
#
# Streamlit's deck chart reports which object was picked and discards deck.gl's
# cursor coordinate, so a hover readout has to be precomputed and looked up.
# These pin the sampling itself: that it refines the database rather than
# replacing it, that it stays on land, and that a hover and a click on the same
# place cannot report different numbers.


def test_sampling_refines_the_database_without_moving_its_nodes(table):
    samples = heatmap.sample_surface(table, "D", "sds", divisor=2)
    rows = heatmap.field_values(table, "D", heatmap.FIELDS["sds"])
    land = rows.loc[heatmap.on_land(rows["lon"], rows["lat"], BAY_AREA.coastline)]
    sampled = {(lat, lon) for lat, lon, _ in samples.points}
    for lat, lon in zip(land["lat"], land["lon"], strict=True):
        assert (
            round(float(lat), heatmap.CELL_DECIMALS),
            round(float(lon), heatmap.CELL_DECIMALS),
        ) in sampled


def test_a_sampled_node_carries_the_databases_own_value(table):
    """At a node the interpolation is the identity, so nothing may drift."""
    samples = heatmap.sample_surface(table, "D", "sds", divisor=2)
    rows = heatmap.field_values(table, "D", heatmap.FIELDS["sds"])
    labels = {(lat, lon): label for lat, lon, label in samples.points}
    checked = 0
    for lat, lon, value in zip(rows["lat"], rows["lon"], rows["value"], strict=True):
        key = (
            round(float(lat), heatmap.CELL_DECIMALS),
            round(float(lon), heatmap.CELL_DECIMALS),
        )
        if key in labels:
            assert labels[key] == f"{value:.3f} g"
            checked += 1
    assert checked > 700


def test_the_divisor_sets_the_cell_size(table):
    coarse = heatmap.sample_surface(table, "D", "sds", divisor=1)
    fine = heatmap.sample_surface(table, "D", "sds", divisor=2)
    assert coarse.half_lat == pytest.approx(BAY_AREA.step / 2)
    assert fine.half_lat == pytest.approx(BAY_AREA.step / 4)
    assert fine.half_lon == pytest.approx(BAY_AREA.step / 4)
    assert len(fine.points) > 3 * len(coarse.points)


def test_sampling_drops_water(table):
    """An invisible cell is still pickable, so a water cell would answer."""
    samples = heatmap.sample_surface(table, "D", "sds", divisor=2)
    lats = np.array([lat for lat, _, _ in samples.points])
    lons = np.array([lon for _, lon, _ in samples.points])
    assert heatmap.on_land(lons, lats, BAY_AREA.coastline).all()


def test_sampling_without_a_coastline_keeps_every_cell(table):
    wet = heatmap.sample_surface(table, "D", "sds", coastline=None, divisor=1)
    dry = heatmap.sample_surface(table, "D", "sds", divisor=1)
    assert len(wet.points) > len(dry.points)


def test_an_unknown_field_cannot_be_sampled(table):
    with pytest.raises(heatmap.HeatmapError, match="unknown field"):
        heatmap.sample_surface(table, "D", "not_a_field")


def test_a_divisor_below_one_is_refused(table):
    with pytest.raises(heatmap.HeatmapError, match="at least 1"):
        heatmap.sample_surface(table, "D", "sds", divisor=0)


def test_the_detail_line_names_the_field_and_says_interpolated(table):
    samples = heatmap.sample_surface(table, "D", "sdsfr_over_sds", divisor=1)
    assert samples.detail == "S_DSfr / S_DS, interpolated"


def test_a_ratio_is_formatted_without_a_unit(table):
    samples = heatmap.sample_surface(table, "D", "sdsfr_over_sds", divisor=1)
    assert all(not label.endswith(" g") for _, _, label in samples.points)
    assert all(len(label.split(".")[-1]) == 3 for _, _, label in samples.points)


# --- the readout layer -------------------------------------------------------


def test_the_readout_cells_are_invisible_but_not_transparent():
    """Alpha 0 is falsy in Streamlit's fade rewrite and becomes opaque."""
    assert map_view.READOUT_FILL[3] == 1
    layer = map_view.readout_layer([(37.3, -122.4, "1.000 g")], 0.01, 0.01)
    assert json.loads(layer.to_json())["getFillColor"][3] == 1


def test_the_readout_answers_clicks():
    layer = map_view.readout_layer([(37.3, -122.4, "1.000 g")], 0.01, 0.01)
    assert json.loads(layer.to_json())["pickable"] is True


def test_the_cell_square_is_an_expression_not_sent_geometry():
    """Four corners per cell on the wire would be four times the bytes."""
    layer = map_view.readout_layer([(37.3, -122.4, "1.000 g")], 0.02, 0.01)
    emitted = json.loads(layer.to_json())
    assert emitted["getPolygon"].startswith("@@=")
    assert "x-0.01" in emitted["getPolygon"]
    assert "y-0.02" in emitted["getPolygon"]
    assert set(emitted["data"][0]) == {
        map_view.CELL_LAT_FIELD,
        map_view.CELL_LON_FIELD,
        map_view.TOOLTIP_TITLE_FIELD,
        map_view.TOOLTIP_DETAIL_FIELD,
    }


def test_a_cell_with_no_size_is_refused():
    with pytest.raises(ValueError, match="half-extents"):
        map_view.readout_layer([(37.3, -122.4, "1.000 g")], 0.0, 0.01)


def test_an_empty_readout_is_refused():
    with pytest.raises(ValueError, match="at least one point"):
        map_view.readout_layer([], 0.01, 0.01)


def test_the_readout_sits_between_the_overlay_and_the_pins(table):
    """Under the pins, so a pin wins the pick wherever the two overlap."""
    overlay = heatmap.render(table, "D", "sds")
    samples = heatmap.sample_surface(table, "D", "sds", divisor=1)
    deck = map_view.site_map(
        SITES,
        None,
        (overlay.data_uri, overlay.bounds),
        readout=(samples.points, samples.half_lat, samples.half_lon, samples.detail),
    )
    assert [layer.id for layer in deck.layers] == [
        map_view.HEATMAP_LAYER_ID,
        map_view.READOUT_LAYER_ID,
        map_view.SITE_LAYER_ID,
    ]


def test_the_readout_is_absent_unless_asked_for(table):
    overlay = heatmap.render(table, "D", "sds")
    deck = map_view.site_map(SITES, None, (overlay.data_uri, overlay.bounds))
    assert map_view.READOUT_LAYER_ID not in [layer.id for layer in deck.layers]


# --- one tooltip for every layer ---------------------------------------------
#
# pydeck has a single template per deck, and Streamlit leaves a {placeholder}
# standing when the picked object has no such field. So the pins and the cells
# have to answer to the same two names.


def deck_tooltip(deck) -> dict:
    """The tooltip the way Streamlit reads it off a Deck.

    Mirrors streamlit.elements.deck_gl_json_chart._get_pydeck_tooltip: the
    widget attribute where pydeck used to keep it, falling back to the
    private one it keeps it in now. Written out here so the test breaks if
    pydeck moves it again, rather than silently checking nothing.
    """
    widget = getattr(deck, "deck_widget", None)
    if widget is not None and isinstance(getattr(widget, "tooltip", None), dict):
        return widget.tooltip
    tooltip = getattr(deck, "_tooltip", None)
    assert isinstance(tooltip, dict), "pydeck no longer exposes the tooltip"
    return tooltip


def test_the_tooltip_template_reads_the_shared_fields():
    html = deck_tooltip(map_view.site_map(SITES))["html"]
    assert "{" + map_view.TOOLTIP_TITLE_FIELD + "}" in html
    assert "{" + map_view.TOOLTIP_DETAIL_FIELD + "}" in html


def test_every_pickable_layer_fills_the_tooltip_fields(table):
    overlay = heatmap.render(table, "D", "sds")
    samples = heatmap.sample_surface(table, "D", "sds", divisor=1)
    deck = map_view.site_map(
        SITES,
        None,
        (overlay.data_uri, overlay.bounds),
        readout=(samples.points, samples.half_lat, samples.half_lon, samples.detail),
    )
    pickable = 0
    for layer in deck.layers:
        emitted = json.loads(layer.to_json())
        if not emitted.get("pickable"):
            continue
        row = emitted["data"][0]
        assert map_view.TOOLTIP_TITLE_FIELD in row
        assert map_view.TOOLTIP_DETAIL_FIELD in row
        pickable += 1
    assert pickable == 2


def test_a_pins_tooltip_is_still_its_name_and_region():
    layer = map_view._site_layer(SITES, None)
    row = json.loads(layer.to_json())["data"][0]
    assert row[map_view.TOOLTIP_TITLE_FIELD] == "Oakland CA"
    assert row[map_view.TOOLTIP_DETAIL_FIELD] == "CA"


# --- reading a click back ----------------------------------------------------


def test_an_empty_selection_names_no_layers():
    assert map_view.picked_layers(None) == set()
    assert map_view.picked_layers({}) == set()
    assert map_view.picked_layers({"objects": {}}) == set()
    assert map_view.picked_layers({"objects": {map_view.SITE_LAYER_ID: []}}) == set()


def test_a_selection_names_the_layer_that_was_picked():
    selection = {"objects": {map_view.READOUT_LAYER_ID: [{"y": 37.7, "x": -122.1}]}}
    assert map_view.picked_layers(selection) == {map_view.READOUT_LAYER_ID}


# --- what a click leaves selected -------------------------------------------
#
# The site and the point are not symmetric, and one case -- the overlay being
# switched off -- cannot be told from the payload alone. This went wrong once
# in the real app: turning the heatmap off deselected the site, because
# Streamlit drops the selection of a layer that has left the deck and that
# arrives looking exactly like a click on bare map.

PIN = {"objects": {map_view.SITE_LAYER_ID: [{"id": 15}]}}
GRID = {
    "objects": {
        map_view.READOUT_LAYER_ID: [
            {map_view.CELL_LAT_FIELD: 37.75, map_view.CELL_LON_FIELD: -122.075}
        ]
    }
}
NOTHING = {"objects": {}}


def test_clicking_the_heatmap_changes_nothing():
    """The grid is pickable so it can answer hovers; clicks must be inert."""
    assert map_view.resolve_selection(GRID, 15, False, True) == (15, True)
    assert map_view.resolve_selection(GRID, None, False, True) == (None, True)


def test_clicking_a_pin_selects_that_site():
    assert map_view.resolve_selection(PIN, None, True, True) == (15, False)


def test_clicking_bare_map_clears_the_site():
    assert map_view.resolve_selection(NOTHING, 15, False, True) == (None, False)


def test_switching_the_overlay_off_keeps_the_site():
    """Streamlit drops a departed layer's selection; that is not a click."""
    assert map_view.resolve_selection(NOTHING, 15, True, False) == (15, False)


def test_bare_map_still_clears_once_the_grid_pick_is_forgotten():
    """The overlay-off case must not become a refusal to ever deselect."""
    assert map_view.resolve_selection(NOTHING, 15, False, False) == (None, False)
    assert map_view.resolve_selection(NOTHING, 15, False, True) == (None, False)


def test_a_fresh_page_selects_nothing():
    assert map_view.resolve_selection(None, None, False, False) == (None, False)


def test_the_readout_never_draws_itself():
    """Opacity 0 is what makes a click inert: the fade paints the picked
    cell solid at alpha 255, and the layer's own opacity multiplies it away
    while leaving picking, which runs in its own pass, untouched."""
    layer = map_view.readout_layer([(37.3, -122.4, "1.000 g")], 0.01, 0.01)
    emitted = json.loads(layer.to_json())
    assert emitted["opacity"] == 0
    assert emitted["pickable"] is True


def test_the_overlay_arrives_faint_enough_to_read_the_map_through():
    """The faintness is in the image; the layer is not scaled."""
    assert heatmap.FILL_ALPHA <= 0.20
    assert map_view.HEATMAP_OPACITY == 1.0


# --- the ramp ----------------------------------------------------------------
#
# Multi-hue, against the project's one-hue rule for magnitude, because the
# brief asks for red to mean a higher value. What the rule is really
# protecting is lightness: a ramp whose lightness wanders is a rainbow and
# cannot be ordered by eye. So the deviation is allowed and the property is
# pinned instead.


def cie_lightness(colour: str) -> float:
    rgb = np.array(to_rgb(colour))
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    y = float(np.dot(linear, [0.2126, 0.7152, 0.0722]))
    return 116 * y ** (1 / 3) - 16 if y > 0.008856 else 903.3 * y


def test_the_ramp_gets_darker_at_every_step():
    steps = [cie_lightness(colour) for colour in heatmap.SEQUENTIAL_RAMP]
    assert all(
        later < earlier for earlier, later in zip(steps, steps[1:], strict=False)
    )


def test_the_ramp_spans_enough_lightness_to_order_by_eye():
    steps = [cie_lightness(colour) for colour in heatmap.SEQUENTIAL_RAMP]
    assert steps[0] - steps[-1] > 40
    gaps = [earlier - later for earlier, later in zip(steps, steps[1:], strict=False)]
    assert min(gaps) > 3


def test_the_ramp_ends_red_because_that_is_what_high_means():
    red, green, blue = to_rgb(heatmap.SEQUENTIAL_RAMP[-1])
    assert red > green and red > blue
    assert red > 0.4


# --- the contour lines -------------------------------------------------------


def test_contours_are_traced_at_the_band_boundaries(table):
    """A line has to fall where the fill changes colour, or it explains
    nothing."""
    overlay = heatmap.render(table, "D", "sds")
    paths = heatmap.contours(table, "D", "sds")
    interior = {round(value, 6) for value in overlay.bounds_of_bands[1:-1]}
    assert {round(path["value"], 6) for path in paths} <= interior
    assert len(interior) == overlay.bands - 1


def test_every_contour_is_a_line_not_a_point(table):
    paths = heatmap.contours(table, "D", "sds")
    assert paths
    assert all(len(path["path"]) >= 2 for path in paths)


def test_contours_stay_on_land(table):
    """Cut rather than clipped: a PathLayer has no alpha channel to mask."""
    paths = heatmap.contours(table, "D", "sds")
    sample = [point for path in paths[:20] for point in path["path"]]
    lons = np.array([lon for lon, _ in sample])
    lats = np.array([lat for _, lat in sample])
    assert heatmap.on_land(lons, lats, BAY_AREA.coastline).mean() > 0.99


def test_a_contour_takes_the_colour_of_the_band_above_it(table):
    paths = heatmap.contours(table, "D", "sds")
    by_value = {}
    for path in paths:
        by_value.setdefault(round(path["value"], 6), path["colour"])
    ordered = [by_value[value] for value in sorted(by_value)]
    # Darker going up, the same way the ramp does.
    assert all(
        sum(later[:3]) < sum(earlier[:3])
        for earlier, later in zip(ordered, ordered[1:], strict=False)
    )


def test_contours_sit_at_the_alpha_the_selection_fade_cannot_lower(table):
    """Streamlit clamps an unselected object to min(alpha, 102). At 102 the
    rewrite is a no-op, which is the only reason a PathLayer is usable here."""
    assert heatmap.CONTOUR_ALPHA <= 102
    paths = heatmap.contours(table, "D", "sds")
    assert all(path["colour"][3] <= 102 for path in paths)


def test_contours_without_a_coastline_are_not_cut(table):
    whole = heatmap.contours(table, "D", "sds", coastline=None)
    clipped = heatmap.contours(table, "D", "sds")
    assert len(whole) < len(clipped), "clipping should break lines into more runs"


def test_an_unknown_field_has_no_contours(table):
    with pytest.raises(heatmap.HeatmapError, match="unknown field"):
        heatmap.contours(table, "D", "not_a_field")


# --- the contour layer -------------------------------------------------------


def test_the_contour_layer_draws_lines_at_a_fixed_pixel_weight(table):
    paths = heatmap.contours(table, "D", "sds")
    emitted = json.loads(map_view.contour_layer(paths).to_json())
    assert emitted["widthUnits"] == "pixels"
    assert emitted["getColor"] == "@@=colour"
    assert emitted["pickable"] is False


def test_the_contour_layer_does_not_take_hovers_from_the_readout(table):
    """Both are over the whole region; only one of them may answer."""
    paths = heatmap.contours(table, "D", "sds")
    assert json.loads(map_view.contour_layer(paths).to_json())["pickable"] is False


def test_an_empty_contour_set_is_refused():
    with pytest.raises(ValueError, match="at least one path"):
        map_view.contour_layer([])


def test_the_contours_sit_over_the_fill_and_under_the_pins(table):
    overlay = heatmap.render(table, "D", "sds")
    samples = heatmap.sample_surface(table, "D", "sds", divisor=1)
    deck = map_view.site_map(
        SITES,
        None,
        (overlay.data_uri, overlay.bounds),
        readout=(samples.points, samples.half_lat, samples.half_lon, samples.detail),
        contours=heatmap.contours(table, "D", "sds"),
    )
    assert [layer.id for layer in deck.layers] == [
        map_view.HEATMAP_LAYER_ID,
        map_view.CONTOUR_LAYER_ID,
        map_view.READOUT_LAYER_ID,
        map_view.SITE_LAYER_ID,
    ]
