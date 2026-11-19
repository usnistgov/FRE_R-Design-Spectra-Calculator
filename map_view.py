"""Zoomable conterminous-U.S. site map for the Streamlit app.

Builds the deck.gl map that :mod:`app` renders in place of a plain site
dropdown: a pannable, zoomable basemap with one pin per precomputed site,
reporting which pin was clicked so the app can develop that site's spectra and
show them underneath.

deck.gl rather than Altair because the map is the primary way into the tool and
has to behave like a map -- scroll to zoom, drag to pan, markers that stay a
sensible size at every zoom. Vega-Lite cannot pan or zoom a cartographic
projection, so a Vega map is a fixed picture of the country. pydeck ships with
Streamlit, so this costs no new dependency.

Nothing here imports Streamlit -- the module only builds a deck and reads a
selection payload back out -- so the map can be built and inspected from a test
or a notebook the same way :mod:`frer.plots` can.

Zooming in is what separates the sites that sit on top of one another at
national zoom (the Los Angeles basin, the Bay Area), which a fixed-extent map
could only resolve through a dropdown.

The basemap tiles are the one part of the page that needs a network
connection, and there is no offline alternative to fall back on: Streamlit
draws a CARTO basemap under every st.pydeck_chart whatever the deck asks for,
reading ``map_style=None`` as "pick one from the theme" rather than "draw
none". The design values themselves never need a network call either way.
"""

import base64

import pydeck as pdk

# deck.gl layer id. Streamlit keys the selection payload by it, so this is what
# a click comes back under; deck.gl also keys its own layer state by id, so
# naming it keeps a re-render updating the existing layer rather than tearing
# it down and rebuilding it.
SITE_LAYER_ID = "precomputed-sites"

# The gridded design-value overlay, drawn under the pins.
HEATMAP_LAYER_ID = "design-value-heatmap"

# The overlay is drawn at full layer opacity, because its alpha is already in
# the image: frer.heatmap inks a faint fill and near-solid contour lines, and
# scaling the layer would dim the two together -- a wash light enough to read
# the streets through would take the lines down with it. There is deliberately
# no control for this; the treatment is fixed so that what the legend says a
# colour means stays true.
HEATMAP_OPACITY = 1.0

# The invisible grid that answers the pointer, drawn between the overlay and
# the pins. See readout_layer for why it can be invisible and still picked.
READOUT_LAYER_ID = "design-value-readout"

# The column carrying the site id. It is what the app needs to load the cached
# entry, and deck.gl hands back the whole row, so the only requirement is that
# the frame passed to site_map carries it.
SITE_ID_FIELD = "id"

# Columns the readout cells carry, and the two the tooltip reads.
#
# pydeck has one tooltip template for the whole deck, not one per layer, so
# the pins and the readout cells have to answer to the same placeholders --
# Streamlit leaves a `{key}` standing in the rendered tooltip when the picked
# object has no such field, which is how a mismatch shows up. Hence two
# generic names rather than `name`/`value`: a title line and a detail line,
# filled differently by each layer.
#
# All four are single characters because they repeat once per cell, and there
# are a few thousand cells. At the default sampling that is the difference
# between about 135 kB of cell data and nearer 180 kB.
CELL_LAT_FIELD = "y"
CELL_LON_FIELD = "x"
TOOLTIP_TITLE_FIELD = "v"
TOOLTIP_DETAIL_FIELD = "d"

# Tile basemap. "light" is CARTO positron through pydeck's default provider: no
# API key, and light to match the theme the app pins. Streamlit substitutes a
# CARTO basemap of its own choosing when this is left unset, so it is named
# here to keep that choice ours rather than inherited.
TILE_BASEMAP = "light"

# Opening view: the whole conterminous U.S.. Zooming out further than the
# country is not useful here, and the hazard models stop at the border.
INITIAL_LATITUDE = 39.5
INITIAL_LONGITUDE = -96.0
INITIAL_ZOOM = 3.3
MIN_ZOOM = 2.5
MAX_ZOOM = 12

MAP_HEIGHT = 620  # px; the map is the page, so it takes most of the viewport

# Pin colours. The app pins Streamlit's light theme (see
# .streamlit/config.toml), so these are fixed rather than theme-aware. An
# unselected pin is the calculator's series blue from app.SERIES_STYLE, so a
# site on the map and its FRE_R spectrum below read as the same thing; the
# selected one goes red, which is what marks it.
PIN_COLOR = (0, 114, 189)
SELECTED_PIN_COLOR = (214, 39, 40)

# Pin heights in pixels. The selected pin is drawn larger as well as redder:
# among 160 pins at national zoom, a colour change alone is easy to lose.
PIN_SIZE = 26
SELECTED_PIN_SIZE = 40

# Columns added to the site frame to drive the icon layer, named for the deck.gl
# accessors that read them. There is deliberately no colour column: see PIN_SVG.
ICON_FIELD = "icon"
SIZE_FIELD = "size"

DEFAULT_PIN = "default"
SELECTED_PIN = "selected"

# One classic teardrop, as a path: a 24x32 pin with a hole through the middle,
# so the basemap shows through it.
PIN_PATH = (
    "M12 0C5.373 0 0 5.373 0 12c0 9 12 20 12 20s12-11 12-20"
    "c0-6.627-5.373-12-12-12zm0 16.5a4.5 4.5 0 110-9 4.5 4.5 0 010 9z"
)


def _hex(rgb) -> str:
    """An (r, g, b) triple as the hex colour an SVG fill wants."""
    return "#{:02x}{:02x}{:02x}".format(*rgb)


# Both states are drawn into one 48x32 atlas -- the blue pin at x=0, the red one
# at x=24 -- and the layer picks between them per row. The obvious alternative
# is one white pin used as a mask with a per-row colour accessor, which is how
# this started; it cannot be used here. While any selection is live, Streamlit
# rewrites the colour accessor of every layer that declares one and fades
# whatever it does not consider selected to 40% alpha. Carrying the colour in
# the icon means this layer declares no colour accessor, so there is nothing for
# it to rewrite and the pins keep their colour whatever is selected.
PIN_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="48" height="32" '
    'viewBox="0 0 48 32">'
    f'<path fill="{_hex(PIN_COLOR)}" fill-rule="evenodd" d="{PIN_PATH}"/>'
    f'<path fill="{_hex(SELECTED_PIN_COLOR)}" fill-rule="evenodd" '
    f'transform="translate(24 0)" d="{PIN_PATH}"/>'
    "</svg>"
)

# Embedded rather than served, so the pins depend on nothing beyond the page.
PIN_ICON_ATLAS = "data:image/svg+xml;base64," + base64.b64encode(
    PIN_SVG.encode("utf-8")
).decode("ascii")


# anchorY is the icon's full height, which puts the pin's point -- rather than
# its middle -- on the site's coordinates, the way a map pin is read. mask is
# False so deck.gl draws each icon in its own colours rather than tinting it.
def _pin_icon(x: int) -> dict:
    return {
        "x": x,
        "y": 0,
        "width": 24,
        "height": 32,
        "anchorX": 12,
        "anchorY": 32,
        "mask": False,
    }


PIN_ICON_MAPPING = {DEFAULT_PIN: _pin_icon(0), SELECTED_PIN: _pin_icon(24)}


def _literal(value: str) -> str:
    """Wrap a string so pydeck sends it as a value rather than as code.

    pydeck reads every bare string layer argument as a deck.gl accessor
    expression and prefixes it with ``@@=``, so an ordinary string -- a data
    URI, a unit name -- reaches the browser as an expression and fails to
    parse there rather than here. Quoting is pydeck's own escape hatch: it
    strips the quotes and passes the literal through.

    It strips *every* quote character, not just the outer pair, so a value
    containing one would come out mangled. None of ours do; the guard is here
    so that stays true.
    """
    if "'" in value:
        raise ValueError(f"pydeck cannot carry a literal containing a quote: {value!r}")
    return f"'{value}'"


def _site_layer(sites, selected_id) -> pdk.Layer:
    """One clickable pin per site, the selected one drawn in the other colour.

    Which icon and what size ride along as ordinary columns rather than as a
    second layer drawn over the top: one layer means one thing to pick, so a
    click cannot land on a highlight instead of the site under it.
    """
    chosen = [site_id == selected_id for site_id in sites[SITE_ID_FIELD]]
    frame = sites.assign(
        **{
            ICON_FIELD: [
                SELECTED_PIN if is_chosen else DEFAULT_PIN for is_chosen in chosen
            ],
            SIZE_FIELD: [
                SELECTED_PIN_SIZE if is_chosen else PIN_SIZE for is_chosen in chosen
            ],
            # The deck's one tooltip template reads these two; see the field
            # constants. A pin's title is its name and its detail the
            # coordinates it was computed at -- the same pair the caption
            # under the map repeats once the pin is clicked.
            TOOLTIP_TITLE_FIELD: sites["name"],
            TOOLTIP_DETAIL_FIELD: [
                f"{lat:.4f}, {lon:.4f}"
                for lat, lon in zip(sites["lat"], sites["lon"], strict=True)
            ],
        }
    )
    # Row order is deliberately left alone. Drawing the selected pin last would
    # keep it above its neighbours, but Streamlit records a selection as an
    # index into this frame and resolves it against whatever order the next
    # render sends; reordering between renders therefore makes that index name
    # a different site, and the selection silently walks to its neighbour on
    # the next rerun. Occlusion in a dense cluster is the lesser problem, and
    # the selected pin is drawn larger anyway.
    return pdk.Layer(
        "IconLayer",
        id=SITE_LAYER_ID,
        data=frame,
        get_position="[lon, lat]",
        get_icon=ICON_FIELD,
        get_size=SIZE_FIELD,
        icon_atlas=_literal(PIN_ICON_ATLAS),
        icon_mapping=PIN_ICON_MAPPING,
        size_units=_literal("pixels"),
        pickable=True,
        # deck.gl's own highlight blends a translucent colour into whichever
        # pin is hovered or picked, which fights the selected/unselected
        # colouring this layer already does for itself. The red pin is the
        # selection indicator; nothing else should tint one.
        auto_highlight=False,
    )


def heatmap_layer(image_uri, bounds, opacity=HEATMAP_OPACITY) -> pdk.Layer:
    """The gridded design-value surface, as a single image.

    A ``BitmapLayer`` rather than drawn cells, and not for performance. While
    a selection is live Streamlit rewrites the colour accessor of every layer
    whose deck.gl type appears in a table in its frontend bundle, clamping
    whatever it does not consider selected to 40% alpha -- the bug fixed for
    the pins in :func:`_site_layer` by carrying their colour in the icon.
    ``ColumnLayer`` is in that table. ``BitmapLayer`` is not in it at all, and
    declares no colour accessor either, so it is immune twice over and the
    overlay keeps its colour when a pin is clicked.

    ``pickable`` is left off deliberately. The pins are the only thing on this
    map that should answer a click, and an overlay drawn across the whole
    region would otherwise swallow every one of them.
    """
    if not isinstance(image_uri, str) or not image_uri:
        raise ValueError("image_uri must be a non-empty string")
    west, south, east, north = (float(value) for value in bounds)
    if west >= east or south >= north:
        raise ValueError(f"bounds are not west/south/east/north: {bounds!r}")

    return pdk.Layer(
        "BitmapLayer",
        id=HEATMAP_LAYER_ID,
        data=None,
        # Quoted for the same reason the pin atlas is: pydeck reads a bare
        # string as a deck.gl expression, so an unquoted data URI reaches the
        # browser as code and takes the whole deck down with it, blank, with
        # only `Unexpected ":" at character 4` to go on.
        image=_literal(image_uri),
        bounds=[west, south, east, north],
        opacity=opacity,
        pickable=False,
    )


# Alpha 1 of 255, not 0, and the difference is the whole trick.
#
# The cells have to be invisible until one is picked, and they have to carry a
# colour accessor to be invisible, which puts them in reach of the selection
# fade that BitmapLayer and the pins dodge. Streamlit's rewrite reads
#
#     [r[0]||0, r[1]||0, r[2]||0, r[3]||255]
#     alpha = isSelected ? Math.max(a[3], 255) : Math.min(a[3], 102)
#
# so alpha 0 is falsy, becomes 255, and the whole grid washes the map grey at
# 40% the moment anything is selected -- which is exactly what it did when
# this was first tried. Alpha 1 is truthy: min(1, 102) leaves it at 1, below
# the threshold of visible.
#
# A picked cell still gets max(1, 255) and would paint solid, which is a mark
# appearing on a click that is meant to do nothing at all. The layer's own
# `opacity` prop settles it: deck.gl applies it after the colour accessor,
# and Streamlit rewrites accessors, not opacity. At 0 the highlight is
# multiplied away while picking -- which runs in its own pass, against its
# own colours -- is untouched. Verified on a live map: the tooltip still
# answers and no square appears.
READOUT_FILL = (*SELECTED_PIN_COLOR, 1)
READOUT_OPACITY = 0.0


def readout_layer(points, half_lat, half_lon, detail="") -> pdk.Layer:
    """An invisible, pickable square per sampled point.

    ``points`` is ``(lat, lon, label)`` per cell -- :attr:`frer.heatmap.
    Samples.points`. The label is shown as-is; nothing is computed in the
    browser, because the tooltip template only substitutes fields.

    Each square's corners are derived in the browser from the row's own
    coordinates rather than sent, which is pydeck's ``@@=`` accessor
    behaviour put to work: a cell costs two numbers on the wire instead of
    ten. Measured on the Bay Area grid that took the deck from 468 kB to
    219 kB.

    Drawn under the pins so a pin wins the pick wherever the two overlap,
    which is the behaviour the site map already depends on.
    """
    if half_lat <= 0 or half_lon <= 0:
        raise ValueError(f"cell half-extents must be positive: {half_lat}, {half_lon}")

    data = [
        {
            CELL_LAT_FIELD: lat,
            CELL_LON_FIELD: lon,
            TOOLTIP_TITLE_FIELD: label,
            TOOLTIP_DETAIL_FIELD: detail,
        }
        for lat, lon, label in points
    ]
    if not data:
        raise ValueError("readout_layer needs at least one point")

    lon_field, lat_field = CELL_LON_FIELD, CELL_LAT_FIELD
    corners = (
        f"[[{lon_field}-{half_lon},{lat_field}-{half_lat}],"
        f"[{lon_field}+{half_lon},{lat_field}-{half_lat}],"
        f"[{lon_field}+{half_lon},{lat_field}+{half_lat}],"
        f"[{lon_field}-{half_lon},{lat_field}+{half_lat}]]"
    )
    return pdk.Layer(
        "PolygonLayer",
        id=READOUT_LAYER_ID,
        data=data,
        get_polygon=corners,
        get_fill_color=list(READOUT_FILL),
        opacity=READOUT_OPACITY,
        stroked=False,
        pickable=True,
        # deck.gl's own hover highlight would make the invisible cells visible
        # under the pointer. The grid answers the tooltip; it never draws.
        auto_highlight=False,
    )


# The contour lines, drawn by deck.gl rather than baked into the overlay so
# they stay crisp at every zoom.
CONTOUR_LAYER_ID = "design-value-contours"

# Line weight in pixels, with a floor and a ceiling. A contour is an
# annotation on the map rather than a thing on the ground, so it holds its
# weight as the map zooms instead of thickening with it.
CONTOUR_WIDTH = 1.5
CONTOUR_WIDTH_MIN = 1
CONTOUR_WIDTH_MAX = 4


def contour_layer(paths) -> pdk.Layer:
    """The band boundaries, as lines deck.gl draws at the viewport's scale.

    ``paths`` is :func:`frer.heatmap.contours` output: a ``path`` of
    ``[lon, lat]`` pairs and a ``colour`` per line. The colour has to come
    from the data rather than from one constant, because each boundary is
    drawn in its own band's hue -- which is what puts this layer in the table
    Streamlit rewrites while a selection is live. ``heatmap.CONTOUR_ALPHA``
    is what makes that rewrite a no-op; the reasoning is there.

    Width is in pixels, not metres: a contour is an annotation on the map,
    not a thing on the ground, so it should not thicken as you zoom in.
    """
    if not paths:
        raise ValueError("contour_layer needs at least one path")
    return pdk.Layer(
        "PathLayer",
        id=CONTOUR_LAYER_ID,
        data=paths,
        get_path="path",
        get_color="colour",
        get_width=CONTOUR_WIDTH,
        width_units=_literal("pixels"),
        width_min_pixels=CONTOUR_WIDTH_MIN,
        width_max_pixels=CONTOUR_WIDTH_MAX,
        cap_rounded=True,
        joint_rounded=True,
        # The readout grid underneath answers the pointer. A pickable line
        # would take hovers off it wherever the two cross.
        pickable=False,
    )


def site_map(
    sites, selected_id=None, overlay=None, readout=None, contours=None
) -> pdk.Deck:
    """Build the zoomable site map.

    ``sites`` needs ``id``, ``name``, ``lat``, and ``lon`` columns; any others
    ride along and are handed back with a click, which is how the
    app gets a site's name without a second lookup. ``selected_id`` is the site
    to colour as selected, or None when nothing has been picked yet.

    ``overlay`` is an optional ``(image_uri, bounds)`` pair -- a rendered
    field from :mod:`frer.heatmap` -- drawn underneath the pins. None leaves
    the map exactly as it was before the overlay existed.

    ``readout`` is an optional ``(points, half_lat, half_lon, detail)`` tuple
    for :func:`readout_layer`, the invisible grid that answers hovers and
    clicks away from the pins. None leaves the overlay a picture.

    ``contours`` is an optional list of traced band boundaries for
    :func:`contour_layer`, drawn over the overlay.

    Vendored state outlines used to be drawn underneath as an offline
    fallback; they went once it turned out Streamlit always draws a tiled
    basemap anyway, which left them nothing to fall back from and, zoomed in
    to a city, a simplified coastline cutting across open water and a grey
    wash over land the tiles had already drawn properly.
    """
    required = {SITE_ID_FIELD, "name", "lat", "lon"}
    missing = required - set(sites.columns)
    if missing:
        raise ValueError(f"sites is missing column(s) {sorted(missing)}")

    layers = []
    if overlay is not None:
        image_uri, bounds = overlay
        # First, so the pins draw on top of it rather than under it.
        layers.append(heatmap_layer(image_uri, bounds))
    if contours:
        # Over the fill, under the pins: the lines explain the wash, and a
        # pin has to stay clickable where one crosses it.
        layers.append(contour_layer(contours))
    if readout is not None:
        layers.append(readout_layer(*readout))
    layers.append(_site_layer(sites, selected_id))

    return pdk.Deck(
        layers=layers,
        map_style=TILE_BASEMAP,
        initial_view_state=pdk.ViewState(
            latitude=INITIAL_LATITUDE,
            longitude=INITIAL_LONGITUDE,
            zoom=INITIAL_ZOOM,
            min_zoom=MIN_ZOOM,
            max_zoom=MAX_ZOOM,
        ),
        tooltip={
            # One template for every layer -- see the field constants. The
            # substituted values are HTML-escaped by Streamlit, so the markup
            # has to live here and the data has to be plain text.
            "html": f"<b>{{{TOOLTIP_TITLE_FIELD}}}</b><br/>{{{TOOLTIP_DETAIL_FIELD}}}",
            "style": {
                "backgroundColor": "#262730",
                "color": "white",
                "fontSize": "12px",
                "padding": "6px 8px",
            },
        },
    )


def clicked_site_id(selection) -> int | None:
    """The site id from an ``st.pydeck_chart`` selection payload, if any.

    Streamlit reports a deck.gl click as
    ``{"objects": {"<layer id>": [{...row...}]}}``, carrying the whole data row
    of every picked object. Returns None for each empty or unexpected shape, so
    a rerun that carries no click cannot be mistaken for one, and takes the last
    entry because that is the most recent pick.
    """
    if not selection:
        return None
    picked = (selection.get("objects") or {}).get(SITE_LAYER_ID) or []
    if not picked:
        return None
    value = picked[-1].get(SITE_ID_FIELD)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def resolve_selection(
    selection, selected_id, on_readout, has_readout
) -> tuple[int | None, bool]:
    """What a map click leaves selected: ``(site id, click was on the grid)``.

    The pins are the only thing on this map that selects anything. The
    readout grid has to be pickable for the tooltip to work, and deck.gl
    picks on click as well as on hover, so a click anywhere on the heatmap
    arrives here as a selection of a layer that means nothing -- and must
    change nothing, rather than deselecting the site the way a click on bare
    map does.

    ``on_readout`` is whether the selection Streamlit is currently holding
    belongs to the grid, which has to be remembered because of the one case
    the payload cannot express. Switching the overlay off removes the layer,
    and Streamlit drops the selection of a layer no longer in the deck --
    arriving here looking exactly like a click on bare map. Deselecting there
    would mean turning off a heatmap threw away someone's site.
    """
    layers = picked_layers(selection)
    if READOUT_LAYER_ID in layers:
        return selected_id, True
    if layers:
        return clicked_site_id(selection), False
    if on_readout and not has_readout:
        return selected_id, False
    return None, False


def picked_layers(selection) -> set[str]:
    """Which layer ids the selection names, empty when nothing is picked.

    The app needs the empty case distinguished from "picked something that is
    not a site": a click on bare map clears the selection, a click on a
    readout cell does not.
    """
    if not selection:
        return set()
    return {key for key, rows in (selection.get("objects") or {}).items() if rows}
