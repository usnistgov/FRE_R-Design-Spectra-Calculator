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

# The column carrying the site id. It is what the app needs to load the cached
# entry, and deck.gl hands back the whole row, so the only requirement is that
# the frame passed to site_map carries it.
SITE_ID_FIELD = "id"

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


def site_map(sites, selected_id=None) -> pdk.Deck:
    """Build the zoomable site map.

    ``sites`` needs ``id``, ``name``, ``lat``, ``lon``, and ``region`` columns;
    any others ride along and are handed back with a click, which is how the
    app gets a site's name without a second lookup. ``selected_id`` is the site
    to colour as selected, or None when nothing has been picked yet.

    The pins are the only layer. Vendored state outlines used to be drawn
    underneath as an offline fallback; they went once it turned out Streamlit
    always draws a tiled basemap anyway, which left them nothing to fall back
    from and, zoomed in to a city, a simplified coastline cutting across open
    water and a grey wash over land the tiles had already drawn properly.
    """
    required = {SITE_ID_FIELD, "name", "lat", "lon", "region"}
    missing = required - set(sites.columns)
    if missing:
        raise ValueError(f"sites is missing column(s) {sorted(missing)}")

    return pdk.Deck(
        layers=[_site_layer(sites, selected_id)],
        map_style=TILE_BASEMAP,
        initial_view_state=pdk.ViewState(
            latitude=INITIAL_LATITUDE,
            longitude=INITIAL_LONGITUDE,
            zoom=INITIAL_ZOOM,
            min_zoom=MIN_ZOOM,
            max_zoom=MAX_ZOOM,
        ),
        tooltip={
            "html": "<b>{name}</b><br/>{region}",
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
