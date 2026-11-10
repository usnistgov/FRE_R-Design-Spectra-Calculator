"""Tests for the site map in ``map_view``.

The map is the only way into a precomputed site, and most of what it has to get
right is a contract with somebody else's code: the payload Streamlit hands back
from a click, and the layer shape pydeck and deck.gl are given. Neither is
checked by anything else, and both have already broken once in ways that were
invisible until the map was driven in a browser.

So these cover the parts with no other guard: that a click payload is read back
correctly and that junk is rejected rather than mistaken for a selection; that
the layer still declares no colour accessor, which is what keeps the pins from
being faded by Streamlit's selection styling; that row order does not depend on
the selection, which is what keeps a selection from walking to a neighbouring
site; and that strings meant as values do not reach the browser as code.

The spectra behind the map are covered by test_single.py and test_benchmark.py.
Nothing here touches the network or the frozen USGS cache.
"""

import json

import pandas as pd
import pytest

import map_view

# Three well-separated sites, enough to tell a selected row from its
# neighbours. The real table is exercised through the app, not from here.
SITES = pd.DataFrame(
    {
        "id": [1, 2, 3],
        "name": ["Atmore AL", "Chicago IL", "Seattle WA"],
        "lat": [31.00, 41.85, 47.60],
        "lon": [-87.50, -87.65, -122.30],
        "region": ["CEUS", "CEUS", "PNW"],
    }
)

SELECTED_ID = 2


def site_layer(selected_id=None) -> dict:
    """The serialised site layer, as the browser receives it."""
    spec = json.loads(map_view.site_map(SITES, selected_id).to_json())
    layers = [
        layer for layer in spec["layers"] if layer.get("id") == map_view.SITE_LAYER_ID
    ]
    assert len(layers) == 1, "expected exactly one site layer"
    return layers[0]


def test_selected_site_uses_the_other_pin():
    """The selected row gets the selected icon and size; the rest do not."""
    rows = {row["id"]: row for row in site_layer(SELECTED_ID)["data"]}

    assert rows[SELECTED_ID]["icon"] == map_view.SELECTED_PIN
    assert rows[SELECTED_ID]["size"] == map_view.SELECTED_PIN_SIZE
    for site_id in (1, 3):
        assert rows[site_id]["icon"] == map_view.DEFAULT_PIN
        assert rows[site_id]["size"] == map_view.PIN_SIZE


@pytest.mark.parametrize("selected_id", [None, 999])
def test_nothing_is_selected_without_a_site_on_the_map(selected_id):
    """No selection, and an id that is not on the map, select nothing."""
    rows = site_layer(selected_id)["data"]
    assert {row["icon"] for row in rows} == {map_view.DEFAULT_PIN}


def test_row_order_does_not_depend_on_the_selection():
    """Selecting a site must not reorder the rows.

    Streamlit records a selection as an index into this frame and resolves it
    against whatever order the next render sends, so reordering between renders
    makes that index name a different site and the selection silently walks to
    its neighbour.
    """
    unselected = [row["id"] for row in site_layer()["data"]]
    selected = [row["id"] for row in site_layer(SELECTED_ID)["data"]]
    assert unselected == selected == list(SITES["id"])


def test_layer_declares_no_colour_accessor():
    """The pins must carry their colour in the icon, not in a colour accessor.

    While a selection is live Streamlit rewrites the colour accessor of every
    layer that declares one and fades whatever it does not consider selected.
    Declaring none is what keeps the pins at full colour either way.
    """
    assert "getColor" not in site_layer(SELECTED_ID)


def test_literal_strings_are_not_sent_as_expressions():
    """Values pydeck would otherwise read as deck.gl accessor expressions.

    pydeck prefixes every bare string layer argument with ``@@=``; a data URI
    sent that way reaches the browser as code and fails to parse there.
    """
    layer = site_layer()
    assert layer["iconAtlas"] == map_view.PIN_ICON_ATLAS
    assert layer["sizeUnits"] == "pixels"


def test_literal_rejects_a_value_it_cannot_carry():
    """pydeck strips every quote, so a value containing one cannot round-trip."""
    with pytest.raises(ValueError, match="quote"):
        map_view._literal("already 'quoted'")


def test_site_map_rejects_a_frame_missing_columns():
    """A frame without the columns the layer reads fails here, not in the page."""
    with pytest.raises(ValueError, match="region"):
        map_view.site_map(SITES.drop(columns=["region"]))


@pytest.mark.parametrize(
    "selection",
    [
        None,
        {},
        {"objects": {}},
        {"objects": {map_view.SITE_LAYER_ID: []}},
        {"objects": {"some-other-layer": [{"id": 5}]}},
        {"objects": {map_view.SITE_LAYER_ID: [{"name": "no id here"}]}},
        {"objects": {map_view.SITE_LAYER_ID: [{"id": "not a number"}]}},
    ],
)
def test_clicked_site_id_reads_no_click(selection):
    """Every empty or unexpected payload reads as no click, never as a site."""
    assert map_view.clicked_site_id(selection) is None


def test_clicked_site_id_reads_a_click():
    """A click comes back as the site id, from the row deck.gl hands over."""
    payload = {"objects": {map_view.SITE_LAYER_ID: [{"id": 42, "name": "Somewhere"}]}}
    assert map_view.clicked_site_id(payload) == 42


def test_clicked_site_id_takes_the_most_recent_pick():
    """With several picked objects the last one is the current selection."""
    payload = {"objects": {map_view.SITE_LAYER_ID: [{"id": 7}, {"id": 99}]}}
    assert map_view.clicked_site_id(payload) == 99
