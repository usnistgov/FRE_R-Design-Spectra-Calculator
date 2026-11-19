"""Streamlit UI for single-site FREr design values.

The page is a map. Its pins are the precomputed sites that ship with the
calculator, plus any site the viewer adds by coordinate -- which runs both
stages of the command-line workflow for that location: pulls the USGS design
values and hazard curves, then develops the FREr and DE design spectra.
The calculation itself lives in :mod:`frer.single`, which this module only
drives -- nothing under ``src/frer`` imports Streamlit, so the package stays
usable from the CLI, the tests, and any other front end.

Run locally with ``streamlit run app.py``.
"""

from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

import map_view
from frer import __version__, heatmap, usgs_api
from frer.cache import cache_path, load_entry
from frer.grid import BAY_AREA, load_grid, load_manifest
from frer.params import (
    DE_ORIENTATION,
    FR_ORIENTATION,
    NSHM_BOUNDS,
    RISK_CATEGORIES,
    SITE_CLASS_VS30,
    VERSION_NSHM,
    VS30_RANGE,
)
from frer.single import (
    RP_PERIODS,
    calc_site,
    design_value_columns,
    frer_mps_columns,
    pull_site,
    return_period_columns,
)
from frer.sites import load_sites
from frer.spectrum import two_point_spectrum

REPO_ROOT = Path(__file__).resolve().parent
SITES_CSV = REPO_ROOT / "sites.csv"
CACHE_TTL = 24 * 3600  # seconds; a day is well inside the USGS revision cadence
CACHE_ENTRIES = 32  # each hazard response is ~750 kB of JSON
VERSIONS = list(VERSION_NSHM)

# The precomputed site list is served entirely from the frozen USGS responses
# committed for the benchmark, so those sites resolve offline with no live call.
# It is the usgs_160 set: every one of its 160 sites has a cache file for each of
# the four site classes BC/C/D/E, holding both NSHM editions and the ASCE 7-22
# design values. Sites are still filtered on what is actually cached (see
# precomputed_sites), so an incomplete cache shortens the list rather than
# offering a site that cannot be calculated.
PRECOMPUTED_CACHE_DIR = REPO_ROOT / "tests" / "usgs_cache"
PRECOMPUTED_SITE_SET = "usgs_160"
# The frozen design values were pulled at risk category II (see frer.pull).
PRECOMPUTED_RISK_CATEGORY = "II"

# Columns the site map needs (see map_view.site_map). The frame is trimmed to
# these before it goes into the chart: deck.gl serialises the data into the page,
# so the unused sites.csv columns would otherwise be sent on every rerun.
MAP_COLUMNS = ["id", "name", "lat", "lon"]

# The gridded heatmap overlay. One database, ASCE 7-28 only: its DE values come
# free with the hazard call that NSHM 2023 answers, where ASCE 7-22 would need a
# second call per node and so a second build of roughly twice the length. The
# overlay therefore states its code version and does NOT follow the code
# version of the single-site results underneath it.
GRID_DIR = REPO_ROOT / "data" / "grid"
GRID_CSV = GRID_DIR / "bay_area_asce7-28.csv.gz"
GRID_JSON = GRID_DIR / "bay_area_asce7-28.json"
GRID_VERSION = "ASCE 7-28"

# Session-state key for the mapped site. Plain state rather than a widget key:
# the map is the only thing that selects a site, so nothing else writes it and
# there is no widget whose creation order has to be worked around.
SITE_STATE_KEY = "mapped_site_id"

# Whether the selection Streamlit is holding belongs to the readout grid
# rather than to a pin. Remembered only so that switching the overlay off,
# which deletes that selection, is not mistaken for a click on bare map; see
# map_view.resolve_selection.
READOUT_PICK_KEY = "mapped_pick_on_readout"

# Sites added by coordinate this session, and the one the sidebar put in front
# of the viewer when it added it. See added_sites and active_site_id.
CUSTOM_SITES_KEY = "added_sites"
CUSTOM_FOCUS_KEY = "added_site_focus"

# Every site is developed for both code versions. The comparison is the
# question this page is for, and a precomputed site answers both out of one
# cache file -- so the only cost is on an added site, which pays a second USGS
# call for the second edition.

# Chart series, in legend order, with their colour and dash pattern. DE is
# black and FREr is the calculator's blue (matching the batch PNGs); the
# multi-period spectrum is solid and the 2-point spectrum dashed.
SERIES_STYLE = {
    "DE: MPS": ("#000000", [1, 0]),
    "DE: 2-point": ("#000000", [6, 4]),
    "FRE_R: MPS": ("#0072BD", [1, 0]),
    "FRE_R: 2-point": ("#0072BD", [6, 4]),
}
PERIOD_MAX = 5.0  # seconds; the range the 2-point spectra are defined over

# Shown under the return-period table. The spectral values carry their
# orientation in their own labels, so the only thing left to say is that the
# return periods do not: every one is read off the RotD50 hazard curve at the
# unscaled RTGM, whether or not the reported ordinate got the max-direction
# factor applied.
RETURN_PERIOD_NOTE = (
    "Note, for comparing return periods with spectral values, return period "
    "calculations are not affected by ground motion orientation "
    f"(i.e., {FR_ORIENTATION} vs {DE_ORIENTATION})."
)

st.set_page_config(page_title="FREr Design Spectra Calculator", layout="wide")


# --- USGS access -----------------------------------------------------------
#
# Cached in memory rather than on disk: Streamlit ignores `ttl` when persisting
# to disk, and Community Cloud clears the disk on restart anyway, so persisting
# would trade expiry for almost no durability. A hazard call takes 12-15 s, so
# the cache is what makes repeated runs feel instant.


@st.cache_data(ttl=CACHE_TTL, max_entries=CACHE_ENTRIES, show_spinner=False)
def fetch_design(lat, lon, site_class, risk_category):
    """Cached USGS design-value call."""
    return usgs_api.call_design_value_api(
        f"{lat:.4f}, {lon:.4f}",
        lat,
        lon,
        risk_category=risk_category,
        site_class=site_class,
    )


@st.cache_data(ttl=CACHE_TTL, max_entries=CACHE_ENTRIES, show_spinner=False)
def fetch_hazard(nshm, lat, lon, vs30):
    """Cached USGS hazard-curve call."""
    return usgs_api.call_hazard_api(nshm, lat, lon, vs30)


def cached_classes(site_id: int) -> list[str]:
    """Site classes with a committed cache file for ``site_id``, canonical order."""
    return [
        site_class
        for site_class in SITE_CLASS_VS30
        if cache_path(PRECOMPUTED_CACHE_DIR, site_id, site_class).exists()
    ]


@st.cache_data(show_spinner=False)
def precomputed_sites(site_set: str):
    """Precomputed-set sites that have committed cache, or None if there are none.

    A site is offered only when at least one of its classes is cached, so every
    entry in the list resolves without a live USGS call.

    ``site_set`` is an argument rather than a read of PRECOMPUTED_SITE_SET so
    that it takes part in the cache key. Read from the module, a change to that
    constant would leave an already-running server serving the previous set's
    list from cache until the process was restarted.
    """
    try:
        sites = load_sites(SITES_CSV, site_set)
    except FileNotFoundError:
        return None
    available = [row for _, row in sites.iterrows() if cached_classes(int(row["id"]))]
    if not available:
        return None
    return pd.DataFrame(available).reset_index(drop=True)


# --- Inputs ----------------------------------------------------------------


def added_sites() -> list[dict]:
    """The sites added by coordinate this session, oldest first.

    Session state and nothing else: they last as long as the browser session,
    nothing is written to the server, and two people on the same deployment
    never see each other's sites. Reloading the page starts over.

    Each record carries what the site was pulled with -- coordinates, site
    class, vs30, risk category -- and the developed results, so a rerun redraws
    it without touching the network.
    """
    return st.session_state.setdefault(CUSTOM_SITES_KEY, [])


def next_added_id(records) -> int:
    """An id for a new added site.

    Negative and descending, so it can neither collide with a precomputed
    site's id nor be reused by a later add within a session -- a reused id
    would hand a stale selection the wrong site.
    """
    return min((record["id"] for record in records), default=0) - 1


def map_frame(mapped, records) -> pd.DataFrame:
    """Every pin on the map: the precomputed sites first, added ones after.

    The order is deliberate. Streamlit records a map selection as an index into
    this frame and resolves it against whatever order the next render sends, so
    appending keeps every existing index pointing at the site it pointed at
    before; putting an added site anywhere else would walk a live selection
    onto its neighbour.
    """
    frames = []
    if mapped is not None:
        frames.append(mapped[MAP_COLUMNS])
    if records:
        frames.append(
            pd.DataFrame(
                [{key: record[key] for key in MAP_COLUMNS} for record in records]
            )
        )
    if not frames:
        return pd.DataFrame({key: [] for key in MAP_COLUMNS})
    return pd.concat(frames, ignore_index=True)


def current_site_id(sites) -> int | None:
    """The site the map has selected, or None if nothing has been picked.

    The page opens with nothing selected, so the area below the map stays
    empty until someone chooses a site. A stored id that is no longer on the
    map is discarded rather than kept, which covers a site list that has
    changed under it.
    """
    ids = {int(site_id) for site_id in sites["id"]}
    if st.session_state.get(SITE_STATE_KEY) not in ids:
        st.session_state[SITE_STATE_KEY] = None
    return st.session_state[SITE_STATE_KEY]


def active_site_id(sites) -> int | None:
    """The site whose values belong under the map.

    A site just added by coordinate, if there is one, and otherwise whatever
    the map has selected. An added site has no selection to live in -- deck.gl
    holds only what the viewer clicked, and a site added from the sidebar was
    never clicked -- so it is held here instead, and it has to win: adding a
    site while a pin was selected would otherwise leave the viewer looking at
    the old pin after a 30-second wait for the new one.

    It gives way as soon as the map reports a pick, which is
    :func:`_apply_map_selection`'s job. A pick, not a deselection: adding a
    site changes the chart's data, and Streamlit drops the selection it was
    holding when that happens -- which arrives looking exactly like a click on
    bare map, and would otherwise throw away the site that was just added, on
    the one path where someone had a pin selected before adding.

    The seam is the other half of that. While this is holding the page, a
    click on bare map reports the deselection the page is already in, so it
    changes nothing; clicking any pin releases it.
    """
    selected = current_site_id(sites)
    focus = st.session_state.get(CUSTOM_FOCUS_KEY)
    if focus is None:
        return selected
    if focus not in {int(site_id) for site_id in sites["id"]}:
        st.session_state.pop(CUSTOM_FOCUS_KEY, None)
        return selected
    return focus


def site_class_input(mapped) -> str:
    """The sidebar's site class, which governs the mapped sites and the overlay.

    An added site is not governed by it: it is pulled at the class chosen when
    it was added and keeps it, because following this selector would mean a
    fresh USGS call per added site every time it moved.

    The classes offered are those every mapped site has cached, so whichever
    pin is clicked next can answer for the class already chosen rather than the
    selector having to reshuffle under the click. With no mapped sites at all
    it offers every class, because the heatmap still reads this.

    It is drawn under the Heatmap section's own field rather than above it: the
    overlay reads both, and the two belong together.
    """
    classes = []
    if mapped is not None:
        classes = shared_site_classes(tuple(int(site_id) for site_id in mapped["id"]))
    return st.sidebar.selectbox(
        "Site class",
        classes or list(SITE_CLASS_VS30),
        help="Applies to the mapped sites and the heatmap. A site you add "
        "yourself keeps the site class it was added with.",
    )


@st.dialog("Add a site")
def add_site_dialog() -> None:
    """The coordinate inputs, in a modal over the map.

    A dialog rather than a panel in the sidebar: the sidebar is where the map
    is steered from, and six inputs that are used once a session crowded out
    the two controls that are used constantly. It opens on the button below
    and closes on the rerun that a successful add triggers.

    The widgets are loose rather than wrapped in ``st.form`` so that vs30 can
    keep following the site class, which a form -- submitting everything at
    once -- would leave a step behind. Only the submit button pulls from USGS.

    A successful add writes the site into session state, points the page at it
    and reruns; nothing is returned.
    """
    records = added_sites()
    default_name = f"Site {len(records) + 1}"

    st.caption(
        "Anywhere in the conterminous U.S. Values are pulled live from USGS, "
        "which takes about 15 seconds per code version."
    )
    left, right = st.columns(2)
    lat = left.number_input(
        "Latitude",
        min_value=NSHM_BOUNDS[1],
        max_value=NSHM_BOUNDS[3],
        value=34.05,
        step=0.01,
        format="%.4f",
    )
    lon = right.number_input(
        "Longitude",
        min_value=NSHM_BOUNDS[0],
        max_value=NSHM_BOUNDS[2],
        value=-118.25,
        step=0.01,
        format="%.4f",
    )
    name = st.text_input(
        "Name (optional)",
        placeholder=default_name,
        help="Shown on the pin and used for the CSV download.",
    )

    classes = list(SITE_CLASS_VS30)
    site_class = st.selectbox(
        "Site class",
        classes,
        index=classes.index("D"),
        key="add_site_class",
    )
    # vs30 defaults to the selected class's value; the class is in the widget key
    # so changing class resets it, while a manual override holds until it does.
    vs30 = st.number_input(
        "v$_{S30}$ (m/s)",
        min_value=float(VS30_RANGE[0]),
        max_value=float(VS30_RANGE[1]),
        value=float(SITE_CLASS_VS30[site_class]),
        step=5.0,
        key=f"add_vs30_{site_class}",
        help="Shear-wave velocity used for the hazard-curve request. Defaults "
        "to the USGS value for the selected site class; override it freely.",
    )
    risk_category = st.selectbox(
        "Risk category",
        RISK_CATEGORIES,
        index=RISK_CATEGORIES.index("II"),
        key="add_risk_category",
        help="Affects the ASCE 7-22 DE values only. FRE_R and the ASCE 7-28 "
        "spectra are computed from the hazard curves and do not use it.",
    )
    if not st.button("Add site", type="primary", use_container_width=True):
        return

    record = {
        "id": next_added_id(records),
        "name": name.strip() or default_name,
        "lat": lat,
        "lon": lon,
        "site_class": site_class,
        "vs30": vs30,
        "risk_category": risk_category,
    }
    try:
        record["results"] = run_analysis({**record, "versions": list(VERSIONS)})
    except ValueError as error:
        st.error(str(error))
        return
    except usgs_api.USGSApiError as error:
        st.error(
            f"The USGS web service did not return usable data: {error}\n\n"
            "Some coordinates genuinely fail on a given model. Try a "
            "neighbouring point or a different site class."
        )
        return

    records.append(record)
    # The map has no selection to put this in -- see active_site_id -- so the
    # page is pointed at it directly, and the rerun is what draws it. It also
    # closes the dialog, which is how a dialog is dismissed from the script.
    st.session_state[CUSTOM_FOCUS_KEY] = record["id"]
    st.rerun()


def add_site_button() -> None:
    """The sidebar button that opens the add-a-site dialog, and what it left.

    The list of what has been added stays in the sidebar rather than going into
    the dialog with the inputs: it is the answer to "what have I got", which is
    a question asked with the dialog shut.
    """
    st.sidebar.header("Add a site")
    if st.sidebar.button(
        "Add a site by coordinate", type="primary", use_container_width=True
    ):
        add_site_dialog()
    records = added_sites()
    if records:
        st.sidebar.caption(
            "Added this session: " + ", ".join(record["name"] for record in records)
        )


@st.cache_data(show_spinner=False)
def shared_site_classes(site_ids: tuple[int, ...]) -> list[str]:
    """Site classes every one of these sites has cached, in canonical order.

    The sidebar offers these before any site has been picked, so it cannot ask
    a particular site what it holds. Taking the classes common to all of them
    means whichever site is clicked next can answer for the class already
    chosen, rather than the selector having to reshuffle under the click.
    """
    shared = None
    for site_id in site_ids:
        classes = set(cached_classes(site_id))
        shared = classes if shared is None else shared & classes
    return [c for c in SITE_CLASS_VS30 if c in (shared or set())]


def data_stamp() -> tuple:
    """What everything derived from the gridded database is cached against.

    The files' identity -- name, size, modification time -- rather than their
    contents, which would mean reading them to decide whether to read them.

    It is a cache key and nothing else. Without it, ``st.cache_data`` keys on
    the arguments alone, so a server that is still running when the database
    changes underneath it keeps serving what it derived from the old one, for
    as long as the process lives. That is not hypothetical: a corrected
    coastline digest was pushed to a deployment and had no effect, because
    the manifest it corrected was being served out of cache.
    """
    return tuple(
        (path.name, stat.st_size, stat.st_mtime_ns)
        for path, stat in ((p, p.stat()) for p in (GRID_CSV, GRID_JSON))
    )


def grid_database():
    """The committed gridded database, or ``(None, None)`` if it is absent.

    Absent rather than fatal: the calculator's single-site flow does not need
    it, so a checkout without it should lose the overlay and nothing else.
    """
    if not (GRID_CSV.exists() and GRID_JSON.exists()):
        return None, None
    return _grid_database(data_stamp())


@st.cache_data(show_spinner=False)
def _grid_database(stamp):
    return load_grid(GRID_CSV), load_manifest(GRID_JSON)


@st.cache_data(show_spinner=False)
def render_overlay(stamp, site_class: str, field_key: str):
    """One rendered field, cached across reruns.

    ``stamp`` is :func:`data_stamp`: it takes no part in the render, only in
    the cache key, so new data cannot be served from an old cache.

    The first render builds the coastline clip, which is the expensive part
    and is then shared by every field and class; after it, switching is
    nearly free. Caching here as well means a rerun from clicking a pin does
    not redo even that.
    """
    table, manifest = grid_database()
    if table is None:
        return None
    heatmap.check_coastline(manifest, BAY_AREA.coastline)
    return heatmap.render(table, site_class, field_key)


@st.cache_data(show_spinner=False)
def contour_paths(stamp, site_class: str, field_key: str):
    """The band boundaries as lines, cached across reruns.

    Traced from the same database the image is drawn from, so the lines fall
    exactly where the fill changes colour.
    """
    table, _ = grid_database()
    if table is None:
        return None
    return heatmap.contours(table, site_class, field_key)


@st.cache_data(show_spinner=False)
def sample_surface(stamp, site_class: str, field_key: str):
    """The hover readout's sampled surface, cached across reruns.

    Derived from the committed database, not stored beside it: it is a pure
    function of a file that is already in the repository, and recomputing it
    costs about fifty milliseconds. A second file holding the same numbers
    would need its own manifest and its own validation, and could go stale
    against the database the day one was rebuilt without the other.
    """
    table, _ = grid_database()
    if table is None:
        return None
    return heatmap.sample_surface(table, site_class, field_key)


def heatmap_field_input():
    """The field the overlay draws, or None when there is no database to draw.

    The overlay is always drawn: the fill is faint enough and the contour
    lines sparse enough that it informs the map rather than covering it, so
    there is neither a switch to turn it off nor an opacity control to reach
    for. Only the field is chosen here.

    It was behind a toggle, off, while the overlay was an opaque wash that got
    in the way of finding a site. What replaced the wash reads as part of the
    map, and the question it answers -- how values vary around here -- is the
    one the map is opened for, so the switch went too.
    """
    st.sidebar.header("Heatmap")
    table, _ = grid_database()
    if table is None:
        st.sidebar.caption(
            "No gridded database in `data/grid`. Build one with "
            "`python scripts/build_grid.py`."
        )
        return None

    return st.sidebar.radio(
        "Field",
        list(heatmap.FIELDS),
        format_func=lambda key: heatmap.FIELDS[key].label,
    )


def heatmap_layers(site_class, field_key):
    """What the overlay draws with; ``(overlay, contours, samples)``.

    Separate from :func:`heatmap_field_input` because it needs the site class,
    which is asked for after the field -- the one sidebar order that puts the
    two selectors the overlay reads next to each other. What it has to say for
    itself therefore lands under both of them, which is where the section
    ends.

    ``(None, None, None)`` is left for the cases that are not a choice -- no
    gridded database, or a field that fails to render -- and means draw no
    overlay.
    """
    if field_key is None:
        return None, None, None
    try:
        stamp = data_stamp()
        overlay = render_overlay(stamp, site_class, field_key)
        lines = contour_paths(stamp, site_class, field_key)
        samples = sample_surface(stamp, site_class, field_key)
    except heatmap.HeatmapError as error:
        st.sidebar.error(str(error))
        return None, None, None
    st.sidebar.caption(
        "Hover anywhere on the shaded area to read an interpolated value."
    )
    return overlay, lines, samples


def render_site_map(
    sites, selected_id, custom_ids=(), overlay=None, contours=None, samples=None
) -> None:
    """Draw the zoomable site map and act on a click.

    ``custom_ids`` are the sites added by coordinate, which draw green.

    A click records the site and reruns, which moves the highlight ring onto it
    and redevelops the values shown underneath.

    The chart keeps one stable key so panning and zooming survive that rerun.
    Re-keying it per site is the obvious way to drop a selection deck.gl is
    still holding, but it would also throw the viewer back out to the whole
    country every time they picked a site, which is the opposite of the point.
    Comparing against the current state is what stops the rerun repeating:
    deck.gl keeps reporting the same selection until it changes.
    """
    event = st.pydeck_chart(
        map_view.site_map(
            sites[MAP_COLUMNS],
            selected_id,
            overlay=(overlay.data_uri, overlay.bounds) if overlay else None,
            readout=(
                (samples.points, samples.half_lat, samples.half_lon, samples.detail)
                if samples is not None
                else None
            ),
            contours=contours,
            custom_ids=custom_ids,
        ),
        height=map_view.MAP_HEIGHT,
        width="stretch",
        key="site_map",
        on_select="rerun",
        selection_mode="single-object",
    )
    if overlay is not None:
        # Under the map rather than beside it: the bar has to be read against
        # the colours it explains, and the sidebar is the wrong side of the
        # page from most of them.
        st.markdown(heatmap.legend_html(overlay, GRID_VERSION), unsafe_allow_html=True)
    added = len(custom_ids)
    st.caption(
        f"{len(sites)} sites. Click a marker for its design spectra; scroll to "
        "zoom, drag to pan. Blue pins come from the USGS responses committed "
        "with the calculator, so no network call is made"
        + (
            f"; the {added} green pin{'' if added == 1 else 's'} you added came "
            "from a live call."
            if added
            else "."
        )
        + (
            " The heatmap covers the San Francisco Bay Area only — zoom in to read it."
            if overlay is not None
            else ""
        )
    )

    if _apply_map_selection(event.selection, samples is not None):
        st.rerun()


def _apply_map_selection(selection, has_readout) -> bool:
    """Store what the map reports; True when the page has to rerun.

    The rule itself is :func:`map_view.resolve_selection`, which is a pure
    function so it can be tested; this is the part that touches session
    state.

    What it compares against is the map's own selection rather than whatever
    the page is currently showing. The add form can point the page at a site
    the map holds no selection for (see :func:`active_site_id`), and comparing
    against that would read as a deselection on every rerun and rerun forever.
    """
    selected_id = st.session_state.get(SITE_STATE_KEY)
    on_readout = bool(st.session_state.get(READOUT_PICK_KEY))
    wanted_site, wanted_readout = map_view.resolve_selection(
        selection, selected_id, on_readout, has_readout
    )
    st.session_state[SITE_STATE_KEY] = wanted_site
    st.session_state[READOUT_PICK_KEY] = wanted_readout
    if wanted_site is not None:
        # The map has reported a pick, so it -- rather than the add form, which
        # has no way of knowing it was overtaken -- is what says which site to
        # show from here on.
        st.session_state.pop(CUSTOM_FOCUS_KEY, None)
    return wanted_site != selected_id


# --- Results ---------------------------------------------------------------


def run_analysis(inputs):
    """Pull and calculate every requested version for an added site.

    Precomputed sites do not come through here: they are read from the committed
    cache by site_results, with no network access and no spinner to justify.
    """
    results = []
    for version_id in inputs["versions"]:
        with st.spinner(
            f"Fetching NSHM {VERSION_NSHM[version_id]} data for {version_id} "
            "from USGS (about 15 seconds)..."
        ):
            entry = pull_site(
                inputs["lat"],
                inputs["lon"],
                inputs["site_class"],
                inputs["vs30"],
                version_id,
                name=inputs["name"],
                risk_category=inputs["risk_category"],
                fetch_design=fetch_design,
                fetch_hazard=fetch_hazard,
            )
        results.append(calc_site(entry, version_id))
    return results


def summary_frame(results) -> pd.DataFrame:
    """One row per version, using the column names of frer_design_vals.csv."""
    rows = []
    for result in results:
        row = {
            "name": result.name,
            "lat": result.lat,
            "lon": result.lon,
            "site_class": result.site_class,
            "vs30": result.vs30,
            "risk_category": result.risk_category,
            "version": result.version_id,
            "nshm": result.nshm,
        }
        # Spectral values carry their orientation in the header; the
        # return-period columns below are all RotD50 and stay unsuffixed.
        row.update(design_value_columns(result))
        row.update(frer_mps_columns(result.fr_spectrum))
        row.update(return_period_columns(result))
        rows.append(row)
    return pd.DataFrame(rows)


def chart_frame(results) -> pd.DataFrame:
    """Long-form (period, Sa, series, version) rows for the spectra chart.

    Altair wants one row per point with the series as a column, rather than the
    wide, one-column-per-series layout the table below uses.
    """
    frames = []
    for result in results:
        fr_t, fr_sa = two_point_spectrum(result.sdsfr, result.sd1fr, t_start=0.0)
        de_t, de_sa = two_point_spectrum(result.sds, result.sd1, t_start=0.0)
        series = {
            "DE: MPS": (result.de_period, result.de_sa),
            "DE: 2-point": (de_t, de_sa),
            "FRE_R: MPS": (result.fr_spectrum.period, result.fr_spectrum.sa),
            "FRE_R: 2-point": (fr_t, fr_sa),
        }
        for label, (period, sa) in series.items():
            frames.append(
                pd.DataFrame(
                    {
                        "Period (s)": pd.Series(period, dtype="float64"),
                        "Sa (g)": pd.Series(sa, dtype="float64"),
                        "Series": label,
                        "Version": result.version_id,
                    }
                )
            )
    return pd.concat(frames, ignore_index=True)


def spectra_chart(result, y_max=None):
    """Interactive linear-scale spectra chart for one code version.

    Each version gets its own chart in its own section rather than one faceted
    chart: Vega-Lite cannot size a faceted spec to its container, so a facet
    renders at a fixed width and is clipped on a narrow screen. ``y_max`` pins
    the y axis across every version so the comparison stays honest, which is
    what lets two charts a scroll apart be read against each other.
    """
    data = chart_frame([result])
    labels = list(SERIES_STYLE)
    y_scale = alt.Scale(zero=True) if y_max is None else alt.Scale(domain=[0, y_max])
    chart = (
        alt.Chart(data)
        .mark_line(clip=True)
        .encode(
            x=alt.X(
                "Period (s):Q",
                scale=alt.Scale(domain=[0, PERIOD_MAX], nice=False),
                title="Period (s)",
            ),
            y=alt.Y("Sa (g):Q", scale=y_scale, title="Sa (g)"),
            color=alt.Color(
                "Series:N",
                sort=labels,
                scale=alt.Scale(
                    domain=labels,
                    range=[SERIES_STYLE[label][0] for label in labels],
                ),
                # Two rows of two, so the labels are not truncated on a
                # narrow screen.
                legend=alt.Legend(title=None, orient="top", columns=2),
            ),
            strokeDash=alt.StrokeDash(
                "Series:N",
                sort=labels,
                scale=alt.Scale(
                    domain=labels,
                    range=[SERIES_STYLE[label][1] for label in labels],
                ),
                # Two rows of two, so the labels are not truncated on a
                # narrow screen.
                legend=alt.Legend(title=None, orient="top", columns=2),
            ),
            tooltip=[
                alt.Tooltip("Version:N"),
                alt.Tooltip("Series:N"),
                alt.Tooltip("Period (s):Q", format=".3f"),
                alt.Tooltip("Sa (g):Q", format=".4f"),
            ],
        )
        .properties(height=420, title=result.version_id)
    )
    # Deliberately not .interactive(): binding pan/zoom to the scales lets the
    # axes be dragged off their domains, which would scroll the 7.5 s and 10 s
    # ordinates into a plot that is meant to show 0-5 s. Tooltips still work.
    return chart


def spectra_frame(results) -> pd.DataFrame:
    """Multi-period FREr and DE ordinates for every version, joined by period."""
    table = None
    for result in results:
        frame = pd.DataFrame({"Period (s)": result.fr_spectrum.period})
        frame[f"{result.version_id} FRE_R Sa (g, {FR_ORIENTATION})"] = (
            result.fr_spectrum.sa
        )
        de = pd.DataFrame(
            {
                "Period (s)": result.de_period,
                f"{result.version_id} DE Sa (g, {DE_ORIENTATION})": result.de_sa,
            }
        )
        frame = frame.merge(de, on="Period (s)", how="outer")
        if table is None:
            table = frame
        else:
            table = table.merge(frame, on="Period (s)", how="outer")
    return table.sort_values("Period (s)").reset_index(drop=True)


def _ratio_to_de(frer_value: float, de_value: float) -> str | None:
    """FREr as a percentage change from the DE value, or None if DE is zero."""
    if not de_value:
        return None
    return f"{frer_value / de_value - 1:+.0%} vs DE"


# Ground-motion column prefixes (see frer.single.return_period_columns) and the
# labels used to present them as table rows. LaTeX subscripts (matching the page
# title and the metric labels) render via st.markdown -- st.dataframe would print
# the raw "$" markup, so the return-period table is emitted as Markdown below.
GM_RP_LABELS = {"frer": "FRE$_R$", "mcer": "MCE$_R$", "de": "DE"}


def return_period_markdown(result) -> str | None:
    """Return periods (years) as a Markdown table string, or None for ASCE 7-22.

    Rendered with st.markdown rather than st.dataframe so the ground-motion
    labels show their subscripts (FRE_R, MCE_R) as LaTeX; a dataframe prints the
    raw markup. None when there are no return periods (ASCE 7-22).
    """
    cols = return_period_columns(result)
    if not cols:
        return None
    header = "| | " + " | ".join(f"{t:.1f} s" for t in RP_PERIODS) + " |"
    sep = "|" + "---|" * (len(RP_PERIODS) + 1)
    lines = [header, sep]
    for gm, label in GM_RP_LABELS.items():
        cells = []
        for t in RP_PERIODS:
            value = cols[f"{gm}_rp_{f'{t:.1f}'.replace('.', 'p')}"]
            cells.append("n/a" if pd.isna(value) else f"{value:.0f}")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def newest_first(results):
    """The results in reverse code order, so the current edition leads.

    VERSION_NSHM is written oldest first -- it is the order a CSV of both
    editions is read in, and the order the calculator has always produced --
    so the page reverses it rather than the data being reordered underneath
    everything else that consumes it.
    """
    return sorted(results, key=lambda r: VERSIONS.index(r.version_id), reverse=True)


def show_version(result, y_max, comparing):
    """One code version, whole: values, return periods, spectra, ordinates.

    Gathered per version rather than per kind of output -- all the metrics,
    then all the charts, then one table of every version's ordinates -- because
    reading a single edition then meant picking its column out of three places
    down the page.

    ``y_max`` pins the chart's y axis across versions, which matters more now
    that the two charts are a scroll apart rather than side by side.
    ``comparing`` says whether the other edition is on the page too, which is
    the only thing either section needs to know about the other.
    """
    st.subheader(f"{result.version_id} - NSHM {result.nshm}")
    cols = st.columns(4)
    cols[0].metric(f"$S_{{DS}}$ (g, {DE_ORIENTATION})", f"{result.sds:.3f}")
    cols[1].metric(f"$S_{{D1}}$ (g, {DE_ORIENTATION})", f"{result.sd1:.3f}")
    cols[2].metric(
        f"$S_{{DSfr}}$ (g, {FR_ORIENTATION})",
        f"{result.sdsfr:.3f}",
        delta=_ratio_to_de(result.sdsfr, result.sds),
        delta_color="off",
    )
    cols[3].metric(
        f"$S_{{D1fr}}$ (g, {FR_ORIENTATION})",
        f"{result.sd1fr:.3f}",
        delta=_ratio_to_de(result.sd1fr, result.sd1),
        delta_color="off",
    )

    rp_table = return_period_markdown(result)
    st.markdown("**Return periods (years)**")
    if rp_table is not None:
        st.markdown(rp_table)
        st.caption(RETURN_PERIOD_NOTE)
        if result.site_class == "E":
            st.warning(
                "Site class E (soft soil) sites often have steep hazard "
                "curves. Therefore, the calculated return periods can be "
                "quite sensitive to small variations in spectral value.",
                icon="⚠️",
            )
    else:
        # Where to send someone whose ASCE 7-22 section has no return periods.
        # The 7-28 section is above it on this page, and when it is the only
        # section there is nothing to point at but the selector the single-site
        # flow has.
        st.caption(
            "Not defined for ASCE 7-22: its design values mix risk-targeted "
            "and deterministic caps that have no single hazard-curve return "
            "period. "
            + (
                "The ASCE 7-28 return periods are above."
                if comparing
                else "Select ASCE 7-28 for return periods."
            )
        )

    st.markdown("**Design spectra**")
    st.altair_chart(spectra_chart(result, y_max), width="stretch")

    st.markdown("**Multi-period spectra**")
    st.dataframe(
        spectra_frame([result]).style.format("{:.4f}"),
        width="stretch",
        hide_index=True,
    )


def show_results(results):
    """Render a section per code version, then one CSV of them all.

    The download stays whole: a file of both editions is what the command-line
    workflow writes, and splitting it per section would turn one artefact into
    two that have to be joined again.
    """
    # A single y limit across versions, so the charts can be compared directly
    # even though only one of them is on screen at a time.
    y_max = float(chart_frame(results)["Sa (g)"].max()) * 1.05
    for result in newest_first(results):
        show_version(result, y_max, len(results) > 1)

    first = results[0]
    st.download_button(
        "Download design values (CSV)",
        summary_frame(results).to_csv(index=False).encode(),
        file_name=(
            f"frer_design_vals - {first.name} - Site Class {first.site_class}.csv"
        ),
        mime="text/csv",
    )


def site_results(site_id, site_class, versions):
    """Develop the design values for one cached site, straight from the cache."""
    entry = load_entry(PRECOMPUTED_CACHE_DIR, site_id, site_class)
    if entry is None:
        raise ValueError("No precomputed data for that site and class; pick another.")
    return [calc_site(entry, version_id) for version_id in versions]


def render_results(sites, site_id, site_class, records) -> None:
    """Show the selected site's design values underneath the map.

    Which half of the page's data a site comes from is settled here: an added
    site carries its own results, and everything else is read from the
    committed cache at the site class the sidebar is set to.
    """
    record = next((r for r in records if r["id"] == site_id), None)
    if record is not None:
        render_added_results(record)
        return
    render_mapped_results(sites, site_id, site_class)


def render_added_results(record) -> None:
    """Show an added site, exactly as it was pulled when it was added.

    The sidebar's site class does not reach it -- see :func:`site_class_input`
    -- so the caption carries the class and vs30 it was pulled with, which are
    otherwise nowhere on the page.
    """
    st.subheader(str(record["name"]))
    st.caption(
        f"Latitude {record['lat']:.4f}, longitude {record['lon']:.4f} · "
        f"site class {record['site_class']} · "
        f"v$_{{S30}}$ {record['vs30']:.0f} m/s · "
        f"risk category {record['risk_category']} · added this session"
    )
    show_results(record["results"])


def render_mapped_results(sites, site_id, site_class) -> None:
    """Show a precomputed site's design values, straight from the cache.

    Redeveloped on every run rather than memoised: a cached site costs well
    under a second, so there is nothing worth the cache.
    """
    row = sites[sites["id"] == site_id].iloc[0]
    st.subheader(str(row["name"]))
    st.caption(
        f"Latitude {row['lat']:.4f}, longitude {row['lon']:.4f} · "
        f"site class {site_class} · risk category {PRECOMPUTED_RISK_CATEGORY}"
    )
    try:
        show_results(site_results(site_id, site_class, list(VERSIONS)))
    except ValueError as e:
        st.error(str(e))


# --- Page ------------------------------------------------------------------

# The title lives in the sidebar rather than over the page: the map is the
# content, and a heading above it costs the map a chunk of the viewport on
# every screen.
st.sidebar.title("FRE$_R$ Calculator")

mapped = precomputed_sites(PRECOMPUTED_SITE_SET)
add_site_button()

records = added_sites()
sites = map_frame(mapped, records)
site_id = active_site_id(sites)

# The heatmap's field, then the site class the overlay and the mapped pins are
# both read at, then what that pair draws.
field_key = heatmap_field_input()
site_class = site_class_input(mapped)
overlay, contour_lines, samples = heatmap_layers(site_class, field_key)
render_site_map(
    sites,
    site_id,
    [record["id"] for record in records],
    overlay,
    contour_lines,
    samples,
)
# Nothing under the map until a pin is clicked or a site is added; the map's
# own caption is what says so, rather than a placeholder that would only be in
# the way once there are values to read.
if site_id is not None:
    render_results(sites, site_id, site_class, records)

# Provenance, as a footer. Anything above the map costs it a band of the
# viewport on every screen, and this is reference matter rather than something
# to read before starting, so it goes last.
st.caption(
    "Risk-targeted Functional Recovery Earthquake (FRE$_R$) design values, as "
    "documented in the 2026 NEHRP Seismic Provisions. Hazard curves and code "
    "design values come from the USGS "
    "[design maps](https://earthquake.usgs.gov/ws/building-codes/) and "
    "[nshmp hazard](https://earthquake.usgs.gov/ws/nshmp/) web services, which "
    f"cover the conterminous United States only. Calculator v{__version__}."
)
