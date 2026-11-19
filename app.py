"""Streamlit UI for single-site FREr design values.

Runs both stages of the command-line workflow for one location: pulls the USGS
design values and hazard curves, then develops the FREr and DE design spectra.
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
MAP_COLUMNS = ["id", "name", "lat", "lon", "region"]

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

# The two site sources, by their label in the sidebar radio.
CUSTOM_SOURCE = "Custom coordinates"
MAPPED_SOURCE = "Mapped values"

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


def site_source():
    """The Site-section source selector; returns (source, mapped sites).

    Only the selector, not the inputs that depend on it, because which inputs
    to ask for depends on the answer.
    """
    st.sidebar.header("Site")

    sites = precomputed_sites(PRECOMPUTED_SITE_SET)
    if sites is None:
        return CUSTOM_SOURCE, None
    source = st.sidebar.radio(
        "Location",
        [CUSTOM_SOURCE, MAPPED_SOURCE],
        index=0,
    )
    return source, sites


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


def custom_inputs():
    """Inputs for an arbitrary location, fetched live from USGS."""
    lat = st.sidebar.number_input(
        "Latitude",
        min_value=NSHM_BOUNDS[1],
        max_value=NSHM_BOUNDS[3],
        value=34.05,
        step=0.01,
        format="%.4f",
    )
    lon = st.sidebar.number_input(
        "Longitude",
        min_value=NSHM_BOUNDS[0],
        max_value=NSHM_BOUNDS[2],
        value=-118.25,
        step=0.01,
        format="%.4f",
    )

    classes = list(SITE_CLASS_VS30)
    site_class = st.sidebar.selectbox("Site class", classes, index=classes.index("D"))

    # vs30 defaults to the selected class's value; the class is in the widget key
    # so changing class resets it, while a manual override holds until it does.
    vs30 = st.sidebar.number_input(
        "v$_{S30}$ (m/s)",
        min_value=float(VS30_RANGE[0]),
        max_value=float(VS30_RANGE[1]),
        value=float(SITE_CLASS_VS30[site_class]),
        step=5.0,
        key=f"vs30_{site_class}",
        help="Shear-wave velocity used for the hazard-curve request. Defaults "
        "to the USGS value for the selected site class; override it freely.",
    )

    st.sidebar.header("Analysis")
    risk_category = st.sidebar.selectbox(
        "Risk category",
        RISK_CATEGORIES,
        index=RISK_CATEGORIES.index("II"),
        help="Affects the ASCE 7-22 DE values only. FRE_R and the ASCE 7-28 "
        "spectra are computed from the hazard curves and do not use it.",
    )
    choice = st.sidebar.radio("Code version", [*VERSIONS, "Both"], index=0)
    versions = VERSIONS if choice == "Both" else [choice]

    run = st.sidebar.button("Calculate", type="primary", use_container_width=True)

    return {
        "mode": "custom",
        "lat": lat,
        "lon": lon,
        "site_class": site_class,
        "vs30": vs30,
        "risk_category": risk_category,
        "versions": versions,
        "name": f"{lat:.4f}, {lon:.4f}",
        "run": run,
    }


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


def mapped_inputs(sites):
    """Sidebar analysis inputs for the mapped flow; ``(site_class, versions)``.

    The site itself is not among them: it is picked on the map, and these are
    asked for up front so they are already set when it is.

    Both code versions, always. Every mapped site is served from the frozen
    USGS responses, so showing the second edition costs nothing but the
    column -- and comparing the two is the question the mapped side is for,
    which made the selector a step between arriving and the answer.
    """
    classes = shared_site_classes(tuple(int(i) for i in sites["id"]))
    site_class = st.sidebar.selectbox("Site class", classes)
    return site_class, list(VERSIONS)


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


def heatmap_inputs(site_class):
    """Sidebar controls for the overlay; ``(overlay, contours, samples)``.

    The overlay is always drawn: the fill is faint enough and the contour
    lines sparse enough that it informs the map rather than covering it, so
    there is neither a switch to turn it off nor an opacity control to reach
    for. Only the field is chosen here.

    It was behind a toggle, off, while the overlay was an opaque wash that got
    in the way of finding a site. What replaced the wash reads as part of the
    map, and the question it answers -- how values vary around here -- is the
    one the map is opened for, so the switch went too.

    ``(None, ...)`` is left for the cases that are not a choice -- no gridded
    database in the checkout, or a field that fails to render -- and means
    draw no overlay.
    """
    st.sidebar.header("Heatmap")
    table, _ = grid_database()
    if table is None:
        st.sidebar.caption(
            "No gridded database in `data/grid`. Build one with "
            "`python scripts/build_grid.py`."
        )
        return None, None, None

    field_key = st.sidebar.radio(
        "Field",
        list(heatmap.FIELDS),
        format_func=lambda key: heatmap.FIELDS[key].label,
    )
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
    sites, selected_id, overlay=None, contours=None, samples=None
) -> None:
    """Draw the zoomable site map and act on a click.

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
    st.caption(
        f"{len(sites)} mapped sites. Click a marker for its design spectra; "
        "scroll to zoom, drag to pan. Every value comes from the USGS responses "
        "committed with the calculator, so no network call is made."
        + (
            " The heatmap covers the San Francisco Bay Area only — zoom in to read it."
            if overlay is not None
            else ""
        )
    )

    if _apply_map_selection(event.selection, selected_id, samples is not None):
        st.rerun()


def _apply_map_selection(selection, selected_id, has_readout) -> bool:
    """Store what the map reports; True when the page has to rerun.

    The rule itself is :func:`map_view.resolve_selection`, which is a pure
    function so it can be tested; this is the part that touches session
    state.
    """
    on_readout = bool(st.session_state.get(READOUT_PICK_KEY))
    wanted_site, wanted_readout = map_view.resolve_selection(
        selection, selected_id, on_readout, has_readout
    )
    st.session_state[SITE_STATE_KEY] = wanted_site
    st.session_state[READOUT_PICK_KEY] = wanted_readout
    return wanted_site != selected_id


# --- Results ---------------------------------------------------------------


def run_analysis(inputs):
    """Pull and calculate every requested version for custom coordinates.

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

    Versions are drawn as separate charts laid out in columns rather than as one
    faceted chart: Vega-Lite cannot size a faceted spec to its container, so a
    facet renders at a fixed width and is clipped on a narrow screen. ``y_max``
    pins the y axis across every version so the comparison stays honest.
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
                # Two rows of two, so the labels are not truncated in the
                # narrower columns used when comparing both code versions.
                legend=alt.Legend(title=None, orient="top", columns=2),
            ),
            strokeDash=alt.StrokeDash(
                "Series:N",
                sort=labels,
                scale=alt.Scale(
                    domain=labels,
                    range=[SERIES_STYLE[label][1] for label in labels],
                ),
                # Two rows of two, so the labels are not truncated in the
                # narrower columns used when comparing both code versions.
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


def show_results(results):
    """Render metrics, the spectra plot, the period table, and a CSV download."""
    # Where to send someone whose ASCE 7-22 panel has no return periods. When
    # both versions are on the page -- always, on the mapped side -- the ones
    # they want are already below, so pointing at the version selector would
    # be pointing at a control that is not there.
    rp_pointer = (
        "The ASCE 7-28 return periods are below."
        if len(results) > 1
        else "Select ASCE 7-28 for return periods."
    )
    for result in results:
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
            st.caption(
                "Not defined for ASCE 7-22: its design values mix risk-targeted "
                "and deterministic caps that have no single hazard-curve return "
                f"period. {rp_pointer}"
            )

    st.subheader("Design spectra")
    # A single y limit across versions, so the panels can be compared directly.
    y_max = float(chart_frame(results)["Sa (g)"].max()) * 1.05
    for column, result in zip(st.columns(len(results)), results, strict=True):
        with column:
            st.altair_chart(spectra_chart(result, y_max), width="stretch")

    st.subheader("Multi-period spectra")
    st.dataframe(
        spectra_frame(results).style.format("{:.4f}"),
        width="stretch",
        hide_index=True,
    )

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


def render_mapped_results(sites, site_id, site_class, versions):
    """Show the selected site's design values underneath the map.

    Redeveloped on every run rather than memoised: a cached site costs well
    under a second, so there is nothing worth the cache.
    """
    row = sites[sites["id"] == site_id].iloc[0]
    st.subheader(str(row["name"]))
    st.caption(
        f"Latitude {row['lat']:.4f}, longitude {row['lon']:.4f} · "
        f"{row['region']} · risk category {PRECOMPUTED_RISK_CATEGORY}"
    )
    try:
        show_results(site_results(site_id, site_class, versions))
    except ValueError as e:
        st.error(str(e))


# --- Page ------------------------------------------------------------------

# The title lives in the sidebar rather than over the page: on the mapped
# side the map is the content, and a heading above it costs the map a chunk of
# the viewport on every screen.
st.sidebar.title("FRE$_R$ Calculator")

source, sites = site_source()

if source == MAPPED_SOURCE:
    site_id = current_site_id(sites)
    site_class, versions = mapped_inputs(sites)
    overlay, contour_lines, samples = heatmap_inputs(site_class)
    render_site_map(sites, site_id, overlay, contour_lines, samples)
    # Nothing under the map until a pin is clicked; the map's own caption is
    # what says so, rather than a placeholder that would only be in the way
    # once there are values to read.
    if site_id is not None:
        render_mapped_results(sites, site_id, site_class, versions)
else:
    # Custom coordinates keep their Calculate button and their results inline:
    # there is no map to click, and a live USGS pull takes about 15 seconds, so
    # it must not fire on every widget change. The results are left in session
    # state when the map is shown instead, so switching back does not re-pull.
    inputs = custom_inputs()

    if inputs["run"]:
        try:
            st.session_state["results"] = run_analysis(inputs)
        except ValueError as e:
            st.session_state.pop("results", None)
            st.error(str(e))
        except usgs_api.USGSApiError as e:
            st.session_state.pop("results", None)
            st.error(
                f"The USGS web service did not return usable data: {e}\n\n"
                "Some coordinates genuinely fail on a given model. Try a "
                "neighbouring point, a different site class, or the other code "
                "version."
            )

    if "results" in st.session_state:
        show_results(st.session_state["results"])
    else:
        st.info(
            "Enter coordinates and options in the sidebar, then select **Calculate**."
        )

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
