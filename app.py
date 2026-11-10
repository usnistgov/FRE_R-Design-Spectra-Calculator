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
from frer import __version__, usgs_api
from frer.cache import cache_path, load_entry
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

# Session-state key for the mapped site. Plain state rather than a widget key:
# the map is the only thing that selects a site, so nothing else writes it and
# there is no widget whose creation order has to be worked around.
SITE_STATE_KEY = "mapped_site_id"

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
    """Sidebar analysis inputs for the mapped flow.

    The site itself is not among them: it is picked on the map, and these are
    asked for up front so they are already set when it is.
    """
    classes = shared_site_classes(tuple(int(i) for i in sites["id"]))
    site_class = st.sidebar.selectbox("Site class", classes)

    st.sidebar.header("Analysis")
    choice = st.sidebar.radio("Code version", [*VERSIONS, "Both"], index=0)
    return site_class, VERSIONS if choice == "Both" else [choice]


def render_site_map(sites, selected_id) -> None:
    """Draw the zoomable site map and act on a click.

    A click records the site and reruns, which moves the highlight ring onto it
    and redevelops the values shown underneath.

    Clicking the map away from any pin clears deck.gl's selection, which comes
    back here as None and deselects the site, emptying the page below the map.
    So the stored site simply tracks whatever the map reports.

    The chart keeps one stable key so panning and zooming survive that rerun.
    Re-keying it per site is the obvious way to drop a selection deck.gl is
    still holding, but it would also throw the viewer back out to the whole
    country every time they picked a site, which is the opposite of the point.
    Comparing against the current site is what stops the rerun repeating:
    deck.gl keeps reporting the same selection until it changes.
    """
    event = st.pydeck_chart(
        map_view.site_map(sites[MAP_COLUMNS], selected_id),
        height=map_view.MAP_HEIGHT,
        width="stretch",
        key="site_map",
        on_select="rerun",
        selection_mode="single-object",
    )
    st.caption(
        f"{len(sites)} mapped sites. Click a marker for its design spectra; "
        "scroll to zoom, drag to pan. Every value comes from the USGS responses "
        "committed with the calculator, so no network call is made."
    )

    clicked = map_view.clicked_site_id(event.selection)
    if clicked != selected_id:
        st.session_state[SITE_STATE_KEY] = clicked
        st.rerun()


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
                "period. Select ASCE 7-28 for return periods."
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
    render_site_map(sites, site_id)
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
