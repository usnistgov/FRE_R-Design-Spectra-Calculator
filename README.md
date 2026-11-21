# FREr Calculator

Risk-targeted **Functional Recovery Earthquake (FRE<sub>R</sub>)** design value calculator.

This tool develops risk-targeted multi-period and 2-point design spectra for
the Functional Recovery Earthquake, as documented in the 2026 NEHRP Seismic
Provisions, from USGS [nshmp hazard](https://earthquake.usgs.gov/ws/nshmp/)
curves.

Everything it reports comes from one source: the hazard curves. There is no
second service and no deterministic cap, so no result mixes two bases.

## Background

The FRE<sub>R</sub> is a risk-targeted ground motion intended for functional-recovery
design. For each site and spectral period, the tool integrates a lognormal
fragility curve with the site hazard curve to find the ground-motion anchor
that produces a target lifetime risk of exceedance (the RTGM approach):

| Design Value | β | PoE at anchor | Risk target | Timespan | Direction |
|---|---|---|---|---|---|
| **FRE<sub>R</sub> (NERHP 2026)** | 0.6 | 25% | 10% | 50 yr | RotD50 (no conversion) |
| **MCE<sub>R</sub> (ASCE 7-28)** | 0.6 | 10% | 1.5% | 50 yr | RotD100 (max-direction factors) |

**ASCE 7-28** is the one code version supported: both FREr and DE spectra are
computed fully probabilistically from **NSHM 2023** hazard curves
(DE = 2/3 × the MCE<sub>R</sub> spectrum).

ASCE 7-22 was supported until 2026-11-20 and has been removed. It worked
differently — its DE values came from the USGS design-maps service, which
mixes risk-targeted and deterministic caps, while only its FREr values were
developed here — so a 7-22 result had two halves on different bases and no
single hazard-curve return period.

### Model revision

The hazard calls are pinned to **CONUS.2023.R1** (`conus-2023.R1`), not to the
service's bare-year path. That path is an alias which follows whatever
revision USGS currently serves, and it has already moved from R1 to R2: a
cache pulled through it in July 2026 silently held R2 curves.

R1 is the revision the committee adopted, so it is the basis of an official
design value. R2 is a later science update — a better estimate of hazard, but
not the adopted basis. See `frer.params.NSHM_EDITION`.

From each multi-period spectrum, 2-point design values are derived per
ASCE 7 §21.4:

- S<sub>DS</sub> = 0.9 × max Sa over 0.2 s ≤ T ≤ 5 s
- S<sub>D1</sub> = max( Sa(1.0 s), 0.9 × max(T·Sa) over 1 s ≤ T ≤ 2 s )

### Orientation: RotD50 vs RotD100

Per the table above, **FRE<sub>R</sub> and DE ordinates are not on the same
basis.** FRE<sub>R</sub> keeps the RotD50 orientation of the underlying hazard
curves; the DE/MCE<sub>R</sub> ordinates are max direction (RotD100), via the
max-direction factors in `frer.spectrum`.

Reading one straight against the other understates FRE<sub>R</sub> by the
max-direction factor (1.2 at short periods, rising to 1.3 at 10 s). Convert one
to the other's orientation before comparing. Every spectral value the tool
reports — CSV column, web-app metric, table header, plot label — carries its
orientation for this reason.

**Return periods are the exception, and deliberately so.** All six
(FRE<sub>R</sub>, MCE<sub>R</sub>, DE at 0.2 s and 1.0 s) are read off the
RotD50 hazard curve at the unscaled RTGM, so they already share one orientation
and are directly comparable as printed — hence no suffix on those columns.

A consequence worth knowing: because the return periods are consistent while the
printed ordinates are not, a site can show S<sub>DSfr</sub> below S<sub>DS</sub>
and *still* have a longer FRE<sub>R</sub> return period at 0.2 s. That is not a
bug — it means FRE<sub>R</sub> genuinely exceeds DE at 0.2 s once both are put in
the same orientation, and it happens wherever the 2-point S<sub>DS</sub> (driven
by the spectral peak over 0.2–5 s) and the 0.2 s ordinate disagree.




## Installation

Requires Python ≥ 3.10.

```bash
python -m venv .venv
.venv/Scripts/pip install -e .        # Windows
# .venv/bin/pip install -e .          # macOS / Linux
```

This installs the `frer` console command and all dependencies. Add `streamlit`
for the web app, or install [requirements.txt](requirements.txt), which pins the
app's full runtime (the library dependencies, Streamlit, and the package
itself).

## Usage

The workflow has two stages: **pull** (download USGS data into a local JSON
cache) and **calc** (develop design values from the cache).

### 1. Pull USGS data

```bash
frer pull
```

- Sites come from [sites.csv](sites.csv): 160 USGS NSHM test sites, all of
  them. There used to be three overlapping "site sets"; `nehrp_34` was the
  same 34 rows as a subset of these, and `frtc_testbed` was 9 street
  addresses, so both are gone.
- Six site classes are pulled per site: BC (760 m/s), C (530), CD (365),
  D (260), DE (185), E (150).
- One call is made per (site, class): the NSHM 2023 hazard curves, from the
  pinned `conus-2023.R1` path.
- **You probably do not need this.** The frozen cache the tests and the web
  app read is built offline from the published data release instead — see
  [Rebuilding the frozen cache](#rebuilding-the-frozen-cache). `frer pull` is
  for points the release does not cover, such as grid nodes.
- Results are cached one JSON file per (site, class) in `results/usgs_cache/`.
  The pull is **resumable**: failed calls are skipped and retried on the next
  run; already-cached data is never re-fetched.
- **Rate limiting is handled automatically.** Each request goes through a
  session that retries transient failures (HTTP 429/5xx, connection errors,
  read timeouts) with exponential backoff + jitter, honoring the server's
  `Retry-After` header. Tune with `--retries` (default 5) and `--backoff`
  (default 1.0 s). A 200 response that reports a server-side calc error is
  *not* retried (some coordinates genuinely fail).
- `--pause` (default 10 s) adds polite spacing between calls, and it is not
  mere politeness: the service throttles sustained use by answering HTTP 200
  with an error body in a fraction of a second, which the retry layer never
  sees because it only watches status codes. Slow high-seismicity sites may
  need a larger `--timeout` (default 30 s). A full pull of all 160 sites at
  six classes is 960 calls — plan on hours.
- Coordinates are sent at four decimal places (about 11 m). This was two
  places until 2026-11-20, which rounded a request by up to 620 m; the service
  computes at the exact point given and snaps to no grid, so that mattered.

### 2. Calculate design values

```bash
frer calc
```

Outputs are written under `results/usgs_160/ASCE 7-28/`:

- `frer_design_vals.csv` — one row per (site, class) with S<sub>DS</sub>,
  S<sub>D1</sub>, S<sub>DSfr</sub>, S<sub>D1fr</sub> and the full FREr
  multi-period spectrum (`frer_mps_*` columns).
  Every spectral column carries the ground-motion orientation its value is
  reported in, as a `_RotD50` / `_RotD100` suffix (`Sds_RotD100`,
  `Sdsfr_RotD50`, `frer_mps_0p20_RotD50`, ...) — see
  [Orientation](#orientation-rotd50-vs-rotd100) below. Return-period columns
  (`frer_rp_0p2`, `mcer_rp_0p2`, `de_rp_0p2`, ...) are deliberately unsuffixed:
  all six are read off the RotD50 hazard curve and share one orientation.
- `Scatter Plots/` — FREr-vs-DE scatter comparisons by site class and region.

Use `--no-plots` to skip figures, `--out-dir` / `--cache-dir` to relocate
outputs and cache.

## Web app

For a single site, the calculator runs as a [Streamlit](https://streamlit.io)
app. [app.py](app.py) performs both stages (pull, then calc) for one location
and displays the FREr and DE design values, return periods, an interactive
spectra chart and the multi-period ordinates, then a CSV to download.

The page is a map, and every site is a pin on it:

- **The 160 `usgs_160` benchmark sites** open as blue pins on a pannable,
  zoomable map of the conterminous U.S. Click a pin and that site's design
  values appear below the map; the site class they are read at is chosen in the
  sidebar. The page opens with nothing selected, so there is nothing below the
  map until a pin is clicked; the selected pin turns red and grows, and clicking
  the map away from any pin clears the selection and empties the page again.
  There is no Calculate button: the values are answered instantly and offline
  from the frozen USGS responses committed under `tests/usgs_cache/`. Zoom in to
  separate the sites that sit on top of one another at national zoom (the Los
  Angeles basin, the Bay Area).
- **Sites you add by coordinate**, from the sidebar's **Add a site by
  coordinate** button, which opens a dialog over the map: any
  latitude/longitude in the conterminous U.S., with its own site class,
  v<sub>S30</sub> and risk category, fetched live from USGS (about 15 seconds
  per code version). The dialog closes when the pull finishes; the site joins
  the map as a green pin and its values open underneath it. Add as many as you
  like — the sidebar lists them — and they last for the browser session, held
  only in that session's memory and gone on reload. The site class under
  **Heatmap** governs the mapped pins and the overlay, not an added site: an
  added site keeps what it was pulled with, which its caption states.

  The map is the only way to pick a site. deck.gl draws its markers into a
  canvas, so they cannot be tabbed to or read by a screen reader; anyone who
  needs a keyboard route can reach any location through the add-a-site dialog,
  whose latitude and longitude are ordinary number inputs, and whose values
  open without a click on the map.

  The map is deck.gl, via the `pydeck` that ships with Streamlit, over a CARTO
  tile basemap (`TILE_BASEMAP` in [map_view.py](map_view.py)). Those tiles are
  the one part of the page that needs a network connection, and there is no
  offline alternative: Streamlit draws a CARTO basemap under every
  `st.pydeck_chart` whatever the deck asks for, reading `map_style=None` as
  "pick one from the theme" rather than "draw none". Without a connection the
  pins are left on an empty background. The mapped sites' design values are
  answered from the committed responses either way; adding a site is the part
  that genuinely needs the network, for its live USGS pull.

Streamlit is a Python web framework, so the app always runs on a Python host —
either your own machine or a server. To run it locally, follow
[Installation](#installation) below (including `streamlit`), then:

```bash
streamlit run app.py
```

**Conterminous U.S. only.** The NSHM models cover latitude 24.4–50.0 and
longitude −125.0 to −65.0. Coordinates outside that box are rejected before
any request is made, because the hazard service answers them with a success
status and unusable curves.

## Project layout

```
├── app.py                     # Streamlit single-site web app
├── map_view.py                # zoomable deck.gl site map for the web app
├── sites.csv                  # the 160 USGS NSHM test sites
├── conftest.py                # puts the repo root on sys.path for pytest
├── src/frer/
│   ├── cli.py                 # `frer` command-line interface
│   ├── usgs_api.py            # nshmp hazard client (pinned to one revision)
│   ├── pull.py                # rate-limited, resumable data pull
│   ├── cache.py               # JSON-per-site cache
│   ├── rtgm.py                # risk-targeted ground motion integration
│   ├── spectrum.py            # RT spectra + 2-point design values
│   ├── single.py              # single-site pull/calc (used by calc.py and app.py)
│   ├── calc.py                # design-value workflow / CSV output
│   ├── plots.py               # spectra and scatter plots
│   ├── sites.py               # sites.csv loading and site-class expansion
│   ├── local_hazard.py        # read a published data release as API responses
│   ├── params.py              # code version, model revision, risk targets
│   └── grid.py                # gridded design values over a region
├── scripts/
│   ├── build_cache_from_release.py  # build the frozen cache (deliberate)
│   ├── update_fixtures.py     # regenerate the reference CSV (deliberate)
│   ├── build_grid.py          # build the gridded database (deliberate)
│   └── validate_grid.py       # check the gridded database
├── tests/
│   ├── test_spectrum.py       # unit tests for the spectrum math
│   ├── test_benchmark.py      # benchmark FREr values vs the reference
│   ├── test_single.py         # single-site API vs the reference + input guards
│   ├── test_pull.py           # pull resilience (retry/backoff) tests
│   ├── test_api_spot_check.py # the frozen cache vs the live service (opt-in)
│   ├── test_map_view.py       # site map layers + click payload contracts
│   ├── test_grid.py           # grid geometry, resume, golden nodes
│   ├── test_validate_grid.py  # the validator, and every way it should fail
│   ├── conftest.py            # `network` marker / --run-network option
│   ├── usgs_cache/            # frozen hazard curves (benchmark input)
│   └── usgs_160/ASCE 7-28/    # reference frer_design_vals.csv
└── results/                   # generated outputs + live usgs_cache/ (git-ignored)
```

## Testing & validation

The benchmark suite lives in `tests/`. Reference design values are stored in
`tests/usgs_160/ASCE 7-28/`, and the hazard curves that produce them are
committed in `tests/usgs_cache/` — so the benchmark runs offline and never
calls the network. Run everything with:

```bash
pytest
```

`test_spectrum.py` covers the deterministic spectrum math. `test_benchmark.py`
recomputes all 960 rows from the frozen cache and asserts the **FREr values**
(`Sdsfr_RotD50`, `Sd1fr_RotD50`, and the 22 `frer_mps_*_RotD50` ordinates)
match the reference within a 2% tolerance. It also checks the return periods
against `frer_return_periods.csv`, which was written by the original MATLAB
tool and is never regenerated — the only genuinely independent check here.

`test_api_spot_check.py` is the one test that exercises the hazard call
itself. The frozen cache is built from a data release rather than pulled, so
nothing else would notice if the URL, the pinned revision or the coordinate
precision broke. It re-pulls ten (site, class) pairs — spread across every
region and every site class — from the live service and compares. It needs the
network, so it is opt-in:

```bash
pytest --run-network -k api_spot_check
```

`test_pull.py` checks the **pull path's resilience** (not the returned values):
against a local stub HTTP server it verifies that rate-limit/5xx/timeout
failures are retried and recovered, that server-side calc errors are surfaced
without retry-hammering, and that `pull_sites` records failures and keeps going.
These run offline. One opt-in smoke test makes a single real USGS call; enable
it with `pytest --run-network`.

### Updating the reference fixtures

The reference CSV is the source of truth and changes only when you intend it
to. When that time comes — after rebuilding the frozen cache, say — regenerate
it from that cache:

```bash
python scripts/update_fixtures.py
```

It recomputes `frer_design_vals.csv` in place under
`tests/usgs_160/ASCE 7-28/` and prints what changed. It never pulls from USGS
and never touches `tests/usgs_cache/`. Review with `git diff -- tests/` and
commit deliberately.

### Rebuilding the frozen cache

`tests/usgs_cache/` is built from the USGS
[CONUS.2023.R1 data release](https://www.sciencebase.gov/catalog/), not pulled
from the web service. All 160 sites are NSHM test sites, so the release covers
every one of them, at every site class:

```bash
python scripts/build_cache_from_release.py --out tests/usgs_cache --adopt
```

Seconds rather than hours, byte-for-byte reproducible, pinned to one model
revision by construction, and about 11 MB rather than 344 MB — the release
carries the total curve where the service response also carries a
per-magnitude breakdown nothing here reads. It needs the release unpacked
locally; point `--release-dir` at it.

## Development

Install the dev tools (linter, formatter, spell checker, pytest):

```bash
.venv/Scripts/pip install -e ".[dev]"    # Windows
# .venv/bin/pip install -e ".[dev]"      # macOS / Linux
```

The full check suite:

```bash
ruff check .        # lint
ruff format --check .   # formatting (drop --check to auto-format)
codespell           # spell-check source and docs
pytest              # unit tests + benchmark
```

All four are configured in [pyproject.toml](pyproject.toml): `ruff` uses an
88-column line length with pycodestyle/Pyflakes/isort/pyupgrade/bugbear rules;
`codespell` skips data artifacts and the cache and whitelists domain terms
(e.g. `afe`, `nshm`).