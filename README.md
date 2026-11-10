# FREr Calculator

Risk-targeted **Functional Recovery Earthquake (FRE<sub>R</sub>)** design value calculator.

This tool pulls seismic hazard curves and code design values from USGS web
services ([design maps](https://earthquake.usgs.gov/ws/designmaps/),
[nshmp hazard](https://earthquake.usgs.gov/ws/nshmp/)), then develops
risk-targeted multi-period and 2-point design spectra for the Functional
Recovery Earthquake as documented in the 2026 NEHRP Seismic Provisions.

## Background

The FRE<sub>R</sub> is a risk-targeted ground motion intended for functional-recovery
design. For each site and spectral period, the tool integrates a lognormal
fragility curve with the site hazard curve to find the ground-motion anchor
that produces a target lifetime risk of exceedance (the RTGM approach):

| Design Value | β | PoE at anchor | Risk target | Timespan | Direction |
|---|---|---|---|---|---|
| **FRE<sub>R</sub> (NERHP 2026)** | 0.6 | 25% | 10% | 50 yr | RotD50 (no conversion) |
| **MCE<sub>R</sub> (ASCE 7-28)** | 0.6 | 10% | 1.5% | 50 yr | RotD100 (max-direction factors) |

Two code versions are supported:

- **ASCE 7-22** — FREr spectra are computed from **NSHM 2018** hazard curves;
  DE values (S<sub>DS</sub>, S<sub>D1</sub>, multi-period design spectrum) are
  taken directly from the USGS ASCE 7-22 design-maps service.
- **ASCE 7-28** — both FREr and DE spectra are computed fully probabilistically
  from **NSHM 2023** hazard curves (DE = 2/3 × the MCE<sub>R</sub> spectrum).

From each multi-period spectrum, 2-point design values are derived per
ASCE 7-22 §21.4:

- S<sub>DS</sub> = 0.9 × max Sa over 0.2 s ≤ T ≤ 5 s
- S<sub>D1</sub> = max( Sa(1.0 s), 0.9 × max(T·Sa) over 1 s ≤ T ≤ 2 s )

### Orientation: RotD50 vs RotD100

Per the table above, **FRE<sub>R</sub> and DE ordinates are not on the same
basis.** FRE<sub>R</sub> keeps the RotD50 orientation of the underlying hazard
curves; the DE/MCE<sub>R</sub> ordinates are max direction (RotD100) — for
ASCE 7-28 via the max-direction factors in `frer.spectrum`, and for ASCE 7-22
because the USGS design-value service reports max-direction values.

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
frer pull --site-set frtc_testbed
```

- Sites come from [sites.csv](sites.csv). Each row belongs to one or more
  "site sets" (`usgs_160`, `nehrp_34`, `frtc_testbed`) via the `*_id`
  columns. Omit `--site-set` to pull every site.
- Sites with an explicit `site_class`/`vs30` use them; otherwise the four
  default classes are pulled: BC (760 m/s), C (530), D (265), E (150).
- Three calls are made per (site, class): ASCE 7-22 design values, NSHM 2018
  hazard curves, and NSHM 2023 hazard curves (select with `--editions`).
- Results are cached one JSON file per (site, class) in `results/usgs_cache/`.
  The pull is **resumable**: failed calls are skipped and retried on the next
  run; already-cached data is never re-fetched.
- **Rate limiting is handled automatically.** Each request goes through a
  session that retries transient failures (HTTP 429/5xx, connection errors,
  read timeouts) with exponential backoff + jitter, honoring the server's
  `Retry-After` header. Tune with `--retries` (default 5) and `--backoff`
  (default 1.0 s). A 200 response that reports a server-side calc error is
  *not* retried (some coordinates genuinely fail — e.g. Reno NV class D on
  NSHM 2018).
- `--pause` (default 1 s) adds polite spacing between calls; raise it to be
  gentler, but backoff already absorbs bursts. Slow high-seismicity sites may
  need a larger `--timeout` (default 30 s). A full pull of all 169 sites still
  takes a while — plan accordingly.

### 2. Calculate design values

```bash
frer calc --site-set frtc_testbed --version asce7-22
frer calc --site-set frtc_testbed --version asce7-28
```

Outputs are written under `results/<site_set>/<version>/`:

- `frer_design_vals.csv` — one row per (site, class) with S<sub>DS</sub>,
  S<sub>D1</sub>, S<sub>DSfr</sub>, S<sub>D1fr</sub>; the `frtc_testbed` set
  also includes the full FREr multi-period spectrum (`frer_mps_*` columns).
  Every spectral column carries the ground-motion orientation its value is
  reported in, as a `_RotD50` / `_RotD100` suffix (`Sds_RotD100`,
  `Sdsfr_RotD50`, `frer_mps_0p20_RotD50`, ...) — see
  [Orientation](#orientation-rotd50-vs-rotd100) below. Return-period columns
  (`frer_rp_0p2`, `mcer_rp_0p2`, `de_rp_0p2`, ...) are deliberately unsuffixed:
  all six are read off the RotD50 hazard curve and share one orientation.
- `Spectra_Plots - loglog/` — DE and FREr spectra plots (for `nehrp_34` and
  `frtc_testbed`).
- `Scatter Plots/` — FREr-vs-DE scatter comparisons by site class and region
  (for `usgs_160`).

Use `--no-plots` to skip figures, `--out-dir` / `--cache-dir` to relocate
outputs and cache.

## Web app

For a single site, the calculator runs as a [Streamlit](https://streamlit.io)
app. [app.py](app.py) performs both stages (pull, then calc) for one location
and displays the FREr and DE design values, an interactive spectra chart, the
multi-period ordinates, and a downloadable CSV. Every option is exposed: code
version (ASCE 7-22, 7-28, or both side by side), site class, v<sub>S30</sub>,
and risk category.

The site is chosen one of two ways, picked in the sidebar:

- **Custom coordinates** — any latitude/longitude in the conterminous U.S.,
  fetched live from USGS (see the timing note below).
- **Mapped values** — the 160 `usgs_160` benchmark sites, dropped as pins on a
  pannable, zoomable map of the conterminous U.S. Click a pin and that site's
  design values appear below the map; site class and code version are chosen in
  the sidebar. The page opens with nothing selected, so there is nothing below
  the map until a pin is clicked; the selected pin turns red and grows, and
  clicking the map away from any pin clears the selection and empties the page
  again. There is no Calculate button: the values are answered instantly and offline
  from the frozen USGS responses committed under `tests/usgs_cache/`. Zoom in to
  separate the sites that sit on top of one another at national zoom (the Los
  Angeles basin, the Bay Area).

  The map is the only way to pick a mapped site. deck.gl draws its markers into
  a canvas, so they cannot be tabbed to or read by a screen reader; anyone who
  needs a keyboard route should use **Custom coordinates**, which reaches the
  same locations through ordinary latitude and longitude inputs.

  The map is deck.gl, via the `pydeck` that ships with Streamlit, over a CARTO
  tile basemap (`TILE_BASEMAP` in [map_view.py](map_view.py)). Those tiles are
  the one part of the page that needs a network connection, and there is no
  offline alternative: Streamlit draws a CARTO basemap under every
  `st.pydeck_chart` whatever the deck asks for, reading `map_style=None` as
  "pick one from the theme" rather than "draw none". Without a connection the
  pins are left on an empty background. The design values behind them are
  answered from the committed responses either way, and **Custom coordinates**
  is the part that genuinely needs the network, for its live USGS pull.

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
├── sites.csv                  # site table with site-set membership
├── conftest.py                # puts the repo root on sys.path for pytest
├── src/frer/
│   ├── cli.py                 # `frer` command-line interface
│   ├── usgs_api.py            # USGS design-maps + nshmp hazard clients
│   ├── pull.py                # rate-limited, resumable data pull
│   ├── cache.py               # JSON-per-site cache
│   ├── rtgm.py                # risk-targeted ground motion integration
│   ├── spectrum.py            # RT spectra + 2-point design values
│   ├── single.py              # single-site pull/calc (used by calc.py and app.py)
│   ├── calc.py                # design-value workflow / CSV output
│   ├── plots.py               # spectra and scatter plots
│   ├── sites.py               # sites.csv loading and site-set expansion
│   └── params.py              # code versions and risk-target parameters
├── scripts/
│   └── update_fixtures.py     # regenerate the reference CSVs (deliberate)
├── tests/
│   ├── test_spectrum.py       # unit tests for the spectrum math
│   ├── test_benchmark.py      # benchmark FREr values vs the reference
│   ├── test_single.py         # single-site API vs the reference + input guards
│   ├── test_pull.py           # pull resilience (retry/backoff) tests
│   ├── test_map_view.py       # site map layers + click payload contracts
│   ├── conftest.py            # `network` marker / --run-network option
│   ├── usgs_cache/            # frozen USGS responses (benchmark input)
│   └── <site_set>/<version>/  # reference frer_design_vals.csv + plots
└── results/                   # generated outputs + live usgs_cache/ (git-ignored)
```

## Testing & validation

The benchmark suite lives in `tests/`. Reference design values are stored under
`tests/<site_set>/<version>/`, and a frozen snapshot of the USGS responses that
produce them is committed in `tests/usgs_cache/` — so the benchmark runs offline
and never calls the network. Run everything with:

```bash
pytest
```

`test_spectrum.py` covers the deterministic spectrum math. `test_benchmark.py`
recomputes each site set from the frozen cache and asserts the **FREr values**
(`Sdsfr_RotD50`, `Sd1fr_RotD50`, and the `frer_mps_*_RotD50` spectrum) match the
reference within a 2% tolerance. The committed fixtures predate the orientation
suffixes and hold the MATLAB reference values, so they are read through
`rename_legacy_spectral_columns` rather than regenerated — regenerating them
from Python would destroy what the benchmark compares against. It deliberately
does **not** check `Sds`/`Sd1`: for ASCE 7-22
those are returned by the USGS design-maps service, which is the source of truth
for its own output, so drift there is not a calculator regression.

`test_pull.py` checks the **pull path's resilience** (not the returned values):
against a local stub HTTP server it verifies that rate-limit/5xx/timeout
failures are retried and recovered, that server-side calc errors are surfaced
without retry-hammering, and that `pull_sites` records failures and keeps going.
These run offline. One opt-in smoke test makes a single real USGS call; enable
it with `pytest --run-network`.

### Updating the reference fixtures

The reference CSVs are the source of truth and change only when you intend them
to. When that time comes (e.g. after an intentional cache refresh, or to adopt
current Python output as the baseline), regenerate them from the frozen cache:

```bash
python scripts/update_fixtures.py            # all cached site sets
python scripts/update_fixtures.py --site-set nehrp_34
```

It recomputes `frer_design_vals.csv` in place under `tests/<site_set>/<version>/`
and prints what changed. It never pulls from USGS and never touches
`tests/usgs_cache/` (refreshing that cache is a separate, manual `frer pull`).
Review with `git diff -- tests/` and commit deliberately.

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