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

## Installation

Requires Python ≥ 3.10.

```bash
python -m venv .venv
.venv/Scripts/pip install -e .        # Windows
# .venv/bin/pip install -e .          # macOS / Linux
```

This installs the `frer` console command and all dependencies
(also listed in [requirements.txt](requirements.txt)).

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
- `Spectra_Plots - loglog/` — DE and FREr spectra plots (for `nehrp_34` and
  `frtc_testbed`).
- `Scatter Plots/` — FREr-vs-DE scatter comparisons by site class and region
  (for `usgs_160`).

Use `--no-plots` to skip figures, `--out-dir` / `--cache-dir` to relocate
outputs and cache.

## Project layout

```
├── sites.csv                  # site table with site-set membership
├── src/frer/
│   ├── cli.py                 # `frer` command-line interface
│   ├── usgs_api.py            # USGS design-maps + nshmp hazard clients
│   ├── pull.py                # rate-limited, resumable data pull
│   ├── cache.py               # JSON-per-site cache
│   ├── rtgm.py                # risk-targeted ground motion integration
│   ├── spectrum.py            # RT spectra + 2-point design values
│   ├── calc.py                # design-value workflow / CSV output
│   ├── plots.py               # spectra and scatter plots
│   ├── sites.py               # sites.csv loading and site-set expansion
│   └── params.py              # code versions and risk-target parameters
├── scripts/
│   └── update_fixtures.py     # regenerate the reference CSVs (deliberate)
├── tests/
│   ├── test_spectrum.py       # unit tests for the spectrum math
│   ├── test_benchmark.py      # benchmark FREr values vs the reference
│   ├── test_pull.py           # pull resilience (retry/backoff) tests
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
(`Sdsfr`, `Sd1fr`, and the `frer_mps_*` spectrum) match the reference within a
2% tolerance. It deliberately does **not** check `Sds`/`Sd1`: for ASCE 7-22
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