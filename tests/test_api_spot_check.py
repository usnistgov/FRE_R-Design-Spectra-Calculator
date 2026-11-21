"""Check the frozen cache against the live USGS hazard service.

The frozen cache under ``tests/usgs_cache`` is no longer pulled from the web
service; it is built offline from the published CONUS.2023.R1 data release
(``scripts/build_cache_from_release.py``). That is faster, reproducible, and
pinned to one model revision -- but it means the rest of the suite never
exercises the hazard call at all, and a change to the URL, the pinned revision,
the coordinate precision or the response parsing would pass every other test.

This is the one test that closes that gap. It re-pulls a sample of (site, site
class) pairs from the live service and asserts the answer matches what the
cache holds. Both sides must agree to within :data:`RTOL`, which is tight
deliberately: these are the same curves from the same revision, so anything
beyond round-off means the two paths have diverged.

It needs the network and the service is slow and rate-limited, so it is opt-in:

    pytest --run-network -k api_spot_check

The sample is random but reproducible (:data:`SEED`), and spans the country and
every site class rather than clustering, because the failure modes worth
catching are regional -- a revision change shows up in the central and eastern
U.S. long before it shows up in California.
"""

import random
from pathlib import Path

import pytest

from frer import usgs_api
from frer.cache import load_entry
from frer.params import DEFAULT_SITE_CLASSES, VERSION_ID, VERSION_NSHM
from frer.single import calc_site, design_value_columns, pull_site
from frer.sites import load_sites

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO_ROOT / "tests" / "usgs_cache"
SITES_CSV = REPO_ROOT / "sites.csv"

# Same tolerance the benchmark uses. The two sides are the same revision of the
# same model, so in practice they agree far more closely than this -- but the
# service rounds its JSON, so an exact comparison would be measuring the
# serialisation rather than the calculation.
RTOL = 0.02

# How many (site, site class) pairs to re-pull. Each is one call to a service
# that wants seconds between requests, so this is a spot check, not a sweep.
SAMPLE_SIZE = 10
SEED = 20261120

# Seconds between calls. The service throttles sustained use by answering 200
# with an error body rather than computing (see frer.usgs_api.PAUSE_RANGE); at
# this sample size a short, polite spacing is enough.
PAUSE = (5.0, 15.0)


def sample_pairs(n=SAMPLE_SIZE, seed=SEED):
    """Pick ``n`` (site row, site class) pairs spread across the country.

    One pair per region-and-class stratum first, so the sample cannot land
    entirely in California or entirely on rock, then the remainder at random.
    """
    sites = load_sites(SITES_CSV)
    rng = random.Random(seed)
    classes = list(DEFAULT_SITE_CLASSES)

    pairs, seen = [], set()

    def take(row, site_class):
        key = (int(row["id"]), site_class)
        if key in seen:
            return False
        seen.add(key)
        pairs.append((row, site_class))
        return True

    # One per region, cycling through site classes so every class appears.
    for i, (_region, group) in enumerate(sorted(sites.groupby("region"))):
        row = group.iloc[rng.randrange(len(group))]
        take(row, classes[i % len(classes)])

    # Any remaining classes, so none is left untested.
    for site_class in classes:
        if len(pairs) >= n:
            break
        if site_class not in {c for _r, c in pairs}:
            take(sites.iloc[rng.randrange(len(sites))], site_class)

    while len(pairs) < n:
        row = sites.iloc[rng.randrange(len(sites))]
        take(row, classes[rng.randrange(len(classes))])

    return pairs[:n]


@pytest.mark.network
def test_cached_curves_match_the_live_service():
    """A sample of cached entries reproduces what the service returns today.

    This is the hazard call's only test against reality. If it fails while the
    rest of the suite passes, suspect the request -- the pinned revision in
    :data:`frer.params.NSHM_EDITION`, the coordinate precision in
    :data:`frer.usgs_api.COORD_PRECISION`, or the URL -- before suspecting the
    arithmetic, which the benchmark covers from the frozen cache.
    """
    session = usgs_api.build_session()
    nshm = VERSION_NSHM[VERSION_ID]
    failures, checked = [], []

    for i, (site, site_class) in enumerate(sample_pairs()):
        cached = load_entry(CACHE_DIR, int(site["id"]), site_class)
        if cached is None:
            failures.append(f"  {site['name']} [{site_class}]: not in the cache")
            continue

        if i:
            usgs_api.pace(PAUSE)
        live = pull_site(
            float(site["lat"]),
            float(site["lon"]),
            site_class,
            DEFAULT_SITE_CLASSES[site_class],
            VERSION_ID,
            name=site["name"],
            session=session,
        )

        want = design_value_columns(calc_site(cached, VERSION_ID))
        got = design_value_columns(calc_site(live, VERSION_ID))
        checked.append(f"{site['name']} [{site_class}]")
        for column, expected in want.items():
            actual = got[column]
            rel = abs(actual - expected) / abs(expected) if expected else abs(actual)
            if rel > RTOL:
                failures.append(
                    f"  {site['name']} [{site_class}] {column}: "
                    f"cache={expected:.6g} live={actual:.6g} ({rel:.2%})"
                )

    # Printed so that a run says what it covered, not just that it passed; the
    # sample is random, and "10 pairs agreed" means little without knowing
    # which. Visible with -s, and in the failure output regardless.
    print(f"\nchecked {len(checked)} pair(s) against the live service:")
    for label in checked:
        print(f"  {label}")

    assert checked, "no (site, class) pair was checked"
    assert not failures, (
        f"{len(failures)} value(s) from {len(checked)} sampled pair(s) differ "
        f"from the live conus-{usgs_api.hazard_edition(nshm)} service by more "
        f"than {RTOL:.0%}:\n" + "\n".join(failures)
    )


@pytest.mark.network
def test_the_request_goes_to_the_pinned_revision():
    """The URL names an explicit revision, not the alias that follows USGS.

    Cheap and offline-ish by comparison with the call above, but it belongs
    here: it is the assertion that would have caught the cache silently moving
    from R1 to R2.
    """
    nshm = VERSION_NSHM[VERSION_ID]
    url = usgs_api.HAZARD_URL.format(
        edition=usgs_api.hazard_edition(nshm), lon=-122.4, lat=37.75, vs30=760
    )
    assert f"conus-{nshm}/" not in url, (
        f"the request uses the bare-year path, which follows whichever "
        f"revision USGS currently serves: {url}"
    )
    assert "conus-2023.R1/" in url, url
