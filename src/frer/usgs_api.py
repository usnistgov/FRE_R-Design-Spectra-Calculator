"""USGS hazard web-service client.

Port of MATLAB ``+usgs/fn_call_USGS_hazard_API.m``, hardened for repeated
calls to a rate-limited service. Requests go through a shared :class:`requests.Session`
whose adapter retries transient failures (HTTP 429/5xx, connection errors,
read timeouts) with exponential backoff + jitter and honors the server's
``Retry-After`` header. Application-level errors (a 200 response whose JSON
body reports failure) are *not* retried. Raw JSON responses are returned so
they can be cached to disk verbatim.
"""

import random
import time

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from . import __version__
from .params import NSHM_EDITION

# Decimal places of latitude and longitude sent to the hazard service.
#
# This was 2 until 2026-11-20, inherited from the original MATLAB URL
# construction, where it was harmless because every site sat on a 0.05 degree
# graticule. It stopped being harmless once the grid went to 0.025 degrees and
# the web app began accepting arbitrary coordinates: two decimals is a 0.005
# degree rounding, up to about 620 m, and the service does not snap to any
# grid, so the answer came back for a different place than the one asked
# about. 745 of the 1,015 nodes in the committed Bay Area grid were requested
# that way.
#
# Four places matches frer.grid.COORD_DECIMALS, which is the precision a node
# is identified by everywhere else in this project, and is about 11 m.
COORD_PRECISION = 4

HAZARD_URL = (
    "https://earthquake.usgs.gov/ws/nshmp/conus-{edition}/dynamic/hazard/"
    "{lon:.{p}f}/{lat:.{p}f}/{vs30:d}".replace("{p}", str(COORD_PRECISION))
)

# "dynamic" is the service that runs a PSHA at the requested coordinates on
# demand, as opposed to the "static" one that serves precomputed curves. It
# does not snap the location to any grid: nshmp-haz builds a site at the exact
# longitude and latitude given and integrates the source model there, so two
# points a hundredth of a degree apart can legitimately differ.
TIMEOUT = 30  # seconds; the nshmp dynamic hazard calc can be slow

USER_AGENT = f"frer-calculator/{__version__} (USGS hazard-curve tool)"

# Transient HTTP statuses worth retrying (rate limiting + gateway/5xx errors).
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})

# Seconds to wait between calls, sampled uniformly. This is not mere
# politeness. The nshmp service throttles sustained use by answering HTTP 200
# with ``status: "error"`` in a fraction of a second rather than computing, and
# a 200 never reaches the retry layer above -- so spacing the calls is the only
# thing standing between a long pull and a run of false failures. A pull at one
# second per call was refused after roughly 320 calls; the original MATLAB tool
# used 10-60 seconds and did not hit this.
PAUSE_RANGE = (10.0, 60.0)

# A 200 whose body reports failure can mean either of two things: this point
# genuinely cannot be computed, or the service is declining to compute right
# now. The first repeats deterministically and the second clears, so a couple
# of widely spaced retries tell them apart without hammering anything.
ERROR_RETRIES = 2
ERROR_BACKOFF = 60.0


def pace(pause=PAUSE_RANGE) -> float:
    """Sleep between API calls; returns the delay used, in seconds.

    ``pause`` is either a number of seconds or a (low, high) pair sampled
    uniformly. Randomising avoids a lockstep request rhythm, which is both
    easier for a rate limiter to spot and more likely to beat against one.
    """
    low, high = pause if isinstance(pause, (tuple, list)) else (pause, pause)
    delay = random.uniform(low, high) if high > low else float(low)
    if delay > 0:
        time.sleep(delay)
    return delay


_DEFAULT_SESSION: requests.Session | None = None


class USGSApiError(Exception):
    """Raised when a USGS web-service call fails (after any retries)."""


def build_session(
    retries: int = 5,
    backoff_factor: float = 1.0,
    backoff_max: float = 60.0,
    user_agent: str = USER_AGENT,
) -> requests.Session:
    """Build a session that retries transient failures with backoff.

    Parameters
    ----------
    retries : maximum retry attempts per request for transient failures.
    backoff_factor : base for exponential backoff; the nth retry waits about
        ``backoff_factor * 2**(n-1)`` seconds (plus jitter), capped at
        ``backoff_max``. A ``Retry-After`` response header overrides this.
    backoff_max : ceiling on a single backoff wait, in seconds.
    user_agent : ``User-Agent`` header identifying this client.
    """
    retry = Retry(
        total=retries,
        connect=retries,
        read=retries,
        status=retries,
        status_forcelist=RETRYABLE_STATUS,
        allowed_methods=frozenset({"GET"}),
        backoff_factor=backoff_factor,
        backoff_jitter=backoff_factor,  # +/- up to backoff_factor seconds
        backoff_max=backoff_max,
        respect_retry_after_header=True,
        raise_on_status=False,  # inspect the final response ourselves
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": user_agent})
    return session


def _session(session: requests.Session | None) -> requests.Session:
    """Return the given session, or a lazily-created shared default."""
    global _DEFAULT_SESSION
    if session is not None:
        return session
    if _DEFAULT_SESSION is None:
        _DEFAULT_SESSION = build_session()
    return _DEFAULT_SESSION


def _get(session, url, timeout, **kwargs):
    """GET ``url``, wrapping transport failures as :class:`USGSApiError`."""
    try:
        return _session(session).get(url, timeout=timeout, **kwargs)
    except requests.RequestException as e:
        # Raised when retries are exhausted (timeouts, connection resets, ...).
        raise USGSApiError(f"request failed after retries: {e}") from e


def hazard_edition(edition) -> str:
    """Resolve an NSHM year to the pinned model revision; pass a string through.

    Callers name the model by its release year, because that is what a code
    version maps to (see :data:`frer.params.VERSION_NSHM`) and what the cache is
    keyed on. The service wants a revision, and the bare year is an alias that
    moves (see :data:`frer.params.NSHM_EDITION`), so the translation happens
    here -- once, on the way out -- rather than at each call site, where a new
    caller would otherwise have to remember to do it.

    A string is returned unchanged, so a caller that genuinely wants a specific
    revision, such as a comparison against a published data release, can ask
    for it by name.
    """
    if isinstance(edition, str):
        return edition
    try:
        return NSHM_EDITION[int(edition)]
    except (KeyError, TypeError, ValueError):
        raise ValueError(
            f"no pinned hazard model revision for NSHM {edition!r}; "
            f"known years are {sorted(NSHM_EDITION)}"
        ) from None


def call_hazard_api(
    edition,
    lat: float,
    lon: float,
    vs30: float,
    timeout: float = TIMEOUT,
    session: requests.Session | None = None,
    error_retries: int = ERROR_RETRIES,
    error_backoff: float = ERROR_BACKOFF,
) -> dict:
    """Call the USGS nshmp dynamic hazard web service; return raw response dict.

    ``edition`` is the NSHM release year (2018 or 2023), which
    :func:`hazard_edition` resolves to the pinned model revision that is
    actually requested -- an explicit revision string is also accepted and used
    as given. Latitude/longitude are formatted to 2 decimal places and vs30 to
    an integer, matching the original MATLAB URL construction. Hazard curves
    live under ``response.hazardCurves``.
    """
    url = HAZARD_URL.format(
        edition=hazard_edition(edition), lon=lon, lat=lat, vs30=int(vs30)
    )
    for attempt in range(error_retries + 1):
        resp = _get(session, url, timeout)
        if resp.status_code == 429:
            raise USGSApiError("Too Many Requests")
        if resp.status_code != 200:
            raise USGSApiError(f"hazard call failed: HTTP {resp.status_code}")
        data = resp.json()
        if data.get("status") == "success":
            break
        # A 200 reporting failure is either a point that cannot be computed or
        # the service declining to compute right now; the retry layer above
        # never sees it, because it only watches status codes. Waiting tells
        # the two apart (see ERROR_RETRIES).
        if attempt == error_retries:
            raise USGSApiError(f"hazard call status: {data.get('status')}")
        time.sleep(error_backoff * (attempt + 1))
    return data
