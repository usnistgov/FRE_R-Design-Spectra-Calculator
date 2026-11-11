"""USGS web-service clients.

Ports of MATLAB ``+usgs/fn_call_USGS_design_value_API.m`` and
``+usgs/fn_call_USGS_hazard_API.m``, hardened for repeated calls to a
rate-limited service. Requests go through a shared :class:`requests.Session`
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

# The older /ws/designmaps/{doc}.json path still works but 301s here (via a
# plain-http hop), so call the current endpoint directly. The response schema is
# unchanged: the design values live under response.data.
DESIGN_MAPS_URL = (
    "https://earthquake.usgs.gov/ws/building-codes/{reference_doc}/calculate"
)
HAZARD_URL = (
    "https://earthquake.usgs.gov/ws/nshmp/conus-{edition}/dynamic/hazard/"
    "{lon:.2f}/{lat:.2f}/{vs30:d}"
)
TIMEOUT = 30  # seconds; the nshmp dynamic hazard calc can be slow

USER_AGENT = f"frer-calculator/{__version__} (USGS design-values tool)"

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


def call_design_value_api(
    site_id,
    lat: float,
    lon: float,
    risk_category: str = "II",
    site_class: str = "D",
    reference_doc: str = "asce7-22",
    timeout: float = TIMEOUT,
    session: requests.Session | None = None,
) -> dict:
    """Call the USGS Design Maps web service and return the raw response dict.

    The returned dict is the full JSON response; the design values of interest
    live under ``response.data`` (ss, s1, sds, sd1, sdc, multi-period spectra).
    """
    resp = _get(
        session,
        DESIGN_MAPS_URL.format(reference_doc=reference_doc),
        timeout,
        params={
            "latitude": lat,
            "longitude": lon,
            "riskCategory": risk_category,
            "siteClass": site_class,
            "title": str(site_id),
        },
    )
    if resp.status_code != 200:
        raise USGSApiError(f"design value call failed: HTTP {resp.status_code}")
    data = resp.json()
    if "response" not in data or "data" not in data.get("response", {}):
        raise USGSApiError(f"design value call returned no data: {data}")
    return data


def call_hazard_api(
    edition: int,
    lat: float,
    lon: float,
    vs30: float,
    timeout: float = TIMEOUT,
    session: requests.Session | None = None,
    error_retries: int = ERROR_RETRIES,
    error_backoff: float = ERROR_BACKOFF,
) -> dict:
    """Call the USGS nshmp dynamic hazard web service; return raw response dict.

    ``edition`` is the NSHM release year (2018 or 2023). Latitude/longitude are
    formatted to 2 decimal places and vs30 to an integer, matching the
    original MATLAB URL construction. Hazard curves live under
    ``response.hazardCurves``.
    """
    url = HAZARD_URL.format(edition=edition, lon=lon, lat=lat, vs30=int(vs30))
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
