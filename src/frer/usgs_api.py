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

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from . import __version__

DESIGN_MAPS_URL = "https://earthquake.usgs.gov/ws/designmaps/{reference_doc}.json"
HAZARD_URL = (
    "https://earthquake.usgs.gov/ws/nshmp/conus-{edition}/dynamic/hazard/"
    "{lon:.2f}/{lat:.2f}/{vs30:d}"
)
TIMEOUT = 30  # seconds; the nshmp dynamic hazard calc can be slow

USER_AGENT = f"frer-calculator/{__version__} (USGS design-values tool)"

# Transient HTTP statuses worth retrying (rate limiting + gateway/5xx errors).
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})

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
) -> dict:
    """Call the USGS nshmp dynamic hazard web service; return raw response dict.

    ``edition`` is the NSHM release year (2018 or 2023). Latitude/longitude are
    formatted to 2 decimal places and vs30 to an integer, matching the
    original MATLAB URL construction. Hazard curves live under
    ``response.hazardCurves``.
    """
    url = HAZARD_URL.format(edition=edition, lon=lon, lat=lat, vs30=int(vs30))
    resp = _get(session, url, timeout)
    if resp.status_code == 429:
        raise USGSApiError("Too Many Requests")
    if resp.status_code != 200:
        raise USGSApiError(f"hazard call failed: HTTP {resp.status_code}")
    data = resp.json()
    if data.get("status") != "success":
        # A 200 with a failure status is a server-side computation error, not a
        # transient one; do not retry it (see Reno NV vs30=265, NSHM 2018).
        raise USGSApiError(f"hazard call status: {data.get('status')}")
    return data
