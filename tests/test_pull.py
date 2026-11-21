"""Resilience tests for the USGS pull path.

These do not check returned *values*; they verify the calling behaviour: that
transient failures (rate limiting, 5xx, dropped connections) are retried and
recovered from, that persistent/application errors are surfaced without
retry-hammering, and that ``pull_sites`` records failures and keeps going
instead of raising. A local stub HTTP server exercises the real
``requests`` + ``urllib3`` retry stack; no live network is used (except the
opt-in ``network`` smoke test at the bottom).
"""

import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from frer import cache, pull, usgs_api
from frer.params import DEFAULT_SITE_CLASSES

# Minimal well-formed response body (shape only; values are irrelevant here).
HAZARD_OK = (200, {"status": "success", "response": {"hazardCurves": [{"imt": "x"}]}})


class _StubServer(ThreadingHTTPServer):
    """HTTP server returning scripted responses and counting requests."""

    def __init__(self):
        super().__init__(("127.0.0.1", 0), _StubHandler)
        self.requests = []  # every path served
        self.counts = {}  # endpoint key -> number of requests
        self.behaviors = {}  # endpoint key -> response tuple or callable(n)->tuple

    @property
    def base(self):
        host, port = self.server_address
        return f"{host}:{port}"

    def next_response(self, path):
        key = "hazard"
        self.counts[key] = self.counts.get(key, 0) + 1
        behavior = self.behaviors.get(key, HAZARD_OK)
        if callable(behavior):
            behavior = behavior(self.counts[key])
        status, body = behavior[0], behavior[1]
        headers = behavior[2] if len(behavior) > 2 else {}
        return status, body, headers


class _StubHandler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802  (name fixed by BaseHTTPRequestHandler)
        self.server.requests.append(self.path)
        status, body, headers = self.server.next_response(self.path)
        payload = json.dumps(body).encode()
        self.send_response(status)
        for key, value in headers.items():
            self.send_header(key, value)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):  # silence the default stderr logging
        pass


@pytest.fixture
def stub(monkeypatch):
    """Run a stub USGS server and point the API client at it."""
    server = _StubServer()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setattr(
        usgs_api,
        "HAZARD_URL",
        f"http://{server.base}/hazard/conus-{{edition}}/"
        f"{{lon:.4f}}/{{lat:.4f}}/{{vs30:d}}",
    )
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture
def mini_sites(tmp_path):
    """A one-row sites.csv sufficient for the pull path."""
    path = tmp_path / "sites.csv"
    path.write_text(
        "id,name,lat,lon,region\n1,Testville,34.00,-118.00,West\n",
        encoding="utf-8",
    )
    return path


def _fast_session(retries):
    # backoff_factor=0 -> no real sleeping between retries, so tests are quick.
    return usgs_api.build_session(retries=retries, backoff_factor=0)


# --- API-client resilience -------------------------------------------------


def test_hazard_success(stub):
    stub.behaviors["hazard"] = HAZARD_OK
    data = usgs_api.call_hazard_api(2023, 34.0, -118.0, 760, session=_fast_session(3))
    assert data["status"] == "success"
    assert stub.counts["hazard"] == 1


def test_hazard_retries_then_succeeds(stub):
    # Rate-limited twice, then OK: the call should transparently recover.
    stub.behaviors["hazard"] = lambda n: HAZARD_OK if n > 2 else (429, {})
    data = usgs_api.call_hazard_api(2023, 34.0, -118.0, 760, session=_fast_session(5))
    assert data["status"] == "success"
    assert stub.counts["hazard"] == 3


def test_hazard_persistent_5xx_raises_after_retries(stub):
    stub.behaviors["hazard"] = (503, {})
    with pytest.raises(usgs_api.USGSApiError):
        usgs_api.call_hazard_api(2023, 34.0, -118.0, 760, session=_fast_session(2))
    assert stub.counts["hazard"] == 3  # 1 initial + 2 retries


def test_hazard_app_error_is_retried_a_few_times_then_raises(stub):
    # A 200 whose body reports failure is either a point that cannot be
    # computed or the service declining to compute. It is tried a bounded
    # number of times -- enough to outlast a throttle, few enough not to
    # hammer a genuine error -- and then surfaced.
    stub.behaviors["hazard"] = (200, {"status": "error"})
    with pytest.raises(usgs_api.USGSApiError):
        usgs_api.call_hazard_api(
            2023, 34.0, -118.0, 760, session=_fast_session(5), error_backoff=0
        )
    assert stub.counts["hazard"] == usgs_api.ERROR_RETRIES + 1


def test_hazard_app_error_that_clears_is_recovered(stub):
    """A throttle presents as a 200 error and clears; the call should survive it.

    This is what a one-second pull ran into: the service answered 200 with a
    failure body in a fraction of a second, the retry layer never saw it
    because it only watches status codes, and a long run turned into a run of
    false failures.
    """
    stub.behaviors["hazard"] = lambda n: (
        HAZARD_OK if n > 1 else (200, {"status": "error"})
    )
    data = usgs_api.call_hazard_api(
        2023, 34.0, -118.0, 760, session=_fast_session(5), error_backoff=0
    )
    assert data["status"] == "success"
    assert stub.counts["hazard"] == 2


def test_connection_error_is_wrapped(monkeypatch):
    # Point at a closed port: transport errors should surface as USGSApiError,
    # not leak a raw requests exception.
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        dead_port = s.getsockname()[1]
    monkeypatch.setattr(
        usgs_api,
        "HAZARD_URL",
        f"http://127.0.0.1:{dead_port}/hazard/conus-{{edition}}/"
        f"{{lon:.4f}}/{{lat:.4f}}/{{vs30:d}}",
    )
    with pytest.raises(usgs_api.USGSApiError):
        usgs_api.call_hazard_api(2023, 34.0, -118.0, 760, session=_fast_session(1))


# --- pull_sites orchestration ----------------------------------------------


def test_pull_happy_path(stub, tmp_path, mini_sites):
    stub.behaviors["hazard"] = HAZARD_OK
    cdir = tmp_path / "cache"
    stats = pull.pull_sites(
        mini_sites, cdir, pause=0, editions=(2023,), session=_fast_session(2)
    )
    assert stats["failed"] == 0
    # One site, one call per site class.
    assert stats["calls"] == len(DEFAULT_SITE_CLASSES)
    entry = cache.load_entry(cdir, 1, "D")
    assert set(entry["hazard_curves"]) == {"2023"}


def test_pull_failure_is_recorded_and_continues(stub, tmp_path, mini_sites):
    # Design succeeds; hazard is persistently down. The run must finish, mark
    # the hazard calls failed, and still write an entry for what did not.
    stub.behaviors["hazard"] = (503, {})
    cdir = tmp_path / "cache"
    stats = pull.pull_sites(
        mini_sites, cdir, pause=0, editions=(2023,), session=_fast_session(1)
    )
    # One failure per site class, and no exception raised.
    assert stats["failed"] == len(DEFAULT_SITE_CLASSES)
    entry = cache.load_entry(cdir, 1, "D")
    assert entry["hazard_curves"] == {}


def test_pull_resumes_without_recalling(stub, tmp_path, mini_sites):
    # A fully-cached site should trigger zero HTTP requests on re-run.
    cdir = tmp_path / "cache"
    for site_class, vs30 in DEFAULT_SITE_CLASSES.items():
        entry = cache.new_entry(
            {
                "id": 1,
                "name": "Testville",
                "lat": 34.0,
                "lon": -118.0,
                "region": "West",
            },
            site_class,
            vs30,
        )
        entry["hazard_curves"] = {"2023": []}
        cache.save_entry(cdir, entry)

    stats = pull.pull_sites(
        mini_sites, cdir, pause=0, editions=(2023,), session=_fast_session(1)
    )
    assert stats["calls"] == 0
    assert stats["skipped"] == len(DEFAULT_SITE_CLASSES)
    assert stub.requests == []


# --- session configuration --------------------------------------------------


def test_build_session_retry_configuration():
    session = usgs_api.build_session(retries=4, backoff_factor=2.0)
    retry = session.get_adapter("https://earthquake.usgs.gov").max_retries
    assert retry.total == 4
    assert {429, 500, 502, 503, 504} <= set(retry.status_forcelist)
    assert retry.respect_retry_after_header is True
    assert "User-Agent" in session.headers


# --- opt-in live smoke test (skipped unless --run-network) ------------------


@pytest.mark.network
def test_live_hazard_smoke():
    """One real hazard call: it should return curves without raising."""
    data = usgs_api.call_hazard_api(2023, 34.05, -118.25, 260)
    assert data["status"] == "success"
    assert data["response"]["hazardCurves"]
