"""Pull USGS design values and hazard curves for sites into the JSON cache.

Port of MATLAB ``pull_hazard_values.m``. Transient failures (rate limiting,
5xx, timeouts) are retried with backoff by the shared session
(:func:`frer.usgs_api.build_session`); any call that still fails is left
un-cached so a re-run retries only what is missing.
"""

import time
from pathlib import Path

from . import cache, usgs_api
from .sites import load_sites, site_class_list

NSHM_YEARS = (2018, 2023)


def pull_sites(
    sites_csv: Path,
    cache_dir: Path,
    site_set: str | None = None,
    pause: float = 1.0,
    limit: int | None = None,
    editions: tuple = NSHM_YEARS,
    timeout: float = usgs_api.TIMEOUT,
    retries: int = 5,
    backoff_factor: float = 1.0,
    session=None,
) -> dict:
    """Pull design values and hazard curves for all (site, class) pairs.

    Parameters
    ----------
    sites_csv : path to the site table.
    cache_dir : directory for the JSON-per-site cache.
    site_set : optionally restrict to one site set (e.g. "frtc_testbed").
    pause : polite spacing, in seconds, between successful API calls. Bursts of
        rate-limit/5xx errors are absorbed by per-request backoff, so this can
        be small; raise it to be gentler on the service.
    limit : optionally stop after this many (site, class) entries (for testing).
    editions : NSHM years to pull hazard curves for.
    timeout : per-request read timeout in seconds.
    retries : retry attempts per request for transient failures.
    backoff_factor : exponential-backoff base for retries (see build_session).
    session : optional pre-built requests session (mainly for testing).

    Returns a summary dict with counts of calls made/failed/skipped.
    """
    sites = load_sites(sites_csv, site_set)
    stats = {"entries": 0, "calls": 0, "failed": 0, "skipped": 0}

    if session is None:
        session = usgs_api.build_session(retries=retries, backoff_factor=backoff_factor)

    n_entries = 0
    for _, row in sites.iterrows():
        for site_class, vs30 in site_class_list(row):
            if limit is not None and n_entries >= limit:
                return stats
            n_entries += 1
            stats["entries"] += 1

            entry = cache.load_entry(cache_dir, row["id"], site_class)
            if entry is None:
                entry = cache.new_entry(row.to_dict(), site_class, vs30)

            label = f"{row['name']} [{site_class}]"

            # Design values (ASCE 7-22)
            if entry["design_values"] is None:
                _rate_limited_call(
                    stats,
                    pause,
                    f"{label} design values (asce7-22)",
                    lambda row=row, site_class=site_class: (
                        usgs_api.call_design_value_api(
                            row["id"],
                            row["lat"],
                            row["lon"],
                            risk_category="II",
                            site_class=site_class,
                            timeout=timeout,
                            session=session,
                        )
                    ),
                    lambda data, entry=entry: entry.__setitem__("design_values", data),
                )
            else:
                stats["skipped"] += 1

            # Hazard curves for each NSHM edition
            for year in editions:
                if str(year) not in entry["hazard_curves"]:
                    _rate_limited_call(
                        stats,
                        pause,
                        f"{label} hazard curves (nshm {year})",
                        lambda y=year, row=row, vs30=vs30: usgs_api.call_hazard_api(
                            y,
                            row["lat"],
                            row["lon"],
                            vs30,
                            timeout=timeout,
                            session=session,
                        ),
                        lambda data, y=year, entry=entry: entry[
                            "hazard_curves"
                        ].__setitem__(str(y), data["response"]["hazardCurves"]),
                    )
                else:
                    stats["skipped"] += 1

            cache.save_entry(cache_dir, entry)
    return stats


def _rate_limited_call(stats, pause, label, call, store):
    """Make one API call (with retries handled by the session), store, and space.

    Transient errors are already retried with backoff inside ``call``; a call
    that reaches here still failed, so it is recorded and skipped (left
    un-cached for a later re-run) rather than aborting the whole pull.
    """
    stats["calls"] += 1
    try:
        data = call()
        store(data)
        print(f"  OK    {label}", flush=True)
    except Exception as e:  # exhausted retries, bad payloads, app-level errors
        stats["failed"] += 1
        print(f"  FAIL  {label}: {e}", flush=True)
    finally:
        # Polite spacing between calls; burst backoff is handled per-request.
        time.sleep(pause)
