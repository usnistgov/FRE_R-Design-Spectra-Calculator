"""Command-line interface for the FREr calculator.

Commands
--------
frer pull : download NSHM hazard curves into the JSON cache
frer calc : develop FREr/DE design values from the cache

Both act on the whole site table and on ASCE 7-28, the one code version this
tool develops values for. ``frer pull`` is for refreshing the cache from the
web service; the frozen benchmark cache is built offline from a published data
release instead (``scripts/build_cache_from_release.py``).
"""

import argparse
from pathlib import Path

from . import usgs_api
from .cache import DEFAULT_CACHE_DIR
from .params import VERSION_ID, VERSION_NSHM
from .sites import SITE_SET


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="frer",
        description="Risk-targeted Functional Recovery Earthquake (FREr) "
        "design value calculator",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # frer pull
    p_pull = sub.add_parser("pull", help="download NSHM hazard curves into the cache")
    p_pull.add_argument(
        "--sites",
        type=Path,
        default=Path("sites.csv"),
        help="site table CSV (default: sites.csv)",
    )
    p_pull.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help=f"JSON cache directory (default: {DEFAULT_CACHE_DIR})",
    )
    p_pull.add_argument(
        "--pause",
        type=float,
        default=usgs_api.PAUSE_RANGE[0],
        help=f"minimum seconds between calls (default: "
        f"{usgs_api.PAUSE_RANGE[0]:.0f}); the service throttles sustained use, "
        "so spacing matters",
    )
    p_pull.add_argument(
        "--pause-max",
        type=float,
        default=usgs_api.PAUSE_RANGE[1],
        help=f"maximum seconds between calls (default: "
        f"{usgs_api.PAUSE_RANGE[1]:.0f}); each wait is drawn uniformly between "
        "--pause and this. Set both to 0 for no spacing",
    )
    p_pull.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="per-call HTTP read timeout in seconds (default: 30); "
        "increase for slow high-seismicity sites",
    )
    p_pull.add_argument(
        "--retries",
        type=int,
        default=5,
        help="retry attempts per call for transient failures "
        "(429/5xx/timeouts) (default: 5)",
    )
    p_pull.add_argument(
        "--backoff",
        type=float,
        default=1.0,
        help="exponential-backoff base in seconds for retries (default: 1.0)",
    )
    p_pull.add_argument(
        "--limit",
        type=int,
        default=None,
        help="stop after N (site, class) entries (for testing)",
    )
    p_pull.add_argument(
        "--editions",
        type=int,
        nargs="+",
        default=sorted(VERSION_NSHM.values()),
        help=f"NSHM years to pull hazard curves for "
        f"(default: {' '.join(str(y) for y in sorted(VERSION_NSHM.values()))})",
    )

    # frer calc
    p_calc = sub.add_parser(
        "calc", help="develop FREr/DE design values from the cached hazard curves"
    )
    p_calc.add_argument(
        "--sites",
        type=Path,
        default=Path("sites.csv"),
        help="site table CSV (default: sites.csv)",
    )
    p_calc.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help=f"JSON cache directory (default: {DEFAULT_CACHE_DIR})",
    )
    p_calc.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results"),
        help="output directory root (default: results)",
    )
    p_calc.add_argument("--no-plots", action="store_true", help="skip generating plots")

    args = parser.parse_args(argv)

    if args.command == "pull":
        from .pull import pull_sites

        stats = pull_sites(
            sites_csv=args.sites,
            cache_dir=args.cache_dir,
            pause=(args.pause, max(args.pause, args.pause_max)),
            limit=args.limit,
            editions=tuple(args.editions),
            timeout=args.timeout,
            retries=args.retries,
            backoff_factor=args.backoff,
        )
        print(
            f"\nDone: {stats['entries']} entries, {stats['calls']} calls, "
            f"{stats['failed']} failed, {stats['skipped']} already cached"
        )
        if stats["failed"]:
            print("Re-run `frer pull` to retry the failed calls.")

    elif args.command == "calc":
        from .calc import calc_design_values

        table = calc_design_values(
            sites_csv=args.sites,
            cache_dir=args.cache_dir,
            out_dir=args.out_dir,
            make_plots=not args.no_plots,
        )
        out = Path(args.out_dir) / SITE_SET / VERSION_ID
        print(f"Wrote {len(table)} rows to {out / 'frer_design_vals.csv'}")


if __name__ == "__main__":
    main()
