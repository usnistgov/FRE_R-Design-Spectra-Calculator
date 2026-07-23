"""Command-line interface for the FREr calculator.

Commands
--------
frer pull : download USGS design values and hazard curves into the JSON cache
frer calc : develop FREr/DE design values from the cache
"""

import argparse
from pathlib import Path

from .cache import DEFAULT_CACHE_DIR
from .sites import SITE_SETS

VERSION_ALIASES = {
    "asce7-22": "ASCE 7-22",
    "asce7-28": "ASCE 7-28",
}


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="frer",
        description="Risk-targeted Functional Recovery Earthquake (FREr) "
        "design value calculator",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # frer pull
    p_pull = sub.add_parser(
        "pull", help="download USGS design values and hazard curves into the cache"
    )
    p_pull.add_argument(
        "--sites",
        type=Path,
        default=Path("sites.csv"),
        help="site table CSV (default: sites.csv)",
    )
    p_pull.add_argument(
        "--site-set",
        choices=SITE_SETS,
        default=None,
        help="restrict to one site set (default: all sites)",
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
        default=1.0,
        help="polite spacing in seconds between calls (default: 1); transient "
        "errors are retried with backoff regardless",
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
        default=[2018, 2023],
        help="NSHM years to pull hazard curves for (default: 2018 2023)",
    )

    # frer calc
    p_calc = sub.add_parser(
        "calc", help="develop FREr/DE design values from cached USGS data"
    )
    p_calc.add_argument(
        "--sites",
        type=Path,
        default=Path("sites.csv"),
        help="site table CSV (default: sites.csv)",
    )
    p_calc.add_argument(
        "--site-set", choices=SITE_SETS, required=True, help="site set to calculate"
    )
    p_calc.add_argument(
        "--version",
        choices=sorted(VERSION_ALIASES),
        required=True,
        help="code version: asce7-22 (NSHM 2018) or asce7-28 (NSHM 2023)",
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
            site_set=args.site_set,
            pause=args.pause,
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

        version_id = VERSION_ALIASES[args.version]
        table = calc_design_values(
            sites_csv=args.sites,
            cache_dir=args.cache_dir,
            site_set=args.site_set,
            version_id=version_id,
            out_dir=args.out_dir,
            make_plots=not args.no_plots,
        )
        out = Path(args.out_dir) / args.site_set / version_id
        print(f"Wrote {len(table)} rows to {out / 'frer_design_vals.csv'}")


if __name__ == "__main__":
    main()
