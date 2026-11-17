"""Quantify what a cache refresh changed, against an archive.

Run after refreshing ``tests/usgs_cache`` to turn "the values moved a little"
into a number. It recomputes design values from the archived entries and from
the live ones and reports the difference per metric and per site, checks that
the halves of the cache that were not meant to change did not, and -- for
ASCE 7-28 -- measures the knock-on effect on return periods, which is where a
small change in Sa is amplified by the slope of the hazard curve.

    python scripts/compare_cache.py --archive vs30-265
    python scripts/compare_cache.py --archive vs30-265 --out results/report.md

The report is small and worth committing; the archive it compares against is
not (see scripts/archive_cache.py).
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

from frer.cache import load_entry
from frer.params import VERSION_NSHM
from frer.single import (
    SD1_COLUMN,
    SD1FR_COLUMN,
    SDS_COLUMN,
    SDSFR_COLUMN,
    calc_site,
    design_value_columns,
    return_period_columns,
)
from frer.sites import load_sites

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO_ROOT / "tests" / "usgs_cache"
ARCHIVE_ROOT = REPO_ROOT / "results" / "archive"
SITES_CSV = REPO_ROOT / "sites.csv"
RP_REFERENCE = (
    REPO_ROOT / "tests" / "usgs_160" / "ASCE 7-28" / "frer_return_periods.csv"
)

VERSIONS = list(VERSION_NSHM)

# ASCE 7-22 takes its DE values from the design-maps service, so only the FREr
# pair moves there; ASCE 7-28 derives everything from the hazard curves.
MOVABLE = {
    "ASCE 7-22": [SDSFR_COLUMN, SD1FR_COLUMN],
    "ASCE 7-28": [SDS_COLUMN, SD1_COLUMN, SDSFR_COLUMN, SD1FR_COLUMN],
}

# The bands test_benchmark.py holds return periods to, mirrored here so the
# report can say which rows newly fall outside them.
RP_RTOL = 0.02
RP_RTOL_SOFT_SOIL = 0.10
RP_COLUMN_MAP = {
    "frer_0p2": "frer_rp_0p2",
    "frer_1p0": "frer_rp_1p0",
    "mcer_0p2": "mcer_rp_0p2",
    "mcer_1p0": "mcer_rp_1p0",
    "DE_0p2": "de_rp_0p2",
    "DE_1p0": "de_rp_1p0",
}


def digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode("utf-8")).hexdigest()


def relative(new: float, old: float) -> float:
    return abs(new - old) / abs(old) if old else float("nan")


def describe(values: pd.Series) -> dict:
    return {
        "max": float(values.max()),
        "p95": float(values.quantile(0.95)),
        "median": float(values.median()),
    }


def compare_entries(archive_dir: Path, site_class: str, sites: pd.DataFrame) -> dict:
    """Recompute from both caches and collect every difference."""
    rows, rp_rows, integrity = [], [], []

    for _, site in sites.iterrows():
        site_id = int(site["id"])
        old = load_entry(archive_dir / "usgs_cache", site_id, site_class)
        new = load_entry(CACHE_DIR, site_id, site_class)
        if old is None or new is None:
            continue

        integrity.append(
            {
                "id": site_id,
                "name": site["name"],
                "design_values_same": digest(old["design_values"])
                == digest(new["design_values"]),
                # An entry whose curves were cleared but not yet refetched is
                # pending, not changed; counting it as changed would make a
                # half-finished pull read like a completed one.
                "refreshed": bool(new["hazard_curves"]),
                "hazard_changed": bool(new["hazard_curves"])
                and digest(old["hazard_curves"]) != digest(new["hazard_curves"]),
                "vs30_old": old.get("vs30"),
                "vs30_new": new.get("vs30"),
            }
        )

        for version in VERSIONS:
            nshm = str(VERSION_NSHM[version])
            if nshm not in old["hazard_curves"] or nshm not in new["hazard_curves"]:
                continue
            before = calc_site(old, version)
            after = calc_site(new, version)
            old_vals = design_value_columns(before)
            new_vals = design_value_columns(after)
            for column in MOVABLE[version]:
                rows.append(
                    {
                        "version": version,
                        "id": site_id,
                        "name": site["name"],
                        "metric": column,
                        "old": old_vals[column],
                        "new": new_vals[column],
                        "rel": relative(new_vals[column], old_vals[column]),
                    }
                )

            if version == "ASCE 7-28":
                old_rp = return_period_columns(before)
                new_rp = return_period_columns(after)
                for column in old_rp:
                    rp_rows.append(
                        {
                            "id": site_id,
                            "name": site["name"],
                            "metric": column,
                            "old": old_rp[column],
                            "new": new_rp[column],
                            "rel": relative(new_rp[column], old_rp[column]),
                        }
                    )

    return {
        "values": pd.DataFrame(rows),
        "return_periods": pd.DataFrame(rp_rows),
        "integrity": pd.DataFrame(integrity),
    }


def matlab_band_check(rp: pd.DataFrame, site_class: str) -> pd.DataFrame:
    """How the refreshed return periods sit against the MATLAB reference.

    Reports, per row, whether it was inside the benchmark's band before the
    refresh and whether it still is -- so the rows that newly need enumerating
    in KNOWN_RP_EXCEPTIONS can be listed rather than discovered by a red test.
    """
    if not RP_REFERENCE.exists() or rp.empty:
        return pd.DataFrame()

    reference = pd.read_csv(RP_REFERENCE)
    reference = reference[reference["site_class"] == site_class]
    band = RP_RTOL_SOFT_SOIL if site_class == "E" else RP_RTOL

    out = []
    for matlab_col, our_col in RP_COLUMN_MAP.items():
        if matlab_col not in reference.columns:
            continue
        subset = rp[rp["metric"] == our_col]
        merged = subset.merge(reference[["name", matlab_col]], on="name", how="inner")
        for _, row in merged.iterrows():
            expected = row[matlab_col]
            if not expected:
                continue
            out.append(
                {
                    "id": row["id"],
                    "name": row["name"],
                    "metric": matlab_col,
                    "matlab": expected,
                    "before": row["old"],
                    "after": row["new"],
                    "rel_before": relative(row["old"], expected),
                    "rel_after": relative(row["new"], expected),
                    "band": band,
                }
            )
    table = pd.DataFrame(out)
    if table.empty:
        return table
    table["was_inside"] = table["rel_before"] <= table["band"]
    table["now_inside"] = table["rel_after"] <= table["band"]
    return table


def section(lines: list, title: str) -> None:
    lines.append(f"\n## {title}\n")


def render(result: dict, bands: pd.DataFrame, archive: str, site_class: str) -> str:
    values, rp, integrity = (
        result["values"],
        result["return_periods"],
        result["integrity"],
    )
    lines = [
        f"# Cache refresh report -- class {site_class} vs `{archive}`",
        "",
        f"Compared {len(integrity)} entries recomputed from both caches "
        f"({int(integrity['refreshed'].sum())} refreshed).",
    ]

    section(lines, "Integrity")
    unchanged_design = int(integrity["design_values_same"].sum())
    changed_hazard = int(integrity["hazard_changed"].sum())
    pending = int((~integrity["refreshed"]).sum())
    lines += [
        f"- `design_values` identical: **{unchanged_design} / {len(integrity)}** "
        "(expected: all -- the design-maps service was never re-queried)",
        f"- `hazard_curves` changed: **{changed_hazard} / {len(integrity)}**",
        f"- recorded vs30: {sorted(set(integrity['vs30_old']))} -> "
        f"{sorted(set(integrity['vs30_new']))}",
    ]
    if pending:
        lines += [
            "",
            f"> **Incomplete: {pending} entr(ies) have no hazard curves yet.** "
            "The pull has not finished or did not finish. Every figure below is "
            "from the entries that do, so the report is a progress check rather "
            "than the final answer.",
        ]
    if unchanged_design != len(integrity):
        lines.append("")
        lines.append("> **The design-value half changed. That was not intended.**")

    section(lines, "Design values")
    lines.append("| version | metric | max | p95 | median | worst site |")
    lines.append("|---|---|---|---|---|---|")
    for version in VERSIONS:
        for metric in MOVABLE[version]:
            subset = values[
                (values["version"] == version) & (values["metric"] == metric)
            ]
            if subset.empty:
                continue
            stats = describe(subset["rel"])
            worst = subset.loc[subset["rel"].idxmax()]
            lines.append(
                f"| {version} | {metric} | {stats['max']:.2%} | {stats['p95']:.2%} "
                f"| {stats['median']:.2%} | {worst['name']} |"
            )

    if not rp.empty:
        section(lines, "Return periods (ASCE 7-28)")
        lines.append(
            "Return periods are 1/AFE read off the hazard curve, so a change in "
            "Sa is amplified by the local slope."
        )
        lines.append("")
        lines.append("| metric | max | p95 | median | worst site |")
        lines.append("|---|---|---|---|---|")
        for metric in sorted(rp["metric"].unique()):
            subset = rp[rp["metric"] == metric]
            stats = describe(subset["rel"])
            worst = subset.loc[subset["rel"].idxmax()]
            lines.append(
                f"| {metric} | {stats['max']:.2%} | {stats['p95']:.2%} "
                f"| {stats['median']:.2%} | {worst['name']} |"
            )

    if not bands.empty:
        section(lines, "Effect on the MATLAB return-period reference")
        newly_out = bands[bands["was_inside"] & ~bands["now_inside"]]
        newly_in = bands[~bands["was_inside"] & bands["now_inside"]]
        lines += [
            f"Band for class {site_class}: {bands['band'].iloc[0]:.0%}. "
            f"Rows compared: {len(bands)}.",
            "",
            f"- inside before -> **outside now: {len(newly_out)}** "
            "(these need adding to `KNOWN_RP_EXCEPTIONS`)",
            f"- outside before -> inside now: {len(newly_in)}",
        ]
        if not newly_out.empty:
            lines += [
                "",
                "| id | site | metric | MATLAB | before | after | was | now |",
                "|---|---|---|---|---|---|---|---|",
            ]
            for _, row in newly_out.sort_values(
                "rel_after", ascending=False
            ).iterrows():
                lines.append(
                    f"| {int(row['id'])} | {row['name']} | {row['metric']} "
                    f"| {row['matlab']:.0f} | {row['before']:.0f} "
                    f"| {row['after']:.0f} | {row['rel_before']:.1%} "
                    f"| {row['rel_after']:.1%} |"
                )

    section(lines, "Largest individual changes")
    top = values.nlargest(10, "rel")
    lines.append("| version | site | metric | old | new | change |")
    lines.append("|---|---|---|---|---|---|")
    for _, row in top.iterrows():
        lines.append(
            f"| {row['version']} | {row['name']} | {row['metric']} "
            f"| {row['old']:.4f} | {row['new']:.4f} | {row['rel']:.2%} |"
        )

    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--archive", required=True, help="archive label to compare against")
    ap.add_argument("--site-class", default="D", help="site class to compare")
    ap.add_argument("--out", type=Path, help="write the report here as Markdown")
    args = ap.parse_args(argv)

    archive_dir = ARCHIVE_ROOT / args.archive
    if not archive_dir.is_dir():
        print(f"No archive at {archive_dir}")
        return 1

    # Only the sites whose class vs30 comes from DEFAULT_SITE_CLASSES; a site
    # with an explicit site_class carries its own and was never affected.
    sites = load_sites(SITES_CSV)
    sites = sites[sites["site_class"].isna()]

    print(f"Comparing class {args.site_class} against {archive_dir.name} ...")
    result = compare_entries(archive_dir, args.site_class, sites)
    if result["integrity"].empty:
        print("No comparable entries found.")
        return 1

    bands = matlab_band_check(result["return_periods"], args.site_class)
    report = render(result, bands, args.archive, args.site_class)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, encoding="utf-8")
        print(f"Wrote {args.out}")
    print()
    print(report)

    ok = bool(result["integrity"]["design_values_same"].all())
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
