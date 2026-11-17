"""Regenerate the vendored land polygon (deliberate, like update_fixtures).

The gridded database covers only land: a design value in the Pacific is not a
deliverable, and roughly a quarter of a rectangular Bay Area grid falls in open
water. Deciding which nodes are on land needs a coastline, and a coastline is
the sort of input that should be committed and pinned rather than fetched at
build time -- otherwise the grid silently changes shape when the upstream file
does.

Source: Natural Earth 1:10m land polygons, public domain. At that scale the
coastline is good to roughly a kilometre, which is ample against a 0.05-degree
(~5.5 km) grid.

    python scripts/vendor_landmask.py --download   # fetch, clip and write
    python scripts/vendor_landmask.py path/to/ne_10m_land.geojson

The whole Natural Earth file is ~10 MB, nearly all of it irrelevant here, so it
is clipped to the Bay Area with a margin and only that is committed.
"""

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT = REPO_ROOT / "src" / "frer" / "data" / "bay_area_land.geojson"

SOURCE_URL = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
    "master/geojson/ne_10m_land.geojson"
)

# The bay_area grid with a margin, so the mask stays valid if the grid grows a
# little without needing a re-vendor.
CLIP = (-123.10, 36.80, -121.40, 38.70)  # west, south, east, north


def clip_ring(ring, box):
    """Sutherland-Hodgman clip of a ring against an axis-aligned rectangle."""
    west, south, east, north = box

    def inside(point, edge):
        x, y = point[0], point[1]
        return {
            "w": x >= west,
            "e": x <= east,
            "s": y >= south,
            "n": y <= north,
        }[edge]

    def intersect(a, b, edge):
        (x1, y1), (x2, y2) = (a[0], a[1]), (b[0], b[1])
        if edge in ("w", "e"):
            x = west if edge == "w" else east
            t = (x - x1) / (x2 - x1)
            return [x, y1 + t * (y2 - y1)]
        y = south if edge == "s" else north
        t = (y - y1) / (y2 - y1)
        return [x1 + t * (x2 - x1), y]

    output = [list(p[:2]) for p in ring]
    for edge in ("w", "e", "s", "n"):
        if not output:
            return []
        current, output = output, []
        for i, point in enumerate(current):
            prev = current[i - 1]
            point_in, prev_in = inside(point, edge), inside(prev, edge)
            if point_in:
                if not prev_in:
                    output.append(intersect(prev, point, edge))
                output.append(point)
            elif prev_in:
                output.append(intersect(prev, point, edge))
    return output


def ring_area(ring):
    """Twice the signed area; used only to discard slivers."""
    total = 0.0
    for i in range(len(ring)):
        x1, y1 = ring[i]
        x2, y2 = ring[(i + 1) % len(ring)]
        total += x1 * y2 - x2 * y1
    return abs(total)


def build(source: Path) -> dict:
    data = json.loads(source.read_text(encoding="utf-8"))
    west, south, east, north = CLIP
    polygons = []
    for feature in data["features"]:
        geom = feature["geometry"]
        parts = (
            [geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"]
        )
        for part in parts:
            xs = [p[0] for p in part[0]]
            ys = [p[1] for p in part[0]]
            if max(xs) < west or min(xs) > east or max(ys) < south or min(ys) > north:
                continue
            exterior = clip_ring(part[0], CLIP)
            if len(exterior) < 4 or ring_area(exterior) < 1e-9:
                continue
            rings = [[[round(x, 6), round(y, 6)] for x, y in exterior]]
            for hole in part[1:]:
                clipped = clip_ring(hole, CLIP)
                if len(clipped) >= 4 and ring_area(clipped) >= 1e-9:
                    rings.append([[round(x, 6), round(y, 6)] for x, y in clipped])
            polygons.append(rings)

    return {
        "type": "Feature",
        "properties": {
            "source": SOURCE_URL,
            "source_name": "Natural Earth 1:10m land (public domain)",
            "clip": {"west": west, "south": south, "east": east, "north": north},
            "note": (
                "Clipped to the Bay Area. Valid only inside 'clip'; a point "
                "outside that box is not classifiable with this file."
            ),
        },
        "geometry": {"type": "MultiPolygon", "coordinates": polygons},
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", nargs="?", type=Path, help="a local ne_10m_land.geojson")
    ap.add_argument(
        "--download", action="store_true", help=f"fetch it from {SOURCE_URL}"
    )
    args = ap.parse_args(argv)

    if args.download:
        print(f"downloading {SOURCE_URL}")
        source = OUT.parent / "ne_10m_land.download.geojson"
        source.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(SOURCE_URL, timeout=180) as response:
            source.write_bytes(response.read())
    elif args.source:
        source = args.source
    else:
        ap.error("give a local file, or pass --download")

    feature = build(source)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(feature, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.download:
        source.unlink()

    digest = hashlib.sha256(OUT.read_bytes()).hexdigest()
    polygons = feature["geometry"]["coordinates"]
    vertices = sum(len(ring) for poly in polygons for ring in poly)
    print(f"wrote {OUT.relative_to(REPO_ROOT)}")
    print(f"  {len(polygons)} polygon(s), {vertices:,} vertices")
    print(f"  {OUT.stat().st_size / 1024:.0f} kB")
    print(f"  sha256 {digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
