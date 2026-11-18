"""Which grid nodes are on land.

A design value in the middle of the Pacific is not a deliverable. A plain
rectangle over the Bay Area puts about a quarter of its nodes in open water,
where the numbers are real enough -- the service answers -- but mean nothing to
anyone, drag the heatmap's colour scale down by a third of its range, and cost
hours of calls to obtain.

So the grid carries a mask: a vendored coastline, committed and pinned, against
which each node is tested. The test is plain ray casting; the polygon is small
enough that no geometry library is needed, and avoiding the dependency keeps
the mask reviewable as data rather than as behaviour.

The vendored file is clipped to a bounding box. Outside that box a point is not
classifiable, and asking raises rather than guessing -- a grid that grows past
the clip must be re-vendored with :mod:`scripts.vendor_landmask`.
"""

import functools
import hashlib
import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_MASK = "bay_area_land"


class MaskError(Exception):
    """The mask cannot answer for this point, or cannot be loaded."""


@functools.lru_cache(maxsize=4)
def _load(name: str):
    """``(polygons, clip, sha256)`` for a vendored mask."""
    path = DATA_DIR / f"{name}.geojson"
    if not path.exists():
        raise MaskError(
            f"no land mask named {name!r} at {path}. Regenerate it with "
            "`python scripts/vendor_landmask.py --download`."
        )
    raw = path.read_bytes()
    # Newlines are normalised before hashing, so the digest describes the
    # coastline rather than the checkout it came from. git treats .geojson as
    # text: with core.autocrlf on Windows the working copy gains a CR on every
    # line while the committed blob keeps LF, so the same file hashes two ways
    # on two machines. That is not a difference in the coastline, but it is
    # enough to fail the check this digest exists for -- a database built on
    # Windows refused to draw on a Linux deployment over a single trailing
    # byte. .gitattributes pins the working copy to LF as well; this is the
    # half that holds even if someone checks the repository out without it.
    feature = json.loads(raw.decode("utf-8"))
    clip = feature["properties"]["clip"]
    polygons = tuple(
        tuple(tuple((float(x), float(y)) for x, y in ring) for ring in poly)
        for poly in feature["geometry"]["coordinates"]
    )
    return polygons, clip, hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()


def digest(name: str = DEFAULT_MASK) -> str:
    """The mask file's SHA-256, recorded in a database's manifest.

    A grid built against one coastline and validated against another is a
    difference worth detecting, so the digest travels with the data.
    """
    return _load(name)[2]


def clip_box(name: str = DEFAULT_MASK) -> dict:
    return dict(_load(name)[1])


def _in_ring(lon: float, lat: float, ring) -> bool:
    """Ray casting: count crossings of a ray heading east from the point."""
    inside = False
    count = len(ring)
    for index in range(count):
        x1, y1 = ring[index]
        x2, y2 = ring[(index + 1) % count]
        if (y1 > lat) != (y2 > lat):
            if lon < x1 + (lat - y1) * (x2 - x1) / (y2 - y1):
                inside = not inside
    return inside


def is_land(lon: float, lat: float, name: str = DEFAULT_MASK) -> bool:
    """True if the point is on land, by the vendored coastline.

    Holes are honoured, so San Francisco Bay reads as water rather than land.
    Raises :class:`MaskError` outside the mask's clip box, where it has no
    coastline to answer with.
    """
    polygons, clip, _ = _load(name)
    if not (
        clip["west"] <= lon <= clip["east"] and clip["south"] <= lat <= clip["north"]
    ):
        raise MaskError(
            f"({lat:.4f}, {lon:.4f}) is outside the {name!r} mask, which covers "
            f"longitude {clip['west']} to {clip['east']} and latitude "
            f"{clip['south']} to {clip['north']}"
        )
    for rings in polygons:
        if _in_ring(lon, lat, rings[0]):
            if any(_in_ring(lon, lat, hole) for hole in rings[1:]):
                continue  # inside a hole: a bay or a lake
            return True
    return False
