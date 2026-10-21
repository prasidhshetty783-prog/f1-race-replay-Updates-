"""
Geometry for the 3D Track Detail view: track ribbon, terrain and buildings.

Pure NumPy (no Qt / OpenGL), so every builder can be unit tested.

Scene coordinates are metres with Y pointing up:

    scene X =  world X * unit_m
    scene Z = -world Y * unit_m          (keeps the map the right way round)
    scene Y = (elevation - base) * unit_m * exaggeration

``unit_m`` converts FastF1 position units (1/10 m) to metres.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree

FASTF1_UNIT_M = 0.1          # FastF1 X/Y/Z are in tenths of a metre
TRACK_HALF_WIDTH_M = 7.0     # F1 tracks are 12-15 m wide


@dataclass
class Mesh:
    """Triangle mesh ready to upload: float32 attributes, uint32 indices."""

    positions: np.ndarray            # (n, 3)
    normals: np.ndarray              # (n, 3)
    uvs: np.ndarray                  # (n, 2)
    indices: np.ndarray              # (m,)

    @property
    def vertex_count(self) -> int:
        return len(self.positions)

    def interleaved(self) -> np.ndarray:
        """position(3) + normal(3) + uv(2) per vertex, float32."""
        return np.hstack([self.positions, self.normals, self.uvs]).astype(np.float32)

    @staticmethod
    def empty() -> "Mesh":
        return Mesh(np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 2)), np.zeros(0, np.uint32))


@dataclass
class TrackFrame:
    """Maps world (FastF1) coordinates to scene metres."""

    base: float                       # elevation that becomes scene Y = 0 (world units)
    exaggeration: float = 1.0
    unit_m: float = FASTF1_UNIT_M

    def to_scene(self, x, y, z=None) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if z is None:
            z = np.full_like(x, self.base)
        z = np.asarray(z, dtype=float)
        return np.column_stack([
            x * self.unit_m,
            (z - self.base) * self.unit_m * self.exaggeration,
            -y * self.unit_m,
        ])


def auto_exaggeration(z_world, unit_m: float = FASTF1_UNIT_M) -> float:
    """Vertical exaggeration so flat circuits still show their relief."""
    z = np.asarray(z_world, dtype=float)
    z = z[np.isfinite(z)]
    if len(z) == 0:
        return 1.0
    relief_m = (z.max() - z.min()) * unit_m
    if relief_m < 8:
        return 4.0
    if relief_m < 25:
        return 2.5
    if relief_m < 60:
        return 1.8
    return 1.3


def _smooth_closed(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values
    kernel = np.ones(window) / window
    padded = np.concatenate([values[-window:], values, values[:window]])
    return np.convolve(padded, kernel, mode="same")[window:-window]


def resample_loop(x, y, z=None, spacing: float = 20.0, smooth_z: int = 9):
    """
    Resample a closed lap evenly by distance (in input units).

    Elevation is lightly smoothed: car position data is noisy in Z.
    Returns (x, y, z) arrays.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.zeros_like(x) if z is None else np.asarray(z, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[ok], y[ok], z[ok]
    if len(x) > 1 and np.hypot(x[0] - x[-1], y[0] - y[-1]) < 1e-6:
        x, y, z = x[:-1], y[:-1], z[:-1]
    if len(x) < 4:
        raise ValueError("need at least 4 points")
    xc, yc, zc = np.append(x, x[0]), np.append(y, y[0]), np.append(z, z[0])
    seg = np.hypot(np.diff(xc), np.diff(yc))
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    n = max(8, int(cum[-1] // spacing))
    t = np.linspace(0, cum[-1], n, endpoint=False)
    rz = np.interp(t, cum, zc)
    return np.interp(t, cum, xc), np.interp(t, cum, yc), _smooth_closed(rz, smooth_z)


def _loop_tangents(xz: np.ndarray) -> np.ndarray:
    d = np.roll(xz, -1, axis=0) - np.roll(xz, 1, axis=0)
    length = np.linalg.norm(d, axis=1, keepdims=True)
    length[length == 0] = 1.0
    return d / length


def _open_tangents(xz: np.ndarray) -> np.ndarray:
    d = np.gradient(xz, axis=0)
    length = np.linalg.norm(d, axis=1, keepdims=True)
    length[length == 0] = 1.0
    return d / length


def ribbon_mesh(centre: np.ndarray, half_width: float, lift: float = 0.0,
                offset: float = 0.0, closed: bool = True) -> Mesh:
    """
    Ribbon (strip) following ``centre`` (n, 3 scene points).

    ``offset`` shifts the strip sideways (positive = left of travel), which is
    how the white edge lines are made from the same centre line. ``closed``
    joins the last point back to the first (a full lap); open strips are used
    for DRS zones.
    """
    c = np.asarray(centre, dtype=float)
    n = len(c)
    if n < 2:
        return Mesh.empty()
    tan = _loop_tangents(c[:, [0, 2]]) if closed else _open_tangents(c[:, [0, 2]])
    # Left of travel in the XZ plane (Y up): rotate the tangent by +90 deg.
    left = np.column_stack([tan[:, 1], np.zeros(n), -tan[:, 0]])
    mid = c + left * offset + np.array([0.0, lift, 0.0])
    lpos = mid + left * half_width
    rpos = mid - left * half_width

    positions = np.empty((2 * n, 3))
    positions[0::2] = lpos
    positions[1::2] = rpos

    # Surface normal = cross(left, forward) with slope taken into account.
    fwd = (np.roll(mid, -1, axis=0) - np.roll(mid, 1, axis=0)) if closed else np.gradient(mid, axis=0)
    normal = np.cross(left, fwd)
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-9)
    normal[normal[:, 1] < 0] *= -1
    normals = np.repeat(normal, 2, axis=0)

    dist = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(mid, axis=0), axis=1))])
    v = dist / max(2 * half_width, 1e-6)
    uvs = np.empty((2 * n, 2))
    uvs[0::2, 0], uvs[1::2, 0] = 0.0, 1.0
    uvs[0::2, 1] = uvs[1::2, 1] = v

    idx = []
    for i in range(n if closed else n - 1):
        a, b = 2 * i, 2 * i + 1
        c2, d = (2 * ((i + 1) % n)), (2 * ((i + 1) % n) + 1)
        idx += [a, b, c2, b, d, c2]
    return Mesh(positions, normals, uvs, np.asarray(idx, dtype=np.uint32))


def heightfield(track: np.ndarray, bounds: Tuple[float, float, float, float],
                resolution: int = 160, corridor_m: float = 18.0,
                sink_m: float = 0.25, k: int = 24) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Terrain heights on a grid, shaped by the track's own elevation.

    The only elevation data we have is along the racing line, so the land in
    between is interpolated (inverse-distance weighting). Near the track the
    ground follows the LOWEST nearby road, just under it, so where a circuit
    crosses itself (e.g. Suzuka) the lower road stays visible and the upper
    one reads as a bridge.

    Args:
        track: (n, 3) scene points of the centre line (Y up).
        bounds: (x0, x1, z0, z1) scene extent of the grid.

    Returns:
        (xs, zs, heights) with heights shaped (len(zs), len(xs)).
    """
    x0, x1, z0, z1 = bounds
    w, h = x1 - x0, z1 - z0
    nx = max(4, int(resolution if w >= h else resolution * w / h))
    nz = max(4, int(resolution if h > w else resolution * h / w))
    xs = np.linspace(x0, x1, nx)
    zs = np.linspace(z0, z1, nz)
    gx, gz = np.meshgrid(xs, zs)
    pts = np.column_stack([gx.ravel(), gz.ravel()])

    tree = cKDTree(track[:, [0, 2]])
    k = min(k, len(track))
    dist, idx = tree.query(pts, k=k)
    heights_k = track[idx, 1]
    weights = 1.0 / (dist ** 2 + 25.0)
    idw = (weights * heights_k).sum(axis=1) / weights.sum(axis=1)

    # Under and right next to the road: follow the lowest road nearby, then
    # blend into the interpolated land over one more corridor width.
    near = np.where(dist <= 2 * corridor_m, heights_k, np.inf).min(axis=1)
    has_near = np.isfinite(near)
    carved = np.where(has_near, near - sink_m, idw)
    t = np.clip((dist[:, 0] - corridor_m) / corridor_m, 0.0, 1.0)
    height = np.where(has_near, carved * (1 - t) + idw * t, idw).reshape(nz, nx)
    return xs, zs, height


def flatten_edges(heights: np.ndarray, level: float, band: float = 0.15) -> np.ndarray:
    """
    Ease the outer ``band`` (fraction of the grid) down to ``level`` so the
    terrain meets the flat surroundings without a cliff.
    """
    nz, nx = heights.shape
    u = np.minimum(np.arange(nx), np.arange(nx)[::-1]) / max(1.0, band * nx)
    v = np.minimum(np.arange(nz), np.arange(nz)[::-1]) / max(1.0, band * nz)
    f = np.clip(np.minimum(u[None, :], v[:, None]), 0.0, 1.0)
    f = f * f * (3 - 2 * f)
    return level + (heights - level) * f


def grid_mesh(xs: np.ndarray, zs: np.ndarray, heights: np.ndarray,
              uv_fn=None) -> Mesh:
    """
    Triangle mesh for a height grid. ``uv_fn(x, z) -> (u, v)`` maps scene
    positions to texture coordinates (defaults to the grid's own 0..1).
    """
    gx, gz = np.meshgrid(xs, zs)
    positions = np.column_stack([gx.ravel(), heights.ravel(), gz.ravel()])

    # Normals from central differences.
    dhdx = np.gradient(heights, xs, axis=1)
    dhdz = np.gradient(heights, zs, axis=0)
    normals = np.column_stack([-dhdx.ravel(), np.ones(heights.size), -dhdz.ravel()])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)

    if uv_fn is None:
        u = (gx - xs[0]) / (xs[-1] - xs[0])
        v = (gz - zs[0]) / (zs[-1] - zs[0])
    else:
        u, v = uv_fn(gx, gz)
    uvs = np.column_stack([np.ravel(u), np.ravel(v)])

    nz, nx = heights.shape
    i = np.arange(nz - 1)[:, None] * nx + np.arange(nx - 1)[None, :]
    a, b, c, d = i, i + 1, i + nx, i + nx + 1
    indices = np.stack([a, c, b, b, c, d], axis=-1).ravel().astype(np.uint32)
    return Mesh(positions, normals, uvs, indices)


def sample_heights(xs: np.ndarray, zs: np.ndarray, heights: np.ndarray,
                   points_xz: np.ndarray) -> np.ndarray:
    """Bilinear lookup of terrain height at scene (x, z) points."""
    p = np.asarray(points_xz, dtype=float)
    fx = np.interp(p[:, 0], xs, np.arange(len(xs)))
    fz = np.interp(p[:, 1], zs, np.arange(len(zs)))
    ix = np.clip(np.floor(fx).astype(int), 0, len(xs) - 2)
    iz = np.clip(np.floor(fz).astype(int), 0, len(zs) - 2)
    tx, tz = fx - ix, fz - iz
    h00 = heights[iz, ix]
    h01 = heights[iz, ix + 1]
    h10 = heights[iz + 1, ix]
    h11 = heights[iz + 1, ix + 1]
    return (h00 * (1 - tx) + h01 * tx) * (1 - tz) + (h10 * (1 - tx) + h11 * tx) * tz


# ---------------------------------------------------------------------------
# Buildings
# ---------------------------------------------------------------------------

def _signed_area(poly: np.ndarray) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _clean_ring(pts: np.ndarray) -> np.ndarray:
    """Indices of a ring's useful corners: no repeats, no straight-through points."""
    idx = [i for i in range(len(pts)) if i == 0 or not np.allclose(pts[i], pts[i - 1])]
    if len(idx) > 1 and np.allclose(pts[idx[0]], pts[idx[-1]]):
        idx.pop()
    scale = float(np.ptp(pts, axis=0).max()) if len(pts) else 1.0
    eps = 1e-9 * scale * scale
    changed = True
    while changed and len(idx) > 3:
        changed = False
        for k in range(len(idx)):
            a, b, c = pts[idx[k - 1]], pts[idx[k]], pts[idx[(k + 1) % len(idx)]]
            if abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) <= eps:
                idx.pop(k)
                changed = True
                break
    return np.asarray(idx, dtype=int)


def triangulate(polygon) -> List[Tuple[int, int, int]]:
    """
    Ear-clipping triangulation of a simple polygon (no holes).

    Returns triangles as index triples into ``polygon``, all anticlockwise.
    Repeated and straight-through points are skipped, which is what makes
    real OpenStreetMap footprints (lots of collinear nodes) triangulate cleanly.
    """
    pts = np.asarray(polygon, dtype=float)
    if len(pts) > 3 and np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    if len(pts) < 3:
        return []
    order = list(_clean_ring(pts))
    if len(order) < 3:
        return []
    if _signed_area(pts[order]) < 0:
        order.reverse()

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def inside(p, a, b, c):
        return cross(a, b, p) >= 0 and cross(b, c, p) >= 0 and cross(c, a, p) >= 0

    triangles = []
    guard = 0
    n = len(order)
    while len(order) > 3 and guard < 4 * n * n:
        guard += 1
        m = len(order)
        clipped = False
        for i in range(m):
            ia, ib, ic = order[i - 1], order[i], order[(i + 1) % m]
            a, b, c = pts[ia], pts[ib], pts[ic]
            if cross(a, b, c) <= 0:          # reflex corner: not an ear
                continue
            blocked = False
            for j in order:
                if j in (ia, ib, ic):
                    continue
                q = pts[j]
                if (np.allclose(q, a) or np.allclose(q, b) or np.allclose(q, c)):
                    continue
                if inside(q, a, b, c):
                    blocked = True
                    break
            if blocked:
                continue
            triangles.append((ia, ib, ic))
            order.pop(i)
            clipped = True
            break
        if not clipped:
            break  # self-intersecting footprint: fan the rest
    if len(order) >= 3:
        for i in range(1, len(order) - 1):
            triangles.append((order[0], order[i], order[i + 1]))

    # Keep every triangle anticlockwise so the whole roof faces the same way.
    fixed = []
    for a, b, c in triangles:
        if cross(pts[a], pts[b], pts[c]) < 0:
            b, c = c, b
        fixed.append((a, b, c))
    return fixed


def extrude_buildings(footprints: Sequence[np.ndarray], heights: Sequence[float],
                      bases: Sequence[float]) -> Mesh:
    """
    Merge building footprints into one mesh of walls + flat roofs.

    Args:
        footprints: scene (x, z) outer rings, one per building.
        heights: building heights in metres (already exaggerated if wanted).
        bases: ground height under each building.
    """
    pos, nrm, idx = [], [], []
    count = 0
    for ring, height, base in zip(footprints, heights, bases):
        ring = np.asarray(ring, dtype=float)
        if len(ring) > 3 and np.allclose(ring[0], ring[-1]):
            ring = ring[:-1]
        if len(ring) < 3 or height <= 0:
            continue
        # Footprint in (x, -z) is a normal maths frame: make it anticlockwise there.
        flat = np.column_stack([ring[:, 0], -ring[:, 1]])
        if _signed_area(flat) < 0:
            ring = ring[::-1]
            flat = flat[::-1]
        top = base + height
        n = len(ring)
        # Walls: one quad per edge with its own outward normal (flat shading).
        for i in range(n):
            p0, p1 = ring[i], ring[(i + 1) % n]
            edge = p1 - p0
            length = float(np.hypot(*edge))
            if length < 1e-6:
                continue
            outward = np.array([-edge[1], 0.0, edge[0]]) / length
            quad = [(p0[0], base, p0[1]), (p1[0], base, p1[1]),
                    (p1[0], top, p1[1]), (p0[0], top, p0[1])]
            pos.extend(quad)
            nrm.extend([outward] * 4)
            idx.extend([count, count + 1, count + 2, count, count + 2, count + 3])
            count += 4
        # Roof
        tris = triangulate(flat)
        if tris:
            pos.extend((p[0], top, p[1]) for p in ring)
            nrm.extend([(0.0, 1.0, 0.0)] * n)
            for a, b, c in tris:
                idx.extend([count + a, count + b, count + c])
            count += n
    if not pos:
        return Mesh.empty()
    positions = np.asarray(pos, dtype=float)
    return Mesh(positions, np.asarray(nrm, dtype=float), np.zeros((len(positions), 2)),
                np.asarray(idx, dtype=np.uint32))


# ---------------------------------------------------------------------------
# Elevation profile + stats
# ---------------------------------------------------------------------------

@dataclass
class ElevationProfile:
    distance_m: np.ndarray      # along the lap from the reference lap's start
    elevation_m: np.ndarray     # relative to the lowest point
    lowest_m: float
    highest_m: float
    climb_m: float              # total ascent over one lap

    @property
    def range_m(self) -> float:
        return self.highest_m - self.lowest_m


def elevation_profile(x, y, z, unit_m: float = FASTF1_UNIT_M, smooth: int = 9) -> ElevationProfile:
    """Distance vs elevation along one lap (metres)."""
    x = np.asarray(x, dtype=float) * unit_m
    y = np.asarray(y, dtype=float) * unit_m
    z = np.asarray(z, dtype=float) * unit_m
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[ok], y[ok], z[ok]
    z = _smooth_closed(z, smooth) if len(z) > smooth * 2 else z
    dist = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
    rise = np.diff(z)
    return ElevationProfile(
        distance_m=dist,
        elevation_m=z - z.min(),
        lowest_m=float(z.min()),
        highest_m=float(z.max()),
        climb_m=float(rise[rise > 0].sum()),
    )


def nearest_on_track(track_xy: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
    """Index of the nearest track sample for each point (world units)."""
    tree = cKDTree(np.asarray(track_xy, dtype=float))
    _, idx = tree.query(np.asarray(points_xy, dtype=float))
    return np.atleast_1d(idx)


def track_headings(centre: np.ndarray) -> np.ndarray:
    """Yaw (degrees, about +Y) that points a -Z-forward model along the lap."""
    tan = _loop_tangents(np.asarray(centre, dtype=float)[:, [0, 2]])
    return np.degrees(np.arctan2(-tan[:, 0], -tan[:, 1]))
