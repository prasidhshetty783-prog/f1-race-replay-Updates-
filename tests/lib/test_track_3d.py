import numpy as np
import pytest

from src.lib import track_3d as t3d


def _oval(n=400, rx=8000.0, ry=4000.0, hill=150.0):
    """A closed lap in FastF1 units (1/10 m) with one hill."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return rx * np.cos(t), ry * np.sin(t), hill * np.clip(np.sin(t), 0, None)


class TestFrameAndResample:
    def test_scene_axes(self):
        frame = t3d.TrackFrame(base=100.0, exaggeration=2.0)
        p = frame.to_scene([10.0], [20.0], [110.0])[0]
        assert p == pytest.approx([1.0, 2.0, -2.0])   # X, Y up (10 units * 0.1 * 2), Z = -y

    def test_exaggeration_is_bigger_for_flat_tracks(self):
        assert t3d.auto_exaggeration([0, 30]) > t3d.auto_exaggeration([0, 1000])

    def test_resample_is_even(self):
        x, y, z = _oval()
        rx, ry, rz = t3d.resample_loop(x, y, z, spacing=50.0)
        steps = np.hypot(np.diff(rx), np.diff(ry))
        assert steps.std() / steps.mean() < 0.05


class TestRibbon:
    def test_closed_ribbon_has_two_vertices_per_point_and_faces_up(self):
        x, y, z = _oval(100)
        centre = t3d.TrackFrame(base=0.0).to_scene(x, y, z)
        mesh = t3d.ribbon_mesh(centre, half_width=7.0)
        assert mesh.vertex_count == 200
        assert len(mesh.indices) == 100 * 6
        assert (mesh.normals[:, 1] > 0).all()
        widths = np.linalg.norm(mesh.positions[0::2] - mesh.positions[1::2], axis=1)
        assert widths == pytest.approx(np.full(100, 14.0), rel=1e-6)

    def test_open_ribbon_does_not_wrap(self):
        centre = np.column_stack([np.arange(10.0), np.zeros(10), np.zeros(10)])
        mesh = t3d.ribbon_mesh(centre, 1.0, closed=False)
        assert len(mesh.indices) == 9 * 6


class TestTerrain:
    def test_ground_sits_just_under_the_road(self):
        x, y, z = _oval()
        frame = t3d.TrackFrame(base=0.0)
        centre = frame.to_scene(x, y, z)
        xs, zs, h = t3d.heightfield(centre, (-1000, 1000, -600, 600), resolution=200, sink_m=0.25)
        road = t3d.sample_heights(xs, zs, h, centre[:, [0, 2]])
        assert np.all(road <= centre[:, 1] + 0.05)          # never above the road
        assert np.median(centre[:, 1] - road) < 1.0          # but hugging it

    def test_lower_road_wins_at_a_crossing(self):
        # Two roads crossing at different heights (like Suzuka's bridge).
        a = np.column_stack([np.linspace(-200, 200, 81), np.zeros(81), np.zeros(81)])
        b = np.column_stack([np.zeros(81), np.full(81, 12.0), np.linspace(-200, 200, 81)])
        xs, zs, h = t3d.heightfield(np.vstack([a, b]), (-300, 300, -300, 300), resolution=121)
        assert t3d.sample_heights(xs, zs, h, np.array([[0.0, 0.0]]))[0] < 0.0

    def test_flatten_edges_reaches_the_level_at_the_border(self):
        h = np.full((50, 80), 10.0)
        flat = t3d.flatten_edges(h, level=-2.0)
        assert flat[0, :] == pytest.approx(-2.0) and flat[:, -1] == pytest.approx(-2.0)
        assert flat[25, 40] == pytest.approx(10.0)

    def test_grid_mesh_shapes(self):
        xs, zs = np.linspace(0, 10, 11), np.linspace(0, 5, 6)
        mesh = t3d.grid_mesh(xs, zs, np.zeros((6, 11)))
        assert mesh.vertex_count == 66 and len(mesh.indices) == 10 * 5 * 6
        assert mesh.uvs.min() == 0.0 and mesh.uvs.max() == 1.0


class TestBuildings:
    def test_triangulate_concave_footprint_with_messy_points(self):
        poly = [(0, 0), (2, 0), (4, 0), (4, 0), (4, 1), (1, 1), (1, 3), (0, 3), (0, 1.5)]
        tris = t3d.triangulate(poly)
        pts = np.asarray(poly, dtype=float)
        area = sum(abs(t3d._signed_area(pts[list(t)])) for t in tris)
        assert area == pytest.approx(6.0)
        assert all(t3d._signed_area(pts[list(t)]) > 0 for t in tris)

    def test_extruded_walls_face_outwards_and_roof_is_on_top(self):
        square = np.array([[0, 0], [10, 0], [10, -10], [0, -10]], dtype=float)  # scene x, z
        mesh = t3d.extrude_buildings([square], [8.0], [2.0])
        assert mesh.positions[:, 1].min() == pytest.approx(2.0)
        assert mesh.positions[:, 1].max() == pytest.approx(10.0)
        centre = np.array([5.0, -5.0])
        walls = mesh.normals[:, 1] == 0
        out = mesh.positions[walls][:, [0, 2]] - centre
        assert (np.einsum("ij,ij->i", out, mesh.normals[walls][:, [0, 2]]) > 0).all()

    def test_degenerate_buildings_are_skipped(self):
        assert t3d.extrude_buildings([np.zeros((2, 2))], [5.0], [0.0]).vertex_count == 0


def test_elevation_profile():
    x, y, z = _oval()
    prof = t3d.elevation_profile(x, y, z, smooth=1)
    assert prof.elevation_m.min() == 0.0
    assert prof.range_m == pytest.approx(15.0, abs=0.1)     # 150 units = 15 m
    assert prof.climb_m == pytest.approx(15.0, abs=0.5)
    assert prof.distance_m[-1] > 0


def test_headings_point_models_along_the_lap():
    centre = np.column_stack([np.zeros(5), np.zeros(5), -np.arange(5.0)])  # moving towards -Z
    assert t3d.track_headings(np.vstack([centre, centre[::-1][1:-1]]))[2] == pytest.approx(0.0, abs=1e-6)
