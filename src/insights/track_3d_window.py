"""
3D Track Detail insight.

The circuit in 3D, live:

* an elevation-aware track ribbon on terrain shaped by the track's own
  elevation data (FastF1 position Z), with DRS zones and the start line,
* the real-world OpenStreetMap map draped on the ground, with buildings
  extruded from their footprints (when map data is available),
* every car, in team colours, and three cameras: orbit, follow a car, and a
  free-flying drone (WASD),
* a Track Detail panel (circuit stats, live conditions, past winners and the
  fastest race lap here) and an elevation profile with every car on it.

Heavy work (meshes, map download, history lookup) runs on background threads,
so the window opens instantly and fills in as data arrives.
"""

import math
import os
import sys
import threading
import time

import numpy as np
from PySide6.QtCore import (
    Property,
    QAbstractListModel,
    QByteArray,
    QModelIndex,
    QObject,
    QPointF,
    QRectF,
    Qt,
    QTimer,
    QUrl,
    Signal,
    Slot,
)
from PySide6.QtGui import QColor, QFont, QLinearGradient, QPainter, QPainterPath, QPen, QVector3D
from PySide6.QtQuick3D import QQuick3DGeometry
from PySide6.QtQuickWidgets import QQuickWidget
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from src.gui.pit_wall_window import PitWallWindow
from src.lib import track_3d as t3d

QML_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "track_3d.qml")

BG = "#101216"
CARD = "#181a1f"
BORDER = "#272a31"
TEXT = "#eceef1"
MUTED = "#8a909a"
ACCENT = "#e10600"

MAP_MARGIN = 0.22          # same area the 2D map uses, so one download serves both
MAP_TEXTURE_PX = 4096      # ground texture resolution along its long side
MAX_BUILDINGS = 12000
DEFAULT_BUILDING_M = 9.0   # OpenStreetMap buildings without a height tag
CAR_SCALE = {"orbit": 6.0, "drone": 2.5, "follow": 1.0}
FOLLOW_TURN_S = 0.25      # how quickly the chase camera swings round corners
LAND_RGB = (18, 19, 22)    # matches the map's land colour (circuit_map.STYLE["land"])

STYLE = f"""
QWidget#root, QScrollArea, QScrollArea > QWidget > QWidget {{ background: {BG}; border: none; }}
QLabel {{ color: {TEXT}; }}
QFrame#card {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 12px; }}
QLabel#title {{ font-size: 18px; font-weight: 700; }}
QLabel#subtitle {{ color: {MUTED}; font-size: 12px; }}
QLabel#cardTitle {{ color: {MUTED}; font-size: 11px; font-weight: 700; letter-spacing: 1px; }}
QLabel#tileValue {{ font-size: 17px; font-weight: 700; }}
QLabel#tileLabel {{ color: {MUTED}; font-size: 11px; }}
QLabel#rowLabel {{ color: {MUTED}; font-size: 12px; }}
QLabel#rowValue {{ font-size: 12px; font-weight: 600; }}
QLabel#small {{ color: {MUTED}; font-size: 11px; }}
QComboBox {{ background: {CARD}; color: {TEXT}; border: 1px solid {BORDER}; border-radius: 6px; padding: 3px 8px; }}
QComboBox QAbstractItemView {{ background: {CARD}; color: {TEXT}; selection-background-color: #33363e; }}
QStatusBar {{ background: #0b0c0f; color: {MUTED}; }}
QStatusBar QLabel {{ color: {MUTED}; }}
"""


# ---------------------------------------------------------------------------
# Qt Quick 3D plumbing
# ---------------------------------------------------------------------------

class MeshGeometry(QQuick3DGeometry):
    """Uploads a :class:`track_3d.Mesh` (position, normal, uv) to Qt Quick 3D."""

    def set_mesh(self, mesh: t3d.Mesh):
        self.clear()
        if mesh.vertex_count:
            data = mesh.interleaved()
            self.setStride(8 * 4)
            self.setVertexData(QByteArray(data.tobytes()))
            self.setIndexData(QByteArray(mesh.indices.astype(np.uint32).tobytes()))
            self.setPrimitiveType(QQuick3DGeometry.PrimitiveType.Triangles)
            attr = QQuick3DGeometry.Attribute
            self.addAttribute(attr.Semantic.PositionSemantic, 0, attr.ComponentType.F32Type)
            self.addAttribute(attr.Semantic.NormalSemantic, 12, attr.ComponentType.F32Type)
            self.addAttribute(attr.Semantic.TexCoord0Semantic, 24, attr.ComponentType.F32Type)
            self.addAttribute(attr.Semantic.IndexSemantic, 0, attr.ComponentType.U32Type)
            lo, hi = mesh.positions.min(axis=0), mesh.positions.max(axis=0)
            self.setBounds(QVector3D(*map(float, lo)), QVector3D(*map(float, hi)))
        self.update()


class CarListModel(QAbstractListModel):
    """One row per car for the QML Repeater3D."""

    KEYS = ("code", "colour", "px", "py", "pz", "yaw", "selected")
    ROLES = {Qt.UserRole + i: key.encode() for i, key in enumerate(KEYS)}

    def __init__(self):
        super().__init__()
        self._rows = []

    def rowCount(self, parent=QModelIndex()):  # noqa: N802
        return 0 if parent.isValid() else len(self._rows)

    def roleNames(self):  # noqa: N802
        return self.ROLES

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or role not in self.ROLES:
            return None
        return self._rows[index.row()][self.ROLES[role].decode()]

    def set_rows(self, rows):
        if [r["code"] for r in rows] != [r["code"] for r in self._rows]:
            self.beginResetModel()
            self._rows = rows
            self.endResetModel()
        elif rows:
            self._rows = rows
            self.dataChanged.emit(self.index(0), self.index(len(rows) - 1))


def _bridge_property(key, kind, signal):
    """Read-only Qt property backed by SceneBridge._v[key]."""
    return Property(kind, lambda self: self._v[key], notify=signal)


class SceneBridge(QObject):
    """State shared with the QML scene."""

    changed = Signal()

    def __init__(self):
        super().__init__()
        self._v = {
            "ready": False, "status": "Waiting for track data… play the replay for a moment.",
            "cameraMode": "orbit", "centre": QVector3D(), "radius": 1500.0,
            "groundSource": QUrl(), "hasGround": False, "hasBuildings": False,
            "followPos": QVector3D(), "followYaw": 0.0, "followCode": "",
            "carScale": CAR_SCALE["orbit"], "corners": [], "showCorners": True,
            "mapCredit": "", "startPos": QVector3D(), "startYaw": 0.0,
            "trackHalfWidth": t3d.TRACK_HALF_WIDTH_M, "outerLevel": 0.0,
        }

    def set(self, **values):
        self._v.update(values)
        self.changed.emit()

    def get(self, key):
        return self._v[key]

    ready = _bridge_property("ready", bool, changed)
    status = _bridge_property("status", str, changed)
    cameraMode = _bridge_property("cameraMode", str, changed)
    centre = _bridge_property("centre", QVector3D, changed)
    radius = _bridge_property("radius", float, changed)
    groundSource = _bridge_property("groundSource", QUrl, changed)
    hasGround = _bridge_property("hasGround", bool, changed)
    hasBuildings = _bridge_property("hasBuildings", bool, changed)
    followPos = _bridge_property("followPos", QVector3D, changed)
    followYaw = _bridge_property("followYaw", float, changed)
    followCode = _bridge_property("followCode", str, changed)
    carScale = _bridge_property("carScale", float, changed)
    corners = _bridge_property("corners", "QVariantList", changed)
    showCorners = _bridge_property("showCorners", bool, changed)
    mapCredit = _bridge_property("mapCredit", str, changed)
    startPos = _bridge_property("startPos", QVector3D, changed)
    startYaw = _bridge_property("startYaw", float, changed)
    trackHalfWidth = _bridge_property("trackHalfWidth", float, changed)
    outerLevel = _bridge_property("outerLevel", float, changed)

    @Slot(str)
    def setCameraMode(self, mode):  # noqa: N802
        if mode in CAR_SCALE:
            self.set(cameraMode=mode, carScale=CAR_SCALE[mode])


class _Results(QObject):
    """Hands results from worker threads back to the GUI thread."""

    scene = Signal(object)
    ground = Signal(object)
    facts = Signal(object)


# ---------------------------------------------------------------------------
# Side panel widgets
# ---------------------------------------------------------------------------

class ElevationProfile(QWidget):
    """Elevation along the lap, with DRS zones, corners and every car."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(118)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.profile = None          # t3d.ElevationProfile over the resampled lap
        self.drs = []                # [(d0, d1)] metres
        self.corners = []            # [(d, label)]
        self.cars = []               # [(d, colour, highlighted)]
        self.exaggeration = 1.0

    def paintEvent(self, event):  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor(BG))
        if self.profile is None or len(self.profile.distance_m) < 2:
            p.setPen(QColor(MUTED))
            p.drawText(self.rect(), Qt.AlignCenter, "Elevation profile appears once the track data arrives")
            return

        left, right, top, bottom = 52, 16, 22, 24
        w = max(10, self.width() - left - right)
        h = max(10, self.height() - top - bottom)
        dist = self.profile.distance_m
        elev = self.profile.elevation_m
        total = float(dist[-1]) or 1.0
        span = max(float(elev.max()), 1.0)

        def px(d):
            return left + w * d / total

        def py(e):
            return top + h - h * e / span

        # DRS zones
        for d0, d1 in self.drs:
            p.fillRect(QRectF(px(d0), top, max(2.0, px(d1) - px(d0)), h), QColor(34, 197, 94, 40))

        # Area under the profile
        path = QPainterPath(QPointF(px(dist[0]), top + h))
        for d, e in zip(dist, elev):
            path.lineTo(px(d), py(e))
        path.lineTo(px(dist[-1]), top + h)
        path.closeSubpath()
        grad = QLinearGradient(0, top, 0, top + h)
        grad.setColorAt(0, QColor(225, 6, 0, 120))
        grad.setColorAt(1, QColor(225, 6, 0, 10))
        p.fillPath(path, grad)
        line = QPainterPath(QPointF(px(dist[0]), py(elev[0])))
        for d, e in zip(dist[1:], elev[1:]):
            line.lineTo(px(d), py(e))
        p.setPen(QPen(QColor("#ff4d45"), 1.6))
        p.drawPath(line)

        # Axis labels
        p.setPen(QColor(MUTED))
        small = QFont()
        small.setPointSize(8)
        p.setFont(small)
        p.drawText(QRectF(0, top - 8, left - 6, 16), Qt.AlignRight | Qt.AlignVCenter, f"{span:.0f} m")
        p.drawText(QRectF(0, top + h - 8, left - 6, 16), Qt.AlignRight | Qt.AlignVCenter, "0 m")
        p.drawText(QRectF(left, 2, w, 16), Qt.AlignLeft | Qt.AlignVCenter,
                   "ELEVATION ALONG THE LAP  ·  DRS zones in green  ·  numbers = corners")

        # Corners (labels that would touch drop to a second row)
        last_x, row = -1e9, 0
        for d, label in sorted(self.corners):
            x = px(d)
            row = 1 - row if x - last_x < 16 else 0
            last_x = x
            p.setPen(QPen(QColor(255, 255, 255, 40), 1))
            p.drawLine(QPointF(x, top), QPointF(x, top + h))
            p.setPen(QColor(MUTED))
            p.drawText(QRectF(x - 14, top + h + 2 + row * 10, 28, 12), Qt.AlignCenter, label)

        # Cars
        for d, colour, highlighted in sorted(self.cars, key=lambda c: c[2]):
            e = float(np.interp(d, dist, elev))
            radius = 5.5 if highlighted else 3.8
            p.setPen(QPen(QColor("#ffffff") if highlighted else QColor(0, 0, 0, 160), 1.4 if highlighted else 1))
            p.setBrush(QColor(colour))
            p.drawEllipse(QPointF(px(d), py(e)), radius, radius)


def _card(title: str = ""):
    frame = QFrame()
    frame.setObjectName("card")
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(14, 12, 14, 12)
    layout.setSpacing(8)
    if title:
        label = QLabel(title.upper())
        label.setObjectName("cardTitle")
        layout.addWidget(label)
    return frame, layout


# ---------------------------------------------------------------------------
# Window
# ---------------------------------------------------------------------------

class Track3DWindow(PitWallWindow):
    """Pit wall insight: the circuit in 3D with live cars and track facts."""

    def __init__(self):
        self._latest = None
        self._geometry = None        # arrays used to place cars
        self._scene_started = False
        self._facts_started = False
        self._session_info = {}
        self._yaw = None
        self._yaw_at = 0.0
        self._results = _Results()
        super().__init__()
        self.setWindowTitle("F1 Race Replay - 3D Track Detail")
        self.resize(1440, 900)
        self._results.scene.connect(self._apply_scene)
        self._results.ground.connect(self._apply_ground)
        self._results.facts.connect(self._apply_facts)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update_cars)
        self._timer.start(33)

    def closeEvent(self, event):  # noqa: N802
        # Tear the QML scene down first so it never sees the Python objects disappear.
        self._timer.stop()
        self.quick.setSource(QUrl())
        super().closeEvent(event)

    # ── UI ───────────────────────────────────────────────────────────────

    def setup_ui(self):
        self.setStyleSheet(STYLE)
        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        top = QHBoxLayout()
        top.setSpacing(0)
        outer.addLayout(top, 1)

        # 3D view
        self.bridge = SceneBridge()
        self.geometries = {name: MeshGeometry() for name in ("ground", "track", "edge", "drs", "building")}
        self.cars = CarListModel()
        self.quick = QQuickWidget()
        self.quick.setResizeMode(QQuickWidget.SizeRootObjectToView)
        self.quick.setClearColor(QColor(BG))
        ctx = self.quick.rootContext()
        ctx.setContextProperty("bridge", self.bridge)
        ctx.setContextProperty("carModel", self.cars)
        for name, geom in self.geometries.items():
            ctx.setContextProperty(f"{name}Geometry", geom)
        self.quick.setSource(QUrl.fromLocalFile(QML_FILE))
        for error in self.quick.errors():
            print(f"3D view error: {error.toString()}")
        top.addWidget(self.quick, 1)

        # Track Detail panel
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(330)
        panel = QWidget()
        panel.setObjectName("root")
        scroll.setWidget(panel)
        top.addWidget(scroll)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        self.title = QLabel("Track Detail")
        self.title.setObjectName("title")
        self.title.setWordWrap(True)
        self.subtitle = QLabel("Waiting for the replay…")
        self.subtitle.setObjectName("subtitle")
        self.subtitle.setWordWrap(True)
        layout.addWidget(self.title)
        layout.addWidget(self.subtitle)

        follow_row = QHBoxLayout()
        follow_label = QLabel("Follow car")
        follow_label.setObjectName("rowLabel")
        self.follow_combo = QComboBox()
        self.follow_combo.addItem("Leader")
        self.follow_combo.setToolTip("Car used by the Follow camera. Clicking a driver on the "
                                     "replay's leaderboard also switches to them.")
        follow_row.addWidget(follow_label)
        follow_row.addWidget(self.follow_combo, 1)
        layout.addLayout(follow_row)

        stats, stats_layout = _card("Circuit")
        self.tiles = QGridLayout()
        self.tiles.setHorizontalSpacing(12)
        self.tiles.setVerticalSpacing(10)
        self.tile_values = {}
        for i, (key, label) in enumerate((
                ("length", "Lap length"), ("laps", "Race laps"),
                ("distance", "Race distance"), ("corners", "Corners"),
                ("relief", "Elevation range"), ("climb", "Climb per lap"))):
            box = QVBoxLayout()
            box.setSpacing(0)
            value = QLabel("—")
            value.setObjectName("tileValue")
            name = QLabel(label)
            name.setObjectName("tileLabel")
            box.addWidget(value)
            box.addWidget(name)
            self.tiles.addLayout(box, i // 2, i % 2)
            self.tile_values[key] = value
        stats_layout.addLayout(self.tiles)
        layout.addWidget(stats)

        conditions, cond_layout = _card("Conditions now")
        self.conditions = QLabel("—")
        self.conditions.setObjectName("rowValue")
        self.conditions.setWordWrap(True)
        cond_layout.addWidget(self.conditions)
        layout.addWidget(conditions)

        history, hist_layout = _card("History here")
        self.history = QLabel("Looking up past winners…")
        self.history.setObjectName("rowValue")
        self.history.setWordWrap(True)
        self.history.setTextFormat(Qt.RichText)
        hist_layout.addWidget(self.history)
        layout.addWidget(history)

        self.credits = QLabel("")
        self.credits.setObjectName("small")
        self.credits.setWordWrap(True)
        layout.addWidget(self.credits)
        layout.addStretch()

        # Elevation profile along the bottom
        self.profile = ElevationProfile()
        outer.addWidget(self.profile)

    # ── Stream ───────────────────────────────────────────────────────────

    def on_telemetry_data(self, data):
        self._latest = data
        if data.get("session_info"):
            self._session_info = data["session_info"]
            self._start_facts()
        geometry = data.get("track_geometry")
        if geometry and not self._scene_started:
            self._scene_started = True
            self.bridge.set(status="Building the 3D circuit…")
            threading.Thread(target=self._build_scene, args=(geometry, dict(self._session_info)),
                             name="track3d-scene", daemon=True).start()

    # ── Scene (worker thread) ────────────────────────────────────────────

    def _build_scene(self, geometry, session_info):
        try:
            x = np.asarray(geometry["x"], dtype=float)
            y = np.asarray(geometry["y"], dtype=float)
            z_raw = geometry.get("z")
            z = np.asarray(z_raw, dtype=float) if z_raw else np.zeros_like(x)
            if not np.isfinite(z).any():
                z = np.zeros_like(x)
            z = np.where(np.isfinite(z), z, np.nanmedian(z))

            rx, ry, rz = t3d.resample_loop(x, y, z, spacing=20.0)
            frame = t3d.TrackFrame(base=float(rz.min()), exaggeration=t3d.auto_exaggeration(rz))
            centre = frame.to_scene(rx, ry, rz)
            headings = t3d.track_headings(centre)
            half = t3d.TRACK_HALF_WIDTH_M

            track = t3d.ribbon_mesh(centre, half, lift=0.30)
            edge_l = t3d.ribbon_mesh(centre, 0.28, lift=0.42, offset=half - 0.45)
            edge_r = t3d.ribbon_mesh(centre, 0.28, lift=0.42, offset=-(half - 0.45))
            edges = _merge(edge_l, edge_r)

            # Distance along the original reference lap, to place DRS zones and corners.
            orig_d = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
            res_d = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(rx), np.diff(ry)))])
            scale_d = res_d[-1] / orig_d[-1] if orig_d[-1] else 1.0
            drs_spans, drs_meshes = [], []
            for start, end in geometry.get("drs_zones") or []:
                try:
                    d0, d1 = orig_d[int(start)] * scale_d, orig_d[int(end)] * scale_d
                except (IndexError, ValueError, TypeError):
                    continue
                sel = (res_d >= d0) & (res_d <= d1)
                if sel.sum() >= 2:
                    # Green stripes just inside both white lines, like the 2D map's DRS marking.
                    for side in (1, -1):
                        drs_meshes.append(t3d.ribbon_mesh(centre[sel], 0.45, lift=0.40,
                                                          offset=side * (half - 1.35), closed=False))
                    drs_spans.append((d0 * t3d.FASTF1_UNIT_M, d1 * t3d.FASTF1_UNIT_M))
            drs = _merge(*drs_meshes) if drs_meshes else t3d.Mesh.empty()

            # Terrain covering the same area the map is drawn for.
            from src.services.circuit_map import MARGIN_FRACTION, frame_for_track  # noqa: F401
            map_frame = frame_for_track(np.column_stack([x, y]), t3d.FASTF1_UNIT_M, 64, MAP_MARGIN)
            bounds = (map_frame.x0 * t3d.FASTF1_UNIT_M, map_frame.x1 * t3d.FASTF1_UNIT_M,
                      -map_frame.y1 * t3d.FASTF1_UNIT_M, -map_frame.y0 * t3d.FASTF1_UNIT_M)
            gx, gz, heights = t3d.heightfield(centre, bounds, resolution=300, sink_m=1.0)
            outer_level = float(heights.min()) - 3.0   # surroundings sit below all terrain
            heights = t3d.flatten_edges(heights, outer_level)

            # Corner labels sit a little above the road.
            tree_idx = t3d.nearest_on_track(np.column_stack([rx, ry]),
                                            np.array([[c["x"], c["y"]] for c in geometry.get("corners") or []]).reshape(-1, 2))
            corners, corner_marks = [], []
            for c, i in zip(geometry.get("corners") or [], tree_idx):
                pos = frame.to_scene([c["x"]], [c["y"]], [rz[i]])[0]
                label = f"{c['number']}{c.get('letter') or ''}"
                corners.append({"x": float(pos[0]), "y": float(pos[1] + 7.0), "z": float(pos[2]), "label": label})
                corner_marks.append((res_d[i] * t3d.FASTF1_UNIT_M, label))

            profile = t3d.elevation_profile(rx, ry, rz)
            self._results.scene.emit({
                "frame": frame, "rx": rx, "ry": ry, "centre": centre, "headings": headings,
                "track": track, "edges": edges, "drs": drs, "drs_spans": drs_spans,
                "grid": (gx, gz, heights), "map_frame": map_frame, "corners": corners,
                "outer_level": outer_level,
                "corner_marks": corner_marks, "profile": profile, "res_d_m": res_d * t3d.FASTF1_UNIT_M,
                "reference": np.column_stack([x, y]), "session_info": session_info,
            })
        except Exception as exc:  # never take the window down
            import traceback
            traceback.print_exc()
            self._results.scene.emit({"error": str(exc)})

    def _apply_scene(self, scene):
        if "error" in scene:
            self.bridge.set(status=f"Couldn't build the 3D circuit: {scene['error']}")
            return
        self._geometry = scene
        gx, gz, heights = scene["grid"]
        mf = scene["map_frame"]
        self.geometries["ground"].set_mesh(t3d.grid_mesh(gx, gz, heights, _uv_for_frame(mf.bounds)))
        self.geometries["track"].set_mesh(scene["track"])
        self.geometries["edge"].set_mesh(scene["edges"])
        self.geometries["drs"].set_mesh(scene["drs"])

        centre = scene["centre"]
        lo, hi = centre.min(axis=0), centre.max(axis=0)
        mid = (lo + hi) / 2
        radius = float(max(np.hypot(hi[0] - lo[0], hi[2] - lo[2]) / 2, 300.0))
        start = centre[0]
        self.bridge.set(
            ready=True, status="", centre=QVector3D(*map(float, mid)), radius=radius,
            outerLevel=float(scene["outer_level"]),
            corners=scene["corners"], startPos=QVector3D(float(start[0]), float(start[1]) + 0.1, float(start[2])),
            startYaw=float(scene["headings"][0]),
        )

        self.profile.profile = scene["profile"]
        self.profile.drs = scene["drs_spans"]
        self.profile.corners = scene["corner_marks"]
        self.profile.exaggeration = scene["frame"].exaggeration
        self.profile.update()
        self._update_panel_stats()

        threading.Thread(target=self._load_ground, args=(scene,), name="track3d-map", daemon=True).start()

    # ── Ground map + buildings (worker thread) ───────────────────────────

    def _load_ground(self, scene):
        try:
            from src.lib.settings import get_settings
            if not get_settings().osm_map_enabled:
                return
            from src.services.circuit_map import ATTRIBUTION, default_cache_dir, load_circuit_map_data, render_map
            info = scene["session_info"] or {}
            data = load_circuit_map_data(
                scene["reference"], 0.0, (0.0, 0.0),
                location=str(info.get("location", "")), country=str(info.get("country", "")),
                event_name=str(info.get("event_name", "")),
                target_px=MAP_TEXTURE_PX, margin_fraction=MAP_MARGIN)
            if data is None:
                print("3D view: no real-world outline matches this layout, plain ground only")
                return
            image = render_map(data.layers, data.frame, data.px_per_m, vignette=False).convert("RGB")
            image = _fade_edges_to(image, LAND_RGB)
            path = os.path.join(default_cache_dir(), f"{data.circuit_id}_3d_ground.png")
            image.save(path)

            gx, gz, heights = scene["grid"]
            unit = t3d.FASTF1_UNIT_M
            footprints, tall, bases = [], [], []
            buildings = [a for a in data.layers.areas if a.kind == "building" and a.outers]
            if len(buildings) > MAX_BUILDINGS:
                tree_pts = scene["reference"]
                from scipy.spatial import cKDTree
                tree = cKDTree(tree_pts)
                centroids = np.array([b.outers[0].mean(axis=0) for b in buildings])
                dist, _ = tree.query(centroids)
                buildings = [buildings[i] for i in np.argsort(dist)[:MAX_BUILDINGS]]
            for b in buildings:
                ring = b.outers[0]
                xz = np.column_stack([ring[:, 0] * unit, -ring[:, 1] * unit])
                footprints.append(xz)
                tall.append(float(b.height_m or DEFAULT_BUILDING_M))
            if footprints:
                cents = np.array([f.mean(axis=0) for f in footprints])
                bases = t3d.sample_heights(gx, gz, heights, cents) - 0.5
            mesh = t3d.extrude_buildings(footprints, tall, bases) if footprints else t3d.Mesh.empty()
            self._results.ground.emit({"path": path, "bounds": data.frame.bounds, "buildings": mesh,
                                       "credit": ATTRIBUTION, "count": len(footprints)})
        except Exception as exc:
            print(f"3D view: real-world map unavailable ({exc})")

    def _apply_ground(self, ground):
        if self._geometry is None:
            return
        gx, gz, heights = self._geometry["grid"]
        self.geometries["ground"].set_mesh(t3d.grid_mesh(gx, gz, heights, _uv_for_frame(ground["bounds"])))
        self.geometries["building"].set_mesh(ground["buildings"])
        self.bridge.set(groundSource=QUrl.fromLocalFile(os.path.abspath(ground["path"])), hasGround=True,
                        hasBuildings=ground["buildings"].vertex_count > 0, mapCredit=ground["credit"])
        self._update_credits()

    # ── Circuit history (worker thread) ──────────────────────────────────

    def _start_facts(self):
        if self._facts_started:
            return
        year, rnd = self._session_info.get("year"), self._session_info.get("round")
        if not year or not rnd:
            return
        self._facts_started = True

        def work():
            from src.services.circuit_facts import load_circuit_facts
            self._results.facts.emit(load_circuit_facts(year, rnd))

        threading.Thread(target=work, name="track3d-facts", daemon=True).start()

    def _apply_facts(self, facts):
        if not facts:
            self.history.setText("<span style='color:#8a909a'>History isn't available offline.</span>")
            return
        rows = []
        fastest = facts.get("fastest_lap")
        if facts.get("first_gp"):
            count = facts.get("races_before")
            before = f" ({count} races before this one)" if isinstance(count, int) else ""
            rows.append(f"First Grand Prix here: <b>{facts['first_gp']}</b>{before}")
        if facts.get("most_wins"):
            rows.append(f"Most wins: <b>{facts['most_wins']['driver']}</b> × {facts['most_wins']['wins']}")
        if fastest:
            m, s = divmod(float(fastest["time_s"]), 60)
            rows.append(f"Fastest race lap here: <b>{int(m)}:{s:06.3f}</b> · "
                        f"{fastest['driver']} ({fastest['season']})")
        winners = facts.get("winners") or []
        if winners:
            rows.append("<br><span style='color:#8a909a'>PAST WINNERS</span>")
            for w in winners[:8]:
                rows.append(f"<span style='color:#8a909a'>{w['season']}</span>&nbsp;&nbsp;{w['driver']} "
                            f"<span style='color:#8a909a'>· {w['team']}</span>")
        self.history.setText("<br>".join(rows) or "No earlier races here.")
        if facts.get("circuit_name"):
            self.title.setText(facts["circuit_name"])
            where = ", ".join(v for v in (facts.get("locality"), facts.get("country")) if v)
            self.subtitle.setText(where)
        self._facts = facts
        self._update_credits()

    # ── Live updates ─────────────────────────────────────────────────────

    def _update_panel_stats(self):
        info = self._session_info or {}
        scene = self._geometry
        if info and not getattr(self, "_facts", None):
            self.title.setText(info.get("circuit_name") or info.get("event_name") or "Track Detail")
            self.subtitle.setText(" · ".join(str(v) for v in (info.get("event_name"), info.get("year")) if v))
        length_m = info.get("circuit_length_m")
        if not length_m and scene is not None:
            length_m = float(scene["profile"].distance_m[-1])
        laps = info.get("total_laps")
        if length_m:
            self.tile_values["length"].setText(f"{float(length_m) / 1000:.3f} km")
        if laps:
            self.tile_values["laps"].setText(str(laps))
        if length_m and laps:
            self.tile_values["distance"].setText(f"{float(length_m) * int(laps) / 1000:.1f} km")
        if scene is not None:
            self.tile_values["corners"].setText(str(len(scene["corners"])) if scene["corners"] else "—")
            prof = scene["profile"]
            self.tile_values["relief"].setText(f"{prof.range_m:.0f} m")
            self.tile_values["climb"].setText(f"{prof.climb_m:.0f} m")
        self._update_credits()

    def _update_credits(self):
        parts = []
        if self._geometry is not None:
            parts.append(f"Elevation: FastF1 position data, shown ×{self._geometry['frame'].exaggeration:g} "
                         "taller so the relief is visible.")
        if self.bridge.get("hasGround"):
            parts.append("Map and buildings © OpenStreetMap contributors.")
        if getattr(self, "_facts", None):
            parts.append("History: Jolpica-F1 (Ergast).")
        self.credits.setText("  ".join(parts))

    def _update_conditions(self, weather):
        if not weather:
            return
        bits = []
        if weather.get("air_temp") is not None:
            bits.append(f"Air {weather['air_temp']:.1f}°C")
        if weather.get("track_temp") is not None:
            bits.append(f"Track {weather['track_temp']:.1f}°C")
        if weather.get("humidity") is not None:
            bits.append(f"Humidity {weather['humidity']:.0f}%")
        if weather.get("wind_speed") is not None:
            bits.append(f"Wind {weather['wind_speed']:.1f} m/s")
        if weather.get("rain_state"):
            bits.append("Raining" if weather["rain_state"] == "RAINING" else "Dry")
        self.conditions.setText("  ·  ".join(bits))

    def _update_cars(self):
        data, scene = self._latest, self._geometry
        if not data or scene is None:
            return
        frame = data.get("frame") or {}
        drivers = frame.get("drivers") or {}
        if not drivers:
            return
        self._update_conditions(frame.get("weather"))
        if self._session_info and self.tile_values["laps"].text() == "—":
            self._update_panel_stats()

        codes = sorted(drivers, key=lambda c: drivers[c].get("position") or 99)
        self._sync_follow_list(codes)
        follow = self._follow_code(data, codes)

        colours = data.get("driver_colors") or {}
        pts = np.array([[drivers[c].get("x", 0.0), drivers[c].get("y", 0.0)] for c in codes], dtype=float)
        idx = t3d.nearest_on_track(np.column_stack([scene["rx"], scene["ry"]]), pts)
        centre, headings, tf = scene["centre"], scene["headings"], scene["frame"]
        rows, marks = [], []
        for code, (x, y), i in zip(codes, pts, idx):
            pos = tf.to_scene([x], [y], [0.0])[0]
            py = float(centre[i][1]) + 0.12
            rows.append({"code": code, "colour": colours.get(code, "#cccccc"), "px": float(pos[0]),
                         "py": py, "pz": float(pos[2]), "yaw": float(headings[i]),
                         "selected": code == follow})
            marks.append((float(scene["res_d_m"][i]), colours.get(code, "#cccccc"), code == follow))
            if code == follow:
                self._set_follow(pos[0], py, pos[2], float(headings[i]), code)
        self.cars.set_rows(rows)
        self.profile.cars = marks
        self.profile.update()

    def _sync_follow_list(self, codes):
        existing = [self.follow_combo.itemText(i) for i in range(1, self.follow_combo.count())]
        if sorted(existing) != sorted(codes):
            current = self.follow_combo.currentText()
            self.follow_combo.blockSignals(True)
            self.follow_combo.clear()
            self.follow_combo.addItem("Leader")
            self.follow_combo.addItems(sorted(codes))
            if current in codes or current == "Leader":
                self.follow_combo.setCurrentText(current)
            self.follow_combo.blockSignals(False)

    def _follow_code(self, data, codes):
        selected = [c for c in (data.get("selected_drivers") or []) if c in codes]
        if selected and selected[-1] != getattr(self, "_last_selected", None):
            # A new click on the replay leaderboard wins over the dropdown.
            self._last_selected = selected[-1]
            self.follow_combo.setCurrentText(selected[-1])
        choice = self.follow_combo.currentText()
        return choice if choice in codes else codes[0]

    def _set_follow(self, x, y, z, yaw, code):
        # The camera sits exactly on the car (so the car stays put in the frame
        # at any playback speed) and eases its heading towards the car's, so it
        # swings round corners instead of snapping. Headings are unwrapped so
        # it never spins the long way round.
        now = time.monotonic()
        if self._yaw is None or code != self.bridge.get("followCode"):
            self._yaw = yaw
        else:
            target = self._yaw + (yaw - self._yaw + 180.0) % 360.0 - 180.0
            dt = min(0.5, max(0.0, now - self._yaw_at))
            self._yaw += (target - self._yaw) * (1.0 - math.exp(-dt / FOLLOW_TURN_S))
        self._yaw_at = now
        self.bridge.set(followPos=QVector3D(float(x), float(y), float(z)), followYaw=float(self._yaw),
                        followCode=code)


def _merge(*meshes: t3d.Mesh) -> t3d.Mesh:
    meshes = [m for m in meshes if m.vertex_count]
    if not meshes:
        return t3d.Mesh.empty()
    offsets = np.cumsum([0] + [m.vertex_count for m in meshes[:-1]])
    return t3d.Mesh(
        np.vstack([m.positions for m in meshes]),
        np.vstack([m.normals for m in meshes]),
        np.vstack([m.uvs for m in meshes]),
        np.concatenate([m.indices + o for m, o in zip(meshes, offsets)]).astype(np.uint32),
    )


def _fade_edges_to(image, rgb, band: float = 0.12):
    """Blend the outer ``band`` of the ground texture into a flat colour."""
    from PIL import Image
    w, h = image.size
    u = np.minimum(np.arange(w), np.arange(w)[::-1]) / max(1.0, band * w)
    v = np.minimum(np.arange(h), np.arange(h)[::-1]) / max(1.0, band * h)
    f = np.clip(np.minimum(u[None, :], v[:, None]), 0.0, 1.0)
    mask = Image.fromarray((f * f * (3 - 2 * f) * 255).astype(np.uint8))
    return Image.composite(image, Image.new("RGB", image.size, rgb), mask)


def _uv_for_frame(bounds):
    """Texture coordinates for the ground, matching the map image's world frame."""
    x0, x1, y0, y1 = bounds
    unit = t3d.FASTF1_UNIT_M

    def uv(gx, gz):
        wx, wy = gx / unit, -gz / unit
        return (wx - x0) / (x1 - x0), (wy - y0) / (y1 - y0)

    return uv


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("3D Track Detail")
    window = Track3DWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
