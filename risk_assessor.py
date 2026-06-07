import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def _segment_intersect_s(Ax, Ay, Bx, By, Cx, Cy, Dx, Dy):
    """
    Return the parameter *s* ∈ [0,1] at which segment AB first crosses CD,
    or None if they do not intersect.

    Parametric form:
        AB(s) = A + s*(B-A),  s ∈ [0,1]
        CD(t) = C + t*(D-C),  t ∈ [0,1]
    Solving gives det = (D-C) × (B-A).
    """
    dABx, dABy = Bx - Ax, By - Ay
    dCDx, dCDy = Dx - Cx, Dy - Cy
    det = dCDx * dABy - dCDy * dABx          # cross product
    if abs(det) < 1e-9:
        return None                            # parallel / collinear
    s = (dCDx * (Cy - Ay) - dCDy * (Cx - Ax)) / det
    t = (dABx * (Cy - Ay) - dABy * (Cx - Ax)) / det
    if 0.0 <= s <= 1.0 and 0.0 <= t <= 1.0:
        return float(s)
    return None


def _poly_entry_s(p0, p1, poly):
    """
    Return the smallest *s* ∈ [0,1] at which segment p0→p1 crosses any edge
    of *poly* (list of (x,y) vertices, auto-closed).  Returns None if no edge
    is crossed.
    """
    n = len(poly)
    min_s = None
    Ax, Ay = p0
    Bx, By = p1
    for i in range(n):
        Cx, Cy = poly[i]
        Dx, Dy = poly[(i + 1) % n]
        s = _segment_intersect_s(Ax, Ay, Bx, By, Cx, Cy, Dx, Dy)
        if s is not None and (min_s is None or s < min_s):
            min_s = s
    return min_s


_EMA_ALPHA = 0.3   # smoothing factor for per-track confidence EMA


class RiskAssessor:
    """
    Assesses traffic risks by checking for vehicle intrusions into danger zones
    during restricted light phases and pedestrian exposure.
    """
    def __init__(self, frame_width=1280, frame_height=720, danger_zone=None):
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        if danger_zone is None:
            # Default to the lower middle part of the frame
            self.danger_zone = (200, 300, 1000, 600)
        else:
            self.danger_zone = danger_zone
            
        self.vehicle_classes = {'car', 'truck', 'bus', 'motorcycle'}
        self.pedestrian_classes = {'person', 'pedestrian'}
        self._ema_confidences = {}   # track_id → smoothed confidence (EMA)

    def assess(self, tracks, light_state):
        """
        Assesses risks for active tracks under the current light state.
        
        Args:
            tracks (list of dicts): list of {"track_id": int, "class": str, "centroid": (x,y),
                                            "bbox": (x1,y1,x2,y2), "speed": float, "direction": (dx,dy)}
            light_state (str): 'green', 'yellow', or 'red'
            
        Returns:
            list of dicts: Active risk events.
        """
        risk_events = []
        vehicle_risk_events = []
        light_active = light_state.lower() in {'red', 'yellow'}
        x1, y1, x2, y2 = self.danger_zone

        def _centroid(track):
            """Accept centroid_meters (KalmanTracker) or centroid (SimpleTracker)."""
            if "centroid_meters" in track:
                return float(track["centroid_meters"][0]), float(track["centroid_meters"][1])
            return float(track["centroid"][0]), float(track["centroid"][1])

        def _velocity_pixels(track):
            """
            Return (vx, vy) in pixels/frame for 1-second extrapolation (30 frames).
            KalmanTracker: direction is a unit vector, speed_kmh → px/frame via scale.
            SimpleTracker: direction is already (Δx, Δy) per frame.
            """
            dx, dy = float(track.get("direction", (0.0, 0.0))[0]), \
                     float(track.get("direction", (0.0, 0.0))[1])
            if "speed_kmh" in track:
                # Unit direction × speed_kmh → m/s → px/s (assume 20 px/m) → px/frame
                speed_ms  = track["speed_kmh"] / 3.6
                speed_pf  = (speed_ms * 20.0) / 30.0   # 20 px/m, 30 fps
                d_norm = np.hypot(dx, dy)
                if d_norm > 1e-9:
                    return dx / d_norm * speed_pf, dy / d_norm * speed_pf
                return 0.0, 0.0
            # SimpleTracker: direction already in px/frame
            return dx, dy

        # Rectangle polygon for geometry helper reuse
        rect_poly = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

        # Prune dead tracks from EMA dict
        active_ids = {t["track_id"] for t in tracks}
        self._ema_confidences = {k: v for k, v in self._ema_confidences.items()
                                  if k in active_ids}

        # 1. Assess vehicle intrusions
        for track in tracks:
            cls = track.get("class", "").lower()
            if cls not in self.vehicle_classes:
                continue

            cx, cy = _centroid(track)
            vx, vy = _velocity_pixels(track)

            # Predicted position after 1 second (30 frames at pixel velocity)
            ex = cx + 30.0 * vx
            ey = cy + 30.0 * vy

            # ── Time-to-danger calculation ─────────────────────────────
            currently_inside = (x1 <= cx <= x2) and (y1 <= cy <= y2)
            time_to_enter = None
            if currently_inside:
                time_to_enter = 0.0
            else:
                s_min = _poly_entry_s((cx, cy), (ex, ey), rect_poly)
                if s_min is not None:
                    # s is fraction of the 1-second trajectory → seconds to entry
                    time_to_enter = float(s_min)

            # No entry into zone predicted → skip
            if time_to_enter is None or not light_active:
                continue

            # Raw then EMA-smoothed confidence
            raw_conf = float(np.clip(1.0 - time_to_enter / 2.0, 0.0, 1.0))
            tid = track["track_id"]
            prev = self._ema_confidences.get(tid, raw_conf)
            smoothed = _EMA_ALPHA * raw_conf + (1.0 - _EMA_ALPHA) * prev
            self._ema_confidences[tid] = smoothed

            # Barrier position: 50 px ahead of bottom-centre along direction
            speed_pf = np.hypot(vx, vy)
            ux, uy = (vx / speed_pf, vy / speed_pf) if speed_pf > 1e-9 else (0.0, 0.0)
            bx1, by1, bx2, by2 = track["bbox"]
            bottom_cx = (bx1 + bx2) / 2.0
            bottom_cy = by2
            barrier_pos = (bottom_cx + 50.0 * ux, bottom_cy + 50.0 * uy)

            event = {
                "type":             "vehicle_intrusion",
                "vehicle_id":       tid,
                "pedestrian_id":    None,
                "barrier_position": barrier_pos,
                "confidence":       round(smoothed, 4),
                "time_to_enter_s":  round(time_to_enter, 4),
            }
            vehicle_risk_events.append(event)

        risk_events.extend(vehicle_risk_events)

        # 2. Assess pedestrian danger (only if a vehicle risk exists)
        if vehicle_risk_events:
            for track in tracks:
                cls = track.get("class", "").lower()
                if cls in self.pedestrian_classes:
                    cx, cy = _centroid(track)
                    vx, vy = _velocity_pixels(track)

                    currently_inside = (x1 <= cx <= x2) and (y1 <= cy <= y2)
                    ex = cx + 30.0 * vx
                    ey = cy + 30.0 * vy
                    moving_into = (x1 <= ex <= x2) and (y1 <= ey <= y2)

                    if currently_inside or moving_into:
                        event = {
                            "type":             "pedestrian_danger",
                            "vehicle_id":       None,
                            "pedestrian_id":    track["track_id"],
                            "barrier_position": None,
                            "confidence":       0.8,
                        }
                        risk_events.append(event)

        return risk_events


def _run_v1_tests():
    print("Testing RiskAssessor (v1) with hardcoded tracks and light states...")

    assessor = RiskAssessor()

    car_track = {
        "track_id": 101,
        "class": "car",
        "centroid": (600.0, 200.0),
        "bbox": (550.0, 150.0, 650.0, 250.0),
        "speed": 5.0,
        "direction": (0.0, 5.0),
    }
    pedestrian_track = {
        "track_id": 201,
        "class": "person",
        "centroid": (400.0, 400.0),
        "bbox": (390.0, 370.0, 410.0, 430.0),
        "speed": 0.5,
        "direction": (0.5, 0.0),
    }
    tracks = [car_track, pedestrian_track]

    print("\n--- Test 1: Light is RED ---")
    events_red = assessor.assess(tracks, "red")
    print(f"Risk events: {events_red}")
    assert len(events_red) == 2
    types = [e["type"] for e in events_red]
    assert "vehicle_intrusion" in types
    assert "pedestrian_danger" in types
    vehicle_event = [e for e in events_red if e["type"] == "vehicle_intrusion"][0]
    # Confidence is now a continuous time-to-danger score in [0,1].
    assert 0.0 < vehicle_event["confidence"] <= 1.0, \
        f"confidence out of range: {vehicle_event['confidence']}"
    assert "time_to_enter_s" in vehicle_event, "time_to_enter_s key missing"
    assert 0.0 <= vehicle_event["time_to_enter_s"] <= 2.0, \
        f"time_to_enter_s out of plausible range: {vehicle_event['time_to_enter_s']}"

    print("\n--- Test 2: Light is GREEN ---")
    events_green = assessor.assess(tracks, "green")
    print(f"Risk events: {events_green}")
    assert len(events_green) == 0

    print("\nAll RiskAssessor (v1) tests passed successfully!")



# ──────────────────────────────────────────────────────────────────────────────
# RiskAssessorV2 – Polygon + Homography Edition
# ──────────────────────────────────────────────────────────────────────────────

import cv2
import warnings
from danger_zone import point_in_danger_zone


class RiskAssessorV2:
    """
    Upgraded risk assessor that operates in real-world metre coordinates.

    Key improvements over RiskAssessor:
    - Danger zone is an arbitrary polygon (list of (x,y) in metres).
    - Uses the ray-casting ``point_in_danger_zone`` check instead of a
      simple rectangle test.
    - Trajectory prediction uses the velocity vector already estimated by the
      Kalman tracker (metres/second), so no FPS assumption is needed.
    - Barrier position is back-projected from metres → image pixels via the
      inverse homography, giving a pixel coordinate suitable for the projector.
    - Confidence is computed from the angular alignment between the vehicle's
      velocity vector and the vector pointing towards the danger-zone centroid.
    - Falls back to the legacy rectangle method when H / polygon are not
      supplied (deprecated path, kept for backward compatibility).

    Args:
        H (np.ndarray | None): 3×3 perspective-transform matrix that maps image
            pixel coordinates to real-world metres (from calibration.py).
        danger_zone_polygon (list[tuple] | None): Ordered (x, y) vertices of
            the danger zone in metres.  At least 3 vertices required.
        frame_width (int): Camera frame width in pixels (legacy fallback only).
        frame_height (int): Camera frame height in pixels (legacy fallback only).
        danger_zone_rect (tuple | None): (x1,y1,x2,y2) rectangle in pixels
            (legacy fallback only).
        lookahead_s (float): Prediction horizon in seconds (default 1.0).
        barrier_ahead_m (float): How far ahead of the vehicle's bottom centre
            (in metres) to place the barrier marker (default 1.5 m).
    """

    VEHICLE_CLASSES    = {'car', 'truck', 'bus', 'motorcycle'}
    PEDESTRIAN_CLASSES = {'person', 'pedestrian'}

    def __init__(
        self,
        H=None,
        danger_zone_polygon=None,
        frame_width=1280,
        frame_height=720,
        danger_zone_rect=None,
        lookahead_s=1.0,
        barrier_ahead_m=1.5,
    ):
        self.H = H
        self.H_inv = np.linalg.inv(H) if H is not None else None
        self.danger_zone_polygon = danger_zone_polygon  # list of (x,y) in metres
        self.lookahead_s = lookahead_s
        self.barrier_ahead_m = barrier_ahead_m

        # Pre-compute polygon centroid (used for confidence scoring)
        if danger_zone_polygon and len(danger_zone_polygon) >= 3:
            xs = [p[0] for p in danger_zone_polygon]
            ys = [p[1] for p in danger_zone_polygon]
            self._poly_cx = sum(xs) / len(xs)
            self._poly_cy = sum(ys) / len(ys)
        else:
            self._poly_cx = self._poly_cy = None

        self._ema_confidences = {}   # track_id → smoothed confidence (EMA)

        # Legacy fallback
        self._legacy = RiskAssessor(
            frame_width=frame_width,
            frame_height=frame_height,
            danger_zone=danger_zone_rect,
        )
        self._use_legacy = H is None or danger_zone_polygon is None
        if self._use_legacy:
            warnings.warn(
                "RiskAssessorV2 is running in legacy (rectangle) mode because "
                "no homography matrix or polygon danger zone was supplied. "
                "Provide 'H' and 'danger_zone_polygon' for the full v2 pipeline.",
                DeprecationWarning,
                stacklevel=2,
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _metres_to_pixels(self, mx, my):
        """Back-project a metre-space point to image pixel coordinates."""
        pt = np.array([[[mx, my]]], dtype=np.float32)
        px = cv2.perspectiveTransform(pt, self.H_inv)
        return float(px[0, 0, 0]), float(px[0, 0, 1])

    def _confidence(self, vx, vy, cx_m, cy_m):
        """
        Angular-alignment confidence.

        Returns 1.0 when the velocity vector points directly at the polygon
        centroid, falling linearly to 0.0 at 90° offset.
        """
        if self._poly_cx is None:
            return 0.8

        v_norm = np.hypot(vx, vy)
        dx = self._poly_cx - cx_m
        dy = self._poly_cy - cy_m
        d_norm = np.hypot(dx, dy)

        if v_norm < 1e-9 or d_norm < 1e-9:
            return 0.8

        cos_theta = np.clip((vx * dx + vy * dy) / (v_norm * d_norm), -1.0, 1.0)
        theta     = np.arccos(cos_theta)          # radians in [0, π]
        return float(np.clip(1.0 - theta / (np.pi / 2.0), 0.0, 1.0))

    def _get_centroid_metres(self, track):
        """
        Return (cx_m, cy_m) from the track dict.

        Priority:
        1. 'centroid_meters'  – provided by KalmanTracker
        2. 'centroid'         – if H available, project pixels → metres
        3. 'centroid'         – returned as-is (no H; units = pixels)
        """
        if 'centroid_meters' in track:
            return float(track['centroid_meters'][0]), float(track['centroid_meters'][1])
        cx_px, cy_px = track.get('centroid', (0.0, 0.0))
        if self.H is not None:
            pt = np.array([[[cx_px, cy_px]]], dtype=np.float32)
            m  = cv2.perspectiveTransform(pt, self.H)
            return float(m[0, 0, 0]), float(m[0, 0, 1])
        return float(cx_px), float(cy_px)

    def _get_velocity_ms(self, track):
        """
        Return (vx, vy) in metres/second.

        Priority:
        1. KalmanTracker provides 'direction' (unit vector) + 'speed_kmh'
        2. SimpleTracker  provides 'direction' (pixel delta/frame) + 'speed' (px/frame)
           → converted using a 0.05 m/px scale fallback (rough; calibrate properly)
        """
        direction = track.get('direction', (0.0, 0.0))
        dx, dy = float(direction[0]), float(direction[1])

        # KalmanTracker path: speed_kmh → m/s
        if 'speed_kmh' in track:
            speed_ms = track['speed_kmh'] / 3.6
            return dx * speed_ms, dy * speed_ms

        # SimpleTracker path: speed in px/frame, direction already per-frame delta
        speed_pf = track.get('speed', 0.0)   # pixels per frame
        FPS      = 30.0
        PX_PER_M = 20.0  # rough default: 1 m ≈ 20 px (replace with calibrated value)
        speed_ms = (speed_pf * FPS) / PX_PER_M
        # direction from SimpleTracker is (Δx, Δy) per frame, normalise first
        d_norm = np.hypot(dx, dy)
        if d_norm > 1e-9:
            ux, uy = dx / d_norm, dy / d_norm
        else:
            ux, uy = 0.0, 0.0
        return ux * speed_ms, uy * speed_ms

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def assess(self, tracks, light_state, timestamp=None):
        """
        Assess risk for a list of active tracks.

        Args:
            tracks (list[dict]): Active track dicts from KalmanTracker or
                SimpleTracker.  Required keys: 'track_id', 'class',
                'centroid' or 'centroid_meters', 'direction', 'bbox'.
                Optional keys: 'speed_kmh' (KalmanTracker) or 'speed'
                (SimpleTracker).
            light_state (str): 'green', 'yellow', or 'red'.
            timestamp (float | None): Wall-clock time (reserved; unused).

        Returns:
            list[dict]: Risk event dicts.  Each event has::

                {
                    "type":           "vehicle_intrusion" | "pedestrian_danger",
                    "vehicle_id":     int | None,
                    "pedestrian_id":  int | None,
                    "barrier_position": (px_x, px_y) in image pixels | None,
                    "confidence":     float in [0, 1],
                    "predicted_pos_m": (x_m, y_m) | None,   # metres
                }
        """
        # ── Legacy fallback ──────────────────────────────────────────────
        if self._use_legacy:
            return self._legacy.assess(tracks, light_state)

        risk_events          = []
        vehicle_risk_events  = []
        light_active = light_state.lower() in {'red', 'yellow'}

        # Prune dead tracks from EMA dict
        active_ids = {t["track_id"] for t in tracks}
        self._ema_confidences = {k: v for k, v in self._ema_confidences.items()
                                  if k in active_ids}

        # ── 1. Vehicle intrusion check ────────────────────────────────────
        for track in tracks:
            cls = track.get('class', '').lower()
            if cls not in self.VEHICLE_CLASSES:
                continue

            cx_m, cy_m = self._get_centroid_metres(track)
            vx, vy     = self._get_velocity_ms(track)

            # Predicted position 1 second ahead.
            # KalmanTrackerV2 pre-computes this; fall back to kinematics.
            if 'predicted_centroid_metres' in track:
                pred_x, pred_y = track['predicted_centroid_metres']
            elif 'predicted_centroid_meters' in track:
                pred_x, pred_y = track['predicted_centroid_meters']
            else:
                pred_x = cx_m + vx * self.lookahead_s
                pred_y = cy_m + vy * self.lookahead_s

            # ── Time-to-danger calculation ─────────────────────────────
            currently_inside = point_in_danger_zone(
                (cx_m, cy_m), self.danger_zone_polygon
            )
            time_to_enter = None
            if currently_inside:
                time_to_enter = 0.0
            else:
                s_min = _poly_entry_s(
                    (cx_m, cy_m), (pred_x, pred_y),
                    self.danger_zone_polygon
                )
                if s_min is not None:
                    time_to_enter = float(s_min)   # s ∈ [0,1] = seconds to entry

            # No zone entry predicted → skip
            if time_to_enter is None or not light_active:
                continue

            # Raw then EMA-smoothed confidence
            raw_conf = float(np.clip(1.0 - time_to_enter / 2.0, 0.0, 1.0))
            tid = track['track_id']
            prev = self._ema_confidences.get(tid, raw_conf)
            smoothed = _EMA_ALPHA * raw_conf + (1.0 - _EMA_ALPHA) * prev
            self._ema_confidences[tid] = smoothed

            # ── Barrier pixel position ─────────────────────────────────
            speed_ms = np.hypot(vx, vy)
            ux, uy = (vx / speed_ms, vy / speed_ms) if speed_ms > 1e-9 else (0.0, 0.0)
            barrier_m_x = cx_m + ux * self.barrier_ahead_m
            barrier_m_y = cy_m + uy * self.barrier_ahead_m
            barrier_px  = self._metres_to_pixels(barrier_m_x, barrier_m_y)

            event = {
                "type":             "vehicle_intrusion",
                "vehicle_id":       tid,
                "pedestrian_id":    None,
                "barrier_position": barrier_px,
                "confidence":       round(smoothed, 4),
                "time_to_enter_s":  round(time_to_enter, 4),
                "predicted_pos_m":  (pred_x, pred_y),
            }
            vehicle_risk_events.append(event)

        risk_events.extend(vehicle_risk_events)

        # ── 2. Pedestrian danger (only when a vehicle risk is active) ────
        if vehicle_risk_events:
            for track in tracks:
                cls = track.get('class', '').lower()
                if cls not in self.PEDESTRIAN_CLASSES:
                    continue

                cx_m, cy_m = self._get_centroid_metres(track)
                vx, vy     = self._get_velocity_ms(track)

                currently_inside = point_in_danger_zone(
                    (cx_m, cy_m), self.danger_zone_polygon
                )
                pred_x = cx_m + vx * self.lookahead_s
                pred_y = cy_m + vy * self.lookahead_s
                moving_into = point_in_danger_zone(
                    (pred_x, pred_y), self.danger_zone_polygon
                )

                if not (currently_inside or moving_into):
                    continue

                event = {
                    "type":             "pedestrian_danger",
                    "vehicle_id":       None,
                    "pedestrian_id":    track["track_id"],
                    "barrier_position": None,
                    "confidence":       0.8,
                    "predicted_pos_m":  (pred_x, pred_y),
                }
                risk_events.append(event)

        return risk_events


# ──────────────────────────────────────────────────────────────────────────────
# Standalone tests for RiskAssessorV2
# ──────────────────────────────────────────────────────────────────────────────

def _run_v2_tests():
    print("\n" + "=" * 60)
    print("Testing RiskAssessorV2 with mock data...")
    print("=" * 60)

    # ── Build a synthetic homography ──────────────────────────────────
    # Simple scaling: 1 pixel = 0.05 m  ↔  1 m = 20 px
    scale = 0.05
    H = np.array([
        [scale, 0.0,   0.0],
        [0.0,   scale, 0.0],
        [0.0,   0.0,   1.0],
    ], dtype=np.float64)

    # Crosswalk polygon in metres (a 4 m × 2 m rectangle 8–12 m ahead)
    poly = [(8.0, 4.0), (12.0, 4.0), (12.0, 6.0), (8.0, 6.0)]

    assessor = RiskAssessorV2(H=H, danger_zone_polygon=poly)

    # ── Track 1: Car approaching crosswalk (KalmanTracker format) ─────
    # Current position: 6 m ahead, moving at 30 km/h straight forward (+y)
    # In 1 second: 6 + (30/3.6)*1 ≈ 6 + 8.33 = 14.33 → outside (past zone)
    # Let's use 10 km/h instead: 6 + 2.78 = 8.78 → INSIDE zone ✓
    car_kalman = {
        "track_id":        1,
        "class":           "car",
        "centroid_meters": (10.0, 2.0),      # 10 m right, 2 m ahead of camera
        "speed_kmh":       10.0,              # 10 km/h ≈ 2.78 m/s
        "direction":       (0.0, 1.0),        # unit vector → moving forward (+y)
        "bbox":            (180.0, 20.0, 220.0, 60.0),
    }

    # ── Track 2: Pedestrian already inside the crosswalk polygon ─────
    ped = {
        "track_id":        2,
        "class":           "person",
        "centroid_meters": (10.0, 5.0),      # inside poly (8-12, 4-6)
        "speed_kmh":       3.0,
        "direction":       (1.0, 0.0),        # walking right
        "bbox":            (190.0, 95.0, 210.0, 120.0),
    }

    # ── Track 3: Car way off to the side – should NOT trigger ────────
    car_safe = {
        "track_id":        3,
        "class":           "car",
        "centroid_meters": (0.5, 0.5),
        "speed_kmh":       5.0,
        "direction":       (1.0, 0.0),        # moving right, away from poly
        "bbox":            (0.0, 0.0, 40.0, 40.0),
    }

    tracks = [car_kalman, ped, car_safe]

    # ── Test A: RED light → intrusion + pedestrian danger expected ────
    print("\n--- Test A: RED light ---")
    events = assessor.assess(tracks, "red")
    for e in events:
        print(f"  {e}")

    types = [e["type"] for e in events]
    assert "vehicle_intrusion" in types,    "Expected vehicle_intrusion on RED"
    assert "pedestrian_danger" in types,    "Expected pedestrian_danger on RED"

    vi = next(e for e in events if e["type"] == "vehicle_intrusion")
    assert vi["vehicle_id"] == 1,           "Wrong vehicle_id"
    assert vi["barrier_position"] is not None, "barrier_position must not be None"
    bx, by = vi["barrier_position"]
    assert isinstance(bx, float) and isinstance(by, float), \
        "barrier_position must be (float, float)"
    assert vi["confidence"] > 0.0,          "Confidence must be positive"
    print(f"  barrier_position (pixels): ({bx:.1f}, {by:.1f})")
    print(f"  confidence: {vi['confidence']:.3f}")

    pd_ev = next(e for e in events if e["type"] == "pedestrian_danger")
    assert pd_ev["pedestrian_id"] == 2,     "Wrong pedestrian_id"

    # safe car must NOT appear
    vehicle_ids = [e["vehicle_id"] for e in events if e["type"] == "vehicle_intrusion"]
    assert 3 not in vehicle_ids, "Safe car (id=3) must not appear in risk events"

    print("  [OK] Test A passed.")

    # ── Test B: GREEN light → no risk events ─────────────────────────
    print("\n--- Test B: GREEN light ---")
    events_g = assessor.assess(tracks, "green")
    print(f"  Events: {events_g}")
    assert len(events_g) == 0, "Expected no risk events on GREEN"
    print("  [OK] Test B passed.")

    # ── Test C: YELLOW light → same as RED ───────────────────────────
    print("\n--- Test C: YELLOW light ---")
    events_y = assessor.assess(tracks, "yellow")
    types_y = [e["type"] for e in events_y]
    assert "vehicle_intrusion" in types_y, "Expected vehicle_intrusion on YELLOW"
    print("  [OK] Test C passed.")

    # ── Test D: Legacy fallback when H=None ──────────────────────────
    print("\n--- Test D: Legacy fallback (no H / no polygon) ---")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        leg = RiskAssessorV2()          # no H, no polygon
    assert any(issubclass(x.category, DeprecationWarning) for x in w), \
        "Expected DeprecationWarning for legacy mode"
    # Build a SimpleTracker-style track for the legacy path
    legacy_tracks = [{
        "track_id": 99,
        "class":    "car",
        "centroid": (600.0, 200.0),
        "bbox":     (550.0, 150.0, 650.0, 250.0),
        "speed":    5.0,
        "direction": (0.0, 5.0),
    }]
    legacy_events = leg.assess(legacy_tracks, "red")
    assert len(legacy_events) >= 1, "Legacy path should detect intrusion"
    print(f"  Legacy events: {legacy_events}")
    print("  [OK] Test D passed.")

    print("\nAll RiskAssessorV2 tests passed successfully!")


if __name__ == '__main__':
    _run_v1_tests()
    _run_v2_tests()
