"""
kalman_tracker_v2.py  —  Experimental constant-velocity tracker
================================================================
This is an *alternative* implementation of the Kalman tracker.
It intentionally coexists with ``kalman_tracker.py`` because the two
modules make different motion-model trade-offs:

  kalman_tracker.py   (primary, used by main.py)
    • Constant-acceleration model: state = [x, y, vx, vy, ax, ay]
    • Better for vehicles that brake / accelerate rapidly
    • Provides ``predicted_bbox_pixels(H_inv)`` for skip-frame visualisation

  kalman_tracker_v2.py  (this file — experimental)
    • Constant-velocity model: state = [x, y, vx, vy]
    • Simpler / fewer parameters to tune; useful as a baseline comparison
    • Operates purely in metre-space from the start (no pixel-centroid path)

To switch the system to this tracker, replace the import in main.py:
    from kalman_tracker_v2 import KalmanTrackerV2 as KalmanTracker
and update the constructor call (it requires ``homography_matrix`` at init).
"""

import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment

# Try to import filterpy; fall back to a minimal numpy implementation if unavailable
try:
    from filterpy.kalman import KalmanFilter as FPKalmanFilter
    FILTERPY_AVAILABLE = True
except ImportError:
    FILTERPY_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Minimal numpy-based Kalman filter (fallback)
# ─────────────────────────────────────────────────────────────────────────────
class _NumpyKalmanFilter:
    """
    Minimal linear Kalman filter for state [x, y, vx, vy] in double-precision.
    Used only when filterpy is not installed.
    """
    def __init__(self, F, H, Q, R, P, x0):
        self.F = F.copy().astype(np.float64)   # State transition matrix
        self.H = H.copy().astype(np.float64)   # Measurement matrix
        self.Q = Q.copy().astype(np.float64)   # Process noise covariance
        self.R = R.copy().astype(np.float64)   # Measurement noise covariance
        self.P = P.copy().astype(np.float64)   # Estimate error covariance
        self.x = x0.copy().astype(np.float64)  # Initial state estimate

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        z = np.asarray(z, dtype=np.float64).reshape(-1, 1)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - self.H @ self.x)
        I = np.eye(self.P.shape[0], dtype=np.float64)
        self.P = (I - K @ self.H) @ self.P


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────

def _make_kalman(dt):
    """
    Builds and returns a Kalman filter for state [x, y, vx, vy]
    using constant-velocity motion model.
    """
    F = np.array([
        [1.0, 0.0, dt,  0.0],
        [0.0, 1.0, 0.0, dt ],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float64)

    H = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ], dtype=np.float64)

    # Process noise Q (tuned for city traffic dynamics)
    # Stdev of acceleration noise ~0.15 m/s^2. Q is discrete-time white noise model.
    q_val = 0.05
    Q = np.array([
        [(dt**3)/3, 0.0,       (dt**2)/2, 0.0      ],
        [0.0,       (dt**3)/3, 0.0,       (dt**2)/2],
        [(dt**2)/2, 0.0,       dt,        0.0      ],
        [0.0,       (dt**2)/2, 0.0,       dt       ],
    ], dtype=np.float64) * q_val

    # Measurement noise covariance R (error of ground contact localization in meters ~0.25m)
    R = np.eye(2, dtype=np.float64) * 0.06

    # Initial state covariance P (uncertainty: 10m on position, 25m/s on velocity)
    P = np.diag([10.0, 10.0, 25.0, 25.0]).astype(np.float64)

    if FILTERPY_AVAILABLE:
        kf = FPKalmanFilter(dim_x=4, dim_z=2)
        kf.F = F
        kf.H = H
        kf.Q = Q
        kf.R = R
        kf.P = P
    else:
        x0 = np.zeros((4, 1), dtype=np.float64)
        kf = _NumpyKalmanFilter(F, H, Q, R, P, x0)

    return kf


def _pixel_to_meters(px, py, H):
    """Transforms pixel coordinates to real world metre coordinates using H."""
    pt = np.array([[[px, py]]], dtype=np.float32)
    result = cv2.perspectiveTransform(pt, H)
    return float(result[0, 0, 0]), float(result[0, 0, 1])


def _iou(boxA, boxB):
    """Compute Intersection-over-Union between two [x1,y1,x2,y2] boxes."""
    ax1, ay1, ax2, ay2 = boxA
    bx1, by1, bx2, by2 = boxB
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    areaA = max(1e-6, (ax2 - ax1) * (ay2 - ay1))
    areaB = max(1e-6, (bx2 - bx1) * (by2 - by1))
    return inter / (areaA + areaB - inter)


def _extract_histogram(frame, bbox, bins=(8, 8, 8)):
    """Extract normalized HSV histogram for appearance model."""
    if frame is None:
        return None
    h_img, w_img = frame.shape[:2]
    x1 = int(max(0, bbox[0]))
    y1 = int(max(0, bbox[1]))
    x2 = int(min(w_img - 1, bbox[2]))
    y2 = int(min(h_img - 1, bbox[3]))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, list(bins), [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return hist.flatten()


def _histogram_similarity(h1, h2):
    if h1 is None or h2 is None:
        return 0.5
    dist = cv2.compareHist(
        h1.astype(np.float32).reshape(-1, 1),
        h2.astype(np.float32).reshape(-1, 1),
        cv2.HISTCMP_BHATTACHARYYA,
    )
    return 1.0 - float(np.clip(dist, 0.0, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# Track
# ─────────────────────────────────────────────────────────────────────────────

class _Track:
    _id_counter = 0

    def __init__(self, bbox, class_name, centroid_m, dt, histogram=None):
        _Track._id_counter += 1
        self.track_id = _Track._id_counter
        self.class_name = class_name
        self.bbox = bbox               # Last observed pixel bbox
        self.missed = 0
        self.age = 0
        self.histogram = histogram

        self.kf = _make_kalman(dt)
        mx, my = centroid_m
        # Initial state: [x, y, vx, vy]
        self.kf.x = np.array([[mx], [my], [0.0], [0.0]], dtype=np.float64)

    @property
    def state(self):
        return self.kf.x.flatten()

    def predict(self):
        self.kf.predict()

    def update_kf(self, measurement_m):
        self.kf.update(np.array([[measurement_m[0]], [measurement_m[1]]], dtype=np.float64))


# ─────────────────────────────────────────────────────────────────────────────
# KalmanTrackerV2
# ─────────────────────────────────────────────────────────────────────────────

class KalmanTrackerV2:
    """
    Constant-velocity Kalman filter multi-object tracker operating in metre-space.
    Uses original pixel-space bounding boxes for IoU matching.
    """
    def __init__(self, homography_matrix, dt=1 / 30.0, max_missed=10,
                 iou_weight=0.7, appearance_weight=0.3, min_iou=0.1):
        self.H = homography_matrix.astype(np.float64)
        self.dt = float(dt)
        self.max_missed = max_missed
        self.iou_weight = iou_weight
        self.appearance_weight = appearance_weight
        self.min_iou = min_iou
        self.tracks: list[_Track] = []
        _Track._id_counter = 0

    def update(self, detections, frame, timestamp=None):
        """
        Runs one cycle of the Kalman tracker.

        Args:
            detections (list[dict]): Each dict must have 'bbox' and 'class_name'.
            frame (np.ndarray | None): Current frame (used for histograms).
            timestamp (float | None): Not used, kept for API compatibility.

        Returns:
            list[dict]: Active tracks.
        """
        # ── 1. Predict step for all active tracks ───────────────────
        for t in self.tracks:
            t.predict()

        # Precompute real-world centroids for all incoming detections
        det_world_coords = []
        det_histograms = []
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cx_px = (x1 + x2) / 2.0
            cy_px = y2  # ground contact (bottom-centre)
            mx, my = _pixel_to_meters(cx_px, cy_px, self.H)
            det_world_coords.append((mx, my))
            det_histograms.append(_extract_histogram(frame, det["bbox"]))

        # ── 2. Association ──────────────────────────────────────────
        n_tracks = len(self.tracks)
        n_dets = len(detections)

        matched_track_ids = set()
        matched_det_ids = set()

        if n_tracks > 0 and n_dets > 0:
            cost = np.ones((n_tracks, n_dets), dtype=np.float64)

            for ti, track in enumerate(self.tracks):
                for di, det in enumerate(detections):
                    if det["class_name"] != track.class_name:
                        cost[ti, di] = 1.0
                        continue

                    # Match in pixel space (IoU on bounding box)
                    iou_score = _iou(track.bbox, det["bbox"])
                    app_score = _histogram_similarity(track.histogram, det_histograms[di])

                    cost[ti, di] = 1.0 - (
                        self.iou_weight * iou_score
                        + self.appearance_weight * app_score
                    )

            row_ind, col_ind = linear_sum_assignment(cost)

            for ti, di in zip(row_ind, col_ind):
                # Check min IoU
                iou_score = _iou(self.tracks[ti].bbox, detections[di]["bbox"])
                if iou_score < self.min_iou:
                    continue

                matched_track_ids.add(ti)
                matched_det_ids.add(di)

                # Update Kalman Filter in meters
                mx, my = det_world_coords[di]
                track = self.tracks[ti]
                track.update_kf((mx, my))
                
                # Update track metadata
                track.bbox = detections[di]["bbox"]
                track.missed = 0
                track.age += 1
                if det_histograms[di] is not None:
                    track.histogram = det_histograms[di]

        # ── 3. Handle unmatched tracks ──────────────────────────────
        for ti, track in enumerate(self.tracks):
            if ti not in matched_track_ids:
                track.missed += 1

        # ── 4. Create new tracks ────────────────────────────────────
        for di, det in enumerate(detections):
            if di not in matched_det_ids:
                mx, my = det_world_coords[di]
                new_track = _Track(
                    bbox=det["bbox"],
                    class_name=det["class_name"],
                    centroid_m=(mx, my),
                    dt=self.dt,
                    histogram=det_histograms[di]
                )
                self.tracks.append(new_track)

        # ── 5. Remove dead tracks ───────────────────────────────────
        self.tracks = [t for t in self.tracks if t.missed <= self.max_missed]

        # ── 6. Build outputs ────────────────────────────────────────
        results = []
        for t in self.tracks:
            s = t.state  # [x, y, vx, vy]
            x_pos, y_pos = float(s[0]), float(s[1])
            vx, vy = float(s[2]), float(s[3])

            speed_ms = np.hypot(vx, vy)
            speed_kmh = speed_ms * 3.6

            # Velocity vector in m/s
            velocity = (vx, vy)

            # Predicted position 1 second ahead (vx * 1.0, vy * 1.0)
            pred_x = x_pos + vx * 1.0
            pred_y = y_pos + vy * 1.0

            results.append({
                "track_id": t.track_id,
                "class": t.class_name,
                "centroid_meters": (x_pos, y_pos),  # Support both name keys
                "centroid_metres": (x_pos, y_pos),
                "bbox": t.bbox,
                "speed_kmh": round(speed_kmh, 2),
                "velocity": velocity,
                "direction": (vx / speed_ms, vy / speed_ms) if speed_ms > 1e-6 else (0.0, 0.0),
                "predicted_centroid_metres": (pred_x, pred_y),
                "predicted_centroid_meters": (pred_x, pred_y),
                "missed": t.missed,
                "age": t.age,
            })

        return results


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic Test Block
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"filterpy available: {FILTERPY_AVAILABLE}")
    print("Running KalmanTrackerV2 synthetic trajectory test...\n")

    # Simple scaling: 1 pixel = 0.1 meters.
    # H = scale * Identity
    scale = 0.1
    H = np.array([
        [scale, 0.0,   0.0],
        [0.0,   scale, 0.0],
        [0.0,   0.0,   1.0]
    ], dtype=np.float64)

    # 10 frames at 30 fps (dt = 1/30s)
    dt = 1 / 30.0
    tracker = KalmanTrackerV2(homography_matrix=H, dt=dt)

    # Move a vehicle in a straight line:
    # Starts at pixel coordinates (100, 100), moves +10 pixels (1 meter) in X per frame.
    # Over 10 frames, it moves 100 pixels (10 meters) in 10 * (1/30) = 0.333 seconds.
    # Velocity = 10 meters / 0.333 seconds = 30 m/s (~108 km/h).
    # Wait, the user request says: "moves 1 metre in 10 frames (0.33 sec) -> speed ~3 m/s (~10.8 km/h)"
    # Let's match this exactly:
    # 1 meter in 10 frames.
    # Each frame, it moves (1 meter / 10 frames) = 0.1 meters.
    # In pixels, since 1 pixel = 0.1 meters, it moves exactly 1 pixel per frame.
    # Over 10 frames, it moves 10 pixels = 1.0 meter.
    # Time elapsed = 10 * (1/30) = 0.333 seconds.
    # Expected speed = 1.0 m / 0.333s = 3.0 m/s = 10.8 km/h.
    
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    print("Simulating diagonal motion of 1 pixel/frame for 30 frames (1 second)...")
    for frame_idx in range(30):
        # We place the bottom-centre pixel at (100 + frame_idx, 200)
        # So x1 = 100 + frame_idx - 10, x2 = 100 + frame_idx + 10, y2 = 200
        # This keeps the bottom-centre ground contact point exactly at (100 + frame_idx, 200)
        cx = 100.0 + frame_idx
        cy = 200.0
        
        # Bbox shape: width 20, height 30
        bbox = [cx - 10.0, cy - 30.0, cx + 10.0, cy]
        
        detections = [{
            "bbox": bbox,
            "class_name": "car",
            "confidence": 0.95
        }]
        
        tracks = tracker.update(detections, frame=dummy_frame)
        if tracks:
            t = tracks[0]
            if frame_idx % 5 == 0:
                print(f"  Frame {frame_idx:2d} | Centroid m: ({t['centroid_metres'][0]:.2f}, {t['centroid_metres'][1]:.2f}) | "
                      f"Speed: {t['speed_kmh']:.2f} km/h | Velocity: ({t['velocity'][0]:.2f}, {t['velocity'][1]:.2f}) m/s")

    # Final velocity check
    # The true speed is 3.0 m/s (10.8 km/h) in the X direction, 0.0 m/s in Y direction.
    # Verify that the tracked output is close to the expected value.
    last_track = tracker.update([], frame=dummy_frame)[0]
    print(f"\nFinal Tracked Speed: {last_track['speed_kmh']:.2f} km/h")
    print(f"Expected Speed: ~10.8 km/h")
    
    assert abs(last_track['speed_kmh'] - 10.8) < 1.0, f"Speed too far from 10.8 km/h: {last_track['speed_kmh']}"
    print("\n[PASS] KalmanTrackerV2 synthetic validation passed!")
