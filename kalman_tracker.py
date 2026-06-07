import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment

# Attempt to import filterpy; fall back to a minimal numpy implementation if unavailable
try:
    from filterpy.kalman import KalmanFilter as FPKalmanFilter
    FILTERPY_AVAILABLE = True
except ImportError:
    FILTERPY_AVAILABLE = False

# ──────────────────────────────────────────────
# Minimal numpy-based Kalman filter (fallback)
# ──────────────────────────────────────────────
class _NumpyKalmanFilter:
    """
    Minimal linear Kalman filter for state [x, y, vx, vy, ax, ay].
    Used only when filterpy is not installed.
    """
    def __init__(self, F, H, Q, R, P, x0):
        self.F = F.copy()   # State transition matrix
        self.H = H.copy()   # Measurement matrix
        self.Q = Q.copy()   # Process noise covariance
        self.R = R.copy()   # Measurement noise covariance
        self.P = P.copy()   # Estimate error covariance
        self.x = x0.copy()  # Initial state estimate

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        z = np.asarray(z).reshape(-1, 1)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - self.H @ self.x)
        I = np.eye(self.P.shape[0])
        self.P = (I - K @ self.H) @ self.P


# ──────────────────────────────────────────────
# Helper utilities
# ──────────────────────────────────────────────

def _make_kalman(dt):
    """
    Builds and returns a Kalman filter for state [x, y, vx, vy, ax, ay]
    using constant-acceleration motion model.

    Measurement: [x, y] (position only).
    """
    # State transition matrix (constant-acceleration kinematics)
    dt2 = 0.5 * dt * dt
    F = np.array([
        [1, 0, dt, 0,  dt2, 0  ],
        [0, 1, 0,  dt, 0,   dt2],
        [0, 0, 1,  0,  dt,  0  ],
        [0, 0, 0,  1,  0,   dt ],
        [0, 0, 0,  0,  1,   0  ],
        [0, 0, 0,  0,  0,   1  ],
    ], dtype=np.float64)

    # Measurement matrix (we only observe x and y)
    H = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
    ], dtype=np.float64)

    # Process noise covariance (tuned for moderate dynamics)
    q = 0.1
    Q = np.eye(6, dtype=np.float64) * q

    # Measurement noise covariance
    r = 1.0
    R = np.eye(2, dtype=np.float64) * r

    # Initial estimate covariance (high uncertainty)
    P = np.eye(6, dtype=np.float64) * 500.0

    if FILTERPY_AVAILABLE:
        kf = FPKalmanFilter(dim_x=6, dim_z=2)
        kf.F = F
        kf.H = H
        kf.Q = Q
        kf.R = R
        kf.P = P
    else:
        x0 = np.zeros((6, 1), dtype=np.float64)
        kf = _NumpyKalmanFilter(F, H, Q, R, P, x0)

    return kf


def _pixel_to_meters(px, py, H):
    """
    Applies perspective transform H to convert image pixel (px, py) → (mx, my) in meters.
    H is a 3×3 float64 homography matrix from cv2.getPerspectiveTransform.
    """
    pt = np.array([[[px, py]]], dtype=np.float32)
    result = cv2.perspectiveTransform(pt, H)
    return float(result[0, 0, 0]), float(result[0, 0, 1])


def _bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


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
    """
    Extracts a normalised 3D HSV histogram from the bbox region of frame.
    Returns a flat numpy array, or None if the crop is empty.
    """
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
    hist = cv2.calcHist(
        [hsv], [0, 1, 2], None, list(bins),
        [0, 180, 0, 256, 0, 256]
    )
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return hist.flatten()


def _histogram_similarity(h1, h2):
    """
    Returns a similarity score in [0, 1] between two histogram vectors
    using the Bhattacharyya coefficient (1 = identical, 0 = completely different).
    Falls back to 0.5 if either histogram is None.
    """
    if h1 is None or h2 is None:
        return 0.5
    # cv2.compareHist returns the Bhattacharyya *distance* (lower = more similar)
    dist = cv2.compareHist(
        h1.astype(np.float32).reshape(-1, 1),
        h2.astype(np.float32).reshape(-1, 1),
        cv2.HISTCMP_BHATTACHARYYA,
    )
    return 1.0 - float(np.clip(dist, 0.0, 1.0))


# ──────────────────────────────────────────────
# Track object
# ──────────────────────────────────────────────

class _Track:
    _id_counter = 0

    def __init__(self, bbox, class_name, centroid_m, dt, histogram=None):
        _Track._id_counter += 1
        self.track_id = _Track._id_counter
        self.class_name = class_name
        self.bbox = bbox               # Most recent observed bbox (pixels)
        self.missed = 0
        self.age = 0
        self.histogram = histogram     # Appearance model

        # Build Kalman filter
        self.kf = _make_kalman(dt)
        mx, my = centroid_m
        if FILTERPY_AVAILABLE:
            self.kf.x = np.array([[mx], [my], [0.], [0.], [0.], [0.]])
        else:
            self.kf.x = np.array([[mx], [my], [0.], [0.], [0.], [0.]])

    @property
    def state(self):
        """Return current Kalman state as flat array [x, y, vx, vy, ax, ay]."""
        if FILTERPY_AVAILABLE:
            return self.kf.x.flatten()
        else:
            return self.kf.x.flatten()

    def predict(self):
        self.kf.predict()

    def update_kf(self, measurement_m):
        mx, my = measurement_m
        if FILTERPY_AVAILABLE:
            self.kf.update(np.array([[mx], [my]]))
        else:
            self.kf.update(np.array([[mx], [my]]))

    def predicted_centroid_meters(self):
        s = self.state
        return s[0], s[1]

    def predicted_bbox_pixels(self, H_inv):
        """
        Predict pixel bbox by back-projecting Kalman-predicted meter centroid
        using the inverse homography H_inv.  Falls back to last known bbox if
        H_inv is None.
        """
        if H_inv is None:
            return self.bbox
        cx_m, cy_m = self.predicted_centroid_meters()
        pt = np.array([[[cx_m, cy_m]]], dtype=np.float32)
        px = cv2.perspectiveTransform(pt, H_inv)
        px_x, px_y = float(px[0, 0, 0]), float(px[0, 0, 1])
        bw = self.bbox[2] - self.bbox[0]
        bh = self.bbox[3] - self.bbox[1]
        return (px_x - bw / 2, px_y - bh / 2, px_x + bw / 2, px_y + bh / 2)


# ──────────────────────────────────────────────
# KalmanTracker
# ──────────────────────────────────────────────

class KalmanTracker:
    """
    Multi-object tracker using per-track Kalman filters (constant-acceleration model),
    Hungarian assignment, and HSV appearance re-identification.
    """

    def __init__(self, dt=1 / 30, max_missed=10, iou_weight=0.7,
                 appearance_weight=0.3, min_iou=0.1):
        """
        Args:
            dt (float): Nominal time step in seconds (default 1/30 for 30 fps).
            max_missed (int): Number of consecutive missed frames before a track is deleted.
            iou_weight (float): Weight for IoU component in the cost matrix.
            appearance_weight (float): Weight for appearance (histogram) component.
            min_iou (float): Minimum IoU for a match to be accepted.
        """
        self.dt = dt
        self.max_missed = max_missed
        self.iou_weight = iou_weight
        self.appearance_weight = appearance_weight
        self.min_iou = min_iou
        self.tracks: list[_Track] = []
        _Track._id_counter = 0   # Reset ID counter on new tracker instance

    # ------------------------------------------------------------------
    def update(self, detections, frame=None, H=None, frame_timestamp=None):
        """
        Run one tracking cycle.

        Args:
            detections (list[dict]): Each dict must have keys 'bbox' and 'class_name'.
                                     Optionally 'confidence'.
            frame (np.ndarray | None): BGR image of the current frame. Used for
                                       appearance histograms. Pass None to disable.
            H (np.ndarray | None): 3×3 perspective transform mapping pixel → meters.
                                   If None, pixel centroids are used as-is (unit = pixels).
            frame_timestamp (float | None): Wall-clock timestamp (unused currently,
                                             reserved for variable-dt extension).

        Returns:
            list[dict]: Active tracks with keys:
                track_id, class, bbox, centroid_meters, speed_kmh, direction.
        """
        # Compute inverse homography once per call (needed for back-projection)
        H_inv = None
        if H is not None:
            H_inv = np.linalg.inv(H)

        # ── 1. Predict step for every existing track ──────────────────
        for t in self.tracks:
            t.predict()

        # ── 2. Build cost matrix (tracks × detections) ────────────────
        n_tracks = len(self.tracks)
        n_dets = len(detections)

        matched_track_ids = set()
        matched_det_ids = set()

        if n_tracks > 0 and n_dets > 0:
            cost = np.ones((n_tracks, n_dets), dtype=np.float64)

            for ti, track in enumerate(self.tracks):
                pred_bbox = track.predicted_bbox_pixels(H_inv)
                for di, det in enumerate(detections):
                    # Class mismatch → maximum cost (never match)
                    if det["class_name"] != track.class_name:
                        cost[ti, di] = 1.0
                        continue

                    iou_score = _iou(pred_bbox, det["bbox"])

                    # Appearance similarity
                    det_hist = _extract_histogram(frame, det["bbox"])
                    app_score = _histogram_similarity(track.histogram, det_hist)

                    # Combined cost: lower is better
                    cost[ti, di] = 1.0 - (
                        self.iou_weight * iou_score
                        + self.appearance_weight * app_score
                    )

            # Hungarian assignment
            row_ind, col_ind = linear_sum_assignment(cost)

            for ti, di in zip(row_ind, col_ind):
                iou_score = _iou(
                    self.tracks[ti].predicted_bbox_pixels(H_inv),
                    detections[di]["bbox"],
                )
                if iou_score < self.min_iou:
                    continue   # Reject weak matches
                matched_track_ids.add(ti)
                matched_det_ids.add(di)

                det = detections[di]
                cx_px, cy_px = _bbox_center(det["bbox"])
                cx_m, cy_m = (
                    _pixel_to_meters(cx_px, cy_px, H) if H is not None
                    else (cx_px, cy_px)
                )

                t = self.tracks[ti]
                t.update_kf((cx_m, cy_m))
                t.bbox = det["bbox"]
                t.missed = 0
                t.age += 1
                # Update appearance histogram (exponential moving average via replacement)
                new_hist = _extract_histogram(frame, det["bbox"])
                if new_hist is not None:
                    t.histogram = new_hist

        # ── 3. Handle unmatched tracks ────────────────────────────────
        for ti, track in enumerate(self.tracks):
            if ti not in matched_track_ids:
                track.missed += 1

        # ── 4. Create new tracks for unmatched detections ─────────────
        for di, det in enumerate(detections):
            if di not in matched_det_ids:
                cx_px, cy_px = _bbox_center(det["bbox"])
                cx_m, cy_m = (
                    _pixel_to_meters(cx_px, cy_px, H) if H is not None
                    else (cx_px, cy_px)
                )
                hist = _extract_histogram(frame, det["bbox"])
                new_track = _Track(
                    bbox=det["bbox"],
                    class_name=det["class_name"],
                    centroid_m=(cx_m, cy_m),
                    dt=self.dt,
                    histogram=hist,
                )
                self.tracks.append(new_track)

        # ── 5. Remove dead tracks ──────────────────────────────────────
        self.tracks = [t for t in self.tracks if t.missed <= self.max_missed]

        # ── 6. Build output ───────────────────────────────────────────
        results = []
        for t in self.tracks:
            s = t.state  # [x, y, vx, vy, ax, ay]
            vx, vy = s[2], s[3]  # velocity in m/s (or px/s if no H)
            speed_ms = float(np.hypot(vx, vy))
            if H is None:
                # Nominal pixel-to-meter scale (0.05m/px) for realistic fallback speeds
                speed_ms = speed_ms * 0.05
            speed_kmh = speed_ms * 3.6  # m/s → km/h

            # Direction unit vector (dx, dy)
            if speed_ms > 1e-6:
                direction = (float(vx / speed_ms), float(vy / speed_ms))
            else:
                direction = (0.0, 0.0)

            results.append({
                "track_id": t.track_id,
                "class": t.class_name,
                "bbox": t.bbox,
                "centroid_meters": (float(s[0]), float(s[1])),
                "speed_kmh": round(speed_kmh, 2),
                "direction": direction,
                "missed": t.missed,
                "age": t.age,
            })

        return results


# ──────────────────────────────────────────────
# Standalone synthetic test
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print(f"filterpy available: {FILTERPY_AVAILABLE}")
    print("Running KalmanTracker synthetic trajectory test...\n")

    tracker = KalmanTracker(dt=1 / 30, max_missed=10)

    # Simulate a car moving diagonally at 1 pixel per frame
    # starting at (100, 100) with bbox size 60×40
    FPS = 30
    DURATION_S = 3       # Simulate 3 seconds
    N_FRAMES = FPS * DURATION_S

    # Synthetic homography: scale 1 pixel = 0.05 m (so 20 px = 1 m)
    # H maps (px, py) → (px * 0.05, py * 0.05)
    scale = 0.05
    H = np.array([
        [scale, 0,     0],
        [0,     scale, 0],
        [0,     0,     1],
    ], dtype=np.float64)

    print(f"Simulating {N_FRAMES} frames ({DURATION_S}s at {FPS} fps)...")
    print(f"Homography scale: 1 pixel = {scale} m\n")

    # A dummy blank frame (appearance histograms will return None but that's fine for testing)
    dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    speeds_kmh = []

    for frame_idx in range(N_FRAMES):
        # Move 1 pixel right and 1 pixel down per frame (constant velocity)
        ox = 100 + frame_idx
        oy = 100 + frame_idx

        # Add small Gaussian noise to simulate real detection jitter
        noise_x = np.random.normal(0, 1.0)
        noise_y = np.random.normal(0, 1.0)

        noisy_x = ox + noise_x
        noisy_y = oy + noise_y

        detections = [{
            "bbox": [noisy_x, noisy_y, noisy_x + 60, noisy_y + 40],
            "class_name": "car",
            "confidence": 0.9,
        }]

        tracks = tracker.update(detections, frame=dummy_frame, H=H)

        if tracks:
            t = tracks[0]
            speeds_kmh.append(t["speed_kmh"])
            if frame_idx % 15 == 0:
                print(f"  Frame {frame_idx:3d} | "
                      f"centroid_m=({t['centroid_meters'][0]:.3f}, {t['centroid_meters'][1]:.3f}) | "
                      f"speed={t['speed_kmh']:.2f} km/h | direction={t['direction']}")

    # -- Assertions --
    # 1. Exactly one track should exist
    assert len(tracker.tracks) == 1, f"Expected 1 track, got {len(tracker.tracks)}"

    # 2. After Kalman settling (skip first 10 frames), speed should be close to expected
    #    1 px/frame @ 30fps = 30 px/s = 30 * scale m/s = 1.5 m/s = 5.4 km/h
    expected_speed_kmh = 1.0 * scale * FPS * 3.6  # = 5.4
    settled_speeds = speeds_kmh[15:]
    mean_speed = float(np.mean(settled_speeds))
    print(f"\nExpected speed (Kalman-settled): {expected_speed_kmh:.2f} km/h")
    print(f"Measured mean speed (frames 15+): {mean_speed:.2f} km/h")
    assert abs(mean_speed - expected_speed_kmh) < 2.5, (
        f"Speed estimate too far from expected: {mean_speed:.2f} vs {expected_speed_kmh:.2f}"
    )

    # 3. Track ID should be 1 (first track)
    assert tracker.tracks[0].track_id == 1

    # 4. Verify track is not marked as missed
    assert tracker.tracks[0].missed == 0

    print("\nAll synthetic tests passed successfully!")
