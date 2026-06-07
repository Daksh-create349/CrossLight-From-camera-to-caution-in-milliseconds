"""
tests/test_kalman_tracker.py
============================
Unit tests for KalmanTracker (constant-acceleration tracker in
kalman_tracker.py — the primary tracker used by main.py).

Uses only synthetic data; no camera or YOLO required.

Run with:
    pytest tests/test_kalman_tracker.py -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from kalman_tracker import KalmanTracker

FPS   = 30
DT    = 1.0 / FPS
SCALE = 0.05   # 1 pixel = 0.05 m  →  20 px = 1 m


@pytest.fixture
def H():
    return np.array([[SCALE,0,0],[0,SCALE,0],[0,0,1]], dtype=np.float64)

@pytest.fixture
def tracker():
    return KalmanTracker(dt=DT, max_missed=10)

@pytest.fixture
def frame():
    return np.zeros((720, 1280, 3), dtype=np.uint8)


def _car(x, y, w=60, h=40):
    return {"bbox": [x-w/2, y-h/2, x+w/2, y+h/2], "class_name": "car", "confidence": 0.9}


# ── Track creation ──────────────────────────────────────────────────────────

class TestTrackCreation:
    def test_first_detection_creates_one_track(self, tracker, frame, H):
        tracks = tracker.update([_car(200,300)], frame=frame, H=H)
        assert len(tracks) == 1

    def test_two_detections_create_two_tracks(self, tracker, frame, H):
        tracks = tracker.update([_car(100,100), _car(800,400)], frame=frame, H=H)
        assert len(tracks) == 2

    def test_track_id_is_positive_int(self, tracker, frame, H):
        tracks = tracker.update([_car(200,200)], frame=frame, H=H)
        assert isinstance(tracks[0]["track_id"], int) and tracks[0]["track_id"] > 0

    def test_required_keys_present(self, tracker, frame, H):
        tracks = tracker.update([_car(200,200)], frame=frame, H=H)
        for key in ("track_id","class","bbox","centroid_meters","speed_kmh","direction"):
            assert key in tracks[0], f"Missing key: {key}"

    def test_initial_speed_near_zero(self, tracker, frame, H):
        tracks = tracker.update([_car(200,200)], frame=frame, H=H)
        assert tracks[0]["speed_kmh"] < 5.0


# ── Track lifecycle ─────────────────────────────────────────────────────────

class TestTrackLifecycle:
    def test_track_persists_across_frames(self, tracker, frame, H):
        for i in range(5):
            tracks = tracker.update([_car(200+i, 300)], frame=frame, H=H)
        assert len(tracks) == 1

    def test_track_id_stays_stable(self, tracker, frame, H):
        first_id = None
        for i in range(8):
            tracks = tracker.update([_car(200+i, 300)], frame=frame, H=H)
            if first_id is None:
                first_id = tracks[0]["track_id"]
        assert tracks[0]["track_id"] == first_id

    def test_missed_counter_increments(self, tracker, frame, H):
        tracker.update([_car(200,200)], frame=frame, H=H)
        tracker.update([], frame=frame, H=H)
        assert tracker.tracks[0].missed == 1

    def test_track_deleted_after_max_missed(self, tracker, frame, H):
        tracker.update([_car(200,200)], frame=frame, H=H)
        for _ in range(11):
            tracks = tracker.update([], frame=frame, H=H)
        assert len(tracks) == 0

    def test_track_recovers_after_one_miss(self, tracker, frame, H):
        tracker.update([_car(200,200)], frame=frame, H=H)
        tracker.update([], frame=frame, H=H)
        tracks = tracker.update([_car(201,200)], frame=frame, H=H)
        assert len(tracks) == 1 and tracker.tracks[0].missed == 0


# ── Speed estimation ─────────────────────────────────────────────────────────

class TestSpeedEstimation:
    def test_moving_car_speed_estimate(self, tracker, frame, H):
        """
        1 px/frame at 30 fps and SCALE=0.05 m/px gives 5.4 km/h.
        After Kalman settling, estimate should be within ±3 km/h.
        """
        speeds = []
        for i in range(60):
            tracks = tracker.update([_car(100+i, 300)], frame=frame, H=H)
            if tracks:
                speeds.append(tracks[0]["speed_kmh"])
        expected = 1.0 * SCALE * FPS * 3.6   # 5.4 km/h
        mean = float(np.mean(speeds[20:]))
        assert abs(mean - expected) < 3.0, f"Speed {mean:.2f} too far from {expected:.2f} km/h"

    def test_stationary_car_near_zero_speed(self, tracker, frame, H):
        for _ in range(40):
            tracks = tracker.update([_car(500,300)], frame=frame, H=H)
        assert tracks[0]["speed_kmh"] < 5.0

    def test_speed_non_negative(self, tracker, frame, H):
        for _ in range(10):
            tracks = tracker.update([_car(300,300)], frame=frame, H=H)
        assert tracks[0]["speed_kmh"] >= 0.0


# ── No-homography fallback ───────────────────────────────────────────────────

class TestNoHomography:
    def test_runs_without_H(self, frame):
        t = KalmanTracker(dt=DT)
        tracks = t.update([_car(200,200)], frame=frame, H=None)
        assert len(tracks) == 1 and "speed_kmh" in tracks[0]
