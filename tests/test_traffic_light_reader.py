"""
tests/test_traffic_light_reader.py
===================================
Unit tests for TrafficLightReader.analyze_crop().

All tests use synthetically generated OpenCV images (no camera / YOLO needed).
The crops are tiny 30×30 images with a bright circle at the centre, matching
the geometry expected by analyze_crop (radius = 0.4 * min(w, h) = 12 px).

Run with:
    pytest tests/test_traffic_light_reader.py -v
"""

import sys
import os

# Allow running from repo root or from tests/ subdirectory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import cv2
import pytest

from traffic_light_reader import TrafficLightReader


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_crop(bgr_circle_color, size=30):
    """
    Return a (size × size) BGR image with a dark background and a bright
    circle drawn at the centre.  The radius is sized so it falls within the
    circular ROI used by analyze_crop (radius = 0.4 * 30 = 12 px).
    """
    crop = np.full((size, size, 3), 20, dtype=np.uint8)   # dark grey casing
    cv2.circle(crop, (size // 2, size // 2), 10, bgr_circle_color, -1)
    return crop


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def reader():
    return TrafficLightReader()


@pytest.fixture
def red_crop():
    return _make_crop((0, 0, 255))       # BGR red


@pytest.fixture
def green_crop():
    return _make_crop((0, 255, 0))       # BGR green


@pytest.fixture
def yellow_crop():
    return _make_crop((0, 220, 255))     # BGR yellow  (HSV: ~30°, high S/V)


@pytest.fixture
def dark_crop():
    """Simulates a traffic light with no active bulb (all dark)."""
    return np.full((30, 30, 3), 15, dtype=np.uint8)


@pytest.fixture
def empty_crop():
    return np.zeros((0, 0, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Tests — analyze_crop
# ---------------------------------------------------------------------------

class TestAnalyzeCrop:

    def test_red_light_detected(self, reader, red_crop):
        state, counts = reader.analyze_crop(red_crop)
        assert state == "red", f"Expected 'red', got '{state}'"

    def test_green_light_detected(self, reader, green_crop):
        state, counts = reader.analyze_crop(green_crop)
        assert state == "green", f"Expected 'green', got '{state}'"

    def test_yellow_light_detected(self, reader, yellow_crop):
        state, counts = reader.analyze_crop(yellow_crop)
        assert state == "yellow", f"Expected 'yellow', got '{state}'"

    def test_dark_crop_returns_unknown(self, reader, dark_crop):
        """An all-dark image should not trigger any colour."""
        state, counts = reader.analyze_crop(dark_crop)
        assert state == "unknown", f"Expected 'unknown' for dark crop, got '{state}'"

    def test_empty_crop_returns_unknown(self, reader, empty_crop):
        """Zero-size crops must not crash and must return 'unknown'."""
        state, counts = reader.analyze_crop(empty_crop)
        assert state == "unknown"

    def test_none_crop_returns_unknown(self, reader):
        state, counts = reader.analyze_crop(None)
        assert state == "unknown"

    def test_counts_dict_has_expected_keys(self, reader, red_crop):
        _, counts = reader.analyze_crop(red_crop)
        assert "red" in counts
        assert "yellow" in counts
        assert "green" in counts
        assert "circle_area" in counts

    def test_counts_are_non_negative_integers(self, reader, green_crop):
        _, counts = reader.analyze_crop(green_crop)
        for key in ("red", "yellow", "green", "circle_area"):
            assert isinstance(counts[key], int), f"{key} should be int"
            assert counts[key] >= 0, f"{key} should be >= 0"

    def test_red_dominates_in_red_crop(self, reader, red_crop):
        _, counts = reader.analyze_crop(red_crop)
        assert counts["red"] > counts["green"], "Red px should exceed green px"
        assert counts["red"] > counts["yellow"], "Red px should exceed yellow px"

    def test_green_dominates_in_green_crop(self, reader, green_crop):
        _, counts = reader.analyze_crop(green_crop)
        assert counts["green"] > counts["red"]
        assert counts["green"] > counts["yellow"]


# ---------------------------------------------------------------------------
# Tests — get_state  (majority vote over detections list)
# ---------------------------------------------------------------------------

class TestGetState:

    def _make_frame_with_tl(self, bgr_circle, frame_h=100, frame_w=100):
        """
        Returns (frame, detections) where the detection bbox covers [5,5,35,35]
        and that region contains a synthetic traffic light crop.
        """
        frame = np.full((frame_h, frame_w, 3), 20, dtype=np.uint8)
        crop = _make_crop(bgr_circle, size=30)
        frame[5:35, 5:35] = crop

        detections = [{
            "class_name": "traffic light",
            "bbox": [5, 5, 35, 35],
            "confidence": 0.9,
        }]
        return frame, detections

    def test_get_state_red(self, reader):
        frame, detections = self._make_frame_with_tl((0, 0, 255))
        state = reader.get_state(detections, frame)
        assert state == "red"

    def test_get_state_green(self, reader):
        frame, detections = self._make_frame_with_tl((0, 255, 0))
        state = reader.get_state(detections, frame)
        assert state == "green"

    def test_get_state_no_tl_detections(self, reader):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        detections = [{"class_name": "car", "bbox": [10, 10, 50, 50], "confidence": 0.8}]
        state = reader.get_state(detections, frame)
        assert state == "unknown"

    def test_get_state_empty_detections(self, reader):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        state = reader.get_state([], frame)
        assert state == "unknown"

    def test_get_state_out_of_bounds_bbox_clipped(self, reader):
        """A detection bbox that extends beyond the frame boundary must not crash."""
        frame = np.zeros((50, 50, 3), dtype=np.uint8)
        detections = [{"class_name": "traffic light", "bbox": [-10, -10, 200, 200]}]
        state = reader.get_state(detections, frame)
        # May be unknown (all dark) — important thing is no exception
        assert state in ("red", "yellow", "green", "unknown")

    def test_majority_vote_two_reds_one_green(self, reader):
        """Majority vote: 2 red detections should beat 1 green."""
        frame = np.full((120, 120, 3), 20, dtype=np.uint8)
        red_crop   = _make_crop((0, 0, 255), size=30)
        green_crop = _make_crop((0, 255, 0), size=30)

        frame[5:35,   5:35]  = red_crop
        frame[5:35,   45:75] = red_crop
        frame[5:35,   85:115] = green_crop

        detections = [
            {"class_name": "traffic light", "bbox": [5, 5, 35, 35],   "confidence": 0.9},
            {"class_name": "traffic light", "bbox": [45, 5, 75, 35],  "confidence": 0.9},
            {"class_name": "traffic light", "bbox": [85, 5, 115, 35], "confidence": 0.9},
        ]
        state = reader.get_state(detections, frame)
        assert state == "red", f"Majority vote should return 'red', got '{state}'"
