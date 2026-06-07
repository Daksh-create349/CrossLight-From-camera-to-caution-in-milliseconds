"""
tests/test_risk_assessor.py
============================
Unit tests for RiskAssessor (v1 rectangle mode) and RiskAssessorV2
(polygon + homography mode).

No camera, YOLO, or OpenCV window needed — pure computation tests.

Run with:
    pytest tests/test_risk_assessor.py -v
"""
import sys, os, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from risk_assessor import RiskAssessor, RiskAssessorV2


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

SCALE = 0.05   # 1 pixel = 0.05 m

@pytest.fixture
def H():
    return np.array([[SCALE,0,0],[0,SCALE,0],[0,0,1]], dtype=np.float64)

@pytest.fixture
def poly():
    """4 m × 2 m crosswalk rectangle in metres (8–12 m ahead, 4–6 m right)."""
    return [(8.0,4.0),(12.0,4.0),(12.0,6.0),(8.0,6.0)]


# ---------------------------------------------------------------------------
# RiskAssessor (v1 — rectangle, pixel-space)
# ---------------------------------------------------------------------------

class TestRiskAssessorV1:
    """Tests for the legacy pixel-rectangle risk assessor."""

    @pytest.fixture
    def assessor(self):
        # Default danger zone: (200, 300, 1000, 600)
        return RiskAssessor()

    @pytest.fixture
    def car_entering(self):
        """Car centred at (600, 200) moving downward at 5 px/frame → will enter zone."""
        return {
            "track_id": 1, "class": "car",
            "centroid": (600.0, 200.0),
            "bbox": (550.0, 150.0, 650.0, 250.0),
            "speed": 5.0, "direction": (0.0, 5.0),
        }

    @pytest.fixture
    def pedestrian_inside(self):
        """Pedestrian already inside the danger zone."""
        return {
            "track_id": 2, "class": "person",
            "centroid": (400.0, 400.0),
            "bbox": (390.0, 370.0, 410.0, 430.0),
            "speed": 0.5, "direction": (0.5, 0.0),
        }

    def test_red_light_vehicle_triggers_intrusion(self, assessor, car_entering):
        events = assessor.assess([car_entering], "red")
        types = [e["type"] for e in events]
        assert "vehicle_intrusion" in types

    def test_green_light_no_events(self, assessor, car_entering):
        events = assessor.assess([car_entering], "green")
        assert len(events) == 0

    def test_yellow_light_triggers_intrusion(self, assessor, car_entering):
        events = assessor.assess([car_entering], "yellow")
        types = [e["type"] for e in events]
        assert "vehicle_intrusion" in types

    def test_pedestrian_risk_requires_vehicle_risk(self, assessor, pedestrian_inside):
        """Pedestrian event must not fire when there is no concurrent vehicle risk."""
        events = assessor.assess([pedestrian_inside], "red")
        types = [e["type"] for e in events]
        assert "pedestrian_danger" not in types

    def test_pedestrian_and_vehicle_both_fire(self, assessor, car_entering, pedestrian_inside):
        events = assessor.assess([car_entering, pedestrian_inside], "red")
        types = [e["type"] for e in events]
        assert "vehicle_intrusion" in types
        assert "pedestrian_danger" in types

    def test_confidence_in_valid_range(self, assessor, car_entering):
        events = assessor.assess([car_entering], "red")
        for e in events:
            assert 0.0 <= e["confidence"] <= 1.0

    def test_barrier_position_tuple(self, assessor, car_entering):
        events = assessor.assess([car_entering], "red")
        vi = next(e for e in events if e["type"] == "vehicle_intrusion")
        bp = vi["barrier_position"]
        assert bp is not None and len(bp) == 2

    def test_safe_car_not_flagged(self, assessor):
        """Car moving away from the danger zone should not trigger any event."""
        safe_car = {
            "track_id": 99, "class": "car",
            "centroid": (600.0, 700.0),     # below zone (y > 600)
            "bbox": (550.0, 650.0, 650.0, 750.0),
            "speed": 5.0, "direction": (0.0, 5.0),  # moving further down
        }
        events = assessor.assess([safe_car], "red")
        assert len(events) == 0

    def test_empty_tracks_returns_no_events(self, assessor):
        assert assessor.assess([], "red") == []


# ---------------------------------------------------------------------------
# RiskAssessorV2 (polygon + homography mode)
# ---------------------------------------------------------------------------

class TestRiskAssessorV2:

    @pytest.fixture
    def assessor(self, H, poly):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return RiskAssessorV2(H=H, danger_zone_polygon=poly)

    @pytest.fixture
    def car_approaching(self):
        """
        Car at (10 m, 2 m) moving at 10 km/h in +y direction.
        In 1 second it reaches ≈ (10, 4.78) — enters poly (y=4..6).
        """
        return {
            "track_id": 1, "class": "car",
            "centroid_meters": (10.0, 2.0),
            "speed_kmh": 10.0,
            "direction": (0.0, 1.0),
            "bbox": (180.0, 20.0, 220.0, 60.0),
        }

    @pytest.fixture
    def ped_inside(self):
        """Pedestrian already inside the polygon."""
        return {
            "track_id": 2, "class": "person",
            "centroid_meters": (10.0, 5.0),
            "speed_kmh": 3.0,
            "direction": (1.0, 0.0),
            "bbox": (190.0, 95.0, 210.0, 120.0),
        }

    @pytest.fixture
    def safe_car(self):
        """Car moving away from the polygon — should not trigger."""
        return {
            "track_id": 3, "class": "car",
            "centroid_meters": (0.5, 0.5),
            "speed_kmh": 5.0,
            "direction": (1.0, 0.0),
            "bbox": (0.0, 0.0, 40.0, 40.0),
        }

    # ── basic triggering ────────────────────────────────────────────────

    def test_red_triggers_vehicle_intrusion(self, assessor, car_approaching):
        events = assessor.assess([car_approaching], "red")
        assert any(e["type"] == "vehicle_intrusion" for e in events)

    def test_green_no_events(self, assessor, car_approaching):
        assert assessor.assess([car_approaching], "green") == []

    def test_yellow_triggers_intrusion(self, assessor, car_approaching):
        events = assessor.assess([car_approaching], "yellow")
        assert any(e["type"] == "vehicle_intrusion" for e in events)

    def test_safe_car_not_flagged(self, assessor, safe_car):
        events = assessor.assess([safe_car], "red")
        assert all(e.get("vehicle_id") != 3 for e in events)

    # ── pedestrian logic ─────────────────────────────────────────────────

    def test_pedestrian_fires_with_vehicle_risk(self, assessor, car_approaching, ped_inside):
        events = assessor.assess([car_approaching, ped_inside], "red")
        types = [e["type"] for e in events]
        assert "pedestrian_danger" in types

    def test_pedestrian_does_not_fire_alone(self, assessor, ped_inside):
        """No vehicle risk → no pedestrian event."""
        events = assessor.assess([ped_inside], "red")
        assert not any(e["type"] == "pedestrian_danger" for e in events)

    # ── output quality ───────────────────────────────────────────────────

    def test_confidence_in_valid_range(self, assessor, car_approaching):
        events = assessor.assess([car_approaching], "red")
        for e in events:
            assert 0.0 <= e["confidence"] <= 1.0

    def test_barrier_position_is_float_tuple(self, assessor, car_approaching):
        events = assessor.assess([car_approaching], "red")
        vi = next(e for e in events if e["type"] == "vehicle_intrusion")
        bp = vi["barrier_position"]
        assert bp is not None
        bx, by = bp
        assert isinstance(bx, float) and isinstance(by, float)

    def test_vehicle_id_correct(self, assessor, car_approaching):
        events = assessor.assess([car_approaching], "red")
        vi = next(e for e in events if e["type"] == "vehicle_intrusion")
        assert vi["vehicle_id"] == 1

    def test_empty_tracks_no_events(self, assessor):
        assert assessor.assess([], "red") == []

    # ── legacy fallback ──────────────────────────────────────────────────

    def test_legacy_mode_raises_deprecation_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            RiskAssessorV2()   # no H, no polygon
        assert any(issubclass(x.category, DeprecationWarning) for x in w)

    def test_legacy_mode_still_detects_intrusion(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            leg = RiskAssessorV2()
        car = {
            "track_id": 99, "class": "car",
            "centroid": (600.0, 200.0),
            "bbox": (550.0, 150.0, 650.0, 250.0),
            "speed": 5.0, "direction": (0.0, 5.0),
        }
        events = leg.assess([car], "red")
        assert len(events) >= 1
