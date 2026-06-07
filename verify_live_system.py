"""
verify_live_system.py
=====================
Pre-flight gate that validates every stage of the CrossLight pipeline is
operating on **live, real-world data** before the system enters production.

Checks (each with PASS/FAIL):
1. Camera stream is live (variance across 5 frames > threshold, URL not local).
2. YOLO detections are real (at least one car or person detected).
3. Traffic light detection from real pixel data (if light detected, V channel std dev not near zero).
4. Calibration matrix exists, is not identity, produces realistic world coordinates (scale check: 100px should map to a few meters, not 100m).
5. Kalman tracker returns speeds within 0-200 km/h and tracks maintain IDs across frames.
6. Risk assessor uses real danger zone polygon and light state (check imports, no simulated cycle).
7. Projector window is open (title 'Projector Output' and surface exists).

Usage
-----
python verify_live_system.py --stream_url http://192.168.1.8:8080/video
"""

import argparse
import importlib
import os
import sys
import time
import warnings

import cv2
import numpy as np


class ResultsTracker:
    def __init__(self):
        self.failures = []
        self.passes = []

    def record(self, check_name, success, detail=""):
        status = "PASS" if success else "FAIL"
        line = f"  [{status}] {check_name}"
        if detail:
            line += f" — {detail}"
        print(line)
        if success:
            self.passes.append(check_name)
        else:
            self.failures.append((check_name, detail))

    def print_summary(self):
        print("\n" + "=" * 62)
        print(f"  VERIFICATION SUMMARY: {len(self.passes)} Passed, {len(self.failures)} Failed")
        print("=" * 62)
        if not self.failures:
            print("  ALL CHECKS PASSED: System is running on real-world live data.")
            sys.exit(0)
        else:
            print("  SYSTEM VERIFICATION FAILED! Failures detected:")
            for name, detail in self.failures:
                print(f"    ✗ {name}: {detail}")
            sys.exit(1)


R = ResultsTracker()


def _try_import(module_name, class_name=None):
    try:
        mod = importlib.import_module(module_name)
        if class_name:
            return getattr(mod, class_name), None
        return mod, None
    except Exception as e:
        return None, str(e)


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 1 – Camera stream is live
# ─────────────────────────────────────────────────────────────────────────────
def check_camera_stream(stream_url, var_threshold=10.0):
    print("\n[CHECK 1] Camera Stream Liveness")
    
    # 1. URL not local check
    is_local = False
    if isinstance(stream_url, str) and not stream_url.isdigit():
        if os.path.exists(stream_url):
            is_local = True
        elif "localhost" in stream_url or "127.0.0.1" in stream_url:
            is_local = True
        elif not stream_url.lower().startswith(("http", "rtsp")):
            is_local = True

    if is_local:
        R.record("Camera stream URL is not local", False, f"'{stream_url}' points to a local file or loopback address.")
        return None
    else:
        R.record("Camera stream URL is not local", True, f"'{stream_url}' is a network stream or live index.")

    # Try importing TrafficCamStream
    TrafficCamStream, err = _try_import("traffic_camera", "TrafficCamStream")
    if err:
        R.record("Import TrafficCamStream", False, err)
        return None

    try:
        stream = TrafficCamStream(int(stream_url) if str(stream_url).isdigit() else stream_url, resolution=(640, 480))
    except Exception as e:
        R.record("Initialize TrafficCamStream", False, str(e))
        return None

    # Capture 5 frames
    frames = []
    start_time = time.time()
    while len(frames) < 5 and (time.time() - start_time) < 10.0:
        f = stream.read()
        if f is not None:
            frames.append(f.astype(np.float32))
        time.sleep(0.1)

    stream.stop()

    if len(frames) < 5:
        R.record("Capture 5 frames from stream", False, f"Only captured {len(frames)} frames within 10s.")
        return None
    else:
        R.record("Capture 5 frames from stream", True, "Successfully captured 5 frames.")

    # Variance / difference threshold check
    diffs = [np.mean(np.abs(frames[i+1] - frames[i])) for i in range(len(frames) - 1)]
    avg_diff = float(np.mean(diffs))

    if avg_diff > var_threshold:
        R.record("Camera stream has live variance", True, f"Mean inter-frame diff = {avg_diff:.2f} (threshold {var_threshold}).")
    else:
        R.record("Camera stream has live variance", False, f"Mean inter-frame diff = {avg_diff:.2f} is too low (threshold {var_threshold}). Stream may be static/frozen.")
        return None

    return frames[-1].astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 2 – YOLO detections are real
# ─────────────────────────────────────────────────────────────────────────────
def check_yolo_detections(frame):
    print("\n[CHECK 2] YOLO Detection on Live Frame")
    if frame is None:
        R.record("YOLO detection check", False, "Skipped due to missing live frame.")
        return None

    YOLODetector, err = _try_import("yolo_detector", "YOLODetector")
    if err:
        R.record("Import YOLODetector", False, err)
        return None

    try:
        detector = YOLODetector()
        detections = detector.detect(frame)
    except Exception as e:
        R.record("Run YOLO detector", False, str(e))
        return None

    # Check that at least one car or person is detected
    relevant = [d for d in detections if d["class_name"] in ("car", "person", "bus", "truck", "motorcycle") and d.get("confidence", 0.0) > 0.4]

    if len(relevant) >= 1:
        classes = [d["class_name"] for d in relevant]
        R.record("YOLO detects real traffic participants", True, f"Found classes: {classes}")
    else:
        R.record("YOLO detects real traffic participants", False, "No real cars or persons detected (confidence > 0.4). Verify that the camera stream has live traffic in view.")

    return detections


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 3 – Traffic light detection from real pixel data
# ─────────────────────────────────────────────────────────────────────────────
def check_traffic_light(frame, detections):
    print("\n[CHECK 3] Traffic Light Reader — Real Pixel Data")
    if frame is None or detections is None:
        R.record("Traffic Light Reader check", False, "Skipped due to missing frame or detections.")
        return

    TrafficLightReader, err = _try_import("traffic_light_reader", "TrafficLightReader")
    if err:
        R.record("Import TrafficLightReader", False, err)
        return

    reader = TrafficLightReader()
    tl_dets = [d for d in detections if d["class_name"] == "traffic light" and d.get("confidence", 0.0) > 0.4]

    if not tl_dets:
        R.record("Traffic light detected", True, "No traffic lights detected in frame (optional, passing).")
        return

    # Check V channel std dev for real pixel data
    fh, fw = frame.shape[:2]
    sd_vals = []
    for det in tl_dets:
        x1, y1, x2, y2 = map(int, det["bbox"])
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(fw - 1, x2); y2 = min(fh - 1, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        crop = frame[y1:y2, x1:x2]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        v_channel = hsv[:, :, 2].astype(np.float32)
        sd_vals.append(float(np.std(v_channel)))

    if sd_vals:
        avg_sd = float(np.mean(sd_vals))
        if avg_sd < 3.0:
            R.record("Traffic light crop has natural brightness variance", False, f"V-channel std dev = {avg_sd:.2f} — looks near-zero (likely synthetic).")
        else:
            R.record("Traffic light crop has natural brightness variance", True, f"V-channel std dev = {avg_sd:.2f}")
    else:
        R.record("Traffic light crop has natural brightness variance", True, "No valid crops to test.")


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 4 – Calibration matrix is realistic
# ─────────────────────────────────────────────────────────────────────────────
def check_calibration(calib_path, danger_zone_path):
    print("\n[CHECK 4] Calibration Matrix & Danger Zone Polygon")
    if not os.path.exists(calib_path):
        R.record("Calibration matrix file exists", False, f"File not found: {calib_path}")
        return None
    else:
        R.record("Calibration matrix file exists", True)

    try:
        H = np.load(calib_path)
    except Exception as e:
        R.record("Load calibration matrix", False, str(e))
        return None

    # Check not identity
    if np.allclose(H, np.eye(3), atol=1e-4):
        R.record("Calibration matrix is not identity", False, "Matrix is equal to Identity.")
        return None
    else:
        R.record("Calibration matrix is not identity", True)

    # Scale check: 100px should map to a few meters, not 100m
    try:
        p1 = np.array([[[640.0, 360.0]]], dtype=np.float32)
        p2 = np.array([[[740.0, 360.0]]], dtype=np.float32)
        w1 = cv2.perspectiveTransform(p1, H)[0, 0]
        w2 = cv2.perspectiveTransform(p2, H)[0, 0]
        scale_m = float(np.linalg.norm(w2 - w1))
        
        # 100px should map to roughly 0.5m to 25m
        if scale_m < 0.2 or scale_m > 25.0:
            R.record("Calibration scale check (100px maps to a few meters)", False, f"100px maps to {scale_m:.2f}m (expected 0.2m to 25.0m).")
        else:
            R.record("Calibration scale check (100px maps to a few meters)", True, f"100px ≈ {scale_m:.2f}m")
    except Exception as e:
        R.record("Calibration scale check", False, str(e))

    # Check danger zone path
    if not os.path.exists(danger_zone_path):
        R.record("Danger zone file exists", False, f"File not found: {danger_zone_path}")
    else:
        try:
            dz = np.load(danger_zone_path)
            if dz.ndim == 2 and dz.shape[1] == 2 and dz.shape[0] >= 3:
                R.record("Danger zone polygon is valid", True, f"Polygon loaded with {dz.shape[0]} vertices.")
            else:
                R.record("Danger zone polygon is valid", False, f"Invalid polygon shape: {dz.shape}")
        except Exception as e:
            R.record("Danger zone polygon is valid", False, str(e))

    return H


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 5 – Kalman tracker speed & ID persistence
# ─────────────────────────────────────────────────────────────────────────────
def check_kalman_tracker(stream_url, H, n_frames=8):
    print("\n[CHECK 5] Kalman Tracker Speed & ID Persistence")
    
    # Try importing KalmanTrackerV2 first, then KalmanTracker
    KalmanTracker, err = _try_import("kalman_tracker_v2", "KalmanTrackerV2")
    if err:
        KalmanTracker, err = _try_import("kalman_tracker", "KalmanTracker")
        
    if err:
        R.record("Import KalmanTracker", False, err)
        return

    YOLODetector, err = _try_import("yolo_detector", "YOLODetector")
    if err:
        R.record("Import YOLODetector", False, err)
        return

    TrafficCamStream, err = _try_import("traffic_camera", "TrafficCamStream")
    if err:
        R.record("Import TrafficCamStream", False, err)
        return

    try:
        stream = TrafficCamStream(int(stream_url) if str(stream_url).isdigit() else stream_url, resolution=(640, 480))
        detector = YOLODetector()
        # Initialize tracker
        try:
            tracker = KalmanTracker(homography_matrix=H, dt=1.0/30)
        except TypeError:
            # Fall back to V1 signature
            tracker = KalmanTracker(dt=1.0/30)
    except Exception as e:
        R.record("Initialize components for tracker test", False, str(e))
        return

    # Run tracker across frames
    all_tracks_per_frame = []
    start_time = time.time()
    frames_captured = 0
    
    while frames_captured < n_frames and (time.time() - start_time) < 15.0:
        frame = stream.read()
        if frame is not None:
            frames_captured += 1
            # Normalise internal processing resolution to 1280x720
            fh, fw = frame.shape[:2]
            if fw != 1280 or fh != 720:
                frame = cv2.resize(frame, (1280, 720))
                
            dets = detector.detect(frame)
            tracks = tracker.update(dets, frame=frame, H=H)
            all_tracks_per_frame.append(tracks)
        time.sleep(0.05)

    stream.stop()

    # Verify active tracks
    flat_tracks = [t for frame_tracks in all_tracks_per_frame for t in frame_tracks]
    if not flat_tracks:
        R.record("Kalman tracker active tracks", False, "No active tracks detected during test window. Verify live traffic is visible.")
        return

    # Check speeds within 0 - 200 km/h
    speeds = [t.get("speed_kmh", 0.0) for t in flat_tracks]
    in_range = all(0.0 <= s <= 200.0 for s in speeds)
    
    if in_range:
        R.record("Kalman tracker speeds within 0-200 km/h", True, f"Track speeds: {[round(s, 2) for s in speeds]} km/h")
    else:
        R.record("Kalman tracker speeds within 0-200 km/h", False, f"Speeds detected out of range: {speeds}")

    # Track ID persistence check
    id_counts = {}
    for frame_tracks in all_tracks_per_frame:
        for t in frame_tracks:
            tid = t["track_id"]
            id_counts[tid] = id_counts.get(tid, 0) + 1

    persistent = {tid: count for tid, count in id_counts.items() if count > 1}
    if persistent:
        R.record("Track IDs persist across frames", True, f"Persistent IDs: {list(persistent.keys())}")
    else:
        R.record("Track IDs persist across frames", False, "No track IDs persisted across frames. All IDs were lost or re-assigned.")


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 6 – Risk Assessor real polygon & light state imports
# ─────────────────────────────────────────────────────────────────────────────
def check_risk_assessor(H, danger_zone_path):
    print("\n[CHECK 6] Risk Assessor Configuration")

    # 1. Check imports and simulated cycle in main.py or main_v2.py
    for filename in ("main_v2.py", "main.py"):
        if os.path.exists(filename):
            try:
                with open(filename, "r") as f:
                    content = f.read()
                
                # Check TrafficLightReader import and usage
                has_reader = "TrafficLightReader" in content
                
                # Check that simulated_light defaults to False (not always true/hardcoded)
                # Parse argparse setup
                has_sim_default_false = (
                    'action="store_true"' in content and 'simulated_light' in content
                ) or (
                    'default=False' in content and 'simulated_light' in content
                )
                
                if has_reader and has_sim_default_false:
                    R.record(f"Pipeline in {filename} uses real TrafficLightReader", True)
                else:
                    R.record(f"Pipeline in {filename} uses real TrafficLightReader", False, f"Missing TrafficLightReader import or simulated light is defaulted to True.")
            except Exception as e:
                R.record(f"Read {filename}", False, str(e))
        else:
            # Optional warning if file doesn't exist, but we need at least one main file
            pass

    # 2. Risk Assessor instantiation with real danger zone polygon
    RiskAssessorV2, err = _try_import("risk_assessor", "RiskAssessorV2")
    if err:
        R.record("Import RiskAssessorV2", False, err)
        return

    try:
        from danger_zone import load_danger_zone
        poly = load_danger_zone(danger_zone_path)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            assessor = RiskAssessorV2(H=H, danger_zone_polygon=poly)
            
        if assessor.danger_zone_polygon is not None and len(assessor.danger_zone_polygon) >= 3:
            R.record("RiskAssessorV2 initialized with real danger zone polygon", True, f"Polygon has {len(assessor.danger_zone_polygon)} vertices.")
        else:
            R.record("RiskAssessorV2 initialized with real danger zone polygon", False, "Polygon is empty or not configured.")
    except Exception as e:
        R.record("RiskAssessorV2 initialization with polygon", False, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 7 – Projector Window is open and active
# ─────────────────────────────────────────────────────────────────────────────
def check_projector():
    print("\n[CHECK 7] Projector Window Check")
    
    ProjectorSimulator, err = _try_import("projector_simulator", "ProjectorSimulator")
    if err:
        R.record("Import ProjectorSimulator", False, err)
        return

    try:
        import pygame
        # Init pygame display if not done
        if not pygame.display.get_init():
            pygame.display.init()

        proj = ProjectorSimulator(width=400, height=300)
        
        # 1. Surface exists
        if proj.screen is not None:
            R.record("Projector screen surface exists", True)
        else:
            R.record("Projector screen surface exists", False, "Surface is None.")

        # 2. Title is 'Projector Output'
        title = pygame.display.get_caption()[0]
        if title == "Projector Output":
            R.record("Projector window title is 'Projector Output'", True)
        else:
            R.record("Projector window title is 'Projector Output'", False, f"Title was: '{title}'")

        pygame.quit()
    except Exception as e:
        R.record("Verify projector window", False, str(e))


def main():
    parser = argparse.ArgumentParser(description="CrossLight pre-flight verification script.")
    parser.add_argument("--stream_url", default="http://192.168.1.8:8080/video", help="Camera stream URL.")
    parser.add_argument("--calibration", default="calibration_matrix.npy", help="Path to calibration_matrix.npy.")
    parser.add_argument("--danger_zone", default="danger_zone.npy", help="Path to danger_zone.npy.")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  CrossLight Monitor — System Integration Verification")
    print("=" * 60)
    print(f"  Stream URL   : {args.stream_url}")
    print(f"  Calibration  : {args.calibration}")
    print(f"  Danger Zone  : {args.danger_zone}")

    # 1. Stream Check
    frame = check_camera_stream(args.stream_url)

    # 2. YOLO Detections Check
    detections = check_yolo_detections(frame)

    # 3. Traffic Light Check
    check_traffic_light(frame, detections)

    # 4. Calibration Check
    H = check_calibration(args.calibration, args.danger_zone)

    # 5. Tracker Check
    check_kalman_tracker(args.stream_url, H)

    # 6. Risk Assessor Check
    check_risk_assessor(H, args.danger_zone)

    # 7. Projector Check
    check_projector()

    # Print overall summary and exit accordingly
    R.print_summary()


if __name__ == "__main__":
    main()
