"""
main_v2.py – CrossLight Monitor (standalone / no web-server variant)
=====================================================================
⚠  LEGACY ENTRY POINT — kept for reference only.
   Use `main.py` for the current version, which adds the async aiohttp
   web server, MJPEG /video_feed endpoint, and WebSocket /ws endpoint
   that the React frontend connects to.

   This file is NOT deleted because main_v2.py is a useful stripped-down
   variant for quick local testing without the web-server dependency.
   It is intentionally NOT the primary entry point.

Original description: Performance-optimized version of the main loop.

Optimizations:
1. Stream Captured at 640x480 resolution (for CPU and network efficiency).
2. YOLO detection is executed every N frames (default N=3); on skipped frames,
   we reuse the last detections and step the Kalman filters forward via prediction
   only, preventing tracking drift and eliminating duplicate measurement updates.
3. Resized output window (640x360 display frame) to save OpenCV drawing overhead.
4. Pygame Projector Simulator runs asynchronously in a daemon thread.
5. On-screen FPS and console statistics (every 30 frames).
"""

import argparse
import sys
import time
import warnings
import threading

import cv2
import numpy as np
import pygame

# ── Local modules ─────────────────────────────────────────────────────────────
from traffic_camera       import TrafficCamStream
from detector             import YOLODetector
from kalman_tracker       import KalmanTracker
from traffic_light_reader  import TrafficLightReader
from risk_assessor        import RiskAssessorV2
from projector_sim        import ProjectorSimulator
from danger_zone          import load_danger_zone


# ── Constants / defaults ─────────────────────────────────────────────────────────────
DEFAULT_STREAM       = "http://192.168.1.8:8080/video"
DEFAULT_CALIB_MATRIX = "calibration_matrix.npy"
DEFAULT_DANGER_ZONE  = "danger_zone.npy"
FRAME_W, FRAME_H     = 1280, 720
FPS                  = 30
LIGHT_CYCLE_S        = 15
PROJ_W, PROJ_H       = 1024, 768

# Class-specific BGR box colours for track overlays
_TRACK_COLORS = {
    "person":        (52,  122, 235),
    "car":           (46,  204, 113),
    "motorcycle":    (15,  196, 241),
    "bus":           (182,  89, 155),
    "truck":         (34,  126, 230),
    "traffic light": (60,   76, 231),
}


# ─────────────────────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser():
    p = argparse.ArgumentParser(
        description="CrossLight Monitor v2 – Optimized safe intersection system"
    )
    p.add_argument(
        "--stream_url",
        default=DEFAULT_STREAM,
        help="Camera stream URL or integer device index (default: %(default)s).",
    )
    p.add_argument(
        "--calibration",
        default=DEFAULT_CALIB_MATRIX,
        help="Path to calibration_matrix.npy (default: %(default)s).",
    )
    p.add_argument(
        "--danger_zone_file",
        default=DEFAULT_DANGER_ZONE,
        help="Path to danger_zone.npy (default: %(default)s).",
    )
    p.add_argument(
        "--no_calibration",
        action="store_true",
        help="Skip loading calibration files and run in pixel-based fallback mode.",
    )
    p.add_argument(
        "--simulated_light",
        action="store_true",
        help="Use a simulated timer-based traffic light instead of HSV detection.",
    )
    p.add_argument(
        "--detection_skip_n",
        type=int,
        default=3,
        help="YOLO detection is run every N frames. Reuses detections on skipped frames (default: 3).",
    )
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Calibration loader
# ─────────────────────────────────────────────────────────────────────────────

def _load_calibration(args):
    if args.no_calibration:
        print("[WARN] --no_calibration flag set. Running in pixel-based fallback mode.")
        return None, None

    H, danger_poly = None, None

    # Homography matrix
    try:
        H = np.load(args.calibration)
        print(f"[OK]  Loaded calibration matrix from '{args.calibration}'.")
    except FileNotFoundError:
        print(f"[WARN] Calibration matrix not found at '{args.calibration}'. "
              "Run calibration.py first for metre-space tracking.")
    except Exception as exc:
        print(f"[WARN] Could not load calibration matrix: {exc}")

    # Danger-zone polygon
    try:
        danger_poly = load_danger_zone(args.danger_zone_file)
        print(f"[OK]  Loaded danger zone polygon ({len(danger_poly)} vertices) "
              f"from '{args.danger_zone_file}'.")
    except FileNotFoundError:
        print(f"[WARN] Danger zone file not found at '{args.danger_zone_file}'. "
              "Run calibration.py first for polygon-based zone checking.")
    except Exception as exc:
        print(f"[WARN] Could not load danger zone: {exc}")

    return H, danger_poly


# ─────────────────────────────────────────────────────────────────────────────
# Drawing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _draw_track(frame, track, H_inv):
    """Draw bounding box, ID label, speed, and direction arrow for one track."""
    x1, y1, x2, y2 = map(int, track["bbox"])
    cls   = track.get("class", "?")
    tid   = track["track_id"]
    color = _TRACK_COLORS.get(cls, (46, 204, 113))

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

    # Speed label
    speed_str = (
        f"{track['speed_kmh']:.1f}km/h"
        if "speed_kmh" in track
        else f"{track.get('speed', 0):.1f}px/f"
    )
    label = f"ID:{tid} {cls} {speed_str}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
    cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
    cv2.putText(frame, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

    # Direction arrow (from centroid)
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    dx, dy = track.get("direction", (0.0, 0.0))
    d_len  = 30
    ex, ey = int(cx + dx * d_len), int(cy + dy * d_len)
    cv2.arrowedLine(frame, (cx, cy), (ex, ey), color, 2,
                    cv2.LINE_AA, tipLength=0.35)


def _draw_danger_zone_polygon(frame, polygon_m, H_inv, has_risk):
    if H_inv is None or polygon_m is None:
        return

    pts_m  = np.array([[p[0], p[1]] for p in polygon_m], dtype=np.float32)
    pts_m  = pts_m.reshape(1, -1, 2)
    pts_px = cv2.perspectiveTransform(pts_m, H_inv)
    pts_px = pts_px.reshape(-1, 1, 2).astype(np.int32)

    color  = (0, 0, 255) if has_risk else (0, 165, 255)

    overlay = frame.copy()
    cv2.fillPoly(overlay, [pts_px], color)
    cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)
    cv2.polylines(frame, [pts_px], isClosed=True, color=color,
                  thickness=2, lineType=cv2.LINE_AA)
    if len(pts_px) > 0:
        lx, ly = int(pts_px[0][0][0]), int(pts_px[0][0][1])
        cv2.putText(frame, "DANGER ZONE", (lx + 5, ly + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def _draw_legacy_danger_zone(frame, danger_zone_rect, has_risk):
    x1, y1, x2, y2 = danger_zone_rect
    color = (0, 0, 255) if has_risk else (0, 165, 255)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
    cv2.putText(frame, "DANGER ZONE", (x1 + 10, y1 + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)


def _draw_traffic_light_hud(frame, light_state):
    cv2.rectangle(frame, (1150, 15), (1265, 180), (30, 30, 30), -1, cv2.LINE_AA)
    cv2.rectangle(frame, (1150, 15), (1265, 180), (80, 80, 80), 1, cv2.LINE_AA)

    r_on = (0, 0, 230) if light_state == "red"    else (0, 0, 55)
    y_on = (0, 200, 220) if light_state == "yellow" else (0, 45, 55)
    g_on = (0, 200, 0) if light_state == "green"  else (0, 45, 0)

    cv2.circle(frame, (1207, 50),  18, r_on, -1, cv2.LINE_AA)
    cv2.circle(frame, (1207, 97),  18, y_on, -1, cv2.LINE_AA)
    cv2.circle(frame, (1207, 144), 18, g_on, -1, cv2.LINE_AA)

    state_color = {"red": (0,0,230), "yellow": (0,200,220), "green": (0,200,0)}.get(
        light_state, (200, 200, 200))
    cv2.putText(frame, light_state.upper(), (1155, 172),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, state_color, 2, cv2.LINE_AA)


def _draw_barrier_debug(frame, barrier_pos_px, event_type):
    if barrier_pos_px is None:
        return
    bx, by = int(barrier_pos_px[0]), int(barrier_pos_px[1])
    label  = "VEHICLE BARRIER" if event_type == "vehicle_intrusion" else "PED DANGER"
    cv2.rectangle(frame, (bx - 55, by - 28), (bx + 55, by + 28),
                  (0, 0, 255), 3, cv2.LINE_AA)
    cv2.putText(frame, label, (bx - 55, by - 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2, cv2.LINE_AA)


_LIGHT_CYCLE = ["green", "yellow", "red"]

def _simulated_light(start_time):
    elapsed = (time.time() - start_time) % (LIGHT_CYCLE_S * 3)
    idx = int(elapsed // LIGHT_CYCLE_S)
    return _LIGHT_CYCLE[idx]


# ─────────────────────────────────────────────────────────────────────────────
# Main Loop
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = _build_parser().parse_args()

    stream_src = (int(args.stream_url)
                  if args.stream_url.isdigit()
                  else args.stream_url)

    print("\n" + "=" * 60)
    print("  CrossLight Monitor v2 (Performance Optimized)")
    print("=" * 60)
    print(f"  Stream : {stream_src}")
    print(f"  YOLO Detection Period : every {args.detection_skip_n} frames")

    # ── Calibration ────────────────────────────────────────────────────────
    H, danger_poly = _load_calibration(args)
    H_inv = np.linalg.inv(H) if H is not None else None
    calibrated = (H is not None and danger_poly is not None)

    if not calibrated:
        print("[INFO] No calibration data. Using pixel-based fallback (rectangle zone).")

    # Legacy rectangle danger zone (fallback)
    LEGACY_ZONE = (200, 300, 1000, 600)

    # ── Component initialisation ───────────────────────────────────────────
    print("[INIT] Starting camera stream at 640x480...")
    stream = TrafficCamStream(stream_src, resolution=(640, 480))

    print("[INIT] Loading YOLO detector...")
    detector = YOLODetector()

    print("[INIT] Building Kalman tracker...")
    tracker = KalmanTracker(dt=1.0 / FPS)

    print("[INIT] Building traffic light reader...")
    light_reader = TrafficLightReader()

    print("[INIT] Building risk assessor...")
    if calibrated:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            assessor = RiskAssessorV2(H=H, danger_zone_polygon=danger_poly)
        print("[OK]  Using RiskAssessorV2 (polygon + homography).")
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            assessor = RiskAssessorV2(danger_zone_rect=LEGACY_ZONE)
        print("[OK]  Using RiskAssessorV2 in legacy rectangle fallback mode.")

    print("[INIT] Opening projector simulator window in background thread...")
    projector = ProjectorSimulator(width=PROJ_W, height=PROJ_H)
    proj_thread = threading.Thread(target=projector.run, daemon=True)
    proj_thread.start()

    start_time = time.time()
    print("\n[OK]  System running. Press 'q' in the CrossLight Monitor window to quit.\n")

    # ── Tracking & Skip Frame State ────────────────────────────────────────
    frame_count = 0
    fps_start_time = time.time()
    detection_times = []
    
    last_detections = []
    active_tracks = []
    
    _last_override_t = 0.0
    _detected_light  = "unknown"

    try:
        while True:
            loop_start = time.time()

            # ── 1. Traffic-light state ─────────────────────────────────────
            if args.simulated_light:
                light_state = _simulated_light(start_time)
            else:
                light_state = _detected_light if _detected_light != "unknown" else "green"

            # ── 2. Read frame ──────────────────────────────────────────────
            frame = stream.read()
            if frame is None:
                placeholder = np.zeros((360, 640, 3), dtype=np.uint8)
                cv2.putText(placeholder, "Waiting for stream...",
                             (180, 180),
                             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200),
                             2, cv2.LINE_AA)
                cv2.imshow("CrossLight Monitor", placeholder)
                projector.set_barriers([])
                if cv2.waitKey(15) & 0xFF == ord("q"):
                    break
                time.sleep(0.01)
                continue

            frame_count += 1

            # Normalise internal processing resolution to 1280x720 (for calibration coordinates match)
            fh, fw = frame.shape[:2]
            if fw != FRAME_W or fh != FRAME_H:
                frame = cv2.resize(frame, (FRAME_W, FRAME_H))

            # ── 3. YOLO detection (Skip logic) ─────────────────────────────
            run_detection = (frame_count % args.detection_skip_n == 1) or (frame_count == 1)
            
            if run_detection:
                det_start = time.time()
                detections = detector.detect(frame)
                det_end = time.time()
                detection_times.append(det_end - det_start)
                last_detections = detections
                
                # ── 5a. Kalman tracker update (with detections) ────────────
                active_tracks = tracker.update(
                    detections=detections,
                    frame=frame,
                    H=H,
                )
            else:
                detections = last_detections
                
                # ── 5b. Kalman tracker prediction (without measurement update) ──
                active_tracks = []
                for t in tracker.tracks:
                    t.predict()
                    # Back-project predicted meter centroid to pixel bbox to update tracking window
                    t.bbox = t.predicted_bbox_pixels(H_inv)
                    
                    s = t.state
                    vx, vy = s[2], s[3]
                    speed_ms = float(np.hypot(vx, vy))
                    speed_kmh = speed_ms * 3.6
                    direction = (float(vx / speed_ms), float(vy / speed_ms)) if speed_ms > 1e-6 else (0.0, 0.0)
                    
                    active_tracks.append({
                        "track_id": t.track_id,
                        "class": t.class_name,
                        "bbox": t.bbox,
                        "centroid_meters": (float(s[0]), float(s[1])),
                        "speed_kmh": round(speed_kmh, 2),
                        "direction": direction,
                        "missed": t.missed,
                        "age": t.age,
                    })

            # ── 4. Traffic light state from frame (unless simulated, only on detection frames) ──
            if not args.simulated_light and run_detection:
                tl_state = light_reader.get_state(detections, frame)
                if tl_state != "unknown":
                    _detected_light = tl_state
                light_state = _detected_light if _detected_light != "unknown" else "green"

            # ── 6. Risk assessment ─────────────────────────────────────────
            risk_events = assessor.assess(active_tracks, light_state)
            has_risk    = len(risk_events) > 0

            # ── 7. Projector output updating ──────────────────────────────
            barrier_positions = []
            if has_risk:
                for event in risk_events:
                    bp = event.get("barrier_position")
                    if bp is None:
                        continue

                    # Map to projector window coordinates.
                    bp_px, bp_py = bp
                    proj_x = (bp_px / FRAME_W) * projector.width
                    proj_y = (bp_py / FRAME_H) * projector.height
                    barrier_positions.append((proj_x, proj_y))

                    # Throttled console log
                    now = time.time()
                    if now - _last_override_t > 1.0:
                        etype = event["type"]
                        vid   = event.get("vehicle_id", "?")
                        conf  = event.get("confidence", 0.0)
                        print(f"[RISK] {etype.upper()} | vehicle_id={vid} "
                              f"| light={light_state.upper()} "
                              f"| confidence={conf:.2f}")
                        _last_override_t = now
            
            projector.set_barriers(barrier_positions)

            # ── 8. Draw overlays on camera frame ──────────────────────────

            # A. Danger zone
            if calibrated:
                _draw_danger_zone_polygon(frame, danger_poly, H_inv, has_risk)
            else:
                _draw_legacy_danger_zone(frame, LEGACY_ZONE, has_risk)

            # B. Track bounding boxes, labels, speed, direction arrows
            for track in active_tracks:
                _draw_track(frame, track, H_inv)

            # C. Barrier debug rectangles
            for event in risk_events:
                _draw_barrier_debug(frame, event.get("barrier_position"),
                                    event["type"])

            # D. Traffic-light HUD
            _draw_traffic_light_hud(frame, light_state)

            # E. FPS Overlay
            elapsed_ms = max((time.time() - loop_start) * 1000, 1)
            fps_text   = f"FPS: {1000/elapsed_ms:.1f}"
            cv2.putText(frame, fps_text, (12, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 255, 180),
                        1, cv2.LINE_AA)

            # F. Calibration mode badge
            badge      = "CALIBRATED" if calibrated else "PIXEL MODE"
            badge_col  = (0, 220, 60) if calibrated else (0, 165, 255)
            cv2.putText(frame, badge, (12, 46),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, badge_col,
                        1, cv2.LINE_AA)

            # G. Risk alert banner
            if has_risk:
                cv2.rectangle(frame, (0, FRAME_H - 40), (FRAME_W, FRAME_H),
                               (0, 0, 180), -1)
                cv2.putText(frame, "!! SIGNAL OVERRIDE: ALL-RED HOLD !!",
                             (FRAME_W // 2 - 290, FRAME_H - 12),
                             cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                             (255, 255, 255), 2, cv2.LINE_AA)

            # ── 9. Resize and Show frame ──────────────────────────────────
            # Resize display frame to half size (640x360) to save window drawing overhead
            display_frame = cv2.resize(frame, (640, 360))
            cv2.imshow("CrossLight Monitor", display_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            # ── 10. FPS & Stats output every 30 frames ─────────────────────
            if frame_count % 30 == 0:
                curr_t = time.time()
                avg_fps = 30.0 / (curr_t - fps_start_time)
                avg_det_ms = np.mean(detection_times) * 1000 if detection_times else 0.0
                print(f"[STATS] Frame {frame_count:4d} | Average FPS: {avg_fps:4.1f} | Avg Det Time: {avg_det_ms:5.1f} ms | Active Tracks: {len(active_tracks)}")
                fps_start_time = curr_t
                detection_times = []

            # Yield CPU – target 30 fps
            sleep_s = max(0.0, (1.0 / FPS) - (time.time() - loop_start))
            if sleep_s > 0:
                time.sleep(sleep_s)

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    finally:
        print("[INFO] Shutting down...")
        stream.stop()
        cv2.destroyAllWindows()
        projector.running = False
        pygame.quit()
        print("[OK]  All resources released. Goodbye.")


if __name__ == "__main__":
    main()
