"""
main.py – CrossLight Monitor (v2 Performance & Web Server Optimized)
====================================================================
Integrates:
  TrafficCamStream   – MJPEG / webcam reader (with resolution / queue size 1 optimizations)
  YOLODetector       – YOLOv8n object detection
  KalmanTracker      – Kalman-filter multi-object tracker (V1 or V2 Velo model)
  TrafficLightReader – HSV-based traffic-light colour classifier
  RiskAssessorV2     – Polygon danger-zone risk assessment
  ProjectorSimulator – Pygame projector overlay running in a separate daemon thread
  aiohttp Web Server – Asynchronous MJPEG stream at /video_feed and WebSocket at /ws on port 5000

Usage
-----
  python main.py --stream_url http://192.168.1.8:8080/video
  python main.py --stream_url 0                        # webcam
  python main.py --stream_url 0 --no_calibration       # skip npy loading
"""

import argparse
import asyncio
import json
import os
import sys
import threading
import time
import warnings

import cv2
import numpy as np
import pygame
from aiohttp import web

# ── Local modules ─────────────────────────────────────────────────────────────
from traffic_camera       import TrafficCamStream
from detector             import YOLODetector
from kalman_tracker       import KalmanTracker
from traffic_light_reader  import TrafficLightReader
from risk_assessor        import RiskAssessorV2
from projector_sim        import ProjectorSimulator
from danger_zone          import load_danger_zone

# ─────────────────────────────────────────────────────────────────────────────
# Constants / defaults
# ─────────────────────────────────────────────────────────────────────────────
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
# Shared State for Asynchronous Web Services
# ─────────────────────────────────────────────────────────────────────────────
class SharedState:
    def __init__(self):
        self.current_frame = None       # Encoded JPEG bytes
        self.tracks = []                # List of tracks for WebSocket
        self.light_state = "green"      # Current light state string
        self.projector_active = False   # True if has active risks
        self.risk_events = []           # List of risk event dicts

shared_state = SharedState()


# ─────────────────────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────────────────────
def _build_parser():
    p = argparse.ArgumentParser(
        description="CrossLight Monitor – AI-powered safe intersection system"
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

    # Direction arrow (from centroid) — fixed 25px length using normalized unit vector
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    dx, dy = track.get("direction", (0.0, 0.0))
    mag = (dx**2 + dy**2) ** 0.5
    if mag > 1e-6:
        ndx, ndy = dx / mag, dy / mag
        d_len = 25
        ex, ey = int(cx + ndx * d_len), int(cy + ndy * d_len)
        cv2.arrowedLine(frame, (cx, cy), (ex, ey), color, 2,
                        cv2.LINE_AA, tipLength=0.4)


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
# Asynchronous Web Server Handlers (CORS-enabled)
# ─────────────────────────────────────────────────────────────────────────────
async def cors_middleware(app, handler):
    async def middleware_handler(request):
        if request.method == "OPTIONS":
            response = web.Response(status=204)
        else:
            response = await handler(request)
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS, PUT, DELETE"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        return response
    return middleware_handler


async def video_feed_handler(request):
    response = web.StreamResponse(
        status=200,
        reason='OK',
        headers={
            'Content-Type': 'multipart/x-mixed-replace; boundary=frame',
            'Cache-Control': 'no-cache, private',
            'Pragma': 'no-cache',
            'Connection': 'close',
        }
    )
    await response.prepare(request)
    try:
        last_frame = None
        while True:
            frame_bytes = shared_state.current_frame
            if frame_bytes is not None and frame_bytes != last_frame:
                header = f"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: {len(frame_bytes)}\r\n\r\n".encode('utf-8')
                await response.write(header)
                await response.write(frame_bytes)
                await response.write(b"\r\n")
                last_frame = frame_bytes
            await asyncio.sleep(1.0 / 30)
    except (asyncio.CancelledError, ConnectionResetError):
        pass
    return response


async def websocket_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    request.app['websockets'].add(ws)

    # Numpy standard type JSON encoder helper
    def serialize_np(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        return str(obj)

    try:
        while not ws.closed:
            payload = {
                "tracks": shared_state.tracks,
                "light_state": shared_state.light_state,
                "projector_active": shared_state.projector_active,
                "risk_events": shared_state.risk_events
            }
            # Clean serialize
            clean_str = json.dumps(payload, default=serialize_np)
            await ws.send_str(clean_str)
            await asyncio.sleep(0.2)
    except Exception:
        pass
    finally:
        request.app['websockets'].discard(ws)
    return ws


# ─────────────────────────────────────────────────────────────────────────────
# Core Vision Pipeline Task
# ─────────────────────────────────────────────────────────────────────────────
async def vision_pipeline_task(args, H, danger_poly, calibrated, LEGACY_ZONE, stream, detector, tracker, light_reader, projector, assessor):
    H_inv = np.linalg.inv(H) if H is not None else None
    start_time = time.time()
    
    frame_count = 0
    fps_start_time = time.time()
    detection_times = []
    
    last_detections = []
    active_tracks = []
    
    _last_override_t = 0.0
    _detected_light  = "unknown"

    print("\n[OK]  Vision pipeline task started.")

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
                await asyncio.sleep(0.01)
                continue

            frame_count += 1

            # Normalise processing resolution to 1280x720 (so it matches calibration map coords)
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
                    t.bbox = t.predicted_bbox_pixels(H_inv)
                    
                    s = t.state
                    vx, vy = s[2], s[3]
                    speed_ms = float(np.hypot(vx, vy))
                    if H is None:
                        # Nominal pixel-to-meter scale (0.05m/px) for realistic fallback speeds
                        speed_ms = speed_ms * 0.05
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
            if calibrated:
                _draw_danger_zone_polygon(frame, danger_poly, H_inv, has_risk)
            else:
                _draw_legacy_danger_zone(frame, LEGACY_ZONE, has_risk)

            for track in active_tracks:
                _draw_track(frame, track, H_inv)

            for event in risk_events:
                _draw_barrier_debug(frame, event.get("barrier_position"),
                                    event["type"])

            _draw_traffic_light_hud(frame, light_state)

            # FPS Overlay
            elapsed_ms = max((time.time() - loop_start) * 1000, 1)
            fps_text   = f"FPS: {1000/elapsed_ms:.1f}"
            cv2.putText(frame, fps_text, (12, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 255, 180),
                        1, cv2.LINE_AA)

            # Calibration mode badge
            badge      = "CALIBRATED" if calibrated else "PIXEL MODE"
            badge_col  = (0, 220, 60) if calibrated else (0, 165, 255)
            cv2.putText(frame, badge, (12, 46),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, badge_col,
                        1, cv2.LINE_AA)

            # Risk alert banner
            if has_risk:
                cv2.rectangle(frame, (0, FRAME_H - 40), (FRAME_W, FRAME_H),
                               (0, 0, 180), -1)
                cv2.putText(frame, "!! SIGNAL OVERRIDE: ALL-RED HOLD !!",
                             (FRAME_W // 2 - 290, FRAME_H - 12),
                             cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                             (255, 255, 255), 2, cv2.LINE_AA)

            # ── 9. Resize and Show frame ──────────────────────────────────
            display_frame = cv2.resize(frame, (640, 360))
            cv2.imshow("CrossLight Monitor", display_frame)

            # ── 10. Update Asynchronous Shared State ──────────────────────
            # Encode frame for /video_feed stream
            ret, jpeg = cv2.imencode('.jpg', display_frame)
            if ret:
                shared_state.current_frame = jpeg.tobytes()

            # Format tracks payload
            ws_tracks = []
            for t in active_tracks:
                centroid_m = t.get("centroid_meters")
                if centroid_m is None and "centroid" in t:
                    centroid_m = t["centroid"]
                ws_tracks.append({
                    "track_id": t["track_id"],
                    "class": t["class"],
                    "speed_kmh": t.get("speed_kmh", 0.0),
                    "centroid_metres": centroid_m
                })

            shared_state.tracks = ws_tracks
            shared_state.light_state = light_state
            shared_state.projector_active = has_risk
            # Remove any raw numpy/cv2 objects from risk events to ensure JSON safety
            clean_risk_events = []
            for ev in risk_events:
                clean_ev = {}
                for k, v in ev.items():
                    if isinstance(v, (np.ndarray, np.generic)):
                        clean_ev[k] = v.tolist()
                    else:
                        clean_ev[k] = v
                clean_risk_events.append(clean_ev)
            shared_state.risk_events = clean_risk_events

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            # ── 11. FPS & Stats output every 30 frames ─────────────────────
            if frame_count % 30 == 0:
                curr_t = time.time()
                avg_fps = 30.0 / (curr_t - fps_start_time)
                avg_det_ms = np.mean(detection_times) * 1000 if detection_times else 0.0
                print(f"[STATS] Frame {frame_count:4d} | Average FPS: {avg_fps:4.1f} | Avg Det Time: {avg_det_ms:5.1f} ms | Active Tracks: {len(active_tracks)}")
                fps_start_time = curr_t
                detection_times = []

            # Yield control to the asyncio event loop
            sleep_s = max(0.001, (1.0 / FPS) - (time.time() - loop_start))
            await asyncio.sleep(sleep_s)

    except asyncio.CancelledError:
        print("[INFO] Vision pipeline task cancelled.")
    except Exception as e:
        print(f"[ERROR] Vision pipeline exception: {e}")
        import traceback
        traceback.print_exc()


# ─────────────────────────────────────────────────────────────────────────────
# Async Application Coordinator
# ─────────────────────────────────────────────────────────────────────────────
async def amain():
    args = _build_parser().parse_args()

    stream_src = (int(args.stream_url)
                  if args.stream_url.isdigit()
                  else args.stream_url)

    print("\n" + "=" * 60)
    print("  CrossLight Monitor v2 (Async & Web Server Integrated)")
    print("=" * 60)
    print(f"  Stream : {stream_src}")
    print(f"  YOLO Detection Period : every {args.detection_skip_n} frames")

    # ── Calibration ────────────────────────────────────────────────────────
    H, danger_poly = _load_calibration(args)
    calibrated = (H is not None and danger_poly is not None)

    if not calibrated:
        print("[INFO] No calibration data. Using pixel-based fallback (rectangle zone).")

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

    # ── Web Application Setup ─────────────────────────────────────────────
    app = web.Application(middlewares=[cors_middleware])
    app['websockets'] = set()
    app.router.add_get('/video_feed', video_feed_handler)
    app.router.add_get('/ws', websocket_handler)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', 5000)
    await site.start()
    print("[INIT] Web server running on http://0.0.0.0:5000")

    # ── Run vision pipeline concurrently with web server ──────────────────
    try:
        await vision_pipeline_task(
            args=args,
            H=H,
            danger_poly=danger_poly,
            calibrated=calibrated,
            LEGACY_ZONE=LEGACY_ZONE,
            stream=stream,
            detector=detector,
            tracker=tracker,
            light_reader=light_reader,
            projector=projector,
            assessor=assessor
        )
    finally:
        print("[INFO] Shutting down application...")
        # Disconnect all active WebSockets
        for ws in list(app['websockets']):
            await ws.close()
        await runner.cleanup()
        stream.stop()
        cv2.destroyAllWindows()
        projector.running = False
        pygame.quit()
        print("[OK]  All resources released.")


def main():
    try:
        asyncio.run(amain())
    except KeyboardInterrupt:
        print("\n[INFO] Keyboard interrupt. Exiting.")


if __name__ == "__main__":
    main()
