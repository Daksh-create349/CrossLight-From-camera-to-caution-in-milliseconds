"""
benchmark.py - CrossLight Pipeline Stage Profiler
==================================================
Times each stage of the vision pipeline independently.
No live camera needed -- uses a synthetic 1280x720 frame.

Run:
    python benchmark.py
"""

import sys, time, warnings, statistics
import numpy as np
import cv2

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

WARMUP_RUNS = 5
BENCH_RUNS  = 30

SEP = "=" * 72
DIV = "-" * 72

def _timer(fn):
    for _ in range(WARMUP_RUNS):
        fn()
    times = []
    for _ in range(BENCH_RUNS):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return times

def _stats(t):
    return dict(mean=statistics.mean(t), mn=min(t), mx=max(t),
                p95=sorted(t)[int(len(t)*0.95)],
                sd=statistics.stdev(t) if len(t)>1 else 0.0)

def _row(label, s):
    tag = "FAST" if s["mean"] < 20 else ("OK" if s["mean"] < 100 else "SLOW")
    print(f"  [{tag:4}]  {label:<45}  mean={s['mean']:7.2f}  "
          f"min={s['mn']:7.2f}  max={s['mx']:7.2f}  p95={s['p95']:7.2f}  sd={s['sd']:5.2f}  (ms)")

# ── Setup ──────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  CrossLight Pipeline Benchmark")
print(f"  Warm-up={WARMUP_RUNS}  Bench={BENCH_RUNS}  Frame=1280x720 BGR uint8")
print(SEP)

rng   = np.random.default_rng(42)
frame = rng.integers(0, 256, (720, 1280, 3), dtype=np.uint8)
SCALE = 0.05
H     = np.array([[SCALE,0,0],[0,SCALE,0],[0,0,1]], dtype=np.float64)
H_inv = np.linalg.inv(H)
poly  = [(8.0,4.0),(12.0,4.0),(12.0,6.0),(8.0,6.0)]

mock_dets = [
    {"class_name":"car",          "bbox":[100,200,300,400],  "confidence":0.91},
    {"class_name":"motorcycle",   "bbox":[400,300,500,450],  "confidence":0.87},
    {"class_name":"person",       "bbox":[600,250,650,450],  "confidence":0.78},
    {"class_name":"car",          "bbox":[800,150,1000,350], "confidence":0.82},
    {"class_name":"person",       "bbox":[200,450,250,650],  "confidence":0.70},
    {"class_name":"traffic light","bbox":[1100,50,1160,200], "confidence":0.94},
]

results = {}

# ── 1. Frame ops ───────────────────────────────────────────────────────────
print("\n[1] Frame Operations")
small = rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)

t = _timer(lambda: cv2.resize(small, (1280, 720)))
results["Resize 640x480 -> 1280x720"] = _stats(t)
_row("Resize 640x480 -> 1280x720", results["Resize 640x480 -> 1280x720"])

t = _timer(lambda: cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80]))
results["JPEG encode (for /video_feed)"] = _stats(t)
_row("JPEG encode (for /video_feed)", results["JPEG encode (for /video_feed)"])

# ── 2. YOLO detection ──────────────────────────────────────────────────────
print("\n[2] YOLOv8n Detection")
try:
    from yolo_detector import YOLODetector
    detector = YOLODetector()
    print("  Warming up YOLOv8n...", end="", flush=True)
    detector.detect(frame)   # pre-warmup outside _timer
    print(" done.")
    t = _timer(lambda: detector.detect(frame))
    results["YOLO detect full frame (1280x720)"] = _stats(t)
    _row("YOLO detect full frame (1280x720)", results["YOLO detect full frame (1280x720)"])
except Exception as e:
    print(f"  YOLO unavailable: {e}")

# ── 3. Kalman tracker ──────────────────────────────────────────────────────
print("\n[3] Kalman Tracker")
from kalman_tracker import KalmanTracker
tracker = KalmanTracker(dt=1/30, max_missed=10)
for _ in range(10):
    tracker.update(mock_dets, frame=frame, H=H)

t = _timer(lambda: tracker.update(mock_dets, frame=frame, H=H))
results["KalmanTracker.update (6 dets + 6 tracks)"] = _stats(t)
_row("KalmanTracker.update (6 dets + 6 tracks)", results["KalmanTracker.update (6 dets + 6 tracks)"])

t = _timer(lambda: tracker.update([], frame=frame, H=H))
results["KalmanTracker.update predict-only (0 dets)"] = _stats(t)
_row("KalmanTracker.update predict-only (0 dets)", results["KalmanTracker.update predict-only (0 dets)"])

# ── 4. Traffic light reader ────────────────────────────────────────────────
print("\n[4] Traffic Light Reader")
from traffic_light_reader import TrafficLightReader
tl = TrafficLightReader()

t = _timer(lambda: tl.get_state(mock_dets, frame))
results["TrafficLightReader.get_state (1 TL bbox)"] = _stats(t)
_row("TrafficLightReader.get_state (1 TL bbox)", results["TrafficLightReader.get_state (1 TL bbox)"])

crop = frame[50:200, 1100:1160]
t = _timer(lambda: tl.analyze_crop(crop))
results["TrafficLightReader.analyze_crop (150x60)"] = _stats(t)
_row("TrafficLightReader.analyze_crop (150x60)", results["TrafficLightReader.analyze_crop (150x60)"])

# ── 5. Risk assessment ─────────────────────────────────────────────────────
print("\n[5] Risk Assessor")
from risk_assessor import RiskAssessorV2
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    assessor = RiskAssessorV2(H=H, danger_zone_polygon=poly)

active_tracks = tracker.update(mock_dets, frame=frame, H=H)

t = _timer(lambda: assessor.assess(active_tracks, "red"))
results["RiskAssessorV2.assess RED (6 tracks)"] = _stats(t)
_row("RiskAssessorV2.assess RED (6 tracks)", results["RiskAssessorV2.assess RED (6 tracks)"])

t = _timer(lambda: assessor.assess(active_tracks, "green"))
results["RiskAssessorV2.assess GREEN (6 tracks)"] = _stats(t)
_row("RiskAssessorV2.assess GREEN (6 tracks)", results["RiskAssessorV2.assess GREEN (6 tracks)"])

# ── 6. Drawing overlays ────────────────────────────────────────────────────
print("\n[6] OpenCV Drawing")

def _draw():
    canvas = frame.copy()
    for tr in active_tracks:
        x1,y1,x2,y2 = map(int, tr["bbox"])
        cv2.rectangle(canvas, (x1,y1),(x2,y2),(46,204,113),2)
        cv2.putText(canvas, f"ID:{tr['track_id']} {tr['speed_kmh']:.1f}km/h",
                    (x1+2,y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255,255,255),1)
    pts = np.array([[p[0],p[1]] for p in poly], np.float32).reshape(1,-1,2)
    pts_px = cv2.perspectiveTransform(pts, H_inv).reshape(-1,1,2).astype(np.int32)
    ov = canvas.copy()
    cv2.fillPoly(ov,[pts_px],(0,0,255))
    cv2.addWeighted(ov,0.18,canvas,0.82,0,canvas)
    cv2.polylines(canvas,[pts_px],True,(0,0,255),2)
    cv2.resize(canvas,(640,360))

t = _timer(_draw)
results["Draw bboxes + polygon + resize (6 tracks)"] = _stats(t)
_row("Draw bboxes + polygon + resize (6 tracks)", results["Draw bboxes + polygon + resize (6 tracks)"])

# ── 7. Full skip-frame composite ───────────────────────────────────────────
print("\n[7] Composite: Full Skip-Frame Cycle (no YOLO)")

def _skip_cycle():
    tracker.update([], frame=frame, H=H)
    tl.get_state(mock_dets, frame)
    tr = tracker.update(mock_dets, frame=frame, H=H)
    assessor.assess(tr, "red")
    _draw()

t = _timer(_skip_cycle)
results["Full skip-frame cycle (no YOLO)"] = _stats(t)
_row("Full skip-frame cycle (no YOLO)", results["Full skip-frame cycle (no YOLO)"])

# ── Summary ────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  SUMMARY  (all times in ms, mean over 30 runs)")
print(DIV)
for label, s in results.items():
    tag = "FAST" if s["mean"] < 20 else ("OK" if s["mean"] < 100 else "SLOW")
    print(f"  [{tag:4}]  {label:<48}  mean={s['mean']:7.2f}")

# FPS math
skip_ms = results.get("Full skip-frame cycle (no YOLO)", {}).get("mean", 0)
yolo_ms = results.get("YOLO detect full frame (1280x720)", {}).get("mean", 0)

if skip_ms > 0 and yolo_ms > 0:
    # detection_skip_n=3: 1 YOLO frame + 2 skip frames per 3
    avg_ms   = (yolo_ms + 2 * skip_ms) / 3
    est_fps  = 1000 / avg_ms
    print(DIV)
    print("  FPS estimate (detection_skip_n=3, excludes asyncio/WebSocket):")
    print(f"    YOLO frame  : {yolo_ms:.1f} ms")
    print(f"    Skip frame  : {skip_ms:.1f} ms")
    print(f"    Avg/frame   : {avg_ms:.1f} ms")
    fps_tag = "FAST" if est_fps >= 25 else ("OK" if est_fps >= 15 else "SLOW")
    print(f"    [{fps_tag}]  Estimated FPS = {est_fps:.1f}")

print(f"{SEP}\n")
