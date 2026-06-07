# CrossLight — From camera to caution in milliseconds

> **Status: active prototype** — built and tested on a single IP camera at a campus intersection. Not production-ready; expect rough edges.

---

## What this is

CrossLight is a real-time intersection safety monitor. It takes a single IP camera feed, detects vehicles, motorcycles, and pedestrians using YOLOv8, tracks them across frames with a Kalman filter, and raises an alert the moment something enters a defined danger zone while the traffic light is red.

The most unusual part is the **projector simulator**: when a risk event fires, it computes a barrier position in image space and renders it in a separate Pygame window — a demo of how a physical projector mounted above an intersection could warn pedestrians by projecting a visible line onto the road surface.

---

## Table of Contents
1. [Why I built this](#why-i-built-this)
2. [System Architecture](#system-architecture)
3. [Core Modules](#core-modules)
4. [Data Flow](#data-flow)
5. [Installation](#installation)
6. [Running the System](#running-the-system)
7. [Calibration](#calibration)
8. [Performance](#performance)
9. [Test Suite](#test-suite)
10. [Known Limitations](#known-limitations)
11. [Honest Roadmap](#honest-roadmap)
12. [License](#license)

---

## Why I built this

I live near a chaotic intersection. Watching near-misses on my commute made me wonder: what would it take to build a system that could catch a red-light violation as it happens and physically warn a pedestrian crossing? Not a simulation, not a paper — an actual pipeline that runs.

The projector idea came from a frustration with dashboards: alerts on a screen don't help the person crossing the road. A projected barrier on the asphalt does.

**What actually broke during development:**
- The HSV traffic-light classifier is genuinely hard to get right. Sunlight washing out the light housing, shadows from overhead cables, and the camera auto-adjusting exposure all break naive colour thresholds. I tuned it for the controlled conditions I had; it would need significant work for real Indian roads.
- Kalman tracking occlusion handling is messier than textbooks suggest. When two vehicles overlap, the Hungarian assignment sometimes swaps IDs and the speed estimate spikes briefly. I mitigated this with an appearance histogram re-ID term but didn't fully solve it.
- Getting the homography calibration to be stable was the hardest 20% of the project. The projector simulator only works if the homography is precise; even a 5-pixel error in the calibration clicks shifts the barrier by half a metre.

**What I'd do differently:**
- Use YOLO's native traffic-light classifier instead of the HSV approach.
- Add a proper re-ID module (e.g., OsNet) rather than HSV histograms.
- Collect a real dataset from the target intersection and fine-tune YOLOv8 on it.

---

## System Architecture

### Backend (Python)
| File | Role |
|------|------|
| `main.py` | **Primary entry point.** Async aiohttp server with MJPEG `/video_feed` + WebSocket `/ws`. |
| `traffic_camera.py` | Threaded MJPEG / webcam reader; single-frame buffer to avoid latency build-up. |
| `yolo_detector.py` | YOLOv8n wrapper; `detect(frame)` returns `[{class_name, bbox, confidence}]`. |
| `detector.py` | Re-export shim — keeps `from detector import YOLODetector` working. |
| `kalman_tracker.py` | Constant-acceleration Kalman tracker (primary). Hungarian assignment + HSV appearance re-ID. |
| `traffic_light_reader.py` | HSV colour classifier on circular ROI of detected traffic-light crops. |
| `risk_assessor.py` | `RiskAssessorV2`: polygon danger-zone check via homography + ray-casting. |
| `projector_simulator.py` | Pygame window; renders pulsing barrier polygons at risk positions. |
| `calibration.py` | Interactive tool for homography + danger-zone polygon setup. |
| `danger_zone.py` | `point_in_danger_zone` ray-casting helper. |

### Frontend (React / Vite)
Lives in `crosslight-frontend/`. Connects to the backend WebSocket at `/ws` and renders:
- Live camera feed (MJPEG via `/video_feed`)
- Traffic-light HUD
- Active tracks table (ID, class, speed, age)
- Risk alerts panel
- Connection status indicator

### Deprecated / Experimental files
| File | Explanation |
|------|-------------|
| `main_v2.py` | Older standalone version without the web server. Kept for quick local testing. See header comment. |
| `kalman_tracker_v2.py` | Constant-velocity alternative tracker. Documented in its own header. |
| `simple_tracker.py` | Pure-IoU tracker without Kalman — early prototype, useful as a baseline. |
| `tracker.py` | Re-export shim for `simple_tracker.py`. |
| `projector_sim.py` | Re-export shim for `projector_simulator.py`. |

---

## Core Modules

### Object Detection (`yolo_detector.py`)
Wraps YOLOv8n via the Ultralytics library. Runs every *N* frames (default: every 3rd) to keep CPU usage reasonable; the Kalman filter predicts positions on skipped frames.

```python
detector = YOLODetector(model_path='yolov8n.pt', confidence=0.35)
detections = detector.detect(frame)
# → [{'class_name': 'car', 'bbox': [x1,y1,x2,y2], 'confidence': 0.87}, ...]
```

### Kalman Tracking (`kalman_tracker.py`)
Constant-acceleration model; state vector `[x, y, vx, vy, ax, ay]` in **metre space** (requires a calibration homography). Provides `predicted_bbox_pixels(H_inv)` for back-projecting the predicted position onto the frame during skip frames.

Association cost = `0.7 × IoU + 0.3 × HSV histogram similarity` — the appearance term helps maintain IDs through partial occlusion.

### Traffic-Light Classification (`traffic_light_reader.py`)
⚠ **See [Known Limitations](#known-limitations) before trusting this in a new environment.**

Crops the YOLO-detected traffic-light bounding box, examines only a central circular ROI (radius = 0.4 × min(w, h)), applies a brightness gate (V > 100 in HSV) to discard dark inactive bulbs, then counts pixels in HSV ranges for red/yellow/green. Returns `'unknown'` unless the dominant colour covers ≥ 15% of the ROI.

### Risk Assessment (`risk_assessor.py`)
`RiskAssessorV2` uses `cv2.perspectiveTransform` + ray-casting (`point_in_danger_zone`) to check whether a vehicle's 1-second predicted trajectory intersects the calibrated danger-zone polygon. Confidence is time-to-entry smoothed by a 0.3-alpha EMA.

### Projector Simulator (`projector_simulator.py`)
Runs in a daemon thread. Receives `(x, y)` barrier positions via a thread-safe queue and renders pulsing semi-transparent red polygons. On a real deployment, this window would be fullscreened onto a projector's second display.

---

## Data Flow

```mermaid
flowchart TD
    A[IP Camera] --> B[TrafficCamStream]
    B -->|Every Nth frame| C[YOLODetector]
    B -->|All frames| D[KalmanTracker]
    C --> D
    C --> E[TrafficLightReader]
    E --> F[light_state]
    D --> G[RiskAssessorV2]
    F --> G
    G --> H[ProjectorSimulator]
    G --> I[WebSocket /ws]
    I --> J[React Frontend]
    B -->|MJPEG| K[/video_feed]
    K --> J
```

---

## Installation

**Prerequisites:** Python ≥ 3.11, Node.js ≥ 20 LTS, a network-reachable camera (MJPEG or H.264 stream). Tested on Windows 11 with an IP Webcam Android app.

```bash
# 1. Clone / open the repo
cd "C:/Users/Daksh/OneDrive/Desktop/Something Revolutionary"

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Install frontend dependencies
cd crosslight-frontend
npm install
cd ..
```

---

## Running the System

```bash
# Terminal 1 — Backend (primary entry point)
python main.py --stream_url http://<CAM_IP>:8080/video

# Terminal 2 — Frontend
cd crosslight-frontend
npm run dev   # → http://localhost:5173
```

**Useful flags:**
```
--no_calibration      Run without .npy files; uses a fallback rectangle zone
--simulated_light     Timer-based traffic light instead of HSV detection
--detection_skip_n 3  Run YOLO every N frames (default: 3)
--stream_url 0        Use local webcam instead of IP camera
```

---

## Calibration

Run once per camera deployment. Requires a known flat rectangle visible in the frame (e.g., a parking-bay marking or a cardboard square placed on the road).

```bash
python calibration.py --stream_url http://<CAM_IP>:8080/video
```

**Phase 1 — Homography:** Click the four corners of your reference rectangle in order (TL → TR → BR → BL). Saves `calibration_matrix.npy`.

**Phase 2 — Danger zone:** Click the polygon vertices of the area to monitor. Press `d` to confirm each point, `f` to finish. Saves `danger_zone.npy`.

Without these files, the system falls back to a hard-coded pixel rectangle and prints a `[WARN]` message.

---

## Performance

The system was profiled using `benchmark.py` (CPU only, no GPU). While an idealized lab benchmark suggests the pipeline could theoretically hit ~50+ FPS (by running YOLO at ~50ms every 3rd frame and fast skip-frames at ~4ms), **real-world performance is typically 4–8 FPS** over a WiFi camera stream.

This massive gap between theoretical and actual performance comes down to three genuine system bottlenecks:

1. **YOLO inference is the ceiling:** Running YOLOv8n at 1280×720 on CPU takes ~50ms per frame. Even with a `detection_skip_n=3`, this puts a hard limit on system responsiveness.
2. **Tracker complexity scales non-linearly:** Our benchmark showed Kalman tracking takes ~1.4ms for 6 active tracks. However, at a busy intersection with **48 active tracks**, the Hungarian matching algorithm's O(n³) complexity causes the tracking step to spike to **~61ms**. At that density, the tracker becomes as expensive as the neural net.
3. **Network/async contention:** The `asyncio` event loop running the WebSocket server competes with the vision pipeline for CPU time. Combined with unstable WiFi IP camera latency (often causing the capture thread to retry), this introduces significant jitter.

**How this needs to be fixed for production:**
- **Wired connections:** The camera stream must be hardwired. WiFi latency destroys the frame buffer.
- **Resolution downscaling:** Dropping YOLO input resolution from 1280×720 to 640×640 roughly halves inference time.
- **Process isolation:** The vision pipeline and the web/websocket server need to be moved to separate processes (e.g., using `multiprocessing`) to prevent I/O blocking from tanking the frame rate.

**Key optimisations currently in place:**
- Frame-skip: YOLO runs every *N* frames; Kalman predicts on skipped frames.
- Projector simulator runs in a daemon thread — never blocks the vision loop.
- `TrafficCamStream` uses a dedicated reader thread with a 1-frame buffer to drop stale frames.

---

## Test Suite

Tests live in `tests/` and use `pytest`. No camera or GPU needed — all tests use synthetic data.

```bash
pip install pytest
pytest tests/ -v
```

| Test file | What it covers |
|-----------|---------------|
| `tests/test_traffic_light_reader.py` | `analyze_crop` on synthetic BGR images; majority-vote `get_state`; edge cases (empty/None crop, OOB bbox) |
| `tests/test_kalman_tracker.py` | Track creation, ID stability, lifecycle (missed/deleted), speed estimation convergence |
| `tests/test_risk_assessor.py` | `RiskAssessor` (v1 rectangle) and `RiskAssessorV2` (polygon + homography); green/red/yellow states; pedestrian coupling; legacy fallback deprecation warning |

Additional test files in the project root (`test_yolo_detector.py`, `test_traffic_cam.py`) use mocks for the YOLO model and OpenCV capture respectively.

---

## Known Limitations

**HSV traffic-light classifier is fragile.** This is the weakest component. The classifier was tuned for the lighting conditions on one specific camera at one time of day. It will likely misclassify under:
- Direct sunlight washing out the housing
- Faded or older signal hardware (common in Indian cities)
- Camera auto-exposure adjusting mid-scene
- Non-standard light colours (some signals use LED arrays with slightly different hues)

The `--simulated_light` flag exists specifically for demos where you need the rest of the system to work reliably without depending on this classifier.

**Single-camera, single-plane assumption.** The homography assumes the tracked objects move on a flat ground plane. This breaks for hilly roads, ramps, or vehicles at significantly different distances from the camera.

**No re-ID across track loss.** If a vehicle is occluded for more than `max_missed` frames (default: 10), it gets a new track ID when it reappears. The HSV histogram re-ID term reduces this but doesn't eliminate it.

**No real-world validation data.** All performance benchmarks in this README are from a single hardware setup. Real-world traffic conditions (especially Indian intersections with mixed vehicle types, lane-sharing, and erratic motion) were not systematically tested.

---

## Honest Roadmap

This is a solo student project. The roadmap is limited to what is realistic.

| What | When |
|------|------|
| Replace HSV classifier with YOLO-based traffic-light state from a fine-tuned model | When I find/make a labelled dataset |
| Add proper re-ID module (OsNet or similar) | After exams |
| Test on real Mumbai intersection footage | When I can get access to camera data |
| Docker containerisation for easier deployment | Before any serious demo |

---

## License

MIT License — use it, break it, improve it.

---

*Built by Daksh — [dakshshrivastav56@gmail.com](mailto:dakshshrivastav56@gmail.com)*
