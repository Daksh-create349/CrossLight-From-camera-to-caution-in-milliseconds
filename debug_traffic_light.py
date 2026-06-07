"""
debug_traffic_light.py
======================
Diagnostic tool for the TrafficLightReader.

For every frame that contains a YOLO 'traffic light' detection
(confidence > 0.5) the script:

  1. Crops the bounding box.
  2. Calls TrafficLightReader.analyze_crop() to classify the colour and get
     per-channel pixel counts.
  3. Prints a detailed summary to the terminal (pixel counts, ratios).
  4. Saves the raw crop as a timestamped PNG inside 'traffic_light_debug/'.
  5. Displays a live window showing the crop next to three false-colour
     overlays (red / yellow / green masks).
  6. Press 's' in the window to save the overlay composite image.
  7. Exits automatically after 60 seconds.

Usage
-----
    python debug_traffic_light.py --stream_url http://192.168.1.8:8080/video
"""

import argparse
import os
import sys
import time
from datetime import datetime

import cv2
import numpy as np

from traffic_camera import TrafficCamStream
from yolo_detector import YOLODetector
from traffic_light_reader import TrafficLightReader


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
RUN_DURATION_S  = 60          # total run time
MIN_DISPLAY_DIM = 80          # up-scale tiny crops so they are visible
DEBUG_DIR       = "traffic_light_debug"

# Colour overlay palette (BGR)
_OVERLAY_COLORS = {
    "red":    (0,   0,   255),
    "yellow": (0,   215, 255),
    "green":  (0,   200,  60),
}
_LABEL_COLOR = {
    "red":    (0,   0,   220),
    "yellow": (0,   180, 220),
    "green":  (0,   180,  40),
    "unknown":(180, 180, 180),
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _upscale(img, min_dim=MIN_DISPLAY_DIM):
    """Upscale img so its smallest dimension is at least min_dim pixels."""
    h, w = img.shape[:2]
    if min(h, w) < min_dim:
        scale = min_dim / max(1, min(h, w))
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    return img


def _build_masks_for_display(crop):
    """
    Re-derive the colour masks from a BGR crop (same logic as TrafficLightReader)
    and return them as colour-tinted BGR images so they are human-readable.
    """
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    v   = hsv[:, :, 2]

    # Brightness gate
    bright_mask = (v > TrafficLightReader.MIN_BRIGHTNESS_V).astype(np.uint8) * 255

    # Circle ROI
    h, w  = crop.shape[:2]
    r     = max(1, int(TrafficLightReader.CIRCLE_RADIUS_K * min(w, h)))
    cy, cx = h // 2, w // 2
    circ  = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(circ, (cx, cy), r, 255, -1)
    roi   = cv2.bitwise_and(circ, bright_mask)

    def _gate(lo, hi):
        m = cv2.inRange(hsv, lo, hi)
        return cv2.bitwise_and(m, roi)

    mask_red = cv2.bitwise_or(
        _gate(np.array([0,   70,  70]), np.array([10,  255, 255])),
        _gate(np.array([160, 70,  70]), np.array([180, 255, 255]))
    )
    mask_yellow = _gate(np.array([15, 70, 70]), np.array([35, 255, 255]))
    mask_green  = _gate(np.array([36, 70, 70]), np.array([95, 255, 255]))

    def _tint(mask, bgr_colour):
        """Fill a 3-channel image with bgr_colour where mask is non-zero."""
        out = np.zeros((*mask.shape, 3), dtype=np.uint8)
        out[mask > 0] = bgr_colour
        # Dim the zero-regions to 20 % of original crop for context
        bg = (crop.astype(np.float32) * 0.20).astype(np.uint8)
        out[mask == 0] = bg[mask == 0]
        return out

    return (
        _tint(mask_red,    _OVERLAY_COLORS["red"]),
        _tint(mask_yellow, _OVERLAY_COLORS["yellow"]),
        _tint(mask_green,  _OVERLAY_COLORS["green"]),
    )


def _labelled_tile(img, label, colour):
    """Add a small one-line header above img."""
    bar_h = 18
    h, w  = img.shape[:2]
    tile  = np.zeros((h + bar_h, w, 3), dtype=np.uint8)
    tile[:bar_h] = (30, 30, 30)
    tile[bar_h:] = img
    cv2.putText(tile, label, (2, bar_h - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, colour, 1, cv2.LINE_AA)
    return tile


def _build_composite(crop, state, counts, det_idx):
    """
    Build a composite image: [crop | red-mask | yellow-mask | green-mask]
    with per-channel pixel counts annotated.
    """
    circle_area = max(1, counts.get("circle_area", 1))
    r_px = counts.get("red",    0)
    y_px = counts.get("yellow", 0)
    g_px = counts.get("green",  0)

    red_img, yellow_img, green_img = _build_masks_for_display(crop)

    # Upscale all panels uniformly
    h_target = max(MIN_DISPLAY_DIM,
                   max(c.shape[0] for c in (crop, red_img, yellow_img, green_img)))
    w_target = max(MIN_DISPLAY_DIM,
                   max(c.shape[1] for c in (crop, red_img, yellow_img, green_img)))

    def _resize_to(img):
        return cv2.resize(img, (w_target, h_target), interpolation=cv2.INTER_NEAREST)

    crop_r      = _resize_to(crop)
    red_r       = _resize_to(red_img)
    yellow_r    = _resize_to(yellow_img)
    green_r     = _resize_to(green_img)

    crop_label  = f"CROP #{det_idx}  state={state.upper()}"
    red_label   = f"RED    {r_px}px ({100*r_px/circle_area:.0f}%)"
    yellow_label= f"YELLOW {y_px}px ({100*y_px/circle_area:.0f}%)"
    green_label = f"GREEN  {g_px}px ({100*g_px/circle_area:.0f}%)"

    state_col = _LABEL_COLOR.get(state, (200, 200, 200))

    tiles = [
        _labelled_tile(crop_r,    crop_label,   state_col),
        _labelled_tile(red_r,     red_label,    _OVERLAY_COLORS["red"]),
        _labelled_tile(yellow_r,  yellow_label, _OVERLAY_COLORS["yellow"]),
        _labelled_tile(green_r,   green_label,  _OVERLAY_COLORS["green"]),
    ]
    return np.hstack(tiles)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Live debug tool for TrafficLightReader."
    )
    parser.add_argument(
        "--stream_url",
        default="http://192.168.1.8:8080/video",
        help="IP camera URL or integer webcam index.",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=RUN_DURATION_S,
        help=f"How many seconds to run (default: {RUN_DURATION_S}).",
    )
    args = parser.parse_args()

    os.makedirs(DEBUG_DIR, exist_ok=True)

    src = int(args.stream_url) if args.stream_url.isdigit() else args.stream_url
    print(f"\n[INFO] Connecting to stream: {src}")
    stream   = TrafficCamStream(src, resolution=(1280, 720))
    detector = YOLODetector()
    reader   = TrafficLightReader()

    print(f"[INFO] Running for {args.duration} s. "
          "Press 's' to save current composite. Press 'q' to quit early.\n")

    WIN_NAME    = "Traffic Light Debug"
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)

    start_time  = time.time()
    frame_count = 0
    last_composite = None   # the most recent composite (for manual save with 's')
    save_count     = 0

    try:
        while time.time() - start_time < args.duration:
            frame = stream.read()
            if frame is None:
                time.sleep(0.02)
                continue

            frame_count += 1
            # Normalise to 1280×720
            fh, fw = frame.shape[:2]
            if fw != 1280 or fh != 720:
                frame = cv2.resize(frame, (1280, 720))

            detections = detector.detect(frame)

            # Filter traffic light detections above threshold
            tl_dets = [
                d for d in detections
                if d.get("class_name") == "traffic light"
                and d.get("confidence", 0.0) > 0.5
            ]

            if not tl_dets:
                # Show a placeholder so the window stays alive
                placeholder = frame.copy()
                cv2.putText(placeholder, "No traffic light detected",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (180, 180, 180), 2, cv2.LINE_AA)
                cv2.imshow(WIN_NAME, placeholder)
            else:
                row_composites = []

                for det_idx, det in enumerate(tl_dets):
                    x1, y1, x2, y2 = map(int, det["bbox"])
                    h_f, w_f = frame.shape[:2]
                    x1 = max(0, min(x1, w_f - 1))
                    y1 = max(0, min(y1, h_f - 1))
                    x2 = max(x1 + 1, min(x2, w_f))
                    y2 = max(y1 + 1, min(y2, h_f))

                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue

                    # Classify via updated reader
                    state, counts = reader.analyze_crop(crop)
                    circle_area   = max(1, counts.get("circle_area", 1))

                    # ── Console output ─────────────────────────────────────
                    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    print(
                        f"[{ts}] Frame {frame_count:4d} | "
                        f"Det #{det_idx} conf={det['confidence']:.2f} | "
                        f"STATE: {state.upper():7s} | "
                        f"RED={counts['red']:4d} ({100*counts['red']/circle_area:4.1f}%)  "
                        f"YELLOW={counts['yellow']:4d} ({100*counts['yellow']/circle_area:4.1f}%)  "
                        f"GREEN={counts['green']:4d} ({100*counts['green']/circle_area:4.1f}%)  "
                        f"CIRCLE={circle_area}px²"
                    )

                    # ── Save raw crop ──────────────────────────────────────
                    ts_file = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    crop_path = os.path.join(
                        DEBUG_DIR,
                        f"tl_{ts_file}_det{det_idx}_{state}.png"
                    )
                    cv2.imwrite(crop_path, crop)

                    # ── Build composite row ────────────────────────────────
                    composite = _build_composite(crop, state, counts, det_idx)
                    row_composites.append(composite)

                if row_composites:
                    # Stack all detection rows vertically
                    # Pad widths to match before vstack
                    max_w = max(r.shape[1] for r in row_composites)
                    padded = []
                    for r in row_composites:
                        if r.shape[1] < max_w:
                            pad = np.zeros(
                                (r.shape[0], max_w - r.shape[1], 3), dtype=np.uint8
                            )
                            r = np.hstack([r, pad])
                        padded.append(r)

                    last_composite = np.vstack(padded)
                    cv2.imshow(WIN_NAME, last_composite)

            # ── Key handling ───────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("[INFO] 'q' pressed – exiting early.")
                break
            elif key == ord('s') and last_composite is not None:
                save_count += 1
                ts_file = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join(DEBUG_DIR, f"overlay_save_{ts_file}_{save_count}.png")
                cv2.imwrite(save_path, last_composite)
                print(f"[SAVE] Composite saved → '{save_path}'")

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    finally:
        stream.stop()
        cv2.destroyAllWindows()
        elapsed = time.time() - start_time
        print(f"\n[DONE] Ran {elapsed:.1f} s | {frame_count} frames processed.")
        print(f"       Debug images saved to '{os.path.abspath(DEBUG_DIR)}'")


if __name__ == "__main__":
    main()
