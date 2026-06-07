"""
calibration.py
==============
Interactive calibration tool for the CrossLight Monitor.

Workflow
--------
Phase 1 — Homography
    Click exactly 4 corners of a known real-world rectangle on the road
    (e.g. a crosswalk or lane markings), going in order:
        1 = Top-Left,  2 = Top-Right,  3 = Bottom-Right,  4 = Bottom-Left
    The window closes automatically after the 4th click.
    Output: calibration_matrix.npy

Phase 2 — Danger Zone
    The same frame re-appears.  Click any number of points that outline
    the danger zone polygon on the image.
    After EACH click, press 'd' to confirm and add the point.
    Press 'f' when finished (need at least 3 points).
    Output: danger_zone.npy  (vertices in metre-space)

Usage
-----
    python calibration.py --stream_url http://192.168.1.8:8080/video
    python calibration.py --stream_url 0            # webcam
    python calibration.py --image frame.jpg         # static image
"""

import argparse
import sys
import time

import cv2
import numpy as np


# ── Real-world size of the 4-point calibration rectangle ─────────────────────
RECT_WIDTH_M  = 4.0   # metres
RECT_HEIGHT_M = 3.0   # metres

# ── Colours ───────────────────────────────────────────────────────────────────
GREEN  = (0,   210,  60)
ORANGE = (0,   165, 255)
RED    = (0,   0,   220)
CYAN   = (220, 220,   0)
WHITE  = (255, 255, 255)
BLACK  = (0,   0,     0)


# ─────────────────────────────────────────────────────────────────────────────
# Frame acquisition
# ─────────────────────────────────────────────────────────────────────────────

def _grab_frame(stream_url, image_path=None, timeout_s=10):
    """Return a single BGR frame from the stream or a static image."""
    if image_path:
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"[ERROR] Cannot read image from '{image_path}'.")
            sys.exit(1)
        print(f"[OK]  Loaded static image: {image_path}")
        return frame

    # Integer device index or HTTP URL
    src = int(stream_url) if str(stream_url).isdigit() else stream_url
    print(f"[INFO] Connecting to stream: {src} ...")

    cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG if isinstance(src, str) else cv2.CAP_ANY)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    frame       = None
    deadline    = time.time() + timeout_s
    attempt     = 0

    while time.time() < deadline:
        ok, f = cap.read()
        attempt += 1
        if ok and f is not None:
            frame = f
            if attempt >= 3:      # skip first 2 frames (often stale buffer)
                break
        time.sleep(0.08)

    cap.release()

    if frame is None:
        print(f"[ERROR] Could not grab a frame from '{src}' within {timeout_s} s.")
        print("        Make sure IP Webcam is running and the URL is correct.")
        sys.exit(1)

    # Normalise to 1280 x 720
    frame = cv2.resize(frame, (1280, 720))
    print("[OK]  Frame captured (1280 x 720).")
    return frame


# ─────────────────────────────────────────────────────────────────────────────
# Drawing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _banner(img, text, color=CYAN):
    """Draw a semi-transparent instruction banner at the top of the image."""
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (img.shape[1], 46), BLACK, -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
    cv2.putText(img, text, (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)


def _draw_phase1(img, pts):
    """Draw clicked calibration points and connecting lines."""
    labels = ["TL", "TR", "BR", "BL"]
    for i, (px, py) in enumerate(pts):
        cv2.circle(img, (px, py), 7, GREEN, -1, cv2.LINE_AA)
        cv2.putText(img, labels[i], (px + 10, py - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREEN, 2, cv2.LINE_AA)
    if len(pts) >= 2:
        for i in range(len(pts) - 1):
            cv2.line(img, pts[i], pts[i + 1], GREEN, 2, cv2.LINE_AA)
    if len(pts) == 4:
        cv2.line(img, pts[3], pts[0], GREEN, 2, cv2.LINE_AA)


def _draw_phase2(img, pts, pending):
    """Draw confirmed polygon vertices and the pending (unconfirmed) click."""
    for i, (px, py) in enumerate(pts):
        cv2.circle(img, (px, py), 6, ORANGE, -1, cv2.LINE_AA)
        cv2.putText(img, f"P{i}", (px + 8, py - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, ORANGE, 1, cv2.LINE_AA)
    if len(pts) >= 2:
        for i in range(len(pts) - 1):
            cv2.line(img, pts[i], pts[i + 1], ORANGE, 2, cv2.LINE_AA)
    if len(pts) >= 3:
        cv2.line(img, pts[-1], pts[0], ORANGE, 1, cv2.LINE_AA)
    if pending:
        cv2.circle(img, pending, 6, RED, -1, cv2.LINE_AA)
        cv2.putText(img, "press 'd' to add", (pending[0] + 10, pending[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, RED, 1, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: 4-point homography
# ─────────────────────────────────────────────────────────────────────────────

def phase1_homography(frame, rect_width_m, rect_height_m):
    """
    Interactively collect 4 image points and compute the homography H that
    maps them to a rect_width_m × rect_height_m rectangle in metres.
    Returns H (3×3 float64).
    """
    WIN  = "Phase 1 – Calibration"
    pts  = []
    done = [False]

    def on_click(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append((x, y))
            print(f"  Point {len(pts)}/4 recorded: ({x}, {y})")
            if len(pts) == 4:
                done[0] = True

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 1280, 720)
    cv2.setMouseCallback(WIN, on_click)

    print("\n" + "=" * 60)
    print("  PHASE 1: Homography Calibration")
    print("=" * 60)
    print("  Click the 4 corners of a known FLAT rectangle on the road.")
    print("  Order: Top-Left -> Top-Right -> Bottom-Right -> Bottom-Left")
    print(f"  The rectangle must be {rect_width_m} m wide x {rect_height_m} m tall.")
    print("  Press 'q' at any time to abort.")
    print()

    while not done[0]:
        display = frame.copy()
        _banner(display,
                f"Click 4 corners: TL -> TR -> BR -> BL  ({len(pts)}/4 done)  |  'q' abort")
        _draw_phase1(display, pts)
        cv2.imshow(WIN, display)
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            print("[ABORT] Calibration aborted by user.")
            cv2.destroyAllWindows()
            sys.exit(0)

    # Show the completed quadrilateral briefly
    display = frame.copy()
    _banner(display, "4 points set!  Computing homography ...", color=GREEN)
    _draw_phase1(display, pts)
    cv2.imshow(WIN, display)
    cv2.waitKey(800)
    cv2.destroyWindow(WIN)

    # Destination rectangle: TL=(0,0) TR=(W,0) BR=(W,H) BL=(0,H)  [metres]
    src = np.float32(pts)
    dst = np.float32([
        [0.0,          0.0          ],
        [rect_width_m, 0.0          ],
        [rect_width_m, rect_height_m],
        [0.0,          rect_height_m],
    ])

    H = cv2.getPerspectiveTransform(src, dst)
    np.save("calibration_matrix.npy", H)
    print("[OK]  Homography matrix saved -> 'calibration_matrix.npy'")
    print(f"      Source pts : {pts}")
    print(f"      Dest rect  : {rect_width_m} m x {rect_height_m} m")
    return H


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: Danger-zone polygon
# ─────────────────────────────────────────────────────────────────────────────

def phase2_danger_zone(frame, H):
    """
    Interactively collect a polygon that outlines the danger zone on the image.
    Transforms the pixel vertices to metre-space via H and saves them.
    """
    WIN     = "Phase 2 – Danger Zone"
    pts     = []          # confirmed pixel vertices
    pending = [None]      # last mouse-click (unconfirmed)

    def on_click(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            pending[0] = (x, y)
            print(f"  Clicked ({x}, {y}) — press 'd' to add, "
                  "or click again to change position.")

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 1280, 720)
    cv2.setMouseCallback(WIN, on_click)

    print("\n" + "=" * 60)
    print("  PHASE 2: Danger Zone Polygon")
    print("=" * 60)
    print("  Click a point, then press 'd' to confirm it.")
    print("  Repeat for each vertex (at least 3 needed).")
    print("  Press 'f' to finish and save.")
    print("  Press 'q' to abort without saving.")
    print()

    while True:
        display = frame.copy()
        n = len(pts)
        _banner(display,
                f"Click -> 'd' to add  |  {n} point(s) so far  "
                f"|  'f' finish (need >=3)  |  'q' abort",
                color=ORANGE)
        _draw_phase2(display, pts, pending[0])
        cv2.imshow(WIN, display)
        key = cv2.waitKey(30) & 0xFF

        if key == ord('q'):
            print("[ABORT] Danger zone selection aborted.")
            cv2.destroyAllWindows()
            sys.exit(0)

        elif key == ord('d'):
            if pending[0] is not None:
                pts.append(pending[0])
                print(f"  Added vertex {len(pts)}: {pending[0]}")
                pending[0] = None
            else:
                print("  [WARN] Click a point first, then press 'd'.")

        elif key == ord('f'):
            if len(pts) < 3:
                print(f"  [WARN] Need at least 3 points (have {len(pts)}). "
                      "Keep adding.")
            else:
                break

    cv2.destroyWindow(WIN)

    # Transform pixel vertices → metre-space
    px_pts = np.array([[p] for p in pts], dtype=np.float32)   # shape (N,1,2)
    m_pts  = cv2.perspectiveTransform(px_pts, H)               # shape (N,1,2)
    m_pts  = m_pts[:, 0, :]                                    # shape (N,2)

    np.save("danger_zone.npy", m_pts)
    print(f"\n[OK]  Danger zone polygon saved -> 'danger_zone.npy'")
    print(f"      {len(pts)} vertices (metres):")
    for i, (mx, my) in enumerate(m_pts):
        print(f"        P{i}: X={mx:.3f} m,  Y={my:.3f} m")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="CrossLight calibration — perspective transform & danger zone."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--stream_url",
                        help="IP camera URL (e.g. http://192.168.1.8:8080/video) "
                             "or integer webcam index.")
    source.add_argument("--image",
                        help="Path to a saved JPEG/PNG image to use instead of a live stream.")
    parser.add_argument("--width_m",  type=float, default=RECT_WIDTH_M,
                        help=f"Calibration rectangle width in metres (default {RECT_WIDTH_M}).")
    parser.add_argument("--height_m", type=float, default=RECT_HEIGHT_M,
                        help=f"Calibration rectangle height in metres (default {RECT_HEIGHT_M}).")
    args = parser.parse_args()

    width_m  = args.width_m
    height_m = args.height_m

    frame = _grab_frame(args.stream_url, args.image)
    H     = phase1_homography(frame, width_m, height_m)
    phase2_danger_zone(frame, H)

    print("\n" + "=" * 60)
    print("  Calibration complete!")
    print("  Files written:")
    print("    calibration_matrix.npy")
    print("    danger_zone.npy")
    print("  You can now run:")
    print("    python main.py --stream_url <url>")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
