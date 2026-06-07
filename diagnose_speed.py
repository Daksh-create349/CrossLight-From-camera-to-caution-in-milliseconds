"""
diagnose_speed.py
=================
Step-by-step diagnostic script for checking homography scaling, real-world
coordinate transformations, and metric speed estimation.

Usage
-----
    python diagnose_speed.py --stream_url http://192.168.1.8:8080/video
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np

# Try to import project classes
try:
    from yolo_detector import YOLODetector
except ImportError:
    try:
        from detector import YOLODetector
    except ImportError:
        YOLODetector = None

try:
    from traffic_camera import TrafficCamStream
except ImportError:
    TrafficCamStream = None


def compute_iou(boxA, boxB):
    """Compute Intersection-over-Union of two bounding boxes [x1, y1, x2, y2]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    unionArea = float(boxAArea + boxBArea - interArea)
    if unionArea <= 0.0:
        return 0.0
    return interArea / unionArea


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose homography scaling and speed calculation step-by-step."
    )
    parser.add_argument(
        "--stream_url",
        default="http://192.168.1.8:8080/video",
        help="IP camera stream URL or webcam index.",
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  CrossLight Speed & Scaling Diagnostic")
    print("=" * 60)

    # ── 1. Load Calibration Matrix ───────────────────────────────────────────
    calib_file = "calibration_matrix.npy"
    if not os.path.isfile(calib_file):
        print(f"[ERROR] '{calib_file}' not found. Run calibration.py first.")
        sys.exit(1)

    H = np.load(calib_file)
    print(f"[OK]  Loaded homography matrix H from '{calib_file}':")
    print(H)

    # ── 2. Initialize Detector & Camera Stream ────────────────────────────────
    if YOLODetector is None:
        print("[ERROR] Could not import YOLODetector class.")
        sys.exit(1)
    detector = YOLODetector()

    src = int(args.stream_url) if args.stream_url.isdigit() else args.stream_url
    print(f"[INFO] Connecting to stream: {src} ...")

    cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG if isinstance(src, str) else cv2.CAP_ANY)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print(f"[ERROR] Could not open video stream '{src}'.")
        sys.exit(1)

    print("[INFO] Waiting for a frame containing at least one car...")
    print("       Press 'q' in the OpenCV window to abort.")

    car_box_1 = None
    frame_1 = None
    t1 = 0.0

    cv2.namedWindow("Diagnostic Feed", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Diagnostic Feed", 1280, 720)

    # Grab first frame with a car
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            time.sleep(0.01)
            continue

        # Resize to system resolution
        frame = cv2.resize(frame, (1280, 720))
        detections = detector.detect(frame)

        # Draw current detections for visual feedback
        draw_img = frame.copy()
        cv2.putText(draw_img, "Searching for a vehicle...", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2, cv2.LINE_AA)

        cars = [d for d in detections if d["class_name"] in ("car", "truck", "bus") and d["confidence"] > 0.5]

        for d in detections:
            x1, y1, x2, y2 = map(int, d["bbox"])
            color = (0, 255, 0) if d["class_name"] == "car" else (255, 255, 0)
            cv2.rectangle(draw_img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(draw_img, f"{d['class_name']} {d['confidence']:.2f}",
                        (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        cv2.imshow("Diagnostic Feed", draw_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[ABORT] Diagnostic aborted.")
            cap.release()
            cv2.destroyAllWindows()
            sys.exit(0)

        if cars:
            # Found a car!
            car_box_1 = cars[0]["bbox"]  # [x1, y1, x2, y2]
            frame_1 = frame
            t1 = time.time()
            print(f"[OK]  Found vehicle at bbox: {car_box_1}")
            break

    # ── 3. Transform contact point to world coordinates (Frame 1) ─────────────
    x1, y1, x2, y2 = car_box_1
    cx1 = (x1 + x2) / 2.0
    cy1 = y2  # Bottom-centre is the ground contact point

    pt1 = np.array([[[cx1, cy1]]], dtype=np.float32)
    world_pt1 = cv2.perspectiveTransform(pt1, H)[0, 0]
    wx1, wy1 = world_pt1
    print(f"      Frame 1 pixel contact point: ({cx1:.1f}, {cy1:.1f})")
    print(f"      Frame 1 world coordinate   : X={wx1:.3f} m, Y={wy1:.3f} m")

    # ── 4. Measure Car Width in Meters ────────────────────────────────────────
    # Bottom-left and bottom-right corners
    bl_pt = np.array([[[x1, y2]]], dtype=np.float32)
    br_pt = np.array([[[x2, y2]]], dtype=np.float32)
    w_bl = cv2.perspectiveTransform(bl_pt, H)[0, 0]
    w_br = cv2.perspectiveTransform(br_pt, H)[0, 0]
    car_width_m = float(np.linalg.norm(w_br - w_bl))

    print(f"      Vehicle pixel width : {x2 - x1:.1f} px")
    print(f"      Vehicle world width : {car_width_m:.3f} m")

    if car_width_m < 1.0 or car_width_m > 3.0:
        print(f"[WARN] Calibration scale is likely wrong! A typical car width is ~1.8 m,")
        print(f"       but the transformed width is {car_width_m:.3f} m.")
        print("       Check the real-world rectangle dimensions used during calibration.")
    else:
        print("[PASS] Vehicle world width is within a realistic range (~1.0 - 3.0 m).")

    # ── 5. Grab next frame after a delay and match the vehicle ────────────────
    print("[INFO] Waiting 0.15 seconds to grab second frame...")
    time.sleep(0.15)

    car_box_2 = None
    t2 = 0.0

    # Read consecutive frames to find the same vehicle
    attempts = 0
    while attempts < 30:
        ok, frame_2 = cap.read()
        if not ok or frame_2 is None:
            time.sleep(0.01)
            continue

        t2 = time.time()
        frame_2 = cv2.resize(frame_2, (1280, 720))
        detections_2 = detector.detect(frame_2)

        cars_2 = [d for d in detections_2 if d["class_name"] in ("car", "truck", "bus") and d["confidence"] > 0.5]

        # Find the vehicle that best matches the first one via IoU
        best_match = None
        best_iou = 0.0

        for c in cars_2:
            iou = compute_iou(car_box_1, c["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_match = c

        # If IoU is very small or zero, fallback to closest center pixel distance
        if best_match is not None and best_iou > 0.1:
            car_box_2 = best_match["bbox"]
            print(f"[OK]  Matched vehicle in Frame 2 (IoU = {best_iou:.2f}) at bbox: {car_box_2}")
            break
        elif cars_2:
            # Fallback to closest Euclidean distance
            min_dist = float("inf")
            closest_car = None
            cx1_curr = (car_box_1[0] + car_box_1[2]) / 2.0
            cy1_curr = (car_box_1[1] + car_box_1[3]) / 2.0
            for c in cars_2:
                box = c["bbox"]
                cx2 = (box[0] + box[2]) / 2.0
                cy2 = (box[1] + box[3]) / 2.0
                dist = np.hypot(cx2 - cx1_curr, cy2 - cy1_curr)
                if dist < min_dist:
                    min_dist = dist
                    closest_car = c
            if min_dist < 150:  # must be within 150px
                car_box_2 = closest_car["bbox"]
                print(f"[OK]  Matched vehicle via center distance ({min_dist:.1f} px) at bbox: {car_box_2}")
                break

        attempts += 1
        time.sleep(0.03)

    cap.release()
    cv2.destroyAllWindows()

    if car_box_2 is None:
        print("[ERROR] Could not match the vehicle in consecutive frames.")
        sys.exit(1)

    # ── 6. Transform contact point (Frame 2) ──────────────────────────────────
    x1_2, y1_2, x2_2, y2_2 = car_box_2
    cx2 = (x1_2 + x2_2) / 2.0
    cy2 = y2_2

    pt2 = np.array([[[cx2, cy2]]], dtype=np.float32)
    world_pt2 = cv2.perspectiveTransform(pt2, H)[0, 0]
    wx2, wy2 = world_pt2

    print(f"      Frame 2 pixel contact point: ({cx2:.1f}, {cy2:.1f})")
    print(f"      Frame 2 world coordinate   : X={wx2:.3f} m, Y={wy2:.3f} m")

    # ── 7. Calculate Speed ───────────────────────────────────────────────────
    dt = t2 - t1
    distance_m = float(np.linalg.norm(world_pt2 - world_pt1))
    speed_ms = distance_m / dt
    speed_kmh = speed_ms * 3.6

    print("\n" + "-" * 60)
    print("  SPEED RESULTS")
    print("-" * 60)
    print(f"  Time Difference (dt) : {dt:.4f} s")
    print(f"  World Distance Moved : {distance_m:.4f} m")
    print(f"  Calculated Speed     : {speed_ms:.2f} m/s ({speed_kmh:.2f} km/h)")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
