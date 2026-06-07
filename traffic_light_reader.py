import cv2
import numpy as np
import sys

class TrafficLightReader:
    """
    Analyzes detected traffic lights in a frame using HSV color histograms
    to classify the current active signal color (red, yellow, or green).
    """
    def __init__(self, threshold_ratio=0.05):
        self.threshold_ratio = threshold_ratio

    # ── New thresholds ────────────────────────────────────────────────────
    # Fraction of the circle area that a colour must exceed to be declared.
    MIN_COLOR_RATIO   = 0.15   # 15 %
    MIN_BRIGHTNESS_V  = 100    # HSV Value channel minimum
    CIRCLE_RADIUS_K   = 0.4    # radius = k * min(w, h)

    def analyze_crop(self, crop):
        """
        Analyzes a cropped traffic light region.

        Improvements over the original implementation:
        - Only examines the central circular region (radius = 0.4 * min(w,h))
          to exclude the dark casing / background that would dilute colour counts.
        - Applies a minimum brightness gate (V > 100) so that dim / inactive
          bulbs are ignored.
        - Returns 'unknown' unless the dominant colour covers ≥ 15 % of the
          examined circle area.

        Args:
            crop (numpy.ndarray): BGR image crop of the traffic light.

        Returns:
            str: 'red', 'yellow', 'green', or 'unknown'.
            dict: pixel counts {'red': int, 'yellow': int, 'green': int,
                                'circle_area': int}  (second return value).
        """
        if crop is None or crop.size == 0:
            return 'unknown', {'red': 0, 'yellow': 0, 'green': 0, 'circle_area': 0}

        h, w = crop.shape[:2]
        if h == 0 or w == 0:
            return 'unknown', {'red': 0, 'yellow': 0, 'green': 0, 'circle_area': 0}

        # ── Build circular ROI mask ───────────────────────────────────────
        radius = int(self.CIRCLE_RADIUS_K * min(w, h))
        radius = max(radius, 1)
        cy, cx = h // 2, w // 2

        circle_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(circle_mask, (cx, cy), radius, 255, -1)
        circle_area = int(cv2.countNonZero(circle_mask))
        if circle_area == 0:
            return 'unknown', {'red': 0, 'yellow': 0, 'green': 0, 'circle_area': 0}

        # ── Convert to HSV and apply brightness gate ──────────────────────
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        v_channel = hsv[:, :, 2]
        bright_mask = (v_channel > self.MIN_BRIGHTNESS_V).astype(np.uint8) * 255

        # Combined: inside circle AND bright enough
        roi_mask = cv2.bitwise_and(circle_mask, bright_mask)

        # ── HSV colour ranges ─────────────────────────────────────────────
        lower_red1 = np.array([0,   70,  70])
        upper_red1 = np.array([10,  255, 255])
        lower_red2 = np.array([160, 70,  70])
        upper_red2 = np.array([180, 255, 255])
        lower_yellow = np.array([15,  70, 70])
        upper_yellow = np.array([35, 255, 255])
        lower_green  = np.array([36,  70, 70])
        upper_green  = np.array([95, 255, 255])

        # ── Build colour masks gated by ROI ──────────────────────────────
        def _masked(lower, upper):
            m = cv2.inRange(hsv, lower, upper)
            return cv2.bitwise_and(m, roi_mask)

        mask_red = cv2.bitwise_or(_masked(lower_red1, upper_red1),
                                   _masked(lower_red2, upper_red2))
        mask_yellow = _masked(lower_yellow, upper_yellow)
        mask_green  = _masked(lower_green,  upper_green)

        red_px    = int(cv2.countNonZero(mask_red))
        yellow_px = int(cv2.countNonZero(mask_yellow))
        green_px  = int(cv2.countNonZero(mask_green))

        counts = {'red': red_px, 'yellow': yellow_px, 'green': green_px,
                  'circle_area': circle_area}

        dominant = max(('red', red_px), ('yellow', yellow_px), ('green', green_px),
                       key=lambda t: t[1])
        colour, best_count = dominant

        if best_count >= self.MIN_COLOR_RATIO * circle_area:
            return colour, counts
        return 'unknown', counts


    def get_state(self, detections, frame):
        """
        Finds all traffic light detections, classifies their color, and resolves
        to a single overall traffic state.
        
        Args:
            detections (list of dict): YOLO detection dictionaries.
            frame (numpy.ndarray): Complete BGR frame image.
            
        Returns:
            str: 'red', 'yellow', 'green', or 'unknown'.
        """
        states = []
        h, w = frame.shape[:2]
        
        for det in detections:
            if det.get("class_name") == "traffic light":
                bbox = det.get("bbox")
                if bbox is None:
                    continue
                
                # Extract coordinates
                x1, y1, x2, y2 = map(int, bbox)
                
                # Clip to frame boundary
                x1 = max(0, min(x1, w - 1))
                y1 = max(0, min(y1, h - 1))
                x2 = max(0, min(x2, w - 1))
                y2 = max(0, min(y2, h - 1))
                
                if x2 <= x1 or y2 <= y1:
                    continue
                
                crop = frame[y1:y2, x1:x2]
                state, _ = self.analyze_crop(crop)   # unpack (colour, counts)
                if state != 'unknown':
                    states.append(state)
                    
        if not states:
            return 'unknown'
            
        # Perform majority voting if multiple traffic lights are detected
        from collections import Counter
        counter = Counter(states)
        most_common_state = counter.most_common(1)[0][0]
        return most_common_state

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Test TrafficLightReader")
    parser.add_argument("--image", type=str, help="Path to a traffic light image.")
    args = parser.parse_args()
    
    reader = TrafficLightReader()
    
    if args.image:
        print(f"Loading image: {args.image}...")
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"Error: Could not load image from {args.image}")
            sys.exit(1)
            
        from detector import YOLODetector
        detector = YOLODetector()
        detections = detector.detect(frame)
        
        print(f"Detections found: {detections}")
        state = reader.get_state(detections, frame)
        print(f"Detected Traffic Light State: {state}")
    else:
        print("No --image argument provided. Running self-test with generated traffic light crops...")
        
        # ── Mock crops: each is a 30×30 square with the active bulb centred ──
        # The new analyze_crop uses a circle of radius int(0.4 * 30) = 12 px
        # centred at (15, 15), so the lit bulb must be at (15, 15).

        # Red: bright red circle at centre
        red_crop = np.full((30, 30, 3), 25, dtype=np.uint8)   # dark casing bg
        cv2.circle(red_crop, (15, 15), 10, (0, 0, 255), -1)   # BGR red

        # Green: bright green circle at centre
        green_crop = np.full((30, 30, 3), 25, dtype=np.uint8)
        cv2.circle(green_crop, (15, 15), 10, (0, 255, 0), -1)  # BGR green

        # Yellow: bright yellow circle at centre
        yellow_crop = np.full((30, 30, 3), 25, dtype=np.uint8)
        cv2.circle(yellow_crop, (15, 15), 10, (0, 220, 255), -1)  # BGR yellow

        # Analyze crops – analyze_crop returns (colour, counts)
        red_state,    red_counts    = reader.analyze_crop(red_crop)
        green_state,  green_counts  = reader.analyze_crop(green_crop)
        yellow_state, yellow_counts = reader.analyze_crop(yellow_crop)

        print(f"Mock Red    result: {red_state:7s} | counts: {red_counts}")
        print(f"Mock Green  result: {green_state:7s} | counts: {green_counts}")
        print(f"Mock Yellow result: {yellow_state:7s} | counts: {yellow_counts}")

        assert red_state    == 'red',    f"Expected 'red',    got {red_state}"
        assert green_state  == 'green',  f"Expected 'green',  got {green_state}"
        assert yellow_state == 'yellow', f"Expected 'yellow', got {yellow_state}"
        print("Self-test passed successfully!")

