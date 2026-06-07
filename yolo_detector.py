import sys
import time
import cv2
import numpy as np
from ultralytics import YOLO
from traffic_camera import TrafficCamStream

class YOLODetector:
    """
    A class to run inference using YOLOv8 on image frames and extract target traffic classes.
    """
    def __init__(self):
        # Load the pretrained YOLOv8n model
        self.model = YOLO('yolov8n.pt')
        # Target classes to keep
        self.target_classes = {'person', 'car', 'motorcycle', 'bus', 'truck', 'traffic light'}
        # Confidence threshold
        self.conf_threshold = 0.5

    def detect(self, frame):
        """
        Runs YOLOv8 inference on a single BGR frame.
        
        Args:
            frame (numpy.ndarray): The input image/frame in BGR format.
            
        Returns:
            list of dicts: List of detections, each of format:
                           {"bbox": [x1, y1, x2, y2], "class_name": str, "confidence": float}
        """
        # Run inference (verbose=False to avoid logging clutter)
        results = self.model(frame, verbose=False)[0]
        
        detections = []
        for box in results.boxes:
            conf = float(box.conf[0])
            if conf < self.conf_threshold:
                continue
                
            cls_id = int(box.cls[0])
            class_name = results.names[cls_id]
            
            if class_name in self.target_classes:
                # Convert bounding box coordinates to floats
                x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "class_name": class_name,
                    "confidence": conf
                })
        return detections

if __name__ == '__main__':
    # Allows passing an alternate stream source/URL via command line arguments
    url = sys.argv[1] if len(sys.argv) > 1 else 'http://192.168.1.100:8080/video'
    
    print(f"Connecting to stream: {url}")
    print("Initializing YOLOv8n detector...")
    detector = YOLODetector()
    
    stream = TrafficCamStream(url)
    print("Press 'q' in the window to quit.")
    
    try:
        while True:
            frame = stream.read()
            if frame is not None:
                # Run YOLO detection
                detections = detector.detect(frame)
                
                # Draw boxes with labels
                for det in detections:
                    x1, y1, x2, y2 = map(int, det["bbox"])
                    class_name = det["class_name"]
                    conf = det["confidence"]
                    
                    label = f"{class_name} {conf:.2f}"
                    
                    # Choose color based on class (rich modern palette)
                    color_map = {
                        'person': (235, 122, 52),      # Modern blue/orange-ish
                        'car': (46, 204, 113),         # Emerald green
                        'motorcycle': (241, 196, 15),   # Bright yellow
                        'bus': (155, 89, 182),         # Amethyst purple
                        'truck': (230, 126, 34),       # Carrot orange
                        'traffic light': (231, 76, 60) # Alizarin red
                    }
                    color = color_map.get(class_name, (46, 204, 113))
                    
                    # Draw bounding box with anti-aliasing
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
                    
                    # Draw label background box
                    (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                    cv2.rectangle(frame, (x1, y1 - text_height - 6), (x1 + text_width, y1), color, -1)
                    
                    # Draw text label with anti-aliasing
                    cv2.putText(frame, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                
                # Display the image
                cv2.imshow('YOLO Traffic Detection', frame)
            
            # cv2.waitKey(1) handles window drawing and checks for keypresses
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # Yield time to avoid spinning the CPU at 100%
            time.sleep(0.005)
            
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        print("Stopping stream and releasing resources...")
        stream.stop()
        cv2.destroyAllWindows()
        print("Exited cleanly.")
