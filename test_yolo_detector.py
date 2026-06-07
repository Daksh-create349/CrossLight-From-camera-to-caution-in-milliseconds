import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from yolo_detector import YOLODetector

class TestYOLODetector(unittest.TestCase):
    @patch('yolo_detector.YOLO')
    def test_detector_inference(self, mock_yolo_cls):
        # Create a mock YOLO model instance
        mock_model = MagicMock()
        mock_yolo_cls.return_value = mock_model
        
        # Setup mock results structure returned by YOLO model
        mock_results = MagicMock()
        
        # Mock bounding box 1: Target class, high confidence (should keep)
        mock_box_1 = MagicMock()
        mock_box_1.conf = [0.8]
        mock_box_1.cls = [0] # class 0 (person)
        mock_box_1.xyxy = [np.array([10.0, 20.0, 30.0, 40.0])]
        
        # Mock bounding box 2: Target class, low confidence (should discard)
        mock_box_2 = MagicMock()
        mock_box_2.conf = [0.4] # Below 0.5 threshold
        mock_box_2.cls = [2] # class 2 (car)
        mock_box_2.xyxy = [np.array([50.0, 60.0, 70.0, 80.0])]
        
        # Mock bounding box 3: Target class, high confidence (should keep)
        mock_box_3 = MagicMock()
        mock_box_3.conf = [0.9]
        mock_box_3.cls = [2] # class 2 (car)
        mock_box_3.xyxy = [np.array([100.0, 110.0, 120.0, 130.0])]
        
        # Mock bounding box 4: Non-target class, high confidence (should discard)
        mock_box_4 = MagicMock()
        mock_box_4.conf = [0.75]
        mock_box_4.cls = [15] # class 15 (cat)
        mock_box_4.xyxy = [np.array([200.0, 210.0, 220.0, 230.0])]
        
        mock_results.boxes = [mock_box_1, mock_box_2, mock_box_3, mock_box_4]
        mock_results.names = {0: 'person', 2: 'car', 15: 'cat'}
        
        mock_model.return_value = [mock_results]
        
        # Initialize detector
        detector = YOLODetector()
        
        # Run detect
        fake_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        detections = detector.detect(fake_frame)
        
        # We expect 2 detections: mock_box_1 (person) and mock_box_3 (car)
        self.assertEqual(len(detections), 2)
        
        # Check person detection
        self.assertEqual(detections[0]["class_name"], 'person')
        self.assertEqual(detections[0]["confidence"], 0.8)
        self.assertEqual(detections[0]["bbox"], [10.0, 20.0, 30.0, 40.0])
        
        # Check car detection
        self.assertEqual(detections[1]["class_name"], 'car')
        self.assertEqual(detections[1]["confidence"], 0.9)
        self.assertEqual(detections[1]["bbox"], [100.0, 110.0, 120.0, 130.0])

if __name__ == '__main__':
    unittest.main()
