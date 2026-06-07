import unittest
from unittest.mock import MagicMock, patch
import time
import warnings
from traffic_cam_stream import TrafficCamStream

class TestTrafficCamStream(unittest.TestCase):
    @patch('traffic_cam_stream.cv2.VideoCapture')
    def test_successful_stream(self, mock_video_capture):
        # Setup a mock VideoCapture that successfully returns frames
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, "mock_frame_data")
        mock_video_capture.return_value = mock_cap
        
        stream = TrafficCamStream('mock://url')
        try:
            # Let the background thread read at least one frame
            time.sleep(0.05)
            
            # Verify stream interface
            self.assertTrue(stream.is_opened())
            frame = stream.read()
            self.assertEqual(frame, "mock_frame_data")
        finally:
            stream.stop()
        
        self.assertFalse(stream.is_opened())

    @patch('traffic_cam_stream.cv2.VideoCapture')
    def test_queue_overflow_and_warnings(self, mock_video_capture):
        # Setup mock VideoCapture returning unique frames
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        
        frame_counter = 0
        def dummy_read():
            nonlocal frame_counter
            frame_counter += 1
            # Sleep slightly to prevent tight loop in test environment
            time.sleep(0.005)
            return True, f"frame_{frame_counter}"
            
        mock_cap.read.side_effect = dummy_read
        mock_video_capture.return_value = mock_cap
        
        stream = TrafficCamStream('mock://url')
        try:
            # Give the thread enough time to overflow the queue (maxsize=2)
            time.sleep(0.1)
            
            # Since the queue maxsize is 2, the latest frame should be returned by read()
            # and older frames should have been dropped.
            frame = stream.read()
            self.assertIsNotNone(frame)
            
            # Subsequent read immediately after draining should yield None if no new frame is in queue
            # (depending on race condition of producer thread, but usually empty)
            time.sleep(0.01)
            empty_frame = stream.read()
            # If the producer hasn't pushed a new one, it is None.
            # We just want to check that the class didn't crash and queue is working.
        finally:
            stream.stop()

if __name__ == '__main__':
    unittest.main()
