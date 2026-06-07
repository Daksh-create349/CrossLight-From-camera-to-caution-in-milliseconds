import threading
import time
import queue
import warnings
import cv2

class TrafficCamStream:
    """
    A class to read frames from an MJPEG/video stream in a background thread.
    
    Attributes:
        url (str): The video stream URL or device ID.
    """
    def __init__(self, url='http://192.168.1.100:8080/video', resolution=(640, 480)):
        self.url = url
        self.width, self.height = resolution
        self._queue = queue.Queue(maxsize=1)
        self._running = False
        self._cap = None
        self._thread = None
        self._lock = threading.Lock()
        
        self.start()

    def start(self):
        """Starts the background thread to read from the stream."""
        with self._lock:
            if not self._running:
                self._running = True
                self._thread = threading.Thread(target=self._read_loop, daemon=True)
                self._thread.start()

    def _read_loop(self):
        """Background thread loop that continuously reads frames and handles reconnection."""
        while self._running:
            with self._lock:
                cap = self._cap
            
            # Recreate VideoCapture if it does not exist
            if cap is None:
                if not self._running:
                    break
                try:
                    if isinstance(self.url, str):
                        new_cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
                    else:
                        new_cap = cv2.VideoCapture(self.url)
                    new_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    new_cap.set(cv2.CAP_PROP_FOURCC,
                                cv2.VideoWriter_fourcc(*'MJPG'))
                    new_cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                    new_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                except Exception as e:
                    if self._running:
                        warnings.warn(f"Exception creating VideoCapture: {e}")
                    new_cap = None
                
                with self._lock:
                    if self._running:
                        self._cap = new_cap
                        cap = new_cap
                    else:
                        if new_cap is not None:
                            new_cap.release()
                        break

            if cap is None:
                if not self._running:
                    break
                time.sleep(2)
                continue

            # Check if VideoCapture is opened
            try:
                is_open = cap.isOpened()
            except Exception:
                is_open = False

            if not is_open:
                if not self._running:
                    break
                warnings.warn("VideoCapture is not opened. Retrying in 2 seconds...")
                time.sleep(2)
                with self._lock:
                    if self._cap is not None:
                        self._cap.release()
                        self._cap = None
                continue

            # Read the next frame
            try:
                grabbed, frame = cap.read()
            except Exception as e:
                if self._running:
                    warnings.warn(f"Exception during cap.read(): {e}")
                grabbed = False
                frame = None

            if not grabbed:
                if not self._running:
                    break
                warnings.warn("read() failed. Waiting 2 seconds and re-creating VideoCapture...")
                time.sleep(2)
                with self._lock:
                    if self._cap is not None:
                        self._cap.release()
                        self._cap = None
                continue

            try:
                self._queue.put_nowait(frame)
            except queue.Full:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._queue.put_nowait(frame)
                except queue.Full:
                    pass

    def read(self):
        """
        Returns the most recent frame from the queue.
        
        Returns:
            numpy.ndarray or None: The latest frame read, or None if no frame is available.
        """
        frame = None
        # Drain the queue to retrieve the absolute most recent frame
        while True:
            try:
                frame = self._queue.get_nowait()
            except queue.Empty:
                break
        return frame

    def stop(self):
        """Releases the capture object and stops the background thread."""
        self._running = False
        with self._lock:
            if self._cap is not None:
                try:
                    self._cap.release()
                except Exception:
                    pass
                self._cap = None
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def is_opened(self):
        """
        Returns whether the video capture is currently opened.
        
        Returns:
            bool: True if opened, False otherwise.
        """
        with self._lock:
            if self._cap is None:
                return False
            try:
                return self._cap.isOpened()
            except Exception:
                return False

if __name__ == '__main__':
    import sys
    
    # Allows passing an alternate stream source/URL via command line arguments
    url = sys.argv[1] if len(sys.argv) > 1 else 'http://192.168.1.100:8080/video'
    
    print(f"Connecting to stream: {url}")
    print("Press 'q' to quit.")
    
    stream = TrafficCamStream(url)
    
    try:
        while True:
            frame = stream.read()
            if frame is not None:
                cv2.imshow('Traffic Cam Stream', frame)
            
            # cv2.waitKey(1) handles window drawing and checks for keypresses
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # Prevent high CPU consumption of the main loop
            time.sleep(0.005)
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        print("Stopping stream and releasing resources...")
        stream.stop()
        cv2.destroyAllWindows()
        print("Exited cleanly.")
