import threading
import time
import cv2


class SharedData:
    """Thread-safe container for sharing the latest processed frame data."""

    def __init__(self):
        self.lock = threading.Lock()
        self.timestamp = None
        self.frame = None
        self.processed_amplitude = None

    def update_frame(self, timestamp, frame):
        """Update the shared frame data."""
        with self.lock:
            self.timestamp = timestamp
            self.frame = frame

    def update_processed(self, timestamp, amplitude):
        """Update the processed amplitude data."""
        with self.lock:
            self.timestamp = timestamp
            self.processed_amplitude = amplitude

    def get_frame(self):
        """Retrieve the current frame data."""
        with self.lock:
            return self.timestamp, self.frame

    def get_processed(self):
        """Retrieve the current processed data."""
        with self.lock:
            return self.timestamp, self.processed_amplitude


class VideoCaptureThread(threading.Thread):
    """Thread for capturing video frames."""

    def __init__(self, camera_url, shared_data, capture_limits):
        super().__init__()
        self.camera_url = camera_url
        self.shared_data = shared_data
        self.capture_limits = capture_limits
        self.stopped = False

    def run(self):
        """Main method to run the thread. Captures frames."""
        cap = cv2.VideoCapture(self.camera_url)

        while not self.stopped:
            ret, frame = cap.read()
            if ret:
                # Apply capture limits
                top, bottom, left, right = self.capture_limits
                cropped_frame = frame[top:bottom, left:right]
                self.shared_data.update_frame(time.time(), cropped_frame)
            else:
                print("Failed to capture frame")
            time.sleep(0.001)  # Short sleep to prevent CPU overuse
        cap.release()

    def stop(self):
        """Stop the thread."""
        self.stopped = True


class VideoProcessThread(threading.Thread):
    """Thread for processing video frames."""

    def __init__(self, shared_data):
        super().__init__()
        self.shared_data = shared_data
        self.stopped = False

    def run(self):
        """Main method to run the thread. Processes frames."""
        while not self.stopped:
            timestamp, frame = self.shared_data.get_frame()
            if frame is not None:
                processed_amplitude = self.process_frame(frame)
                if processed_amplitude is not None:
                    self.shared_data.update_processed(timestamp, processed_amplitude)
            time.sleep(0.001)  # Short sleep to prevent CPU overuse

    def process_frame(self, frame):
        """Process a single frame from the video stream."""
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresholded = cv2.threshold(gray_frame, 245, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(
            thresholded,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if len(contours) < 2:
            return None

        # Keep the 4 largest contours
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:4]
        # Use all 4 markers
        markers_contours = contours[:4]

        coords = []
        for cnt in markers_contours:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX, cY = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                coords.append((cX, cY))

        if len(coords) == 4:
            frame_height = frame.shape[0]
            avg_y = frame_height // 2 - int(sum(y for _, y in coords) / len(coords))
            return avg_y

        return None

    def stop(self):
        """Stop the thread."""
        self.stopped = True

