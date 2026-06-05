import cv2
import time


class WebcamHandler:
    """Handles webcam capture, display, and FPS regulation."""

    def __init__(self, target_fps=12):
        """
        Initialize webcam handler.

        Args:
            target_fps: Target frames per second (default 12)
        """
        self.target_fps = target_fps
        self.frame_delay = 1.0 / target_fps
        self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            raise RuntimeError("Failed to open webcam. Check if camera is available.")

        # Set camera properties for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def capture(self):
        """
        Generator that yields frames from webcam at target FPS.

        Yields:
            frame: BGR image from webcam
        """
        last_frame_time = time.time()

        while True:
            ret, frame = self.cap.read()

            if not ret:
                print("Error reading frame from webcam")
                break

            # Regulate FPS
            elapsed_time = time.time() - last_frame_time
            sleep_time = self.frame_delay - elapsed_time

            if sleep_time > 0:
                time.sleep(sleep_time)

            last_frame_time = time.time()

            yield frame

    def display(self, frame, window_name="Emotion Detection"):
        """
        Display frame and check for user input.

        Args:
            frame: Image to display
            window_name: Title of the window

        Returns:
            bool: True if user pressed 'q' (should quit), False otherwise
        """
        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            return True

        return False

    def release(self):
        """Clean up resources."""
        self.cap.release()
        cv2.destroyAllWindows()
