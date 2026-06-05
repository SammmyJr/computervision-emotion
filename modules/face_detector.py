import cv2


class FaceDetector:
    """Detects faces in images using OpenCV Haar Cascade classifier."""

    def __init__(self):
        """Initialize the face detector with pre-trained Haar Cascade."""
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.cascade = cv2.CascadeClassifier(cascade_path)

        if self.cascade.empty():
            raise RuntimeError("Failed to load Haar Cascade classifier")

    def detect_face(self, frame):
        """
        Detect faces in a frame and return the first one.

        Args:
            frame: Input image (BGR format from OpenCV)

        Returns:
            tuple: (x, y, w, h) coordinates of first detected face, or None if no face found
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        if len(faces) > 0:
            x, y, w, h = faces[0]  # Return only the first face
            return (x, y, w, h)

        return None
