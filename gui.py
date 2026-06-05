import torch
import cv2

from modules.cnn import CNN
from modules.face_detector import FaceDetector
from modules.emotion_predictor import EmotionPredictor
from modules.webcam_handler import WebcamHandler


def draw_overlay(frame, face_coords, emotion, confidence):
    """
    Draw emotion prediction overlay on frame.

    Args:
        frame: Image to draw on (modified in place)
        face_coords: tuple (x, y, w, h)
        emotion: str (emotion name)
        confidence: float (0.0-1.0)

    Returns:
        frame: Modified frame with overlay
    """
    x, y, w, h = face_coords

    # Draw green bounding box around face
    cv2.rectangle(frame, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)

    # Prepare text: "Emotion (confidence%)"
    text = f"{emotion.capitalize()} {confidence * 100:.0f}%"

    # Draw text above the bounding box
    text_position = (x, y - 10)
    cv2.putText(
        frame,
        text,
        text_position,
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.8,
        color=(0, 255, 0),
        thickness=2,
    )

    return frame


def main():
    """Main entry point for real-time emotion detection."""
    try:
        # Device setup
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {device}")

        # Initialize model
        print("Loading trained model...")
        model = CNN(in_channels=3, num_classes=8).to(device)
        model.load_state_dict(torch.load("models/EmotionCNN.pt", map_location=device))
        print("Model loaded successfully!")

        # Initialize components
        print("Initializing face detector...")
        face_detector = FaceDetector()

        print("Initializing emotion predictor...")
        emotion_predictor = EmotionPredictor(model, device)

        print("Starting webcam...")
        webcam_handler = WebcamHandler(target_fps=60)

        print("Real-time emotion detection started! Press 'q' to quit.")
        print("-" * 50)

        # Main loop
        for frame in webcam_handler.capture():
            # Detect face
            face_coords = face_detector.detect_face(frame)

            if face_coords is not None:
                # Predict emotion
                emotion, confidence = emotion_predictor.predict(frame, face_coords)

                # Draw overlay
                frame = draw_overlay(frame, face_coords, emotion, confidence)

                print(f"Emotion: {emotion:10s} | Confidence: {confidence:.2%}")
            else:
                # No face detected - show empty frame
                pass

            # Display frame and check for quit
            should_quit = webcam_handler.display(frame)
            if should_quit:
                print("\nQuitting...")
                break

        # Cleanup
        webcam_handler.release()
        print("Done!")

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
