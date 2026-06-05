import torch
import cv2
import numpy as np
from torchvision import transforms


class EmotionPredictor:
    """Predicts emotions from face crops using trained CNN model."""

    EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

    def __init__(self, model, device):
        """
        Initialize emotion predictor with trained model.

        Args:
            model: Loaded PyTorch model
            device: Device to run inference on (cpu, cuda, mps)
        """
        self.model = model
        self.device = device
        self.model.eval()

        # Same transform as used during training
        self.transform = transforms.ToTensor()

    def predict(self, frame, face_coords):
        """
        Predict emotion for a face in the frame.

        Args:
            frame: Full frame (BGR format)
            face_coords: tuple (x, y, w, h) from face detector

        Returns:
            tuple: (emotion_name, confidence_score)
                   emotion_name: str (e.g., "happy")
                   confidence_score: float (0.0-1.0)
        """
        x, y, w, h = face_coords

        # Crop and resize face to 48x48
        face_crop = frame[y : y + h, x : x + w]
        face_crop = cv2.resize(face_crop, (48, 48))

        # Convert BGR to RGB (model expects RGB)
        face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

        # Transform to tensor and add batch dimension
        face_tensor = self.transform(face_crop).unsqueeze(0).to(self.device)

        # Run inference
        with torch.no_grad():
            output = self.model(face_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, class_idx = torch.max(probabilities, dim=1)

        emotion = self.EMOTIONS[class_idx.item()]
        confidence_score = confidence.item()

        return emotion, confidence_score
