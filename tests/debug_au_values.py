#!/usr/bin/env python3
"""
Debug script to see raw AU values from OpenFace-3.0
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from model.MLT import MLT

WEIGHTS_PATH = "./weights/stage2_epoch_7_loss_1.1606_acc_0.5589.pth"

AU_NAMES = ["AU1", "AU2", "AU4", "AU6", "AU9", "AU12", "AU25", "AU26"]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Simple face detector
class FaceDetector:
    def __init__(self):
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.cascade = cv2.CascadeClassifier(cascade_path)

    def get_face(self, frame, padding=0.2):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
        if len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda f: f[2]*f[3])
        pad_w, pad_h = int(w * padding), int(h * padding)
        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(frame.shape[1], x + w + pad_w)
        y2 = min(frame.shape[0], y + h + pad_h)
        return frame[y1:y2, x1:x2]


def main():
    print("Loading model...")
    device = torch.device("cpu")
    model = MLT()
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model.eval()

    face_detector = FaceDetector()

    print("Opening camera...")
    cap = cv2.VideoCapture(0)

    print("\nPress SPACE to capture and analyze, Q to quit\n")
    print("=" * 70)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Show live feed
        display = frame.copy()
        cv2.putText(display, "SPACE=Capture, Q=Quit", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Debug AU Values", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord(' '):
            # Capture and analyze
            face = face_detector.get_face(frame)

            if face is None:
                print("\n[!] No face detected\n")
                continue

            # Prepare input
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face_pil = Image.fromarray(face_rgb)
            input_tensor = transform(face_pil).unsqueeze(0)

            # Run model
            with torch.no_grad():
                emotion_out, gaze_out, au_out = model(input_tensor)

            # Get RAW values (no clipping!)
            au_raw = au_out.squeeze().numpy()
            emotion_raw = emotion_out.squeeze().numpy()

            print("\n" + "=" * 70)
            print("RAW AU OUTPUT (no clipping/sigmoid):")
            print("-" * 70)
            for i, (name, val) in enumerate(zip(AU_NAMES, au_raw)):
                bar_len = int(abs(val) * 20)
                if val >= 0:
                    bar = "+" * bar_len
                    print(f"  {name}: {val:+8.4f}  |{'':>20}|{bar}")
                else:
                    bar = "-" * bar_len
                    print(f"  {name}: {val:+8.4f}  |{bar:>20}|")

            print("-" * 70)
            print(f"  Min: {au_raw.min():.4f}, Max: {au_raw.max():.4f}, Mean: {au_raw.mean():.4f}")

            # Apply sigmoid to see 0-1 range
            au_sigmoid = 1 / (1 + np.exp(-au_raw))
            print("\nAFTER SIGMOID (0-1 range):")
            print("-" * 70)
            for i, (name, val) in enumerate(zip(AU_NAMES, au_sigmoid)):
                bar = "#" * int(val * 30)
                highlight = " <-- SMILE" if name in ["AU6", "AU12"] else ""
                print(f"  {name}: {val:.4f}  |{bar:30}|{highlight}")

            # Emotion
            emotion_probs = np.exp(emotion_raw) / np.sum(np.exp(emotion_raw))
            emotions = ["Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger", "Contempt"]
            top_emotion = emotions[np.argmax(emotion_probs)]
            print(f"\nEMOTION: {top_emotion} ({emotion_probs.max():.1%})")

            print("=" * 70)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
