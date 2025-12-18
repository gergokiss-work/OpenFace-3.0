#!/usr/bin/env python3
"""
Live Camera Test for OpenFace-3.0 Smile Detection - V2 with Face Detection

This version crops the face before sending to the model.

Usage:
    cd /Users/gergokiss/work/gergo/OpenFace-3.0
    source venv/bin/activate
    python tests/live_camera_test_v2.py
"""

import sys
import os
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from model.MLT import MLT

# ============================================================================
# Configuration
# ============================================================================

WEIGHTS_PATH = "./weights/stage2_epoch_7_loss_1.1606_acc_0.5589.pth"
DEVICE = "cpu"

AU_NAMES = ["AU1", "AU2", "AU4", "AU6", "AU9", "AU12", "AU25", "AU26"]

EMOTION_LABELS = ["Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger", "Contempt"]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# ============================================================================
# Face Detection (using OpenCV Haar Cascade - simple and fast)
# ============================================================================

class SimpleFaceDetector:
    def __init__(self):
        # Use OpenCV's built-in face detector
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

    def detect(self, frame):
        """Detect faces and return list of (x, y, w, h) bounding boxes."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(100, 100)
        )
        return faces

    def get_largest_face(self, frame, padding=0.2):
        """Get the largest face with padding."""
        faces = self.detect(frame)

        if len(faces) == 0:
            return None, None

        # Get largest face
        largest = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = largest

        # Add padding
        pad_w = int(w * padding)
        pad_h = int(h * padding)

        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(frame.shape[1], x + w + pad_w)
        y2 = min(frame.shape[0], y + h + pad_h)

        # Crop face
        face_crop = frame[y1:y2, x1:x2]
        bbox = (x1, y1, x2, y2)

        return face_crop, bbox


# ============================================================================
# Visualization
# ============================================================================

def draw_au_bars(frame, au_values, x_start=10, y_start=30, bar_width=150, bar_height=20):
    for i, (name, value) in enumerate(zip(AU_NAMES, au_values)):
        y = y_start + i * (bar_height + 5)

        if name in ["AU6", "AU12"]:
            color = (0, 255, 0)
            text_color = (0, 255, 0)
        else:
            color = (100, 100, 100)
            text_color = (200, 200, 200)

        cv2.rectangle(frame, (x_start, y), (x_start + bar_width, y + bar_height), (50, 50, 50), -1)
        value_width = int(bar_width * min(max(value, 0), 1.0))
        cv2.rectangle(frame, (x_start, y), (x_start + value_width, y + bar_height), color, -1)
        cv2.rectangle(frame, (x_start, y), (x_start + bar_width, y + bar_height), (100, 100, 100), 1)

        label = f"{name}: {value:.3f}"
        cv2.putText(frame, label, (x_start + bar_width + 10, y + bar_height - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

    return frame


def draw_smile_indicator(frame, au6, au12, x=10, y=250):
    duchenne_ratio = au6 / max(au12, 0.001)

    if au12 < 0.2:
        smile_type = "No Smile"
        color = (100, 100, 100)
    elif au6 < 0.1:
        smile_type = "Social Smile (no eye)"
        color = (0, 165, 255)
    elif duchenne_ratio >= 0.2:
        smile_type = "DUCHENNE SMILE!"
        color = (0, 255, 0)
    else:
        smile_type = "Partial Smile"
        color = (0, 255, 255)

    cv2.rectangle(frame, (x, y), (x + 280, y + 100), (30, 30, 30), -1)
    cv2.rectangle(frame, (x, y), (x + 280, y + 100), color, 2)

    cv2.putText(frame, smile_type, (x + 10, y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(frame, f"AU6 (eye):   {au6:.4f}", (x + 10, y + 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(frame, f"AU12 (mouth): {au12:.4f}", (x + 10, y + 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(frame, f"Ratio: {duchenne_ratio:.4f}", (x + 10, y + 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    return frame


def draw_latency(frame, latencies, x=430, y=30):
    cv2.rectangle(frame, (x, y), (x + 200, y + 150), (30, 30, 30), -1)
    cv2.putText(frame, "LATENCY (ms)", (x + 10, y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    labels = ["Capture", "Face Det", "Preprocess", "Inference", "Total"]
    for i, (label, val) in enumerate(zip(labels, latencies)):
        color = (0, 200, 255) if label == "Inference" else (150, 150, 150)
        if label == "Total":
            color = (0, 255, 0) if val < 100 else (0, 255, 255) if val < 200 else (0, 0, 255)
        cv2.putText(frame, f"{label}: {val:6.1f}", (x + 10, y + 45 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    return frame


def draw_emotion(frame, emotion_logits, x=10, y=360):
    probs = torch.softmax(torch.tensor(emotion_logits), dim=0).numpy()
    emotion_idx = np.argmax(probs)
    emotion = EMOTION_LABELS[emotion_idx]
    confidence = probs[emotion_idx]

    color = (0, 255, 0) if emotion == "Happy" else (200, 200, 200)
    cv2.putText(frame, f"Emotion: {emotion} ({confidence:.0%})", (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return frame


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 60)
    print("OpenFace-3.0 Live Camera Test - V2 (with face detection)")
    print("=" * 60)

    if not os.path.exists(WEIGHTS_PATH):
        print(f"ERROR: Weights not found at {WEIGHTS_PATH}")
        return 1

    # Load model
    print("Loading MLT model...")
    device = torch.device(DEVICE)
    model = MLT()
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model = model.to(device)
    model.eval()
    print("Model loaded!")

    # Initialize face detector
    print("Initializing face detector...")
    face_detector = SimpleFaceDetector()
    print("Face detector ready!")

    # Open camera
    print("Opening camera...")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("ERROR: Could not open camera")
        return 1

    print("\nControls: Q=Quit, S=Save, R=Reset peaks\n")

    # Tracking
    peak_au6 = 0.0
    peak_au12 = 0.0
    save_count = 0
    frame_count = 0
    fps_time = time.time()
    fps = 0

    try:
        while True:
            # === CAPTURE ===
            t0 = time.perf_counter()
            ret, frame = cap.read()
            t1 = time.perf_counter()

            if not ret:
                break

            frame_count += 1
            display_frame = frame.copy()

            # === FACE DETECTION ===
            face_crop, bbox = face_detector.get_largest_face(frame, padding=0.15)
            t2 = time.perf_counter()

            if face_crop is None:
                cv2.putText(display_frame, "No face detected", (200, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow("OpenFace-3.0 V2", display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            # Draw face bounding box
            x1, y1, x2, y2 = bbox
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # === PREPROCESS (cropped face only!) ===
            face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            face_pil = Image.fromarray(face_rgb)
            input_tensor = transform(face_pil).unsqueeze(0).to(device)
            t3 = time.perf_counter()

            # === INFERENCE ===
            with torch.no_grad():
                emotion_output, gaze_output, au_output = model(input_tensor)
            t4 = time.perf_counter()

            # Extract values
            au_values = au_output.squeeze().cpu().numpy()
            emotion_logits = emotion_output.squeeze().cpu().numpy()

            au6 = float(au_values[3])
            au12 = float(au_values[5])

            # Update peaks
            peak_au6 = max(peak_au6, au6)
            peak_au12 = max(peak_au12, au12)

            # Latencies
            latencies = [
                (t1 - t0) * 1000,  # Capture
                (t2 - t1) * 1000,  # Face detection
                (t3 - t2) * 1000,  # Preprocess
                (t4 - t3) * 1000,  # Inference
                (t4 - t0) * 1000,  # Total
            ]

            # FPS
            if time.time() - fps_time >= 1.0:
                fps = frame_count
                frame_count = 0
                fps_time = time.time()

            # Draw visualizations
            display_frame = draw_au_bars(display_frame, au_values)
            display_frame = draw_smile_indicator(display_frame, au6, au12)
            display_frame = draw_latency(display_frame, latencies)
            display_frame = draw_emotion(display_frame, emotion_logits)

            cv2.putText(display_frame, f"FPS: {fps}", (10, 470),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
            cv2.putText(display_frame, f"Peak AU6: {peak_au6:.3f} | Peak AU12: {peak_au12:.3f}",
                        (300, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            cv2.imshow("OpenFace-3.0 V2", display_frame)

            # Key handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                save_count += 1
                cv2.imwrite(f"tests/capture_v2_{save_count:03d}.jpg", display_frame)
                print(f"\nSaved capture_v2_{save_count:03d}.jpg")
                print(f"  AU values: {au_values}")
                print(f"  AU6={au6:.4f}, AU12={au12:.4f}, Ratio={au6/max(au12,0.001):.4f}")
                print(f"  Latency: {latencies[-1]:.1f}ms total")
            elif key == ord('r'):
                peak_au6 = 0.0
                peak_au12 = 0.0
                print("Peaks reset")

    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"\nSession summary:")
    print(f"  Peak AU6: {peak_au6:.4f}")
    print(f"  Peak AU12: {peak_au12:.4f}")
    print(f"  Peak Ratio: {peak_au6/max(peak_au12, 0.001):.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
