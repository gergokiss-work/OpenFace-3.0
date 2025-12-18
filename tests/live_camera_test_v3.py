#!/usr/bin/env python3
"""
Live Camera Test for OpenFace-3.0 - V3 Enhanced Display

Shows:
- Main camera feed with face bounding box
- Cropped face that goes to the model
- All 8 AU values with visual bars
- Smile detection status
- Emotion prediction
- Latency breakdown
- Real-time graphs

Usage:
    cd /Users/gergokiss/work/gergo/OpenFace-3.0
    source venv/bin/activate
    python tests/live_camera_test_v3.py
"""

import sys
import os
import time
from pathlib import Path
from collections import deque

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

AU_CONFIG = [
    {"name": "AU1",  "desc": "Inner Brow Raiser", "color": (150, 150, 150)},
    {"name": "AU2",  "desc": "Outer Brow Raiser", "color": (150, 150, 150)},
    {"name": "AU4",  "desc": "Brow Lowerer",      "color": (150, 150, 150)},
    {"name": "AU6",  "desc": "Cheek Raiser",      "color": (0, 255, 0)},      # GREEN - smile eye
    {"name": "AU9",  "desc": "Nose Wrinkler",     "color": (150, 150, 150)},
    {"name": "AU12", "desc": "Lip Corner Puller", "color": (0, 255, 255)},    # YELLOW - smile mouth
    {"name": "AU25", "desc": "Lips Part",         "color": (150, 150, 150)},
    {"name": "AU26", "desc": "Jaw Drop",          "color": (150, 150, 150)},
]

EMOTION_LABELS = ["Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger", "Contempt"]
EMOTION_COLORS = [
    (200, 200, 200),  # Neutral - gray
    (0, 255, 0),      # Happy - green
    (255, 100, 100),  # Sad - blue
    (0, 255, 255),    # Surprise - yellow
    (255, 0, 255),    # Fear - magenta
    (0, 100, 0),      # Disgust - dark green
    (0, 0, 255),      # Anger - red
    (128, 0, 128),    # Contempt - purple
]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# ============================================================================
# Face Detection
# ============================================================================

class SimpleFaceDetector:
    def __init__(self):
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
        )
        return faces

    def get_largest_face(self, frame, padding=0.2):
        faces = self.detect(frame)
        if len(faces) == 0:
            return None, None

        largest = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = largest

        pad_w = int(w * padding)
        pad_h = int(h * padding)

        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(frame.shape[1], x + w + pad_w)
        y2 = min(frame.shape[0], y + h + pad_h)

        face_crop = frame[y1:y2, x1:x2]
        bbox = (x1, y1, x2, y2)

        return face_crop, bbox


# ============================================================================
# Panel Drawing Functions
# ============================================================================

def create_info_panel(width, height):
    """Create a dark panel for information display."""
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    panel[:] = (30, 30, 30)
    return panel


def draw_section_header(panel, text, x, y, width):
    """Draw a section header with underline."""
    cv2.putText(panel, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.line(panel, (x, y + 5), (x + width - 20, y + 5), (80, 80, 80), 1)


def draw_au_panel(panel, au_values, x, y, width=280, bar_height=22):
    """Draw all AU values with bars and descriptions."""
    draw_section_header(panel, "ACTION UNITS", x, y, width)

    y_offset = y + 25

    for i, (au_cfg, value) in enumerate(zip(AU_CONFIG, au_values)):
        name = au_cfg["name"]
        desc = au_cfg["desc"]
        color = au_cfg["color"]

        # Clamp value
        value = max(0, min(1, value))

        # Background bar
        bar_x = x
        bar_y = y_offset + i * (bar_height + 4)
        bar_width = 120

        cv2.rectangle(panel, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)

        # Value bar
        value_width = int(bar_width * value)
        if value_width > 0:
            cv2.rectangle(panel, (bar_x, bar_y), (bar_x + value_width, bar_y + bar_height), color, -1)

        # Border
        cv2.rectangle(panel, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (80, 80, 80), 1)

        # AU name and value
        text_color = color if name in ["AU6", "AU12"] else (180, 180, 180)
        cv2.putText(panel, f"{name}", (bar_x + bar_width + 5, bar_y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 1)
        cv2.putText(panel, f"{value:.3f}", (bar_x + bar_width + 45, bar_y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 1)

        # Description (smaller, dimmer)
        cv2.putText(panel, desc, (bar_x + bar_width + 95, bar_y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1)

    return y_offset + 8 * (bar_height + 4)


def draw_smile_panel(panel, au6, au12, x, y, width=280):
    """Draw smile detection status."""
    draw_section_header(panel, "SMILE DETECTION", x, y, width)

    y_offset = y + 25

    # Calculate metrics
    duchenne_ratio = au6 / max(au12, 0.001)

    # Determine smile status
    if au12 < 0.15:
        status = "NO SMILE"
        status_color = (100, 100, 100)
        confidence = 0
    elif au6 < 0.08:
        status = "SOCIAL SMILE"
        status_color = (0, 165, 255)  # Orange
        confidence = min(au12 * 2, 1.0)
    elif duchenne_ratio >= 0.15:
        status = "DUCHENNE SMILE"
        status_color = (0, 255, 0)  # Green
        confidence = min((au6 + au12) / 2 * 2, 1.0)
    else:
        status = "PARTIAL SMILE"
        status_color = (0, 255, 255)  # Yellow
        confidence = min(au12 * 1.5, 1.0)

    # Status box
    cv2.rectangle(panel, (x, y_offset), (x + width - 10, y_offset + 35), status_color, 2)
    cv2.putText(panel, status, (x + 10, y_offset + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

    # Metrics
    y_offset += 45
    metrics = [
        ("AU6 (eye):", f"{au6:.4f}", (0, 255, 0)),
        ("AU12 (mouth):", f"{au12:.4f}", (0, 255, 255)),
        ("Ratio (AU6/AU12):", f"{duchenne_ratio:.4f}", (200, 200, 200)),
        ("Confidence:", f"{confidence:.1%}", status_color),
    ]

    for label, value, color in metrics:
        cv2.putText(panel, label, (x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
        cv2.putText(panel, value, (x + 120, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        y_offset += 20

    return y_offset + 10


def draw_emotion_panel(panel, emotion_logits, x, y, width=280):
    """Draw emotion prediction with all probabilities."""
    draw_section_header(panel, "EMOTION", x, y, width)

    y_offset = y + 25

    # Get probabilities
    probs = torch.softmax(torch.tensor(emotion_logits), dim=0).numpy()
    top_idx = np.argmax(probs)

    # Draw bars for all emotions
    bar_width = 100
    bar_height = 16

    for i, (label, prob) in enumerate(zip(EMOTION_LABELS, probs)):
        color = EMOTION_COLORS[i]
        is_top = (i == top_idx)

        # Background
        cv2.rectangle(panel, (x, y_offset), (x + bar_width, y_offset + bar_height), (50, 50, 50), -1)

        # Value bar
        val_width = int(bar_width * prob)
        if val_width > 0:
            bar_color = color if is_top else tuple(c // 2 for c in color)
            cv2.rectangle(panel, (x, y_offset), (x + val_width, y_offset + bar_height), bar_color, -1)

        # Border (highlighted if top)
        border_color = (255, 255, 255) if is_top else (60, 60, 60)
        cv2.rectangle(panel, (x, y_offset), (x + bar_width, y_offset + bar_height), border_color, 1)

        # Label and value
        text_color = (255, 255, 255) if is_top else (120, 120, 120)
        cv2.putText(panel, label, (x + bar_width + 5, y_offset + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)
        cv2.putText(panel, f"{prob:.1%}", (x + bar_width + 70, y_offset + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)

        y_offset += bar_height + 3

    return y_offset + 10


def draw_latency_panel(panel, latencies, x, y, width=280):
    """Draw latency breakdown."""
    draw_section_header(panel, "LATENCY (ms)", x, y, width)

    y_offset = y + 25

    labels = ["Capture", "Face Detect", "Preprocess", "Inference", "TOTAL"]

    for i, (label, val) in enumerate(zip(labels, latencies)):
        if label == "TOTAL":
            color = (0, 255, 0) if val < 100 else (0, 255, 255) if val < 200 else (0, 0, 255)
            cv2.putText(panel, f"{label}:", (x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.putText(panel, f"{val:6.1f}", (x + 100, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        else:
            color = (0, 200, 255) if label == "Inference" else (150, 150, 150)
            cv2.putText(panel, f"{label}:", (x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1)
            cv2.putText(panel, f"{val:6.1f}", (x + 100, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        y_offset += 20

    return y_offset


def draw_peaks_panel(panel, peak_au6, peak_au12, x, y, width=280):
    """Draw peak values."""
    draw_section_header(panel, "PEAK VALUES", x, y, width)

    y_offset = y + 25

    cv2.putText(panel, f"Peak AU6:", (x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1)
    cv2.putText(panel, f"{peak_au6:.4f}", (x + 80, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

    y_offset += 20
    cv2.putText(panel, f"Peak AU12:", (x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1)
    cv2.putText(panel, f"{peak_au12:.4f}", (x + 80, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

    y_offset += 20
    ratio = peak_au6 / max(peak_au12, 0.001)
    cv2.putText(panel, f"Peak Ratio:", (x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1)
    cv2.putText(panel, f"{ratio:.4f}", (x + 80, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    return y_offset + 20


def draw_history_graph(panel, history_au6, history_au12, x, y, width=280, height=80):
    """Draw real-time graph of AU6 and AU12."""
    draw_section_header(panel, "HISTORY (last 100 frames)", x, y, width)

    y_offset = y + 20
    graph_height = height

    # Graph background
    cv2.rectangle(panel, (x, y_offset), (x + width - 10, y_offset + graph_height), (40, 40, 40), -1)
    cv2.rectangle(panel, (x, y_offset), (x + width - 10, y_offset + graph_height), (60, 60, 60), 1)

    # Grid lines
    for i in range(1, 4):
        gy = y_offset + int(graph_height * i / 4)
        cv2.line(panel, (x, gy), (x + width - 10, gy), (50, 50, 50), 1)

    # Draw AU6 line (green)
    if len(history_au6) > 1:
        points = []
        for i, val in enumerate(history_au6):
            px = x + int((width - 10) * i / max(len(history_au6) - 1, 1))
            py = y_offset + graph_height - int(graph_height * min(val, 1))
            points.append((px, py))
        for i in range(len(points) - 1):
            cv2.line(panel, points[i], points[i+1], (0, 255, 0), 1)

    # Draw AU12 line (yellow)
    if len(history_au12) > 1:
        points = []
        for i, val in enumerate(history_au12):
            px = x + int((width - 10) * i / max(len(history_au12) - 1, 1))
            py = y_offset + graph_height - int(graph_height * min(val, 1))
            points.append((px, py))
        for i in range(len(points) - 1):
            cv2.line(panel, points[i], points[i+1], (0, 255, 255), 1)

    # Legend
    cv2.putText(panel, "AU6", (x + 5, y_offset + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
    cv2.putText(panel, "AU12", (x + 35, y_offset + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)

    return y_offset + graph_height + 10


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 60)
    print("OpenFace-3.0 Live Camera Test - V3 Enhanced Display")
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

    # Face detector
    print("Initializing face detector...")
    face_detector = SimpleFaceDetector()

    # Camera
    print("Opening camera...")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("ERROR: Could not open camera")
        return 1

    print("\nControls: Q=Quit, S=Save, R=Reset peaks\n")

    # State
    peak_au6 = 0.0
    peak_au12 = 0.0
    save_count = 0
    frame_count = 0
    fps_time = time.time()
    fps = 0

    # History for graph
    history_au6 = deque(maxlen=100)
    history_au12 = deque(maxlen=100)

    # Layout
    CAM_WIDTH = 480
    CAM_HEIGHT = 360
    PANEL_WIDTH = 300
    FACE_DISPLAY_SIZE = 150

    try:
        while True:
            # === CAPTURE ===
            t0 = time.perf_counter()
            ret, frame = cap.read()
            t1 = time.perf_counter()

            if not ret:
                break

            frame_count += 1

            # Resize camera frame for display
            cam_display = cv2.resize(frame, (CAM_WIDTH, CAM_HEIGHT))

            # === FACE DETECTION ===
            face_crop, bbox = face_detector.get_largest_face(frame, padding=0.15)
            t2 = time.perf_counter()

            # Create info panel
            panel = create_info_panel(PANEL_WIDTH, CAM_HEIGHT + FACE_DISPLAY_SIZE + 20)

            if face_crop is None:
                # No face detected
                cv2.putText(cam_display, "NO FACE DETECTED", (CAM_WIDTH//2 - 100, CAM_HEIGHT//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                # Empty face box
                face_display = np.zeros((FACE_DISPLAY_SIZE, FACE_DISPLAY_SIZE, 3), dtype=np.uint8)
                cv2.putText(face_display, "No Face", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)

                au_values = np.zeros(8)
                emotion_logits = np.zeros(8)
                latencies = [(t1-t0)*1000, (t2-t1)*1000, 0, 0, (t2-t0)*1000]
            else:
                # Draw bounding box on camera display
                x1, y1, x2, y2 = bbox
                scale_x = CAM_WIDTH / frame.shape[1]
                scale_y = CAM_HEIGHT / frame.shape[0]
                bx1, by1 = int(x1 * scale_x), int(y1 * scale_y)
                bx2, by2 = int(x2 * scale_x), int(y2 * scale_y)
                cv2.rectangle(cam_display, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
                cv2.putText(cam_display, "FACE", (bx1, by1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # Prepare face display
                face_display = cv2.resize(face_crop, (FACE_DISPLAY_SIZE, FACE_DISPLAY_SIZE))

                # === PREPROCESS ===
                face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                face_pil = Image.fromarray(face_rgb)
                input_tensor = transform(face_pil).unsqueeze(0).to(device)
                t3 = time.perf_counter()

                # === INFERENCE ===
                with torch.no_grad():
                    emotion_output, gaze_output, au_output = model(input_tensor)
                t4 = time.perf_counter()

                au_values = au_output.squeeze().cpu().numpy()
                emotion_logits = emotion_output.squeeze().cpu().numpy()
                latencies = [(t1-t0)*1000, (t2-t1)*1000, (t3-t2)*1000, (t4-t3)*1000, (t4-t0)*1000]

            # Extract AU6 and AU12
            au6 = float(au_values[3])
            au12 = float(au_values[5])

            # Update peaks and history
            peak_au6 = max(peak_au6, au6)
            peak_au12 = max(peak_au12, au12)
            history_au6.append(au6)
            history_au12.append(au12)

            # FPS
            if time.time() - fps_time >= 1.0:
                fps = frame_count
                frame_count = 0
                fps_time = time.time()

            # Draw panels
            y_pos = 10
            y_pos = draw_au_panel(panel, au_values, 10, y_pos) + 10
            y_pos = draw_smile_panel(panel, au6, au12, 10, y_pos) + 10
            y_pos = draw_emotion_panel(panel, emotion_logits, 10, y_pos) + 10

            # Create bottom panel for latency, peaks, and graph
            bottom_panel = create_info_panel(CAM_WIDTH, FACE_DISPLAY_SIZE + 20)
            draw_latency_panel(bottom_panel, latencies, 10, 10)
            draw_peaks_panel(bottom_panel, peak_au6, peak_au12, 170, 10)
            draw_history_graph(bottom_panel, history_au6, history_au12, 10, 120, CAM_WIDTH - 20, 40)

            # Add face display to panel
            cv2.putText(panel, "MODEL INPUT (224x224)", (10, CAM_HEIGHT + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
            panel[CAM_HEIGHT + 20:CAM_HEIGHT + 20 + FACE_DISPLAY_SIZE, 10:10 + FACE_DISPLAY_SIZE] = face_display

            # FPS on camera
            cv2.putText(cam_display, f"FPS: {fps}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Combine displays
            # Top row: camera + panel
            top_row = np.hstack([cam_display, panel[:CAM_HEIGHT, :]])
            # Bottom row: bottom panel + face panel continuation
            bottom_left = bottom_panel
            bottom_right = panel[CAM_HEIGHT:, :]

            # Make bottom_right same height as bottom_left
            if bottom_right.shape[0] < bottom_left.shape[0]:
                pad = np.zeros((bottom_left.shape[0] - bottom_right.shape[0], bottom_right.shape[1], 3), dtype=np.uint8)
                pad[:] = (30, 30, 30)
                bottom_right = np.vstack([bottom_right, pad])
            elif bottom_right.shape[0] > bottom_left.shape[0]:
                bottom_right = bottom_right[:bottom_left.shape[0], :]

            bottom_row = np.hstack([bottom_left, bottom_right])

            # Final combined
            combined = np.vstack([top_row, bottom_row])

            cv2.imshow("OpenFace-3.0 V3 - Enhanced", combined)

            # Key handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                save_count += 1
                filename = f"tests/capture_v3_{save_count:03d}.jpg"
                cv2.imwrite(filename, combined)
                print(f"\nSaved: {filename}")
                print(f"  AU values: {au_values}")
                print(f"  AU6={au6:.4f}, AU12={au12:.4f}")
            elif key == ord('r'):
                peak_au6 = 0.0
                peak_au12 = 0.0
                history_au6.clear()
                history_au12.clear()
                print("Reset peaks and history")

    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"\nSession Summary:")
    print(f"  Peak AU6 (eye): {peak_au6:.4f}")
    print(f"  Peak AU12 (mouth): {peak_au12:.4f}")
    print(f"  Peak Ratio: {peak_au6/max(peak_au12, 0.001):.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
