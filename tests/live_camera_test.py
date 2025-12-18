#!/usr/bin/env python3
"""
Live Camera Test for OpenFace-3.0 Smile Detection

Tests AU6 (Cheek Raiser) and AU12 (Lip Corner Puller) in real-time
to evaluate smile detection capabilities.

Usage:
    cd /Users/gergokiss/work/gergo/OpenFace-3.0
    source venv/bin/activate
    python tests/live_camera_test.py

Controls:
    q - Quit
    s - Save current frame with AU values
    r - Reset peak values
"""

import sys
import os
import time
from pathlib import Path

# Add parent directory to path for imports
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
DEVICE = "cpu"  # Use "cuda" if available and want GPU

# AU indices in OpenFace output
AU_NAMES = ["AU1", "AU2", "AU4", "AU6", "AU9", "AU12", "AU25", "AU26"]
AU_DESCRIPTIONS = [
    "Inner Brow Raiser",
    "Outer Brow Raiser",
    "Brow Lowerer",
    "Cheek Raiser (SMILE-EYE)",      # Key for Duchenne
    "Nose Wrinkler",
    "Lip Corner Puller (SMILE-MOUTH)", # Key for smile
    "Lips Part",
    "Jaw Drop"
]

# Emotion labels
EMOTION_LABELS = ["Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger", "Contempt"]

# Image transform for model input
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# ============================================================================
# Visualization helpers
# ============================================================================

def draw_au_bars(frame, au_values, x_start=10, y_start=30, bar_width=150, bar_height=20):
    """Draw AU value bars on frame."""
    for i, (name, desc, value) in enumerate(zip(AU_NAMES, AU_DESCRIPTIONS, au_values)):
        y = y_start + i * (bar_height + 5)

        # Highlight AU6 and AU12 (smile-related)
        if name in ["AU6", "AU12"]:
            color = (0, 255, 0)  # Green for smile AUs
            text_color = (0, 255, 0)
        else:
            color = (100, 100, 100)  # Gray for others
            text_color = (200, 200, 200)

        # Background bar
        cv2.rectangle(frame, (x_start, y), (x_start + bar_width, y + bar_height), (50, 50, 50), -1)

        # Value bar
        value_width = int(bar_width * min(value, 1.0))
        cv2.rectangle(frame, (x_start, y), (x_start + value_width, y + bar_height), color, -1)

        # Border
        cv2.rectangle(frame, (x_start, y), (x_start + bar_width, y + bar_height), (100, 100, 100), 1)

        # Label
        label = f"{name}: {value:.2f}"
        cv2.putText(frame, label, (x_start + bar_width + 10, y + bar_height - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

    return frame


def draw_smile_indicator(frame, au6, au12, x=10, y=250):
    """Draw smile detection indicator."""
    # Calculate Duchenne ratio
    duchenne_ratio = au6 / max(au12, 0.01)

    # Determine smile type
    if au12 < 0.3:
        smile_type = "No Smile"
        color = (100, 100, 100)
    elif au6 < 0.15:
        smile_type = "Social Smile (no eye)"
        color = (0, 165, 255)  # Orange
    elif duchenne_ratio >= 0.25:
        smile_type = "DUCHENNE SMILE!"
        color = (0, 255, 0)  # Green
    else:
        smile_type = "Partial Smile"
        color = (0, 255, 255)  # Yellow

    # Draw indicator box
    cv2.rectangle(frame, (x, y), (x + 250, y + 80), (30, 30, 30), -1)
    cv2.rectangle(frame, (x, y), (x + 250, y + 80), color, 2)

    # Draw text
    cv2.putText(frame, smile_type, (x + 10, y + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(frame, f"AU6 (eye): {au6:.2f}", (x + 10, y + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(frame, f"AU12 (mouth): {au12:.2f}", (x + 10, y + 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    return frame


def draw_emotion(frame, emotion_logits, x=10, y=340):
    """Draw predicted emotion."""
    probs = torch.softmax(torch.tensor(emotion_logits), dim=0).numpy()
    emotion_idx = np.argmax(probs)
    emotion = EMOTION_LABELS[emotion_idx]
    confidence = probs[emotion_idx]

    # Color based on emotion
    if emotion == "Happy":
        color = (0, 255, 0)
    elif emotion == "Neutral":
        color = (200, 200, 200)
    else:
        color = (0, 165, 255)

    cv2.putText(frame, f"Emotion: {emotion} ({confidence:.0%})", (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return frame


def draw_fps(frame, fps, x=10, y=470):
    """Draw FPS counter."""
    cv2.putText(frame, f"FPS: {fps:.1f}", (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
    return frame


def draw_instructions(frame, x=10, y=450):
    """Draw control instructions."""
    cv2.putText(frame, "Q: Quit | S: Save frame | R: Reset peaks", (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
    return frame


def draw_latency(frame, capture_ms, preprocess_ms, inference_ms, postprocess_ms, total_ms, x=430, y=30):
    """Draw latency breakdown."""
    cv2.rectangle(frame, (x, y), (x + 200, y + 130), (30, 30, 30), -1)
    cv2.rectangle(frame, (x, y), (x + 200, y + 130), (80, 80, 80), 1)

    cv2.putText(frame, "LATENCY (ms)", (x + 10, y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.putText(frame, f"Capture:    {capture_ms:6.1f}", (x + 10, y + 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
    cv2.putText(frame, f"Preprocess: {preprocess_ms:6.1f}", (x + 10, y + 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
    cv2.putText(frame, f"Inference:  {inference_ms:6.1f}", (x + 10, y + 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)  # Highlight inference
    cv2.putText(frame, f"Postprocess:{postprocess_ms:6.1f}", (x + 10, y + 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

    # Total with color based on performance
    if total_ms < 50:
        total_color = (0, 255, 0)  # Green - good
    elif total_ms < 100:
        total_color = (0, 255, 255)  # Yellow - ok
    else:
        total_color = (0, 0, 255)  # Red - slow

    cv2.putText(frame, f"TOTAL:      {total_ms:6.1f}", (x + 10, y + 125),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, total_color, 2)

    return frame


def draw_peak_values(frame, peak_au6, peak_au12, x=270, y=250):
    """Draw peak AU values."""
    cv2.rectangle(frame, (x, y), (x + 150, y + 60), (30, 30, 30), -1)
    cv2.putText(frame, "Peak Values:", (x + 10, y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    cv2.putText(frame, f"AU6: {peak_au6:.2f}", (x + 10, y + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(frame, f"AU12: {peak_au12:.2f}", (x + 10, y + 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return frame


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 60)
    print("OpenFace-3.0 Live Camera Test")
    print("=" * 60)
    print()

    # Check if running in correct directory
    if not os.path.exists(WEIGHTS_PATH):
        print(f"ERROR: Weights not found at {WEIGHTS_PATH}")
        print("Please run from OpenFace-3.0 directory:")
        print("  cd /Users/gergokiss/work/gergo/OpenFace-3.0")
        print("  source venv/bin/activate")
        print("  python tests/live_camera_test.py")
        return 1

    # Load model
    print(f"Loading model from {WEIGHTS_PATH}...")
    device = torch.device(DEVICE)
    model = MLT()
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model = model.to(device)
    model.eval()
    print("Model loaded successfully!")
    print()

    # Open camera
    print("Opening camera...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Could not open camera")
        return 1

    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Camera opened successfully!")
    print()
    print("Controls:")
    print("  Q - Quit")
    print("  S - Save current frame")
    print("  R - Reset peak values")
    print()
    print("Starting live feed...")
    print()

    # Tracking variables
    frame_count = 0
    fps = 0
    last_fps_time = time.time()
    peak_au6 = 0.0
    peak_au12 = 0.0
    save_count = 0

    # Latency tracking
    latency_capture = 0.0
    latency_preprocess = 0.0
    latency_inference = 0.0
    latency_postprocess = 0.0
    latency_total = 0.0

    try:
        while True:
            # === CAPTURE ===
            t_start = time.perf_counter()
            ret, frame = cap.read()
            t_capture = time.perf_counter()

            if not ret:
                print("Failed to grab frame")
                break

            frame_count += 1

            # === PREPROCESS ===
            # Convert BGR to RGB for model
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)

            # Transform and predict
            input_tensor = transform(pil_image).unsqueeze(0).to(device)
            t_preprocess = time.perf_counter()

            # === INFERENCE ===
            with torch.no_grad():
                emotion_output, gaze_output, au_output = model(input_tensor)
            t_inference = time.perf_counter()

            # === POSTPROCESS ===
            # Extract values
            au_values = au_output.squeeze().cpu().numpy()
            emotion_logits = emotion_output.squeeze().cpu().numpy()

            au6 = float(au_values[3])   # Cheek Raiser
            au12 = float(au_values[5])  # Lip Corner Puller
            t_postprocess = time.perf_counter()

            # Calculate latencies (in ms)
            latency_capture = (t_capture - t_start) * 1000
            latency_preprocess = (t_preprocess - t_capture) * 1000
            latency_inference = (t_inference - t_preprocess) * 1000
            latency_postprocess = (t_postprocess - t_inference) * 1000
            latency_total = (t_postprocess - t_start) * 1000

            # Update peaks
            peak_au6 = max(peak_au6, au6)
            peak_au12 = max(peak_au12, au12)

            # Calculate FPS
            current_time = time.time()
            if current_time - last_fps_time >= 1.0:
                fps = frame_count / (current_time - last_fps_time)
                frame_count = 0
                last_fps_time = current_time

            # Draw visualizations
            frame = draw_au_bars(frame, au_values)
            frame = draw_smile_indicator(frame, au6, au12)
            frame = draw_emotion(frame, emotion_logits)
            frame = draw_peak_values(frame, peak_au6, peak_au12)
            frame = draw_latency(frame, latency_capture, latency_preprocess,
                                 latency_inference, latency_postprocess, latency_total)
            frame = draw_fps(frame, fps)
            frame = draw_instructions(frame)

            # Show frame
            cv2.imshow("OpenFace-3.0 Live Test", frame)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("Quitting...")
                break
            elif key == ord('s'):
                save_count += 1
                filename = f"tests/capture_{save_count:03d}.jpg"
                cv2.imwrite(filename, frame)
                print(f"\nSaved: {filename}")
                print(f"  ALL AU values: {au_values}")
                print(f"  AU6 (eye)={au6:.4f}, AU12 (mouth)={au12:.4f}, Ratio={au6/max(au12,0.01):.4f}")
                print(f"  Latency: capture={latency_capture:.1f}ms, preprocess={latency_preprocess:.1f}ms, "
                      f"inference={latency_inference:.1f}ms, total={latency_total:.1f}ms")
            elif key == ord('r'):
                peak_au6 = 0.0
                peak_au12 = 0.0
                print("Peak values reset")

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    finally:
        cap.release()
        cv2.destroyAllWindows()

    # Print summary
    print()
    print("=" * 60)
    print("Session Summary")
    print("=" * 60)
    print(f"Peak AU6 (eye involvement): {peak_au6:.3f}")
    print(f"Peak AU12 (smile intensity): {peak_au12:.3f}")
    print(f"Peak Duchenne ratio: {peak_au6/max(peak_au12, 0.01):.3f}")
    print()

    if peak_au12 > 0.5 and peak_au6 > 0.2:
        print("Result: Duchenne smile detected during session!")
    elif peak_au12 > 0.5:
        print("Result: Smile detected, but limited eye involvement")
    else:
        print("Result: No significant smile detected")

    return 0


if __name__ == "__main__":
    sys.exit(main())
