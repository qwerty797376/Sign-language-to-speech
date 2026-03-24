"""
STEP 1: DATA COLLECTION — Compatible with MediaPipe 0.10.30+
=============================================================
Usage:
    python 1_collect_data.py

Controls:
    TAB    - cycle to next gesture class
    SPACE  - start collecting 200 samples for current class
    Q      - quit and save
"""

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import mediapipe as mp
import numpy as np
import json
import time

# ── New MediaPipe 0.10+ API ───────────────────────────────────────────────────
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import HandLandmarkerOptions, HandLandmarker
from mediapipe.tasks.python import BaseOptions

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATA_DIR         = "dataset"
SAMPLES_PER_CLASS = 200

GESTURE_CLASSES = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    # "hello","greetings","thank_you", "yes", "no", "please",
    # "help", "sorry", "good", "bad", "water",
    # "food", "more", "stop", "go", "i_love_you",
    # "A", "B", "C", "D", "E", "F", "G", "H", "I",
    # "J", "K", "L", "M", "N", "O", "P", "Q", "R",
    # "S", "T", "U", "V", "W", "X", "Y", "Z",
    "end"
]


# ─────────────────────────────────────────────────────────────────────────────

MODEL_PATH = "hand_landmarker.task"   # downloaded below if missing


def download_model():
    """Download the hand landmarker model if not present."""
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand_landmarker.task ...")
        import urllib.request
        url = ("https://storage.googleapis.com/mediapipe-models/"
               "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task")
        urllib.request.urlretrieve(url, MODEL_PATH)
        print("✓ Downloaded hand_landmarker.task")


def extract_landmarks_from_result(detection_result):
    if not detection_result.hand_landmarks:
        return None
    features = []
    for i in range(2):
        if i < len(detection_result.hand_landmarks):
            hand = detection_result.hand_landmarks[i]
            wx, wy, wz = hand[0].x, hand[0].y, hand[0].z
            for lm in hand:
                features.extend([lm.x - wx, lm.y - wy, lm.z - wz])
        else:
            features.extend([0.0] * 63)  # pad if second hand missing
    return features  # 126 features # 63 values


def draw_landmarks_on_frame(frame, detection_result):
    """Draw hand skeleton on frame using new API."""
    if not detection_result.hand_landmarks:
        return

    h, w = frame.shape[:2]
    for hand_landmarks in detection_result.hand_landmarks:
        # Draw connections
        connections = mp.solutions.hands.HAND_CONNECTIONS if hasattr(mp, 'solutions') else [
            (0,1),(1,2),(2,3),(3,4),
            (0,5),(5,6),(6,7),(7,8),
            (0,9),(9,10),(10,11),(11,12),
            (0,13),(13,14),(14,15),(15,16),
            (0,17),(17,18),(18,19),(19,20),
            (5,9),(9,13),(13,17),
        ]
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
        for a, b in connections:
            cv2.line(frame, pts[a], pts[b], (0, 200, 100), 2)
        for pt in pts:
            cv2.circle(frame, pt, 4, (0, 255, 200), -1)


def collect_data():
    download_model()
    os.makedirs(DATA_DIR, exist_ok=True)

    data_file = os.path.join(DATA_DIR, "raw_data.json")
    if os.path.exists(data_file):
        with open(data_file) as f:
            all_data = json.load(f)
        print(f"Loaded existing data: {sum(len(v) for v in all_data.values())} samples")
    else:
        all_data = {cls: [] for cls in GESTURE_CLASSES}

    # Add any new classes not in existing data
    for cls in GESTURE_CLASSES:
        if cls not in all_data:
            all_data[cls] = []

    # ── Setup HandLandmarker (new API) ────────────────────────────────────────
    options = HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    current_class  = GESTURE_CLASSES[0]
    collecting     = False
    collected_count = 0

    print("\n" + "="*60)
    print("HAND GESTURE DATA COLLECTOR")
    print("="*60)
    print("TAB=Next class | SPACE=Start collecting | Q=Quit")
    print(f"Current class: {current_class}")
    print("="*60 + "\n")

    with HandLandmarker.create_from_options(options) as detector:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame     = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            result   = detector.detect(mp_image)
            features = extract_landmarks_from_result(result)
            draw_landmarks_on_frame(frame, result)

            if collecting and features:
                all_data[current_class].append(features)
                collected_count += 1
                if collected_count >= SAMPLES_PER_CLASS:
                    collecting      = False
                    print(f"✓ Collected {collected_count} samples for '{current_class}'")
                    collected_count = 0

            count  = len(all_data.get(current_class, []))
            status = "COLLECTING" if collecting else "READY - press SPACE"
            color  = (0, 80, 255) if collecting else (0, 220, 100)

            cv2.rectangle(frame, (0, 0), (640, 85), (15, 15, 25), -1)
            cv2.putText(frame, f"Class: {current_class}   Saved: {count}", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (200, 200, 255), 2)
            cv2.putText(frame, status, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            if collecting:
                prog = int((collected_count / SAMPLES_PER_CLASS) * 620)
                cv2.rectangle(frame, (0, 470), (prog, 480), (0, 255, 100), -1)

            cv2.imshow("Gesture Data Collector", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord(' '):
                if features:
                    collecting      = True
                    collected_count = 0
                    print(f"Recording '{current_class}'...")
                else:
                    print("No hand detected — show your hand first!")
            elif key == 9:   # TAB
                idx           = GESTURE_CLASSES.index(current_class)
                current_class = GESTURE_CLASSES[(idx + 1) % len(GESTURE_CLASSES)]
                print(f"Switched to: {current_class}")

    cap.release()
    cv2.destroyAllWindows()

    with open(data_file, "w") as f:
        json.dump(all_data, f)

    total = sum(len(v) for v in all_data.values())
    print(f"\n✓ Saved to {data_file}  |  Total samples: {total}")
    for cls, samples in all_data.items():
        if samples:
            print(f"  {cls}: {len(samples)}")


if __name__ == "__main__":
    collect_data()