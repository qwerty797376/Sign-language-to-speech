"""
GESTURE-TO-VOICE — ASL Letter by Letter Mode
=============================================
Signs letters one by one to build words and sentences.

HOW TO USE:
    Sign letters A-Z to spell words
    Show 'E' gesture  -> finishes current word
    Show 'Z' gesture  -> speaks full sentence aloud
    Press C key       -> clear everything
    Press B key       -> delete last letter
    Press Q key       -> quit

EXAMPLE:
    Sign W, A, I, T -> show E  -> word "WAIT" added
    Sign F, O, R    -> show E  -> word "FOR" added
    Show Z                     -> speaks "WAIT FOR" aloud

Install:
    pip install ai-edge-litert mediapipe opencv-python gtts playsound pyttsx3
"""

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import mediapipe as mp
import numpy as np
import json
import time
import threading
import queue
import collections
from pathlib import Path

# ── MediaPipe new API ─────────────────────────────────────────────────────────
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import HandLandmarkerOptions, HandLandmarker
from mediapipe.tasks.python import BaseOptions

# ── TFLite backend ───────────────────────────────────────────────────────────
Interpreter = None

try:
    from ai_edge_litert.interpreter import Interpreter
    print("Using: ai-edge-litert")
except ImportError:
    pass

if Interpreter is None:
    try:
        from tflite_runtime.interpreter import Interpreter
        print("Using: tflite-runtime")
    except ImportError:
        pass

if Interpreter is None:
    try:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter
        print("Using: tensorflow lite")
    except ImportError:
        pass

if Interpreter is None:
    print("ERROR: Run this -> pip install ai-edge-litert")
    exit(1)

# ── TTS ───────────────────────────────────────────────────────────────────────
try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False

# ── CONFIG ────────────────────────────────────────────────────────────────────
TFLITE_MODEL         = "models/gesture_model.tflite"
LABELS_FILE          = "models/label_encoder.json"
HAND_MODEL_PATH      = "hand_landmarker.task"

CONFIDENCE_THRESHOLD = 0.85
SMOOTHING_WINDOW     = 10
GESTURE_HOLD_FRAMES  = 20
AUTO_SPEAK_DELAY     = 3.0

WORD_END_LABEL       = 'E'    # show E to finish a word
SENTENCE_END_LABEL   = 'Z'    # show Z to speak full sentence
# ─────────────────────────────────────────────────────────────────────────────


def download_hand_model():
    if not os.path.exists(HAND_MODEL_PATH):
        print("Downloading hand_landmarker.task (~25MB)...")
        import urllib.request
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/"
            "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
            HAND_MODEL_PATH
        )
        print("Download complete.")


# ─────────────────────────────────────────────────────────────────────────────

class TTSEngine:
    def __init__(self):
        self.pending = None
        if TTS_AVAILABLE:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 150)
            self.engine.setProperty('volume', 1.0)
            for v in self.engine.getProperty('voices'):
                if 'english' in v.name.lower() or 'en' in v.id.lower():
                    self.engine.setProperty('voice', v.id)
                    break
            print("TTS: pyttsx3")
        else:
            print("TTS: gTTS (internet required)")

    def speak(self, text):
        if text.strip():
            print(f'[SPEAK]: "{text}"')
            self.pending = text

    def flush(self):
        """Called every frame from main loop."""
        if self.pending:
            text         = self.pending
            self.pending = None
            threading.Thread(target=self._say, args=(text,), daemon=True).start()

    def _say(self, text):
        if TTS_AVAILABLE:
            try:
                engine = pyttsx3.init()
                engine.setProperty('rate', 150)
                engine.setProperty('volume', 1.0)
                engine.say(text)
                engine.runAndWait()
                engine.stop()
            except Exception as e:
                print(f"pyttsx3 error: {e}")
                self._gtts_fallback(text)
        else:
            self._gtts_fallback(text)

    def _gtts_fallback(self, text):
        try:
            from gtts import gTTS
            import tempfile
            tmp = tempfile.mktemp(suffix='.mp3')
            gTTS(text=text, lang='en', slow=False).save(tmp)
            if os.name == 'nt':
                os.system(f'start /min wmplayer "{tmp}"')
            else:
                os.system(f'mpg123 -q {tmp}')
        except Exception as e:
            print(f"gTTS error: {e}")

    def stop(self):
        pass


# ─────────────────────────────────────────────────────────────────────────────

class GesturePredictor:
    def __init__(self, model_path, labels_path):
        with open(labels_path) as f:
            self.labels = json.load(f)

        self.interp  = Interpreter(model_path=str(model_path))
        self.interp.allocate_tensors()
        self.in_idx  = self.interp.get_input_details()[0]['index']
        self.out_idx = self.interp.get_output_details()[0]['index']

        # Detect expected input size from model
        self.input_size = self.interp.get_input_details()[0]['shape'][1]
        print(f"Model input size : {self.input_size} features")
        print(f"Gestures         : {list(self.labels.values())}")

    def predict(self, features):
        x = np.array(features, dtype=np.float32).reshape(1, -1)
        self.interp.set_tensor(self.in_idx, x)
        self.interp.invoke()
        probs = self.interp.get_tensor(self.out_idx)[0]
        idx   = int(np.argmax(probs))
        return self.labels.get(str(idx), "?"), float(probs[idx])


# ─────────────────────────────────────────────────────────────────────────────

class GestureSmoother:
    def __init__(self, window=10, hold=20):
        self.history      = collections.deque(maxlen=window)
        self.hold         = hold
        self.stable_count = 0
        self.last_stable  = None

    def update(self, label, conf):
        self.history.append((label, conf) if conf >= CONFIDENCE_THRESHOLD else ("", 0))
        labels = [l for l, c in self.history if l]
        if not labels:
            self.stable_count = 0
            return None, 0
        best     = max(set(labels), key=labels.count)
        frac     = labels.count(best) / len(labels)
        avg_conf = np.mean([c for l, c in self.history if l == best])
        if best == self.last_stable:
            self.stable_count += 1
        else:
            self.stable_count = 0
            self.last_stable  = best
        return best, float(avg_conf * frac)

    @property
    def is_stable(self):
        return self.stable_count >= self.hold

    def reset(self):
        self.history.clear()
        self.stable_count = 0
        self.last_stable  = None


# ─────────────────────────────────────────────────────────────────────────────

def extract_landmarks(detection_result, input_size):
    """Auto-detects whether model needs 63 or 126 features."""
    if not detection_result.hand_landmarks:
        return None

    if input_size == 63:
        # Single hand — 63 features
        hand = detection_result.hand_landmarks[0]
        wx, wy, wz = hand[0].x, hand[0].y, hand[0].z
        features = []
        for lm in hand:
            features.extend([lm.x - wx, lm.y - wy, lm.z - wz])
        return features

    else:
        # Two hands — 126 features
        features = []
        for i in range(2):
            if i < len(detection_result.hand_landmarks):
                hand = detection_result.hand_landmarks[i]
                wx, wy, wz = hand[0].x, hand[0].y, hand[0].z
                for lm in hand:
                    features.extend([lm.x - wx, lm.y - wy, lm.z - wz])
            else:
                features.extend([0.0] * 63)
        return features


def draw_hand(frame, detection_result):
    if not detection_result.hand_landmarks:
        return
    h, w = frame.shape[:2]
    CONN = [
        (0,1),(1,2),(2,3),(3,4),
        (0,5),(5,6),(6,7),(7,8),
        (0,9),(9,10),(10,11),(11,12),
        (0,13),(13,14),(14,15),(15,16),
        (0,17),(17,18),(18,19),(19,20),
        (5,9),(9,13),(13,17),
    ]
    for hand in detection_result.hand_landmarks:
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand]
        for a, b in CONN:
            cv2.line(frame, pts[a], pts[b], (0, 200, 100), 2)
        for pt in pts:
            cv2.circle(frame, pt, 5, (0, 255, 180), -1)


def draw_ui(frame, gesture, conf, current_word, sentence, status_msg, fps):
    h, w = frame.shape[:2]

    # ── Top bar — detected gesture ────────────────────────────────
    cv2.rectangle(frame, (0, 0), (w, 90), (15, 15, 25), -1)
    bar_w = int(conf * (w - 20))
    cv2.rectangle(frame, (10, 70), (10 + bar_w, 80), (0, 220, 120), -1)
    cv2.rectangle(frame, (10, 70), (w - 10, 80), (60, 60, 60), 1)

    if gesture == SENTENCE_END_LABEL:
        color = (0, 80, 255)       # red = speak now
    elif gesture == WORD_END_LABEL:
        color = (0, 180, 255)      # orange = end word
    elif conf >= CONFIDENCE_THRESHOLD:
        color = (0, 255, 150)      # green = confident
    else:
        color = (100, 100, 200)    # blue = not sure

    cv2.putText(frame, str(gesture), (10, 55),
                cv2.FONT_HERSHEY_DUPLEX, 1.6, color, 2)
    cv2.putText(frame, f"{conf*100:.0f}%", (w - 85, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.putText(frame, f"FPS:{fps:.0f}", (w - 75, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (130, 130, 130), 1)

    # ── Bottom panel ──────────────────────────────────────────────
    cv2.rectangle(frame, (0, h - 130), (w, h), (20, 20, 35), -1)

    # Current word being spelled
    word_display = "".join(current_word) if current_word else "(spelling...)"
    cv2.putText(frame, f"Word: {word_display}", (10, h - 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 200), 2)

    # Full sentence so far
    sentence_display = " ".join(sentence[-6:]) if sentence else "(empty)"
    if len(sentence) > 6:
        sentence_display = "... " + sentence_display
    cv2.putText(frame, f"Sentence: {sentence_display}", (10, h - 68),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (220, 220, 255), 2)

    # Hint
    cv2.putText(frame,
                f"E=End Word  Z=Speak Sentence  B=Backspace  C=Clear  Q=Quit",
                (10, h - 38), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 140, 140), 1)
    cv2.putText(frame,
                f"{len(sentence)} word(s) | {len(current_word)} letter(s)",
                (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 180, 100), 1)

    # Status message
    if status_msg:
        cv2.putText(frame, status_msg, (10, h - 145),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 220, 0), 2)


# ─────────────────────────────────────────────────────────────────────────────

def run():
    # Check files
    for fpath, name in [(TFLITE_MODEL, "gesture_model.tflite"),
                         (LABELS_FILE,  "label_encoder.json")]:
        if not Path(fpath).exists():
            print(f"ERROR: '{name}' not found in models/ folder!")
            return

    download_hand_model()

    tts       = TTSEngine()
    predictor = GesturePredictor(TFLITE_MODEL, LABELS_FILE)
    smoother  = GestureSmoother(SMOOTHING_WINDOW, GESTURE_HOLD_FRAMES)

    # Auto detect num_hands from model input size
    num_hands = 2 if predictor.input_size == 126 else 1

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=HAND_MODEL_PATH),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=num_hands,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS,          30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

    sentence        = []       # list of completed words
    current_word    = []       # letters being spelled right now
    status_msg      = ""
    status_timer    = 0
    last_spoken     = None
    last_speak_time = 0
    fps_counter     = collections.deque(maxlen=30)

    print("\n" + "="*55)
    print("  GESTURE-TO-VOICE — LETTER MODE")
    print("="*55)
    print(f"  Model input : {predictor.input_size} features ({num_hands} hand)")
    print(f"  Word end    : show '{WORD_END_LABEL}' gesture")
    print(f"  Speak       : show '{SENTENCE_END_LABEL}' gesture")
    print("="*55)
    print("  Spell letters -> show E to finish word")
    print("  Show Z to speak full sentence aloud")
    print("="*55 + "\n")

    with HandLandmarker.create_from_options(options) as detector:
        while cap.isOpened():
            t0  = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                break

            frame    = cv2.flip(frame, 1)
            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result   = detector.detect(mp_image)

            draw_hand(frame, result)
            features = extract_landmarks(result, predictor.input_size)
            gesture, conf = "—", 0.0

            if features:
                raw_label, raw_conf = predictor.predict(features)
                gesture, conf       = smoother.update(raw_label, raw_conf)
                if gesture is None:
                    gesture, conf = "—", 0.0

                now = time.time()
                if (smoother.is_stable
                        and gesture != "—"
                        and conf >= CONFIDENCE_THRESHOLD
                        and gesture != last_spoken
                        and (now - last_speak_time) > AUTO_SPEAK_DELAY):

                    # ── Z = speak full sentence ───────────────────
                    if gesture == SENTENCE_END_LABEL:
                        # First finish any current word
                        if current_word:
                            word = "".join(current_word)
                            sentence.append(word)
                            current_word = []
                        if sentence:
                            text = " ".join(sentence)
                            tts.speak(text)
                            print(f'Speaking: "{text}"')
                            status_msg   = f'Speaking: "{text}"'
                            status_timer = 150
                            sentence     = []
                            last_spoken  = None
                        else:
                            status_msg   = "Sign some letters first!"
                            status_timer = 90

                    # ── E = finish current word ───────────────────
                    elif gesture == WORD_END_LABEL:
                        if current_word:
                            word = "".join(current_word)
                            sentence.append(word)
                            print(f'Word: "{word}"  ->  {sentence}')
                            status_msg   = f'Word added: "{word}"'
                            status_timer = 80
                            current_word = []
                            last_spoken  = None
                        else:
                            status_msg   = "No letters signed yet!"
                            status_timer = 60

                    # ── Normal letter ─────────────────────────────
                    else:
                        current_word.append(gesture)
                        print(f'Letter: "{gesture}"  word: {"".join(current_word)}')
                        status_msg      = f'Letter: "{gesture}"'
                        status_timer    = 40
                        last_spoken     = gesture
                        last_speak_time = now

                    smoother.reset()

            else:
                smoother.reset()
                gesture, conf = "—", 0.0

            # FPS
            fps_counter.append(time.perf_counter() - t0)
            fps = 1.0 / (np.mean(fps_counter) + 1e-9)

            # Status timer
            if status_timer > 0:
                status_timer -= 1
            else:
                status_msg = ""

            # Speak pending TTS
            tts.flush()

            draw_ui(frame, gesture, conf, current_word, sentence, status_msg, fps)
            cv2.imshow("Gesture-to-Voice", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            elif key == ord('b'):
                # Backspace — remove last letter
                if current_word:
                    removed      = current_word.pop()
                    status_msg   = f'Removed letter: "{removed}"'
                    status_timer = 60
                elif sentence:
                    # Remove last word if no letters
                    removed      = sentence.pop()
                    status_msg   = f'Removed word: "{removed}"'
                    status_timer = 60

            elif key == ord('c'):
                # Clear everything
                sentence     = []
                current_word = []
                last_spoken  = None
                status_msg   = "Cleared"
                status_timer = 60

            elif key == ord('s'):
                # Speak sentence with keyboard
                if current_word:
                    sentence.append("".join(current_word))
                    current_word = []
                if sentence:
                    text = " ".join(sentence)
                    tts.speak(text)
                    status_msg   = f'Speaking: "{text}"'
                    status_timer = 150
                    sentence     = []
                    last_spoken  = None

    cap.release()
    cv2.destroyAllWindows()
    tts.stop()
    print("\nStopped.")



if __name__ == "__main__":
    run()