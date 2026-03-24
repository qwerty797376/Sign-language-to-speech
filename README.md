# 🤟 Hand Gesture to Voice — Complete System
### Real-time Sign Language Recognition for Differently-Abled Communication

---

## 🏗 System Architecture

```
📷 Webcam
   │
   ▼
┌─────────────────────────────┐
│   MediaPipe Hands (Google)  │  ← Detects 21 hand landmarks in ~5ms
│   (pre-trained, no training │    Runs entirely on CPU
│    needed from your side)   │
└─────────────┬───────────────┘
              │ 63 features (21 landmarks × xyz)
              ▼
┌─────────────────────────────┐
│   Gesture Classifier (MLP)  │  ← YOUR trained model (~2ms inference)
│   TFLite INT8 quantized     │    TFLite makes it 10x faster than Keras
└─────────────┬───────────────┘
              │ gesture label + confidence
              ▼
┌─────────────────────────────┐
│   Gesture Smoother          │  ← Reduces flicker, requires stable hold
│   (sliding window + hold)   │
└─────────────┬───────────────┘
              │ confirmed gesture
              ▼
┌─────────────────────────────┐
│   Sentence Builder          │  ← Accumulates words into sentences
│   + pyttsx3 TTS Engine      │    Speaks offline, ~0ms latency
└─────────────────────────────┘
```

---

## 📁 Project Structure

```
sign-language-project/
├── model_training/
│   ├── 1_collect_data.py      ← Collect gesture training data
│   └── 2_train_model.py       ← Train + export model
├── app/
│   ├── 3_gesture_to_voice.py  ← Main real-time app
│   └── 4_web_dashboard.py     ← Optional browser dashboard
├── dataset/
│   └── raw_data.json          ← Created by step 1
├── models/
│   ├── gesture_model.h5       ← Keras model (created by step 2)
│   ├── gesture_model.tflite   ← TFLite model (faster, used at runtime)
│   └── label_encoder.json     ← Gesture names
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

```bash
# 1. Clone / download this project
cd sign-language-project

# 2. Install dependencies
pip install -r requirements.txt

# For macOS (audio):
brew install mpg123

# For Linux (audio):
sudo apt install espeak python3-espeak
```

---

## 🎯 Step-by-Step: Training Your Model

### WHAT MODEL TO TRAIN
You are training a **lightweight MLP (Multi-Layer Perceptron)** neural network.

| Property            | Value                                 |
|---------------------|---------------------------------------|
| Input               | 63 features (21 landmarks × x,y,z)   |
| Architecture        | MLP: 256 → 128 → 64 → N classes      |
| Training time       | ~2-5 minutes on CPU                   |
| Inference speed     | ~2ms per frame (TFLite)               |
| Recommended data    | 150-300 samples per gesture           |
| Expected accuracy   | 95-99% with good data                 |

**Why MLP and not CNN/LSTM?**
- MediaPipe already extracts the landmarks — you don't need pixel-level features
- MLP on 63 numbers = extremely fast, no GPU needed
- For dynamic gestures (movement over time), use the LSTM option in train script

---

### STEP 1 — Collect Training Data

```bash
cd model_training
python 1_collect_data.py
```

**How to collect good data:**

| Tip | Why |
|-----|-----|
| Collect 200+ samples per gesture | More data = better accuracy |
| Vary lighting conditions | Bright, dim, side-lit |
| Vary hand distance from camera | 30cm to 80cm |
| Vary hand angle slightly | ±15° tilt |
| Multiple people if possible | Improves generalization |
| Keep background simple | Reduces false detections |

**Controls during collection:**
- `TAB` — cycle to next gesture class
- `SPACE` — start auto-collecting 200 samples
- `Q` — quit and save

**Recommended gesture set (start small):**
```
hello, thank_you, yes, no, please, help, water, food, stop, more
```
Then add ASL alphabet (A-Z) once the basics work well.

---

### STEP 2 — Train the Model

```bash
python 2_train_model.py
```

Expected output:
```
Loading data from dataset/raw_data.json...
  Dataset shape: (3000, 63)
  Classes: ['hello', 'help', 'no', 'please', 'stop', ...]
  Num classes: 10

Training...
Epoch 1/100 - accuracy: 0.45 - val_accuracy: 0.52
Epoch 20/100 - accuracy: 0.94 - val_accuracy: 0.91
...
✓ Test Accuracy: 97.33%
✓ TFLite model saved: models/gesture_model.tflite (48.2 KB)
```

**Target metrics:**
- ✅ Test accuracy > 95% — ready to use
- ⚠️  Test accuracy 90-95% — collect more data for weak classes
- ❌ Test accuracy < 90% — check data quality, ensure 150+ samples/class

---

### STEP 3 — Run the Application

```bash
cd ../app
python 3_gesture_to_voice.py
```

**Controls:**
| Key | Action |
|-----|--------|
| `SPACE` | Manually add current gesture to sentence |
| `S` | Speak the full sentence aloud |
| `C` | Clear sentence |
| `B` | Backspace (remove last word) |
| `Q` | Quit |

**Auto-speak:** After holding a gesture stable for 3 seconds, it auto-adds to sentence.

---

### STEP 4 (Optional) — Web Dashboard

```bash
pip install flask
python 4_web_dashboard.py
# Open http://localhost:5000 in browser
```

Works great on a tablet mounted nearby — the person can see their gestures recognized in real-time with big text.

---

## 🔧 Tuning for Low Latency

All these settings are at the top of `3_gesture_to_voice.py`:

```python
CONFIDENCE_THRESHOLD = 0.85   # ↑ raise to reduce false positives
SMOOTHING_WINDOW     = 10     # ↓ lower for faster response (more jitter)
GESTURE_HOLD_FRAMES  = 20     # ↓ lower to accept gestures faster
AUTO_SPEAK_DELAY     = 3.0    # ↓ lower for quicker auto-speak
```

**For fastest response** (experienced users who hold gestures cleanly):
```python
CONFIDENCE_THRESHOLD = 0.80
SMOOTHING_WINDOW     = 5
GESTURE_HOLD_FRAMES  = 10
AUTO_SPEAK_DELAY     = 1.5
```

---

## 📊 Model Performance Guide

### Getting 99%+ accuracy:

1. **Data balance** — same number of samples per class
2. **Good lighting** — avoid shadows on hand
3. **Wrist in frame** — the normalization uses wrist position
4. **Augment data** — in `2_train_model.py` you can add:
   ```python
   # Add Gaussian noise for robustness
   X_augmented = X_train + np.random.normal(0, 0.01, X_train.shape)
   X_train = np.vstack([X_train, X_augmented])
   y_train = np.hstack([y_train, y_train])
   ```

### Confusion matrix interpretation:
After training, check which classes get confused with each other. Common confusions:
- 'M' and 'N' in ASL — very similar hand shapes
- 'A' and 'S' — collect more distinct samples
- Solution: collect more samples in varied conditions for confused classes

---

## 🌐 Language Support

Change TTS language in `3_gesture_to_voice.py`:

```python
# pyttsx3 — use system voices
voices = engine.getProperty('voices')
# Pick voice by locale: en, hi, fr, es, de, ar, etc.

# gTTS — change language code
gTTS(text, lang='hi')   # Hindi
gTTS(text, lang='ar')   # Arabic
gTTS(text, lang='fr')   # French
```

---

## 🚀 Hardware Recommendations

| Use Case | Recommended Hardware |
|----------|---------------------|
| Laptop/desktop | Any webcam, Intel i5+, no GPU needed |
| Portable device | Raspberry Pi 4 + Pi Camera (use tflite) |
| Tablet | iPad/Android + web dashboard mode |
| Best latency | USB 3.0 webcam (1080p 60fps) |

**Raspberry Pi optimization:**
```python
# In 3_gesture_to_voice.py, set:
model_complexity=0   # already set — fastest MediaPipe model
# Use TFLite (already default)
# Set resolution to 320x240 for Pi:
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
```

---

## 🆘 Troubleshooting

| Problem | Solution |
|---------|----------|
| Camera not opening | Change `cv2.VideoCapture(0)` to `(1)` or `(2)` |
| No sound on Linux | `sudo apt install espeak` or `sudo apt install ffmpeg` |
| Low FPS | Lower resolution, use `model_complexity=0` |
| Too many false detections | Raise `CONFIDENCE_THRESHOLD` to 0.90 |
| Gesture not recognized | Collect more data for that class |
| pyttsx3 error on Mac | `pip install pyobjc` |

---

## 📜 License
Free for educational and assistive technology use.
Built with ❤️ to help differently-abled people communicate.
