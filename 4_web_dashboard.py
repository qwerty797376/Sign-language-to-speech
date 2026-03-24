"""
OPTIONAL: WEB DASHBOARD (Flask)
================================
Streams camera feed + gesture results to a browser.
Useful for tablets, wall displays, or remote monitoring.

Usage:
    pip install flask
    python 4_web_dashboard.py
    Open: http://localhost:5000
"""

from flask import Flask, Response, render_template_string, jsonify
import cv2
import mediapipe as mp
import numpy as np
import json
import time
import threading
from pathlib import Path

# Reuse predictor + smoother from main app
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))

app = Flask(__name__)

# ── Global state ─────────────────────────────────────────────────
state = {
    "gesture":    "—",
    "confidence": 0.0,
    "sentence":   [],
    "fps":        0.0,
}
frame_lock   = threading.Lock()
latest_frame = None

# ── HTML Template ─────────────────────────────────────────────────
HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Gesture-to-Voice Dashboard</title>
  <style>
    * { margin:0; padding:0; box-sizing:border-box; }
    body {
      background: #0a0a1a;
      color: #e0e0ff;
      font-family: 'Courier New', monospace;
      display: flex; flex-direction: column; align-items: center;
      min-height: 100vh; padding: 20px; gap: 20px;
    }
    h1 { color: #00ffaa; font-size: 1.8rem; letter-spacing: 4px; margin-top:10px; }
    .grid { display: grid; grid-template-columns: 640px 1fr; gap: 20px; width:100%; max-width:1000px; }
    .camera-box {
      border: 2px solid #00ffaa33;
      border-radius: 12px; overflow: hidden; background:#000;
    }
    .camera-box img { width: 100%; display: block; }
    .info-panel { display: flex; flex-direction: column; gap: 16px; }
    .card {
      background: #12122a;
      border: 1px solid #ffffff15;
      border-radius: 12px; padding: 20px;
    }
    .gesture-display {
      font-size: 3rem; font-weight: bold;
      color: #00ffaa; text-align: center;
      text-shadow: 0 0 20px #00ffaa66;
      min-height: 80px; display:flex;
      align-items:center; justify-content:center;
    }
    .conf-bar-wrap { background: #1a1a35; border-radius:6px; height:12px; overflow:hidden; }
    .conf-bar { height:100%; background: linear-gradient(90deg, #00ffaa, #00aaff); border-radius:6px; transition: width 0.15s; }
    .label { font-size:0.75rem; color:#888; text-transform:uppercase; letter-spacing:2px; margin-bottom:6px; }
    .sentence-box {
      font-size: 1.1rem; color: #aaeeff; min-height: 60px;
      word-break: break-word; line-height: 1.6;
    }
    .btn {
      padding: 10px 20px; border-radius: 8px; border: 1px solid #00ffaa44;
      background: #0a1a14; color: #00ffaa; cursor: pointer;
      font-family: inherit; letter-spacing: 1px;
      transition: all 0.2s;
    }
    .btn:hover { background: #00ffaa22; }
    .btn-row { display: flex; gap: 10px; flex-wrap: wrap; }
    .fps { font-size: 0.8rem; color: #555; text-align:right; }
  </style>
</head>
<body>
  <h1>🤟 GESTURE-TO-VOICE</h1>
  <div class="grid">
    <div class="camera-box">
      <img src="/video_feed" alt="Camera Feed">
    </div>
    <div class="info-panel">
      <div class="card">
        <div class="label">Detected Gesture</div>
        <div class="gesture-display" id="gesture">—</div>
        <div class="label">Confidence</div>
        <div class="conf-bar-wrap">
          <div class="conf-bar" id="conf-bar" style="width:0%"></div>
        </div>
        <div style="text-align:right;font-size:0.8rem;color:#555;margin-top:4px" id="conf-pct">0%</div>
      </div>
      <div class="card">
        <div class="label">Sentence Builder</div>
        <div class="sentence-box" id="sentence">—</div>
        <div class="btn-row" style="margin-top:12px">
          <button class="btn" onclick="api('speak')">🔊 Speak</button>
          <button class="btn" onclick="api('add')">➕ Add Word</button>
          <button class="btn" onclick="api('backspace')">⌫ Undo</button>
          <button class="btn" onclick="api('clear')">🗑 Clear</button>
        </div>
      </div>
      <div class="fps" id="fps">FPS: —</div>
    </div>
  </div>

  <script>
    async function poll() {
      try {
        const r = await fetch('/state');
        const d = await r.json();
        document.getElementById('gesture').textContent = d.gesture || '—';
        const pct = Math.round((d.confidence || 0) * 100);
        document.getElementById('conf-bar').style.width = pct + '%';
        document.getElementById('conf-pct').textContent = pct + '%';
        document.getElementById('sentence').textContent =
          d.sentence && d.sentence.length ? d.sentence.join(' ') : '—';
        document.getElementById('fps').textContent = 'FPS: ' + (d.fps || 0).toFixed(1);
      } catch(e) {}
      setTimeout(poll, 100);
    }
    async function api(action) {
      await fetch('/action/' + action, { method: 'POST' });
    }
    poll();
  </script>
</body>
</html>
"""


mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils


def gen_frames():
    global latest_frame, state
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Try to load model
    predictor = None
    try:
        from app_3_gesture_to_voice import GesturePredictor, GestureSmoother, extract_landmarks
        predictor = GesturePredictor("models/gesture_model.tflite", "models/label_encoder.json")
        smoother  = GestureSmoother()
    except Exception as e:
        print(f"Model not loaded: {e}")

    with mp_hands.Hands(
        max_num_hands=1, model_complexity=0,
        min_detection_confidence=0.6, min_tracking_confidence=0.5
    ) as hands:
        prev = time.time()
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            gesture, conf = "—", 0.0
            if result.multi_hand_landmarks and predictor:
                hand_lm  = result.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)
                features = extract_landmarks(hand_lm)
                gesture, conf = predictor.predict(np.array(features))

            now = time.time()
            fps = 1.0 / (now - prev + 1e-9)
            prev = now

            state["gesture"]    = gesture
            state["confidence"] = float(conf)
            state["fps"]        = round(fps, 1)

            _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')

    cap.release()


@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/state')
def get_state():
    return jsonify(state)

@app.route('/action/<action>', methods=['POST'])
def action(action):
    if action == 'clear':
        state['sentence'] = []
    elif action == 'add':
        if state['gesture'] != '—':
            state['sentence'].append(state['gesture'])
    elif action == 'backspace':
        if state['sentence']:
            state['sentence'].pop()
    elif action == 'speak':
        text = ' '.join(state['sentence'])
        if text:
            threading.Thread(
                target=lambda: __import__('pyttsx3').init().say(text) or
                               __import__('pyttsx3').init().runAndWait(),
                daemon=True
            ).start()
    return jsonify({"ok": True})


if __name__ == '__main__':
    print("🌐 Dashboard: http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, threaded=True)
