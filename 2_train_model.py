"""
STEP 2: MODEL TRAINING
=======================
After collecting data, run this to train the gesture classifier.

Usage:
    pip install tensorflow scikit-learn numpy matplotlib
    python 2_train_model.py

Output:
    - gesture_model.h5       (Keras model)
    - gesture_model.tflite   (TFLite for fast inference)
    - label_encoder.json     (class index → gesture name)
    - training_history.png   (accuracy/loss curves)

Architecture: Lightweight MLP optimized for real-time inference.
Inference speed: ~2ms per frame on CPU — near-zero latency.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, regularizers

# ── CONFIG ──────────────────────────────────────────────────────────────────
DATA_FILE    = "dataset/raw_data.json"
MODEL_OUT    = "models/gesture_model.h5"
TFLITE_OUT   = "models/gesture_model.tflite"
LABELS_OUT   = "models/label_encoder.json"
PLOT_OUT     = "models/training_history.png"

EPOCHS       = 100
BATCH_SIZE   = 32
MIN_SAMPLES  = 20    # skip classes with too few samples
DROPOUT_RATE = 0.4
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs("models", exist_ok=True)


def load_dataset(data_file, min_samples=20):
    with open(data_file, "r") as f:
        raw = json.load(f)

    X, y = [], []
    skipped = []
    for cls_name, samples in raw.items():
        if len(samples) < min_samples:
            skipped.append(f"{cls_name}({len(samples)})")
            continue
        for sample in samples:
            X.append(sample)
            y.append(cls_name)

    if skipped:
        print(f"⚠  Skipped (too few samples): {', '.join(skipped)}")

    return np.array(X, dtype=np.float32), np.array(y)


def build_model(input_dim, num_classes):
    """
    Lightweight MLP — fast enough for real-time inference on CPU/mobile.
    For better accuracy with more data, switch to the LSTM variant below.
    """
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),

        # Batch norm on input for robustness to lighting/scale variation
        layers.BatchNormalization(),

        layers.Dense(256, activation='relu',
                     kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(DROPOUT_RATE),

        layers.Dense(128, activation='relu',
                     kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(DROPOUT_RATE),

        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),

        layers.Dense(num_classes, activation='softmax'),
    ], name="GestureClassifier_MLP")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def build_lstm_model(seq_len, feature_dim, num_classes):
    """
    Use this for DYNAMIC gestures (sequences of frames).
    Set SEQUENCE_LENGTH > 1 in collect_data.py to use this.
    """
    model = models.Sequential([
        layers.Input(shape=(seq_len, feature_dim)),
        layers.LSTM(128, return_sequences=True),
        layers.Dropout(0.3),
        layers.LSTM(64),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax'),
    ], name="GestureClassifier_LSTM")

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def plot_training(history, output_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history.history['accuracy'],     label='Train', color='#00C9A7', lw=2)
    ax1.plot(history.history['val_accuracy'], label='Val',   color='#FF6B6B', lw=2)
    ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Accuracy')
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(history.history['loss'],     label='Train', color='#00C9A7', lw=2)
    ax2.plot(history.history['val_loss'], label='Val',   color='#FF6B6B', lw=2)
    ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Loss')
    ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Training plot saved to {output_path}")


def export_tflite(keras_model, output_path):
    """Convert to TFLite for ~10x faster CPU inference."""
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]  # INT8 quantization
    tflite_model = converter.convert()
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    size_kb = os.path.getsize(output_path) / 1024
    print(f"✓ TFLite model saved: {output_path} ({size_kb:.1f} KB)")


def train():
    print("\n" + "="*60)
    print("GESTURE MODEL TRAINING")
    print("="*60)

    # 1. Load data
    print(f"\nLoading data from {DATA_FILE}...")
    X, y_raw = load_dataset(DATA_FILE, MIN_SAMPLES)
    print(f"  Dataset shape: {X.shape}")
    print(f"  Classes: {np.unique(y_raw).tolist()}")

    # 2. Encode labels
    le = LabelEncoder()
    y  = le.fit_transform(y_raw)
    num_classes = len(le.classes_)
    print(f"  Num classes: {num_classes}")

    # Save label mapping
    label_map = {int(i): str(cls) for i, cls in enumerate(le.classes_)}
    with open(LABELS_OUT, 'w') as f:
        json.dump(label_map, f, indent=2)
    print(f"✓ Labels saved to {LABELS_OUT}")

    # 3. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, stratify=y_train, random_state=42
    )
    print(f"\n  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # 4. Build model
    model = build_model(input_dim=X.shape[1], num_classes=num_classes)
    model.summary()

    # 5. Callbacks
    cbs = [
        callbacks.EarlyStopping(
            monitor='val_accuracy', patience=15, restore_best_weights=True
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6
        ),
        callbacks.ModelCheckpoint(
            MODEL_OUT, save_best_only=True, monitor='val_accuracy', verbose=0
        ),
    ]

    # 6. Train
    print("\n▶ Training...\n")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=cbs,
        verbose=1
    )

    # 7. Evaluate
    print("\n" + "="*60)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"✓ Test Accuracy: {test_acc*100:.2f}%")
    print(f"  Test Loss:     {test_loss:.4f}")

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=le.classes_
    ))

    # 8. Export TFLite
    export_tflite(model, TFLITE_OUT)

    # 9. Plot
    plot_training(history, PLOT_OUT)

    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print(f"   Model:   {MODEL_OUT}")
    print(f"   TFLite:  {TFLITE_OUT}")
    print(f"   Labels:  {LABELS_OUT}")
    print("="*60)


if __name__ == "__main__":
    train()
