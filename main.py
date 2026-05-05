"""
GRYOLO — Gesture-Controlled IoT Pipeline
=========================================
Architecture:
  Webcam
    ├── YOLOv11 Face         → identity (Jay / Adam / Aditya)
    └── YOLOv11 Pose         → wrist keypoints → hand ROI
              └── MediaPipe Hands  → 21 landmarks (42 coords/frame)
                        └── Gesture Transformer  → gesture label
                                  └── Per-user Profile DB
                                            └── IoT Action Dispatcher

Controls: Q to quit
"""

import cv2
import numpy as np
import mediapipe as mp
import torch
import torch.nn as nn
import math
import json
from ultralytics import YOLO
from collections import deque, Counter
import time

# ── MODEL PATHS ───────────────────────────────────────────────────────────────
FACE_MODEL_PATH     = "runs/detect/runs/train/jay_adam_Aditya_face-3/weights/best.pt"  # swap when 3-person model ready
POSE_MODEL_PATH     = "yolo11n-pose.pt"
GESTURE_MODEL_PATH  = "gesture_model.pt"
GESTURE_LABELS_PATH = "gesture_labels.json"

# ── IDENTITY CONFIG ───────────────────────────────────────────────────────────
FACE_CONFIDENCE_THRESHOLD = 0.80
IDENTITY_VOTE_WINDOW      = 30
UNKNOWN_LABEL             = "Unknown"

# ── GESTURE CONFIG ────────────────────────────────────────────────────────────
FRAMES_PER_SAMPLE  = 30        # must match gesture_collect.py
INPUT_SIZE         = 42        # 21 landmarks x 2 (x, y)
GESTURE_CONFIDENCE = 0.70      # minimum softmax prob to accept gesture
GESTURE_COOLDOWN   = 2.0       # seconds between repeated action triggers

# ── TRANSFORMER CONFIG (must match gesture_train.py exactly) ──────────────────
D_MODEL  = 128
N_HEADS  = 4
N_LAYERS = 3
D_FF     = 512
DROPOUT  = 0.1

# ── WRIST KEYPOINT INDICES (COCO) ─────────────────────────────────────────────
LEFT_WRIST_IDX  = 9
RIGHT_WRIST_IDX = 10

# ── PER-USER ACTION PROFILES ──────────────────────────────────────────────────
# gesture names must match GESTURES list in gesture_collect.py exactly
USER_PROFILES = {
    "Jay": {
        "Jay_thumbs_up": "lights_on",
        "Jay_open_palm":  "lights_off",
        "Jay_wave":       "volume_up",
        "Jay_fist":       "volume_down",
        "Jay_point":      "play_pause",
    },
    "Adam": {
        "thumbs_up": "play_pause",
        "open_palm":  "lights_on",
        "wave":       "lights_off",
        "fist":       "volume_down",
        "point":      "volume_up",
    },
    "Aditya": {
        "thumbs_up": "volume_up",
        "open_palm":  "play_pause",
        "wave":       "lights_on",
        "fist":       "lights_off",
        "point":      "volume_down",
    },
}

identity_lock_counter = 0

# ── IoT ACTION DISPATCHER (mocked — replace with real Pi calls) ───────────────
def dispatch_action(user: str, gesture: str, action: str):
    """
    Mock dispatcher — prints to terminal.
    Replace body with real Raspberry Pi call when ready:
        import requests
        requests.post("http://raspberrypi.local/action", json={"action": action})
    """
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {user} → {gesture} → ACTION: {action.upper()}")
    if action == "lights_on":
        print("         💡 Lights ON")
    elif action == "lights_off":
        print("         🌑 Lights OFF")
    elif action == "volume_up":
        print("         🔊 Volume UP")
    elif action == "volume_down":
        print("         🔈 Volume DOWN")
    elif action == "play_pause":
        print("         ⏯  Play / Pause")

# ── TRANSFORMER MODEL (mirrors gesture_train.py exactly) ─────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=30, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class GestureTransformer(nn.Module):
    def __init__(self, input_size, d_model, n_heads, n_layers, d_ff,
                 num_classes, dropout=0.1):
        super().__init__()
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # ← Adam's attention pooling — must match gesture_train.py
        self.attn_pool = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Softmax(dim=1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )

    def forward(self, x, return_attention=False):
        x = self.input_projection(x)
        x = self.pos_encoding(x)
        x = self.transformer(x)
        weights = self.attn_pool(x)        # (batch, seq_len, 1)
        pooled  = (x * weights).sum(dim=1) # weighted sum
        out = self.classifier(pooled)
        if return_attention:
            return out, weights.squeeze(-1)
        return out


def load_gesture_model(device):
    """Load trained Transformer + label map from disk."""
    with open(GESTURE_LABELS_PATH, "r") as f:
        labels = json.load(f)

    model = GestureTransformer(
        input_size=INPUT_SIZE,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        d_ff=D_FF,
        num_classes=len(labels),
        dropout=DROPOUT
    ).to(device)

    model.load_state_dict(torch.load(GESTURE_MODEL_PATH, map_location=device))
    model.eval()
    print(f"✅ Gesture model loaded — classes: {labels}")
    return model, labels


# ── LANDMARK EXTRACTOR (mirrors gesture_collect.py) ───────────────────────────
def extract_landmarks(hand_landmarks) -> np.ndarray:
    """Returns 42 floats: [x0, y0, x1, y1, ..., x20, y20]"""
    coords = []
    for lm in hand_landmarks.landmark:
        coords.append(lm.x)
        coords.append(lm.y)
    return np.array(coords, dtype=np.float32)


def normalize_landmarks(sequence):
    """Must match gesture_train.py exactly — centers on wrist, scales by hand size."""
    sequence = sequence.reshape(21, 2)
    wrist = sequence[0:1, :]
    sequence = sequence - wrist
    scale = np.linalg.norm(sequence[12, :]) + 1e-6
    sequence = sequence / scale
    return sequence.reshape(42)

# ── GESTURE INFERENCE ─────────────────────────────────────────────────────────
def infer_gesture(sequence: list, model, labels: list, device) -> tuple:
    """
    Takes a list of FRAMES_PER_SAMPLE landmark arrays (each 42 floats).
    Returns (gesture_label, confidence).
    """
    while len(sequence) < FRAMES_PER_SAMPLE:
        sequence.append(np.zeros(INPUT_SIZE, dtype=np.float32))

    seq_array = np.array(sequence[:FRAMES_PER_SAMPLE])

    # ← Apply same normalization as gesture_train.py
    seq_array = np.array([normalize_landmarks(frame) for frame in seq_array])

    x = torch.FloatTensor([seq_array]).to(device)

    with torch.no_grad():
        logits = model(x)
        probs  = torch.softmax(logits, dim=1)[0]
        conf, idx = probs.max(dim=0)

    return labels[idx.item()], conf.item()


# ── HAND ROI FROM WRIST KEYPOINT ──────────────────────────────────────────────
def get_hand_roi(wrist_kp, frame_w, frame_h, roi_size=220):
    x, y = int(wrist_kp[0]), int(wrist_kp[1])
    half = roi_size // 2
    x1 = max(0, x - half)
    y1 = max(0, y - half)
    x2 = min(frame_w, x + half)
    y2 = min(frame_h, y + half)
    return (x1, y1, x2, y2)


# ── DRAW UTILS ────────────────────────────────────────────────────────────────
def draw_label_box(frame, text, x, y, color=(0, 255, 0)):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(frame, (x, y - th - 8), (x + tw + 8, y + 4), color, -1)
    cv2.putText(frame, text, (x + 4, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

def draw_hud(frame, identity, gesture, conf, action, fps):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 60), (20, 20, 20), -1)

    cv2.putText(frame, f"User: {identity}",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 100), 2)

    gesture_display = f"Gesture: {gesture} ({conf:.0%})" if gesture != "NONE" else "Gesture: —"
    cv2.putText(frame, gesture_display,
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

    action_text = f"Action: {action}" if action else "Action: —"
    cv2.putText(frame, action_text,
                (w // 2 - 80, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 200, 0), 2)

    cv2.putText(frame, f"FPS: {fps:.1f}",
                (w - 110, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)


# ── MAIN PIPELINE ─────────────────────────────────────────────────────────────
def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("Loading models...")

    face_model = YOLO(FACE_MODEL_PATH)
    pose_model = YOLO(POSE_MODEL_PATH)
    gesture_model, gesture_labels = load_gesture_model(device)

    mp_hands   = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands      = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("❌ Camera index 0 failed, trying 0...")
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ No camera found. Lock your iPhone and try again.")
        return
    print("✅ Pipeline running — press Q to quit\n")

    # ── State ─────────────────────────────────────────────────────────────────
    identity_votes   = deque(maxlen=IDENTITY_VOTE_WINDOW)
    current_identity = UNKNOWN_LABEL
    current_gesture  = "NONE"
    current_conf     = 0.0
    last_action      = None
    last_action_time = 0.0

    # Rolling 30-frame landmark buffer — feeds the Transformer
    landmark_buffer = deque(maxlen=FRAMES_PER_SAMPLE)

    fps_timer   = time.time()
    fps         = 0.0
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)   # mirror so left/right feel natural
        frame_count += 1
        h_frame, w_frame = frame.shape[:2]
        now = time.time()

        # FPS counter
        if now - fps_timer >= 1.0:
            fps = frame_count / (now - fps_timer)
            frame_count = 0
            fps_timer = now

        # ── 1. FACE IDENTIFICATION ─────────────────────────────────────────
        face_results      = face_model(frame, verbose=False)
        detected_identity = UNKNOWN_LABEL

        for box in face_results[0].boxes:
            conf = float(box.conf[0])
            if conf >= FACE_CONFIDENCE_THRESHOLD:
                cls  = int(box.cls[0])
                name = face_model.names[cls].capitalize()
                detected_identity = name
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                draw_label_box(frame, f"{name} {conf:.0%}", x1, y1 - 5)

        identity_votes.append(detected_identity)
        vote_result = Counter(identity_votes).most_common(1)[0]
        new_identity = vote_result[0]

        # Only switch identity if new person detected consistently
        if new_identity == current_identity:
            identity_lock_counter = 0
        elif new_identity != UNKNOWN_LABEL:
            identity_lock_counter += 1
            if identity_lock_counter >= 20:  # 20 frames before switching
                current_identity = new_identity
                identity_lock_counter = 0
        else:
            identity_lock_counter = 0

        # ── 2. POSE ESTIMATION → WRIST KEYPOINTS ──────────────────────────
        pose_results = pose_model(frame, verbose=False)
        wrist_points = []

        for person in pose_results:
            if person.keypoints is not None and len(person.keypoints.xy) > 0:
                kps = person.keypoints.xy[0]
                if len(kps) > RIGHT_WRIST_IDX:
                    for idx in [LEFT_WRIST_IDX, RIGHT_WRIST_IDX]:
                        kp = kps[idx]
                        if kp[0] > 0 and kp[1] > 0:
                            wrist_points.append((float(kp[0]), float(kp[1])))

        pose_annotated = pose_results[0].plot(conf=False, labels=False)
        frame = cv2.addWeighted(frame, 0.4, pose_annotated, 0.6, 0)

        # ── 3. MEDIAPIPE HANDS → LANDMARK EXTRACTION ──────────────────────
        rois = [(0, 0, w_frame, h_frame)]

        hand_detected_this_frame = False

        for (rx1, ry1, rx2, ry2) in rois:
            roi_frame = frame[ry1:ry2, rx1:rx2]
            if roi_frame.size == 0:
                continue

            roi_rgb     = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2RGB)
            hand_result = hands.process(roi_rgb)

            if hand_result.multi_hand_landmarks:
                for hand_lm in hand_result.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        roi_frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(255, 100, 0),
                                               thickness=2, circle_radius=3),
                        mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2)
                    )
                    frame[ry1:ry2, rx1:rx2] = roi_frame

                    # Push landmarks into rolling buffer
                    raw = extract_landmarks(hand_lm)
                    landmark_buffer.append(normalize_landmarks(raw))
                    hand_detected_this_frame = True
                    break

            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (100, 100, 255), 1)

        # No hand visible → push zeros to keep buffer advancing
        if not hand_detected_this_frame:
            landmark_buffer.append(np.zeros(INPUT_SIZE, dtype=np.float32))

        # ── 4. TRANSFORMER GESTURE INFERENCE ──────────────────────────────
        # Only run once buffer is full (30 frames = ~1 second)
        if len(landmark_buffer) == FRAMES_PER_SAMPLE:
            gesture, conf = infer_gesture(
                list(landmark_buffer), gesture_model, gesture_labels, device
            )
            if conf >= GESTURE_CONFIDENCE:
                current_gesture = gesture
                current_conf    = conf
            else:
                current_gesture = "NONE"
                current_conf    = 0.0

        # ── 5. ACTION DISPATCH ─────────────────────────────────────────────
        if (current_identity != UNKNOWN_LABEL
                and current_gesture != "NONE"
                and current_identity in USER_PROFILES):

            profile = USER_PROFILES[current_identity]
            action  = profile.get(current_gesture)

            if action and (now - last_action_time) >= GESTURE_COOLDOWN:
                dispatch_action(current_identity, current_gesture, action)
                last_action      = action
                last_action_time = now

        # ── 6. HUD ────────────────────────────────────────────────────────
        draw_hud(frame, current_identity, current_gesture,
                 current_conf, last_action, fps)

        cv2.imshow("GRYOLO Pipeline", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    hands.close()
    cv2.destroyAllWindows()
    print("\n👋 Pipeline stopped.")


if __name__ == "__main__":
    main()