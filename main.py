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
import os
import threading
from ultralytics import YOLO
from collections import deque, Counter
import time

# ── MODEL PATHS ───────────────────────────────────────────────────────────────
FACE_MODEL_PATH     = "runs/detect/runs/train/jay_adam_Aditya_face-4/weights/best.pt"
POSE_MODEL_PATH     = "yolo11n-pose.pt"
GESTURE_MODEL_PATH  = "gesture_model.pt"
GESTURE_LABELS_PATH = "gesture_labels.json"

# ── IDENTITY CONFIG ───────────────────────────────────────────────────────────
FACE_CONFIDENCE_THRESHOLD = 0.65
IDENTITY_VOTE_WINDOW      = 30   # frames to vote over
IDENTITY_LOCK_FRAMES      = 20   # consecutive frames before switching identity
UNKNOWN_LABEL             = "Unknown"

# ── GESTURE CONFIG ────────────────────────────────────────────────────────────
FRAMES_PER_SAMPLE  = 20
INPUT_SIZE         = 42
GESTURE_CONFIDENCE = 0.65
GESTURE_COOLDOWN   = 1.5

# ── TRANSFORMER CONFIG (must match gesture_train.py exactly) ──────────────────
D_MODEL  = 128
N_HEADS  = 4
N_LAYERS = 3
D_FF     = 512
DROPOUT  = 0.1

# ── PER-USER ACTION PROFILES ──────────────────────────────────────────────────
USER_PROFILES = {
    "Jay": {
        "thumbs_up": "next_track",
        "open_palm":  "prev_track",
        "wave":       "volume_up",
        "fist":       "volume_down",
        "point":      "play_pause",
    },
    "Adam": {
        "thumbs_up": "play_pause",
        "open_palm":  "next_track",
        "wave":       "prev_track",
        "fist":       "volume_down",
        "point":      "volume_up",
    },
    "Aditya": {
        "thumbs_up": "volume_up",
        "open_palm":  "play_pause",
        "wave":       "next_track",
        "fist":       "prev_track",
        "point":      "volume_down",
    },
}

# ── IoT ACTION DISPATCHER ─────────────────────────────────────────────────────
def dispatch_action(user: str, gesture: str, action: str):
    # print who did what gesture and what action fired with a timestamp
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {user} → {gesture} → ACTION: {action.upper()}")

    # check if Spotify process is currently running without launching it
    def spotify_running():
        # pgrep searches all running processes, returns process ID if found, empty if not
        return bool(os.popen("pgrep -x Spotify").read().strip())

    # check if Google Chrome process is currently running
    def chrome_running():
        return bool(os.popen("pgrep -x 'Google Chrome'").read().strip())

    if action == "play_pause":
        def smart_play_pause():
            if spotify_running():
                # toggle Spotify directly regardless of playing or paused state
                os.system("osascript -e 'tell application \"Spotify\" to playpause'")
                print("         Spotify Play/Pause")
                # stop here, no need to touch Chrome
                return
            # if Spotify is not open fall through to YouTube
            if chrome_running():
                # bring Chrome to front briefly so it can receive the keyboard shortcut
                os.system("osascript -e 'tell application \"Google Chrome\" to activate'")
                # send spacebar key code 49 to Chrome, which toggles YouTube play pause
                os.system("osascript -e 'tell application \"System Events\" to key code 49'")
                print("         YouTube Play/Pause")
        # run in background thread so camera loop keeps running without waiting
        threading.Thread(target=smart_play_pause, daemon=True).start()

    elif action == "next_track":
        def next_t():
            if spotify_running():
                # send Apple Event to Spotify to skip to next song, no focus needed
                os.system("osascript -e 'tell application \"Spotify\" to next track'")
                print("         Spotify Next Track")
            elif chrome_running():
                # activate Chrome then send right arrow key code 124 to seek forward in YouTube
                os.system("osascript -e 'tell application \"Google Chrome\" to activate'")
                os.system("osascript -e 'tell application \"System Events\" to key code 124'")
                print("         YouTube Next (arrow right)")
        threading.Thread(target=next_t, daemon=True).start()

    elif action == "prev_track":
        def prev_t():
            if spotify_running():
                # send Apple Event to Spotify to go back to previous song
                os.system("osascript -e 'tell application \"Spotify\" to previous track'")
                print("         Spotify Previous Track")
            elif chrome_running():
                # activate Chrome then send left arrow key code 123 to seek backward in YouTube
                os.system("osascript -e 'tell application \"Google Chrome\" to activate'")
                os.system("osascript -e 'tell application \"System Events\" to key code 123'")
                print("         YouTube Previous (arrow left)")
        threading.Thread(target=prev_t, daemon=True).start()

    elif action == "volume_up":
        def ramp_up():
            if spotify_running():
                os.system("""osascript -e '
                tell application "Spotify"
                    -- get current volume level
                    set currentVol to sound volume
                    -- target is 20 points higher
                    set targetVol to currentVol + 20
                    -- cap at 100 so it never goes over max
                    if targetVol > 100 then set targetVol to 100
                    -- loop one step at a time from current to target
                    repeat with v from (currentVol + 1) to targetVol
                        set sound volume to v
                        -- small delay between each step makes it sound smooth
                        delay 0.01
                    end repeat
                end tell'""")
                print("         Spotify Volume UP")
            else:
                # if Spotify is not open, raise Mac system volume instead
                os.system("osascript -e 'set volume output volume (output volume of (get volume settings) + 10)'")
                print("         System Volume UP")
        threading.Thread(target=ramp_up, daemon=True).start()

    elif action == "volume_down":
        def ramp_down():
            if spotify_running():
                os.system("""osascript -e '
                tell application "Spotify"
                    -- get current volume level
                    set currentVol to sound volume
                    -- target is 20 points lower
                    set targetVol to currentVol - 20
                    -- floor at 0 so it never goes below minimum
                    if targetVol < 0 then set targetVol to 0
                    -- loop one step at a time from current down to target
                    repeat with v from (currentVol - 1) to targetVol by -1
                        set sound volume to v
                        -- small delay between each step makes it sound smooth
                        delay 0.01
                    end repeat
                end tell'""")
                print("         Spotify Volume DOWN")
            else:
                # if Spotify is not open, lower Mac system volume instead
                os.system("osascript -e 'set volume output volume (output volume of (get volume settings) - 10)'")
                print("         System Volume DOWN")
        threading.Thread(target=ramp_down, daemon=True).start()
        
# ── TRANSFORMER MODEL ─────────────────────────────────────────────────────────
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
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.attn_pool = nn.Sequential(nn.Linear(d_model, 1), nn.Softmax(dim=1))
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 32), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(32, num_classes)
        )

    def forward(self, x, return_attention=False):
        x = self.input_projection(x)
        x = self.pos_encoding(x)
        x = self.transformer(x)
        weights = self.attn_pool(x)
        pooled  = (x * weights).sum(dim=1)
        out = self.classifier(pooled)
        if return_attention:
            return out, weights.squeeze(-1)
        return out


def load_gesture_model(device):
    with open(GESTURE_LABELS_PATH, "r") as f:
        labels = json.load(f)
    model = GestureTransformer(
        input_size=INPUT_SIZE, d_model=D_MODEL, n_heads=N_HEADS,
        n_layers=N_LAYERS, d_ff=D_FF, num_classes=len(labels), dropout=DROPOUT
    ).to(device)
    model.load_state_dict(torch.load(GESTURE_MODEL_PATH, map_location=device))
    model.eval()
    print(f"✅ Gesture model loaded — classes: {labels}")
    return model, labels


# ── LANDMARK EXTRACTOR ────────────────────────────────────────────────────────
def extract_landmarks(hand_landmarks) -> np.ndarray:
    coords = []
    for lm in hand_landmarks.landmark:
        coords.append(lm.x)
        coords.append(lm.y)
    return np.array(coords, dtype=np.float32)


def normalize_landmarks(sequence):
    sequence = sequence.reshape(21, 2)
    wrist = sequence[0:1, :]
    sequence = sequence - wrist
    scale = np.linalg.norm(sequence[12, :]) + 1e-6
    sequence = sequence / scale
    return sequence.reshape(42)


# ── GESTURE INFERENCE ─────────────────────────────────────────────────────────
def infer_gesture(sequence: list, model, labels: list, device) -> tuple:
    while len(sequence) < FRAMES_PER_SAMPLE:
        sequence.append(np.zeros(INPUT_SIZE, dtype=np.float32))
    seq_array = np.array(sequence[:FRAMES_PER_SAMPLE])
    seq_array = np.array([normalize_landmarks(frame) for frame in seq_array])
    x = torch.FloatTensor(seq_array).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs  = torch.softmax(logits, dim=1)[0]
        conf, idx = probs.max(dim=0)
    return labels[idx.item()], conf.item()


# ── DRAW UTILS ────────────────────────────────────────────────────────────────
def draw_label_box(frame, text, x, y, color=(0, 255, 0)):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(frame, (x, y - th - 8), (x + tw + 8, y + 4), color, -1)
    cv2.putText(frame, text, (x + 4, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)


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

    face_model    = YOLO(FACE_MODEL_PATH)
    pose_model    = YOLO(POSE_MODEL_PATH)
    gesture_model, gesture_labels = load_gesture_model(device)

    mp_hands   = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands      = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=4,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("❌ No camera found.")
        return
    print("✅ Pipeline running — press Q to quit\n")

    # ── State (all local to main — no globals) ────────────────────────────────
    identity_votes        = deque(maxlen=IDENTITY_VOTE_WINDOW)
    current_identity      = UNKNOWN_LABEL
    identity_lock_counter = 0          # ← local, accumulates correctly
    current_gesture       = "NONE"
    current_conf          = 0.0
    last_action           = None
    last_action_time      = {}         # per-person cooldown
    landmark_buffers      = {}         # per-person landmark deque

    fps_timer   = time.time()
    fps         = 0.0
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_count += 1
        h_frame, w_frame = frame.shape[:2]
        now = time.time()

        if now - fps_timer >= 1.0:
            fps = frame_count / (now - fps_timer)
            frame_count = 0
            fps_timer = now

        # ── 1. FACE IDENTIFICATION ─────────────────────────────────────────
        face_results      = face_model(frame, verbose=False)
        detected_identity = UNKNOWN_LABEL
        face_centers      = []

        for box in face_results[0].boxes:
            conf = float(box.conf[0])
            if conf >= FACE_CONFIDENCE_THRESHOLD:
                cls  = int(box.cls[0])
                name = face_model.names[cls].capitalize()
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detected_identity = name
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                draw_label_box(frame, f"{name} {conf:.0%}", x1, y1 - 5)
                face_centers.append((name, (x1 + x2) / 2, (y1 + y2) / 2))

        # ── Identity vote + lock ──────────────────────────────────────────
        identity_votes.append(detected_identity)
        new_identity = Counter(identity_votes).most_common(1)[0][0]

        if new_identity == current_identity:
            # Same person confirmed → reset counter
            identity_lock_counter = 0
        elif new_identity != UNKNOWN_LABEL:
            # Different person detected → accumulate lock counter
            identity_lock_counter += 1
            if identity_lock_counter >= IDENTITY_LOCK_FRAMES:
                # Only switch after IDENTITY_LOCK_FRAMES consecutive frames
                print(f"  Identity switched: {current_identity} → {new_identity}")
                current_identity      = new_identity
                identity_lock_counter = 0
                # Clear old person's gesture buffer on identity switch
                landmark_buffers.clear()
        else:
            # Unknown detected → do not switch, just reset counter
            identity_lock_counter = 0

        # ── 2. POSE ESTIMATION ────────────────────────────────────────────
        pose_results   = pose_model(frame, verbose=False)
        pose_annotated = pose_results[0].plot(conf=False, labels=False)
        frame = cv2.addWeighted(frame, 0.4, pose_annotated, 0.6, 0)

        # ── 3. MEDIAPIPE HANDS → per-person landmark buffers ──────────────
        rgb         = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_result = hands.process(rgb)

        def closest_person(wx, wy):
            """Return the name of the face closest to this wrist position."""
            best_name, best_dist = UNKNOWN_LABEL, float('inf')
            for name, fx, fy in face_centers:
                dist = ((wx - fx) ** 2 + (wy - fy) ** 2) ** 0.5
                if dist < best_dist:
                    best_dist, best_name = dist, name
            # Fall back to current_identity if no face detected this frame
            return best_name if best_name != UNKNOWN_LABEL else current_identity

        if hand_result.multi_hand_landmarks:
            for hand_lm in hand_result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(255, 100, 0), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2)
                )
                wrist   = hand_lm.landmark[0]
                wx, wy  = wrist.x * w_frame, wrist.y * h_frame
                person  = closest_person(wx, wy)

                if person not in landmark_buffers:
                    landmark_buffers[person] = deque(maxlen=FRAMES_PER_SAMPLE)
                landmark_buffers[person].append(normalize_landmarks(extract_landmarks(hand_lm)))
        else:
            # No hand visible — push zeros for current identity only
            if current_identity != UNKNOWN_LABEL:
                if current_identity not in landmark_buffers:
                    landmark_buffers[current_identity] = deque(maxlen=FRAMES_PER_SAMPLE)
                landmark_buffers[current_identity].append(
                    np.zeros(INPUT_SIZE, dtype=np.float32)
                )

        # ── 4. TRANSFORMER GESTURE INFERENCE (per person) ─────────────────
        person_gestures = {}
        for person, buf in landmark_buffers.items():
            if len(buf) == FRAMES_PER_SAMPLE:
                gesture, conf = infer_gesture(
                    list(buf), gesture_model, gesture_labels, device
                )
                if conf >= GESTURE_CONFIDENCE:
                    person_gestures[person] = (gesture, conf)

        # ── 5. ACTION DISPATCH ─────────────────────────────────────────────
        current_gesture = "NONE"
        current_conf    = 0.0

        # only dispatch if hand is actually detected this frame
        if hand_result.multi_hand_landmarks and current_identity in person_gestures and current_identity in USER_PROFILES:
            gesture, conf = person_gestures[current_identity]
            current_gesture = gesture
            current_conf    = conf

            profile    = USER_PROFILES[current_identity]
            action     = profile.get(gesture)
            last_fired = last_action_time.get(current_identity, 0)

            if action and (now - last_fired) >= GESTURE_COOLDOWN:
                dispatch_action(current_identity, gesture, action)
                last_action                        = action
                last_action_time[current_identity] = now
                landmark_buffers[current_identity].clear()

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