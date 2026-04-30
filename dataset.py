import cv2
import os
import time
import urllib.request
import json

# ── CONFIG ──────────────────────────────────────────────
SAVE_FOLDER = "Adityaface"
TARGET_COUNT = 1500
INTERVAL = 0.3
# ────────────────────────────────────────────────────────

PHASES = [
    (0,    125, "Phase 1:  Straight, neutral expression"),
    (125,  250, "Phase 2:  Turn head LEFT slowly"),
    (250,  375, "Phase 3:  Turn head RIGHT slowly"),
    (375,  500, "Phase 4:  Tilt head up and down"),
    (500,  625, "Phase 5:  Move closer (fill the frame)"),
    (625,  750, "Phase 6:  Move further back"),
    (750,  875, "Phase 7:  Smile, serious, surprised, talking"),
    (875,  1000, "Phase 8:  Dim lighting"),
    (1000, 1125, "Phase 9:  Window lighting"),
    (1125, 1250, "Phase 10: Gray background"),
    (1250, 1375, "Phase 11: Flashlight (side lighting)"),
    (1375, 1500, "Phase 12: Natural random movement — just act normal"),
]

PROGRESS_FILE = f"{SAVE_FOLDER}_progress.json"

def get_phase_label(count):
    for start, end, label in PHASES:
        if start <= count < end:
            return f"{label}  [{count - start}/{end - start}]"
    return "Complete"

def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            data = json.load(f)
            return data.get("count", 0)
    return 0

def save_progress(count):
    with open(PROGRESS_FILE, "w") as f:
        json.dump({"count": count, "folder": SAVE_FOLDER}, f)

# Download DNN model if not present
if not os.path.exists("deploy.prototxt"):
    print("Downloading face detector...")
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
        "deploy.prototxt"
    )
    urllib.request.urlretrieve(
        "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
        "face.caffemodel"
    )

net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "face.caffemodel")
os.makedirs(SAVE_FOLDER, exist_ok=True)

# Separate folder for full frames (YOLO trains on these)
FULL_FRAME_FOLDER = f"{SAVE_FOLDER}_full"
os.makedirs(FULL_FRAME_FOLDER, exist_ok=True)

cap = cv2.VideoCapture(0)

count = load_progress()
if count > 0:
    print(f"▶ Resuming from photo {count}/{TARGET_COUNT}")
else:
    print(f"📸 Starting fresh → '{SAVE_FOLDER}'")

print("Controls: SPACE = start/pause | Q = save & quit")

last_capture_time = 0
started = False
paused = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h_frame, w_frame = frame.shape[:2]

    # Face detection
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 177, 123))
    net.setInput(blob)
    detections = net.forward()

    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.6:
            box = detections[0, 0, i, 3:7] * [w_frame, h_frame, w_frame, h_frame]
            x1, y1, x2, y2 = box.astype(int)
            faces.append((x1, y1, x2 - x1, y2 - y1))

    # Draw face box
    color = (0, 255, 0) if (started and not paused) else (100, 100, 100)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        # Show confidence hint
        conf_text = f"Face detected"
        cv2.putText(frame, conf_text, (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # No face warning
    if len(faces) == 0 and started and not paused:
        cv2.putText(frame, "⚠ No face detected — reposition",
                    (10, h_frame // 2), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2)

    # Status overlays
    cv2.putText(frame, f"Captured: {count}/{TARGET_COUNT}",
                (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.putText(frame, get_phase_label(count),
                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

    if not started:
        cv2.putText(frame, "Press SPACE to start",
                    (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    elif paused:
        cv2.putText(frame, "PAUSED — Press SPACE to resume",
                    (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)
        cv2.putText(frame, "Change environment now",
                    (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 2)
    else:
        cv2.putText(frame, "RECORDING — SPACE to pause | Q to quit",
                    (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    # Progress bar
    bar_w = int((count / TARGET_COUNT) * (w_frame - 20))
    cv2.rectangle(frame, (10, h_frame - 20), (w_frame - 10, h_frame - 10),
                  (50, 50, 50), -1)
    cv2.rectangle(frame, (10, h_frame - 20), (10 + bar_w, h_frame - 10),
                  (0, 255, 0), -1)

    # Phase markers on progress bar
    for start, end, _ in PHASES:
        marker_x = 10 + int((start / TARGET_COUNT) * (w_frame - 20))
        cv2.line(frame, (marker_x, h_frame - 22),
                 (marker_x, h_frame - 8), (255, 255, 255), 1)

    cv2.imshow(f"Capturing — {SAVE_FOLDER}", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord(' '):
        if not started:
            started = True
            print("▶ Started!")
        elif paused:
            paused = False
            print("▶ Resumed!")
        else:
            paused = True
            save_progress(count)
            print(f"⏸ Paused at {count} — progress saved.")

    if key == ord('q'):
        save_progress(count)
        print(f"💾 Saved at {count}/{TARGET_COUNT}. Run again to resume.")
        break

    # Auto-capture — saves BOTH cropped face AND full frame
    if started and not paused and len(faces) > 0 and count < TARGET_COUNT:
        now = time.time()
        if now - last_capture_time >= INTERVAL:
            x, y, w, h = faces[0]

            # 1. Save cropped face (with padding)
            pad = int(max(w, h) * 0.2)
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(w_frame, x + w + pad)
            y2 = min(h_frame, y + h + pad)
            face_crop = frame[y1:y2, x1:x2]
            crop_filename = os.path.join(SAVE_FOLDER,
                            f"{SAVE_FOLDER}_{count:04d}.jpg")
            cv2.imwrite(crop_filename, face_crop)

            # 2. Save full frame (YOLO sees full spatial context)
            full_filename = os.path.join(FULL_FRAME_FOLDER,
                            f"{SAVE_FOLDER}_full_{count:04d}.jpg")
            cv2.imwrite(full_filename, frame)

            count += 1
            last_capture_time = now
            print(f"  {get_phase_label(count)} — saved {count}")
            save_progress(count)

    if count >= TARGET_COUNT:
        save_progress(count)
        print(f"\n✅ Done! {TARGET_COUNT} photos saved to '{SAVE_FOLDER}'")
        break

cap.release()
cv2.destroyAllWindows()