import cv2
import os
import time

# ── CONFIG ──────────────────────────────────────────────
SAVE_FOLDER = "Aface"        # Change to "Aface" for Adam
TARGET_COUNT = 200
INTERVAL = 0.5               # seconds between captures
# ────────────────────────────────────────────────────────

os.makedirs(SAVE_FOLDER, exist_ok=True)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

count = 0
last_capture_time = 0

print(f"📸 Starting capture → saving to '{SAVE_FOLDER}'")
print("Press SPACE to start | Q to quit early")

started = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Progress display
    status = f"Captured: {count}/{TARGET_COUNT}"
    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if not started:
        cv2.putText(frame, "Press SPACE to start", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

    cv2.imshow("Face Capture", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord(' '):
        started = True
        print("▶ Capturing started!")

    if key == ord('q'):
        print("Quit early.")
        break

    # Auto-capture logic
    if started and len(faces) > 0 and count < TARGET_COUNT:
        now = time.time()
        if now - last_capture_time >= INTERVAL:
            face_crop = frame[y:y+h, x:x+w]
            filename = os.path.join(SAVE_FOLDER, f"{SAVE_FOLDER}_{count:04d}.jpg")
            cv2.imwrite(filename, face_crop)
            count += 1
            last_capture_time = now
            print(f"  Saved {count}/{TARGET_COUNT}")

    if count >= TARGET_COUNT:
        print(f"\n✅ Done! {TARGET_COUNT} photos saved to '{SAVE_FOLDER}'")
        break

cap.release()
cv2.destroyAllWindows()