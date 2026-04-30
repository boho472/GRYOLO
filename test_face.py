from ultralytics import YOLO
import cv2

face_model = YOLO("runs/detect/runs/train/jay_adam_face-29/weights/best.pt")
pose_model = YOLO("yolo11n-pose.pt")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    face_results = face_model(frame, verbose=False)
    pose_results = pose_model(frame, verbose=False)

    # Draw pose first, then overlay face boxes manually
    annotated = pose_results[0].plot()

    for box in face_results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        label = f"{face_model.names[cls]} {conf:.2f}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Jay & Adam — Face + Pose", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()