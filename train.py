from ultralytics import YOLO

model = YOLO("yolo11n.pt")  # downloads automatically on first run

model.train(
    data="faces.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    name="jay_adam_face",
    patience=10,  # stops early if no improvement
    device="mps"
)