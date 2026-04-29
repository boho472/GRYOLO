import torch
torch.cuda.empty_cache()
from ultralytics import YOLO

def main():
    # Confirm GPU
    print("CUDA available:", torch.cuda.is_available())
    print("GPU:", torch.cuda.get_device_name(0))

    # Load model
    model = YOLO("yolo11s.pt")

    # Train
    model.train(
        data="faces.yaml",
        epochs=100,
        imgsz=960,

        # GPU settings
        batch=16,
        device=0,
        workers=8,
        amp=True,

        # Training settings
        patience=20,
        cos_lr=True,
        optimizer="AdamW",

        # Augmentations
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        translate=0.1,
        scale=0.8,
        fliplr=0.5,

        # Output
        project="runs/train",
        name="jay_adam_face",
        save=True,
        verbose=True
    )

# 🔥 THIS LINE FIXES YOUR ERROR
if __name__ == "__main__":
    main()