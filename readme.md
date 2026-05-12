# GRYOLO — Gesture-Controlled Media Pipeline

A real-time, hands-free media control system that uses computer vision to identify users by face and interpret hand gestures, then dispatches per-user media actions to Spotify or YouTube.

## Architecture

```
Webcam
  ├── YOLOv11 Face         → identity (Jay / Adam / Aditya)
  └── YOLOv11 Pose         → wrist keypoints → hand ROI
            └── MediaPipe Hands  → 21 landmarks (42 coords/frame)
                      └── Gesture Transformer  → gesture label
                                └── Per-user Profile DB
                                          └── IoT Action Dispatcher
```

## Gesture → Action Mapping

Each recognized user has their own gesture-to-action profile:

| Gesture     | Jay         | Adam        | Aditya      |
|-------------|-------------|-------------|-------------|
| Thumbs Up   | Next Track  | Play/Pause  | Volume Up   |
| Open Palm   | Prev Track  | Next Track  | Play/Pause  |
| Wave        | Volume Up   | Prev Track  | Next Track  |
| Fist        | Volume Down | Volume Down | Prev Track  |
| Point       | Play/Pause  | Volume Up   | Volume Down |

Actions target Spotify if it is running, otherwise fall back to YouTube in Google Chrome.

## Requirements

- **Mac with Apple Silicon (M1 or later)** or a machine with an **NVIDIA GPU**
- **Python 3.11** — MediaPipe is not compatible with Python 3.12 or later
- Spotify or Google Chrome with YouTube open for media control

## Installation

**Step 1 — Clone the repository:**
```bash
git clone https://github.com/boho472/GRYOLO.git
cd GRYOLO
git checkout main
```

**Step 2 — Create and activate a Python 3.11 virtual environment:**
```bash
python3.11 -m venv venv
source venv/bin/activate
```

**Step 3 — Install required packages:**
```bash
pip install ultralytics mediapipe==0.10.14 opencv-python numpy torch torchvision scikit-learn
```

## Running

```bash
python3 main.py
```

Press **Q** to quit the pipeline window.

## Models

| File | Purpose |
|------|---------|
| `yolo11n-pose.pt` | YOLOv11 pose estimation (wrist keypoints) |
| `gesture_model.pt` | Trained Gesture Transformer (sequence classifier) |
| `gesture_labels.json` | Class label index for the gesture model |
| `runs/.../best.pt` | Fine-tuned YOLOv11 face detector (Jay / Adam / Aditya) |