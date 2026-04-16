import cv2
import os
import shutil
import random
import urllib.request

# ── CONFIG ──────────────────────────────────────────────
CLASSES = {"Jface": 0, "Aface": 1}
SPLIT_RATIO = 0.8  # 80% train, 20% val
# ────────────────────────────────────────────────────────

# Download DNN face detector if not present
if not os.path.exists("deploy.prototxt"):
    print("Downloading face detector model...")
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
        "deploy.prototxt"
    )
    urllib.request.urlretrieve(
        "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
        "face.caffemodel"
    )

net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "face.caffemodel")

# Create folder structure
for split in ["train", "val"]:
    os.makedirs(f"dataset/images/{split}", exist_ok=True)
    os.makedirs(f"dataset/labels/{split}", exist_ok=True)

def detect_face(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None, None
    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104, 177, 123))
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.6:
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            x1, y1, x2, y2 = box.astype(int)
            # Convert to YOLO format (normalized cx, cy, w, h)
            cx = ((x1 + x2) / 2) / w
            cy = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            return img, (cx, cy, bw, bh)
    return img, None

total_saved = 0
total_skipped = 0

for folder, class_id in CLASSES.items():
    images = [f for f in os.listdir(folder) if f.endswith(".jpg")]
    random.shuffle(images)

    split_idx = int(len(images) * SPLIT_RATIO)
    splits = {"train": images[:split_idx], "val": images[split_idx:]}

    print(f"\nProcessing {folder} (class {class_id})...")

    for split, files in splits.items():
        for fname in files:
            src_path = os.path.join(folder, fname)
            img, bbox = detect_face(src_path)

            if bbox is None:
                total_skipped += 1
                continue

            # Save image
            dest_img = f"dataset/images/{split}/{folder}_{fname}"
            shutil.copy(src_path, dest_img)

            # Save YOLO annotation
            dest_label = f"dataset/labels/{split}/{folder}_{fname.replace('.jpg', '.txt')}"
            with open(dest_label, "w") as f:
                cx, cy, bw, bh = bbox
                f.write(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

            total_saved += 1

# Write faces.yaml
with open("faces.yaml", "w") as f:
    f.write(f"""path: dataset
train: images/train
val: images/val

nc: 2
names: ['Jay', 'Adam']
""")

print(f"\n✅ Done!")
print(f"   Saved  : {total_saved} images with annotations")
print(f"   Skipped: {total_skipped} images (no face detected)")
print(f"   Config : faces.yaml ready")