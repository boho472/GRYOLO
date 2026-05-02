import cv2
import os
import shutil

# --- CONFIG ---
SOURCE_FOLDERS = {
    "adam": ("Adamface_full", 0),
    "jay": ("Jayface_full", 1),
    "aditya": ("Adityaface_full", 2),
}

OUTPUT = "dataset"

# --- Create dataset structure ---
for split in ["train", "val"]:
    os.makedirs(f"{OUTPUT}/images/{split}", exist_ok=True)
    os.makedirs(f"{OUTPUT}/labels/{split}", exist_ok=True)

# --- Load face detector ---
net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "face.caffemodel")

# --- Process images ---
import random

for person, (folder, class_id) in SOURCE_FOLDERS.items():
    images = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    random.shuffle(images)

    split_idx = int(len(images) * 0.8)
    splits = {
        "train": images[:split_idx],
        "val": images[split_idx:]
    }

    for split, img_list in splits.items():
        for img_name in img_list:
            path = os.path.join(folder, img_name)
            image = cv2.imread(path)

            if image is None:
                continue

            h, w = image.shape[:2]

            # Detect face
            blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104, 177, 123))
            net.setInput(blob)
            detections = net.forward()

            best_box = None
            best_conf = 0

            for i in range(detections.shape[2]):
                conf = detections[0, 0, i, 2]
                if conf > 0.6 and conf > best_conf:
                    box = detections[0, 0, i, 3:7] * [w, h, w, h]
                    x1, y1, x2, y2 = box.astype(int)
                    best_box = (x1, y1, x2 - x1, y2 - y1)
                    best_conf = conf

            if best_box is None:
                continue  # skip if no face found

            x, y, bw, bh = best_box

            # Normalize for YOLO
            x_center = (x + bw / 2) / w
            y_center = (y + bh / 2) / h
            width = bw / w
            height = bh / h

            # Save image
            new_name = f"{person}_{img_name}"
            out_img = os.path.join(OUTPUT, "images", split, new_name)
            shutil.copy(path, out_img)

            # Save label
            label_path = os.path.join(OUTPUT, "labels", split, new_name.rsplit(".", 1)[0] + ".txt")
            with open(label_path, "w") as f:
                f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

print("✅ Dataset built from existing photos!")