import numpy as np
import os
import torch
import torch.nn as nn
import math
import json
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATA_DIR        = "gesture_data"
MODEL_SAVE_PATH = "gesture_model.pt"
LABELS_SAVE_PATH = "gesture_labels.json"

# Gesture classes — order matters, index = class ID
GESTURES = ["wave", "fist", "point", "thumbs_up", "open_palm"]

# People — add names here as more people collect data
PEOPLE = ["Jay", "Adam", "Aditya"]

EPOCHS         = 80
BATCH_SIZE     = 16       # safe for small datasets (200 samples), fine for large ones too
LEARNING_RATE  = 0.0001
TEST_SPLIT     = 0.2
INPUT_SIZE     = 42       # 21 landmarks x 2 (x, y)
D_MODEL        = 64
N_HEADS        = 4
N_LAYERS       = 3
D_FF           = 128
DROPOUT        = 0.1


# ── POSITIONAL ENCODING ───────────────────────────────────────────────────────
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


# ── TRANSFORMER MODEL ─────────────────────────────────────────────────────────
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
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        x = self.input_projection(x)
        x = self.pos_encoding(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.classifier(x)
        return x


# ── DATASET ───────────────────────────────────────────────────────────────────
class GestureDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ── DATA LOADER ───────────────────────────────────────────────────────────────
def load_data():
    """
    Loads gesture data from two file naming conventions:

    Per-person files (new format — recommended):
        gesture_data/Jay_fist.npy
        gesture_data/Adam_fist.npy
        gesture_data/Aditya_fist.npy

    Legacy shared files (old format — still supported):
        gesture_data/fist.npy

    All files for the same gesture are merged into one dataset.
    Gesture label = gesture name (filename), not person name.
    """
    all_X, all_y, label_names = [], [], []
    print("\nLoading gesture data...")

    for label_idx, gesture in enumerate(GESTURES):
        gesture_samples = []

        # Per-person files
        for person in PEOPLE:
            path = os.path.join(DATA_DIR, f"{person}_{gesture}.npy")
            if os.path.exists(path):
                data = np.load(path)
                gesture_samples.append(data)
                print(f"  {person}_{gesture}: {len(data)} samples")

        # Legacy shared file
        legacy_path = os.path.join(DATA_DIR, f"{gesture}.npy")
        if os.path.exists(legacy_path):
            data = np.load(legacy_path)
            gesture_samples.append(data)
            print(f"  {gesture} (shared): {len(data)} samples")

        if not gesture_samples:
            print(f"  {gesture}: no data found — skipping")
            continue

        merged = np.concatenate(gesture_samples, axis=0)
        all_X.append(merged)
        all_y.append(np.full(len(merged), label_idx))
        label_names.append(gesture)
        print(f"  → {gesture} total: {len(merged)} samples")

    if not all_X:
        raise RuntimeError("No gesture data found in gesture_data/")

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    print(f"\nTotal: {len(X)} samples across {len(label_names)} gestures")
    return X, y, label_names


# ── TRAINING LOOP ─────────────────────────────────────────────────────────────
def train(model, train_loader, val_loader, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    best_val_acc = 0
    print(f"\n{'Epoch':>6} | {'Loss':>8} | {'Train Acc':>9} | {'Val Acc':>7}")
    print("-" * 40)

    for epoch in range(EPOCHS):
        model.train()
        total_loss, correct, total = 0, 0, 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            out = model(X_batch)
            loss = criterion(out, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += out.argmax(1).eq(y_batch).sum().item()
            total += y_batch.size(0)

        scheduler.step()

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                out = model(X_batch)
                val_correct += out.argmax(1).eq(y_batch).sum().item()
                val_total += y_batch.size(0)

        train_acc = 100 * correct / total
        val_acc   = 100 * val_correct / val_total
        avg_loss  = total_loss / len(train_loader)

        print(f"{epoch+1:>6} | {avg_loss:>8.4f} | {train_acc:>8.1f}% | {val_acc:>6.1f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

    print(f"\nBest validation accuracy: {best_val_acc:.1f}%")
    print(f"Model saved to: {MODEL_SAVE_PATH}")


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 50)
    print("  GESTURE TRANSFORMER TRAINING")
    print("=" * 50)

    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    X, y, label_names = load_data()

    # Save label order — inference script reads this to map index → gesture name
    with open(LABELS_SAVE_PATH, "w") as f:
        json.dump(label_names, f)
    print(f"Labels saved: {label_names}")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SPLIT, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train)} samples | Val: {len(X_val)} samples")

    train_loader = DataLoader(GestureDataset(X_train, y_train),
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(GestureDataset(X_val, y_val),
                              batch_size=BATCH_SIZE)

    model = GestureTransformer(
        input_size=INPUT_SIZE,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        d_ff=D_FF,
        num_classes=len(label_names),
        dropout=DROPOUT
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    train(model, train_loader, val_loader, device)
    print("\nDone. Run main.py to test live.")


if __name__ == "__main__":
    main()