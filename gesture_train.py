import numpy as np
import os
import torch
import torch.nn as nn
import math
import json
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATA_DIR         = "gesture_data"
MODEL_SAVE_PATH  = "gesture_model.pt"
LABELS_SAVE_PATH = "gesture_labels.json"

GESTURES = ["fist", "open_palm", "point", "thumbs_up", "wave"]
PEOPLE   = ["Jay", "Adi"]

EPOCHS        = 360
BATCH_SIZE    = 16
LEARNING_RATE = 0.0008
TEST_SPLIT    = 0.2
INPUT_SIZE    = 42
D_MODEL       = 128
N_HEADS       = 4
N_LAYERS      = 3
D_FF          = 512
DROPOUT       = 0.1


def augment(sequence):
    sequence = sequence + np.random.normal(0, 0.01, sequence.shape)
    scale = np.random.uniform(0.9, 1.1)
    sequence = sequence * scale
    if np.random.rand() > 0.5:
        shift = np.random.randint(-2, 3)
        sequence = np.roll(sequence, shift, axis=0)
    return sequence


def normalize_landmarks(sequence):
    sequence = sequence.reshape(sequence.shape[0], 21, 2)
    wrist = sequence[:, 0:1, :]
    sequence = sequence - wrist
    scale = np.linalg.norm(sequence[:, 12, :], axis=1, keepdims=True) + 1e-6
    sequence = sequence / scale[:, None, :]
    return sequence.reshape(sequence.shape[0], 42)


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


class GestureDataset(Dataset):
    def __init__(self, X, y, augment=False):
        self.X = X
        self.y = y
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.augment:
            x = augment(x)
        return torch.FloatTensor(x), torch.LongTensor([self.y[idx]])[0]


def load_data():
    all_X, all_y, label_names = [], [], []
    print("\nLoading gesture data...")

    for label_idx, gesture in enumerate(GESTURES):
        gesture_samples = []

        for person in PEOPLE:
            path = os.path.join(DATA_DIR, f"{person}_{gesture}.npy")
            if os.path.exists(path):
                data = np.load(path)
                gesture_samples.append(data)
                print(f"  {person}_{gesture}: {len(data)} samples")

        legacy_path = os.path.join(DATA_DIR, f"{gesture}.npy")
        if os.path.exists(legacy_path):
            data = np.load(legacy_path)
            gesture_samples.append(data)
            print(f"  {gesture} (shared): {len(data)} samples")

        if not gesture_samples:
            print(f"  {gesture}: no data found — skipping")
            continue

        merged = np.concatenate(gesture_samples, axis=0)
        merged = np.array([normalize_landmarks(seq) for seq in merged])
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

        if epoch % 20 == 0 and epoch < EPOCHS - 20:
            with torch.no_grad():
                for X_batch, _ in val_loader:
                    X_batch = X_batch.to(device)
                    _, attn = model(X_batch, return_attention=True)
                    attn = attn[0].detach().cpu().numpy()
                    print(f"  Attention: {np.round(attn, 3)}  sum={attn.sum():.3f}")
                    break

    print(f"\nBest validation accuracy: {best_val_acc:.1f}%")
    print(f"Model saved to: {MODEL_SAVE_PATH}")


def main():
    print("=" * 50)
    print("  GESTURE TRANSFORMER TRAINING")
    print("=" * 50)

    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    X, y, label_names = load_data()

    with open(LABELS_SAVE_PATH, "w") as f:
        json.dump(label_names, f)
    print(f"Labels saved: {label_names}")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SPLIT, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train)} samples | Val: {len(X_val)} samples")

    train_loader = DataLoader(GestureDataset(X_train, y_train, augment=True),
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(GestureDataset(X_val, y_val, augment=False),
                              batch_size=BATCH_SIZE)

    model = GestureTransformer(
        input_size=INPUT_SIZE, d_model=D_MODEL, n_heads=N_HEADS,
        n_layers=N_LAYERS, d_ff=D_FF, num_classes=len(label_names), dropout=DROPOUT
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    train(model, train_loader, val_loader, device)
    print("\nDone. Run main.py to test live.")


if __name__ == "__main__":
    main()