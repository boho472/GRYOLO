import numpy as np
import os
import torch
import torch.nn as nn
import math
import json
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# folder where gesture npy files are saved from gesture_collect.py
DATA_DIR = "gesture_data"

# where to save the trained model and label mapping after training
MODEL_SAVE_PATH = "gesture_model.pt"
LABELS_SAVE_PATH = "gesture_labels.json"

# base gesture names only — no person prefix
# the loader will automatically pick up Jay_fist.npy, Adi_fist.npy, Adam_fist.npy etc.
GESTURES = ["fist", "open_palm", "point", "thumbs_up", "wave"]

# how many times the model sees all the training data
EPOCHS = 360

# how many samples to process at once during training
BATCH_SIZE = 32

# how fast the model learns
LEARNING_RATE = 0.0008

# 20 percent of data is used for validation to check accuracy
TEST_SPLIT = 0.2

# 21 landmarks times 2 coordinates x and y equals 42 numbers per frame
INPUT_SIZE = 42

# internal size the transformer works with
D_MODEL = 128

# number of attention heads
N_HEADS = 4

# how many transformer encoder layers to stack
N_LAYERS = 3

# size of the feedforward network inside each transformer layer
D_FF = 512

# dropout randomly turns off neurons during training to prevent memorization
DROPOUT = 0.1


# ── DATA AUGMENTATION ─────────────────────────────────────────────────────────
def augment(sequence):
    # small random noise
    sequence = sequence + np.random.normal(0, 0.01, sequence.shape)
    # slight scaling
    scale = np.random.uniform(0.9, 1.1)
    sequence = sequence * scale
    # optional slight time shift
    if np.random.rand() > 0.5:
        shift = np.random.randint(-2, 3)
        sequence = np.roll(sequence, shift, axis=0)
    return sequence


# ── NORMALIZATION ─────────────────────────────────────────────────────────────
def normalize_landmarks(sequence):
    # sequence: (frames, 42)
    sequence = sequence.reshape(sequence.shape[0], 21, 2)
    # center on wrist (landmark 0)
    wrist = sequence[:, 0:1, :]
    sequence = sequence - wrist
    # scale by hand size (distance wrist -> middle finger tip)
    scale = np.linalg.norm(sequence[:, 12, :], axis=1, keepdims=True) + 1e-6
    sequence = sequence / scale[:, None, :]
    return sequence.reshape(sequence.shape[0], 42)


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
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )
        self.attn_pool = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Softmax(dim=1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )

    def forward(self, x, return_attention=False):
        x = self.input_projection(x)
        x = self.pos_encoding(x)
        x = self.transformer(x)
        weights = self.attn_pool(x)         # (batch, seq_len, 1)
        pooled = (x * weights).sum(dim=1)
        out = self.classifier(pooled)
        if return_attention:
            return out, weights.squeeze(-1)
        return out


# ── DATASET ───────────────────────────────────────────────────────────────────
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


# ── DATA LOADER ───────────────────────────────────────────────────────────────
def load_data():
    """
    Auto-discovers all .npy files in gesture_data/ and merges them by base gesture.

    Supports any naming convention:
        Jay_fist.npy   → class "fist"
        Adi_fist.npy   → class "fist"   (merged with Jay)
        Adam_fist.npy  → class "fist"   (merged with Jay + Adi)
        fist.npy       → class "fist"   (legacy shared file)

    Only gestures listed in GESTURES are loaded. Anything else is skipped.
    """
    all_X, all_y = [], []
    label_map = {}  # base_gesture → label_idx

    print("\nLoading gesture data...")

    for gesture_file in sorted(os.listdir(DATA_DIR)):
        if not gesture_file.endswith(".npy"):
            continue

        full_name = gesture_file.replace(".npy", "")  # e.g. "Jay_fist"

        # strip person prefix to get base gesture name
        # Jay_fist → fist | Adi_thumbs_up → thumbs_up | fist → fist
        parts = full_name.split("_", 1)
        base_gesture = parts[1] if len(parts) == 2 else parts[0]

        # skip anything not in our gesture list
        if base_gesture not in GESTURES:
            print(f"  Skipping {gesture_file} (not in GESTURES list)")
            continue

        # assign consistent label index across all people
        if base_gesture not in label_map:
            label_map[base_gesture] = len(label_map)

        label_idx = label_map[base_gesture]
        path = os.path.join(DATA_DIR, gesture_file)
        data = np.load(path)
        data = np.array([normalize_landmarks(seq) for seq in data])

        all_X.append(data)
        all_y.append(np.full(len(data), label_idx))
        print(f"  {full_name}: {len(data)} samples → class '{base_gesture}' (label {label_idx})")

    if not all_X:
        raise RuntimeError("No gesture data found in gesture_data/ — check your .npy files exist")

    # build label list ordered by index
    label_names = [k for k, v in sorted(label_map.items(), key=lambda x: x[1])]

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)

    print(f"\nTotal: {len(X)} samples across {len(label_names)} gestures: {label_names}")
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

        # print attention weights every 20 epochs to monitor training
        if (epoch % 20 == 0) and (epoch < EPOCHS - 20):
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(device)
                    outputs, attn = model(X_batch, return_attention=True)
                    attn = attn[0].detach().cpu().numpy()
                    print(f"  Attention sum: {attn.sum():.4f}  weights: {attn.round(3)}")
                    break

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

    # save label order — main.py reads this to map index → gesture name
    with open(LABELS_SAVE_PATH, "w") as f:
        json.dump(label_names, f)
    print(f"Labels saved to {LABELS_SAVE_PATH}: {label_names}")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SPLIT, random_state=42, stratify=y
    )

    train_loader = DataLoader(GestureDataset(X_train, y_train, augment=True),
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(GestureDataset(X_val, y_val, augment=False),
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
    print("\nDone. Run main.py to test the full pipeline.")


if __name__ == "__main__":
    main()