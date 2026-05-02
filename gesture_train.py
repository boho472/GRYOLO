import numpy as np
import os
import torch
import torch.nn as nn
import math
import json
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt # so i can see how its predicting attention across the frames, not necessary for training

# folder where gesture npy files are saved from gesture_collect.py
DATA_DIR = "gesture_data"

# where to save the trained model and label mapping after training
# MODEL_SAVE_PATH = "gesture_model.pt"
# LABELS_SAVE_PATH = "gesture_labels.json"
MODEL_SAVE_PATH = "gesture_model_NOISES.pt"
LABELS_SAVE_PATH = "gesture_labels_NOISES.json"

# list of gestures we are training on, must match what was collected
GESTURES = ["Jay_fist", "Jay_open_palm", "Jay_point", "Jay_thumbs_up", "Jay_wave"]

# how many times the model sees all the training data
EPOCHS = 360

# how many samples to process at once during training
BATCH_SIZE = 32

# how fast the model learns, smaller means slower but more stable
LEARNING_RATE = 0.0008 # tested it on Adam desktop
# 0.0008 is best so far,
# this works best over 0.001 and it gets low Accuracy and under 0.00001 and it gets bad again too

# 20 percent of data is used for validation to check accuracy
TEST_SPLIT = 0.2

# 21 landmarks times 2 coordinates x and y equals 42 numbers per frame
INPUT_SIZE = 42

# internal size the transformer works with, all layers use this dimension
D_MODEL = 128 # tested 256 and it works better than 128

# number of attention heads, each head looks at the sequence differently
# D_MODEL must be divisible by N_HEADS so 128 divided by 8 works
N_HEADS = 4  # double it to match me double D_model

# how many transformer encoder layers to stack
N_LAYERS = 3

# size of the feedforward network inside each transformer layer
D_FF = 512 #og 128 now testing 2048 to got with 

# dropout randomly turns off neurons during training to prevent memorization
DROPOUT = 0.1

# data augmentation adds variety to the training data so the model learns more general patterns and doesn't just memorize the samples it sees
def augment(sequence):
    # small random noise
    sequence = sequence + np.random.normal(0, 0.01, sequence.shape)

    # slight scaling
    scale = np.random.uniform(0.9, 1.1)
    sequence = sequence * scale

    # optional: slight time shift
    if np.random.rand() > 0.5:
        shift = np.random.randint(-2, 3)
        sequence = np.roll(sequence, shift, axis=0)

    return sequence

# NORMALIZATION
# we want the model to focus on the hand pose, not where the hand is in the
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

# POSITIONAL ENCODING
# transformer has no built in sense of order so we add position information
# each frame gets a unique signal so the model knows frame 1 comes before frame 2

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=30, dropout=0.1):
        super().__init__()
        
        
        self.dropout = nn.Dropout(dropout)

        # create a matrix of position signals using sine and cosine waves
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        # even positions use sine, odd positions use cosine
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # add position signal to each frame in the sequence
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# TRANSFORMER MODEL
# takes 30 frames of hand landmarks and classifies which gesture it is
# uses self attention to figure out which frames matter most for each gesture

class GestureTransformer(nn.Module):
    def __init__(self, input_size, d_model, n_heads, n_layers, d_ff,
                 num_classes, dropout=0.1):
        super().__init__()
        
        

        # step 1: convert 42 input features to 64 dimensions the transformer expects
        self.input_projection = nn.Linear(input_size, d_model)

        # step 2: add positional encoding so model knows frame order
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)

        # step 3: stack transformer encoder layers
        # each layer uses self attention to compare all frames against each other
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

        # step 4: classification head that outputs a probability for each gesture
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )

    def forward(self, x, return_attention=False):
        # x comes in as batch of sequences, shape is batch x 30 frames x 42 coords
        x = self.input_projection(x)    # expand to 64 dims
        x = self.pos_encoding(x)        # add position info
        x = self.transformer(x)         # self attention across all 30 frames
        weights = self.attn_pool(x)          # (batch, seq_len, 1)
        pooled = (x * weights).sum(dim=1)

        out = self.classifier(pooled)

        if return_attention:
            return out, weights.squeeze(-1)  # shape: (batch, seq_len)

        return out


# DATASET
# wraps numpy arrays into pytorch format so the dataloader can use them

class GestureDataset(Dataset):
    def __init__(self, X, y, augment=False):   # ✅ accept augment
        self.X = X
        self.y = y
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]

        if self.augment:
            x = augment(x)   # apply augmentation dynamically

        return torch.FloatTensor(x), torch.LongTensor([self.y[idx]])[0]


def load_data():
    # loads all gesture npy files and combines them into one dataset
    all_X, all_y, label_names = [], [], []
    print("\nLoading gesture data...")

    for label_idx, gesture in enumerate(GESTURES):
        path = os.path.join(DATA_DIR, f"{gesture}.npy")
        if not os.path.exists(path):
            print(f"  Missing: {path} skipping")
            continue
        data = np.load(path)
        data = np.array([normalize_landmarks(seq) for seq in data])
        all_X.append(data)
        # assign the same label number to every sample of this gesture
        all_y.append(np.full(len(data), label_idx))
        label_names.append(gesture)
        print(f"  {gesture}: {len(data)} samples")

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    print(f"\nTotal: {len(X)} samples, {len(label_names)} gestures")
    return X, y, label_names


def train(model, train_loader, val_loader, device):
    # crossentropyloss measures how wrong the predictions are
    criterion = nn.CrossEntropyLoss()

    # adam optimizer adjusts the model weights to reduce the loss each epoch
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # scheduler reduces learning rate every 20 epochs to fine tune the model
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    best_val_acc = 0
    print(f"\n{'Epoch':>6} | {'Loss':>8} | {'Train Acc':>9} | {'Val Acc':>7}")
    print("-" * 40)

    for epoch in range(EPOCHS):
        model.train()
        total_loss, correct, total = 0, 0, 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # clear old gradients, run forward pass, calculate loss, update weights
            optimizer.zero_grad()
            out = model(X_batch)
            loss = criterion(out, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += out.argmax(1).eq(y_batch).sum().item()
            total += y_batch.size(0)

        scheduler.step()

        # check accuracy on validation set without updating weights
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                out = model(X_batch)
                val_correct += out.argmax(1).eq(y_batch).sum().item()
                val_total += y_batch.size(0)

        train_acc = 100 * correct / total
        val_acc = 100 * val_correct / val_total
        avg_loss = total_loss / len(train_loader)

        print(f"{epoch+1:>6} | {avg_loss:>8.4f} | {train_acc:>8.1f}% | {val_acc:>6.1f}%")

        # save the model whenever validation accuracy improves
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            
        if ((epoch % 20 == 0) and (epoch < EPOCHS - 20)):  # every 20 epochs
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(device)

                    outputs, attn = model(X_batch, return_attention=True)

                    attn = attn[0].detach().cpu().numpy()

                    print("Attention:", attn)
                    print("Sum:", attn.sum())

                    # # 🔥 PLOT IT
                    # plt.plot(attn)
                    # plt.title("Attention over frames")
                    # plt.xlabel("Frame")
                    # plt.ylabel("Attention weight")
                    # plt.show()

                    break

    print(f"\nBest validation accuracy: {best_val_acc:.1f}%")
    print(f"Model saved to: {MODEL_SAVE_PATH}")


def main():
    print("="*50)
    print("  GESTURE TRANSFORMER TRAINING")
    print("="*50)

    # use apple gpu on mac, nvidia gpu if available, otherwise cpu
    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    X, y, label_names = load_data()

    # save label order to json so inference script knows which number means which gesture
    with open(LABELS_SAVE_PATH, "w") as f:
        json.dump(label_names, f)

    # split into 80 percent training and 20 percent validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SPLIT, random_state=42, stratify=y
    )   
    
   

    train_loader = DataLoader(GestureDataset(X_train, y_train, augment=True ),
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(GestureDataset(X_val, y_val, augment=False),
                            batch_size=BATCH_SIZE)

    # build the transformer model with the config settings defined at the top
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
    print("\nDone. Run gesture_inference.py to test live.")
    model.eval()

    
if __name__ == "__main__":
    main()
    