import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
from tqdm import tqdm

# ============================================================
# 🔹 MODEL ARCHITECTURE
# ============================================================

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = x.mean(dim=2)
        y = self.fc(y)
        y = y.view(b, c, 1)
        return x * y


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7):
        super(ResBlock, self).__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.se = SEBlock(out_channels)

    def forward(self, x):
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += identity
        return F.relu(out)


class TemporalAttentionPooling(nn.Module):
    def __init__(self, in_features):
        super(TemporalAttentionPooling, self).__init__()
        self.attn = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        weights = F.softmax(self.attn(x), dim=1)
        return torch.sum(weights * x, dim=1)


class ECGAttentionNet(nn.Module):
    def __init__(self, input_channels=12, num_classes=2):
        super(ECGAttentionNet, self).__init__()
        self.layer1 = ResBlock(input_channels, 32)
        self.layer2 = ResBlock(32, 64)
        self.layer3 = ResBlock(64, 128)
        self.global_pool = TemporalAttentionPooling(128)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = F.max_pool1d(x, 2)
        x = self.layer2(x)
        x = F.max_pool1d(x, 2)
        x = self.layer3(x)
        x = x.permute(0, 2, 1)
        x = self.global_pool(x)
        x = self.dropout(x)
        return self.fc(x)


# ============================================================
# 🔹 TRAINING + EVALUATION
# ============================================================

def train_one_epoch(model, dataloader, optimizer, criterion, scaler, device):
    model.train()
    total_loss = 0
    for xb, yb in tqdm(dataloader, leave=False):
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            outputs = model(xb)
            loss = criterion(outputs, yb)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for xb, yb in dataloader:
            xb, yb = xb.to(device), yb.to(device)
            outputs = model(xb)
            preds.extend(torch.argmax(outputs, 1).cpu().numpy())
            targets.extend(yb.cpu().numpy())

    acc = accuracy_score(targets, preds)
    f1 = f1_score(targets, preds, average='macro')
    prec = precision_score(targets, preds, average='macro', zero_division=0)
    rec = recall_score(targets, preds, average='macro', zero_division=0)
    return acc, f1, prec, rec, confusion_matrix(targets, preds)


# ============================================================
# 🔹 MAIN SCRIPT (SAFE TO IMPORT)
# ============================================================

if __name__ == "__main__":
    print("✅ Using device:", "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    DATA_PATH = "data/processed/ptbxl_preprocessed.pkl"
    SAVE_PATH = "weights/best_model_v4.pth"
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    with open(DATA_PATH, "rb") as f:
        data = pickle.load(f)

    X_train = torch.tensor(np.array(data["X_train"]), dtype=torch.float32).permute(0, 2, 1)
    y_train = torch.tensor(np.array(data["y_train"]), dtype=torch.long)
    X_test = torch.tensor(np.array(data["X_test"]), dtype=torch.float32).permute(0, 2, 1)
    y_test = torch.tensor(np.array(data["y_test"]), dtype=torch.long)

    print(f"Train: {len(X_train)} | Test: {len(X_test)} | Classes: {len(np.unique(y_train))}")

    # ✅ FIXED DATALOADER (no deadlock on Windows)
    train_ds = TensorDataset(X_train, y_train)
    test_ds = TensorDataset(X_test, y_test)
    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size=64, num_workers=0, pin_memory=True)

    model = ECGAttentionNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda')

    best_f1 = 0
    patience = 8
    patience_counter = 0
    num_epochs = 80

    for epoch in range(1, num_epochs + 1):
        loss = train_one_epoch(model, train_dl, optimizer, criterion, scaler, device)
        acc, f1, prec, rec, cm = evaluate(model, test_dl, device)

        print(f"Epoch [{epoch}/{num_epochs}] - TrainLoss: {loss:.4f} | Val_Acc: {acc:.4f} | Val_F1: {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"💾 Saved best model (epoch {epoch}) - F1: {f1:.4f}  Acc: {acc:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("⏸ Early stopping triggered.")
                break

    # Final evaluation
    model.load_state_dict(torch.load(SAVE_PATH))
    acc, f1, prec, rec, cm = evaluate(model, test_dl, device)
    print(f"\n🔍 Final Evaluation on test set:")
    print(f"Accuracy: {acc:.4f} | F1 (macro): {f1:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f}")

    os.makedirs("reports", exist_ok=True)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("reports/confusion_matrix_v4.png", dpi=300)
    print("✅ Confusion matrix saved: reports/confusion_matrix_v4.png")
