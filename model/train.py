import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import BALANCED_CLASSES, get_loaders


class CharCNN(nn.Module):
    def __init__(self, num_classes: int = len(BALANCED_CLASSES)):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # 28 -> 14
        x = self.pool(F.relu(self.conv2(x)))   # 14 -> 7
        x = self.pool(F.relu(self.conv3(x)))   # 7  -> 3
        x = x.flatten(1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)

    def features(self, x):
        """Returns the last conv feature map, for Grad-CAM."""
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        return x


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        n += images.size(0)
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, n = 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        correct += (logits.argmax(1) == labels).sum().item()
        n += images.size(0)
    return correct / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--out", type=str, default="../backend/checkpoint.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, test_loader = get_loaders(args.data_dir, args.batch_size)
    model = CharCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
        test_acc = evaluate(model, test_loader, device)
        scheduler.step()
        print(
            f"epoch {epoch}/{args.epochs}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  test_acc={test_acc:.4f}"
        )
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(
                {"model_state": model.state_dict(), "classes": BALANCED_CLASSES, "test_acc": test_acc},
                args.out,
            )
            print(f"  saved new best checkpoint ({test_acc:.4f}) -> {args.out}")

    print(f"Done. Best test accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()
