"""
Minimal CIFAR-10 target model for membership inference experiments.

This script trains a small CNN on CIFAR-10 and saves the model weights
along with the indices of the training samples. These indices are the
"members" set used later by the membership inference attack.

Usage (inside the container):
    python scripts/train_cifar10.py --epochs 20 --out /workspace/data/target.pt
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


class SmallCNN(nn.Module):
    """A compact CNN used as the target model in Shokri-style MIA setups."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def get_loaders(data_dir: str, train_size: int, batch_size: int, seed: int):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])
    full_train = datasets.CIFAR10(data_dir, train=True, download=True,
                                  transform=transform)
    test = datasets.CIFAR10(data_dir, train=False, download=True,
                            transform=transform)

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(full_train))[:train_size]
    train_subset = Subset(full_train, indices.tolist())

    train_loader = DataLoader(train_subset, batch_size=batch_size,
                              shuffle=True, num_workers=2)
    test_loader = DataLoader(test, batch_size=batch_size,
                             shuffle=False, num_workers=2)
    return train_loader, test_loader, indices


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return loss_sum / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total, correct = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return correct / total


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CIFAR-10 target model")
    parser.add_argument("--data", default="/workspace/data")
    parser.add_argument("--out", default="/workspace/data/target.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--train-size", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] using device: {device}")

    train_loader, test_loader, member_indices = get_loaders(
        args.data, args.train_size, args.batch_size, args.seed
    )
    print(f"[info] train size: {len(member_indices)}  test size: {len(test_loader.dataset)}")

    model = SmallCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, device)
        te_acc = evaluate(model, test_loader, device)
        print(f"epoch {epoch:02d}  train_loss={tr_loss:.4f}  "
              f"train_acc={tr_acc:.4f}  test_acc={te_acc:.4f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "member_indices": member_indices,
            "config": vars(args),
        },
        args.out,
    )
    print(f"[done] saved model + member indices to {args.out}")


if __name__ == "__main__":
    main()
