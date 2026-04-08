"""
Shokri-style Membership Inference Attack (MIA) reproduction on CIFAR-10.

This is a scaled-down reproduction of the shadow-model attack proposed in:
    Shokri, R., Stronati, M., Song, C., & Shmatikov, V. (2017).
    Membership Inference Attacks Against Machine Learning Models. IEEE S&P.

Pipeline:
    1. Train one TARGET model on a random subset of CIFAR-10.
    2. Train N SHADOW models on disjoint subsets drawn from CIFAR-10's
       training split. Each shadow gets its own "member" and "non-member"
       set (the complement inside CIFAR-10 train).
    3. Query each shadow with its member and non-member sets and record
       the softmax confidence vectors, labelled 1 (in training) or 0.
    4. Train a per-class attack MLP on these confidence vectors.
    5. Evaluate the attack on the target model's members and a disjoint
       non-member set drawn from the remaining CIFAR-10 train samples.

Metrics reported:
    - Target model train/test accuracy (to show how overfit it is)
    - Attack accuracy on the target
    - Attack AUC
    - TPR @ FPR = 0.1% and 1% (low-FPR regime, following Carlini 2022)

The defaults are chosen so the whole pipeline runs on a single CPU in
roughly 20-40 minutes. They are *not* the full Shokri setup, and the
numbers are therefore expected to be lower than the original paper. The
comparison with the original numbers is discussed in the midterm report.

Run (from repo root):
    python scripts/run_mia.py --out data/results_mia.json
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms
from sklearn.metrics import roc_auc_score, roc_curve


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class SmallCNN(nn.Module):
    """Compact CNN used as both target and shadow model."""

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


class AttackMLP(nn.Module):
    """Per-class attack classifier, input = softmax confidence vector."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_classes, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Training / evaluation helpers
# ---------------------------------------------------------------------------

def train_model(model: nn.Module, loader: DataLoader, epochs: int, lr: float,
                device: torch.device) -> None:
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()


@torch.no_grad()
def accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total, correct = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += x.size(0)
    return correct / max(total, 1)


@torch.no_grad()
def softmax_scores(model: nn.Module, loader: DataLoader,
                   device: torch.device):
    """Return softmax vectors and labels for every sample in the loader."""
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        x = x.to(device)
        p = F.softmax(model(x), dim=1).cpu().numpy()
        probs.append(p)
        labels.append(y.numpy())
    return np.concatenate(probs), np.concatenate(labels)


# ---------------------------------------------------------------------------
# MIA pipeline
# ---------------------------------------------------------------------------

def build_cifar10(data_dir: str):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])
    train = datasets.CIFAR10(data_dir, train=True, download=True,
                             transform=transform)
    return train


def loader_from_indices(dataset, indices, batch_size: int, shuffle: bool):
    return DataLoader(Subset(dataset, indices.tolist()),
                      batch_size=batch_size, shuffle=shuffle, num_workers=0)


def train_and_collect(dataset, member_idx, nonmember_idx, epochs, lr,
                      batch_size, device, tag):
    model = SmallCNN().to(device)
    train_loader = loader_from_indices(dataset, member_idx, batch_size, True)
    train_model(model, train_loader, epochs, lr, device)

    eval_mem = loader_from_indices(dataset, member_idx, 256, False)
    eval_non = loader_from_indices(dataset, nonmember_idx, 256, False)
    train_acc = accuracy(model, eval_mem, device)
    test_acc = accuracy(model, eval_non, device)
    print(f"  [{tag}] train_acc={train_acc:.4f}  test_acc={test_acc:.4f}")

    mem_probs, mem_lbls = softmax_scores(model, eval_mem, device)
    non_probs, non_lbls = softmax_scores(model, eval_non, device)

    return {
        "model": model,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "mem_probs": mem_probs, "mem_labels": mem_lbls,
        "non_probs": non_probs, "non_labels": non_lbls,
    }


def train_attack_classifiers(shadow_results, num_classes, device, epochs=30):
    """Train one attack MLP per class (as in Shokri et al., 2017)."""
    # Aggregate shadow data per class
    per_class = {c: {"X": [], "y": []} for c in range(num_classes)}
    for sh in shadow_results:
        for probs, lbls, is_mem in (
            (sh["mem_probs"], sh["mem_labels"], 1),
            (sh["non_probs"], sh["non_labels"], 0),
        ):
            for p, c in zip(probs, lbls):
                per_class[int(c)]["X"].append(p)
                per_class[int(c)]["y"].append(is_mem)

    attack_models = {}
    for c in range(num_classes):
        X = np.asarray(per_class[c]["X"], dtype=np.float32)
        y = np.asarray(per_class[c]["y"], dtype=np.int64)
        if len(y) == 0:
            continue
        ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        loader = DataLoader(ds, batch_size=128, shuffle=True)

        m = AttackMLP(num_classes=num_classes).to(device)
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        m.train()
        for _ in range(epochs):
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                F.cross_entropy(m(xb), yb).backward()
                opt.step()
        attack_models[c] = m
    return attack_models


@torch.no_grad()
def attack_scores(attack_models, probs, labels, device):
    """Return P(member) for each sample using its per-class attack MLP."""
    scores = np.zeros(len(labels), dtype=np.float32)
    for c, m in attack_models.items():
        mask = labels == c
        if not mask.any():
            continue
        x = torch.from_numpy(probs[mask].astype(np.float32)).to(device)
        m.eval()
        p = F.softmax(m(x), dim=1)[:, 1].cpu().numpy()
        scores[mask] = p
    return scores


def tpr_at_fpr(y_true, y_score, target_fpr):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    idx = np.searchsorted(fpr, target_fpr, side="right") - 1
    idx = max(idx, 0)
    return float(tpr[idx])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--out", default="data/results_mia.json")
    parser.add_argument("--num-shadows", type=int, default=4)
    parser.add_argument("--train-size", type=int, default=2500,
                        help="member set size per model (target + each shadow)")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--attack-epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cpu")
    t0 = time.time()

    print("[step 1/4] loading CIFAR-10")
    train_ds = build_cifar10(args.data)
    n = len(train_ds)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)

    # Partition indices: 1 target + N shadows, each with member+nonmember
    needed = (1 + args.num_shadows) * 2 * args.train_size
    if needed > n:
        raise RuntimeError(
            f"Not enough CIFAR-10 samples: need {needed}, have {n}")
    chunks = [perm[i * args.train_size:(i + 1) * args.train_size]
              for i in range(2 * (1 + args.num_shadows))]

    target_mem, target_non = chunks[0], chunks[1]
    shadows = [(chunks[2 + 2 * i], chunks[3 + 2 * i])
               for i in range(args.num_shadows)]

    # -------- Target --------
    print(f"[step 2/4] training target model (train_size={args.train_size}, "
          f"epochs={args.epochs})")
    target = train_and_collect(train_ds, target_mem, target_non,
                               args.epochs, args.lr, args.batch_size,
                               device, tag="target")

    # -------- Shadows --------
    print(f"[step 3/4] training {args.num_shadows} shadow models")
    shadow_results = []
    for i, (mi, ni) in enumerate(shadows):
        print(f" shadow {i + 1}/{args.num_shadows}")
        shadow_results.append(
            train_and_collect(train_ds, mi, ni, args.epochs, args.lr,
                              args.batch_size, device, tag=f"shadow-{i+1}"))

    # -------- Attack classifier --------
    print("[step 4/4] training attack classifiers and evaluating on target")
    attack_models = train_attack_classifiers(
        shadow_results, num_classes=10, device=device,
        epochs=args.attack_epochs)

    # Evaluate on target member vs non-member
    mem_scores = attack_scores(attack_models, target["mem_probs"],
                               target["mem_labels"], device)
    non_scores = attack_scores(attack_models, target["non_probs"],
                               target["non_labels"], device)

    y_true = np.concatenate([np.ones_like(mem_scores),
                             np.zeros_like(non_scores)])
    y_score = np.concatenate([mem_scores, non_scores])

    pred = (y_score >= 0.5).astype(int)
    attack_acc = float((pred == y_true).mean())
    auc = float(roc_auc_score(y_true, y_score))
    tpr_at_1 = tpr_at_fpr(y_true, y_score, 0.01)
    tpr_at_01 = tpr_at_fpr(y_true, y_score, 0.001)

    wall = time.time() - t0
    results = {
        "config": vars(args),
        "wall_seconds": round(wall, 1),
        "target": {
            "train_acc": target["train_acc"],
            "test_acc": target["test_acc"],
            "generalization_gap": target["train_acc"] - target["test_acc"],
        },
        "shadows": [
            {"train_acc": s["train_acc"], "test_acc": s["test_acc"]}
            for s in shadow_results
        ],
        "attack": {
            "accuracy": attack_acc,
            "auc": auc,
            "tpr_at_fpr_1pct": tpr_at_1,
            "tpr_at_fpr_0.1pct": tpr_at_01,
        },
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(json.dumps(results, indent=2))
    print(f"\nsaved to {args.out}")
    print(f"total wall time: {wall:.1f}s")


if __name__ == "__main__":
    main()
