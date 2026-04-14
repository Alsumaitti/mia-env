"""
Preliminary Experiment Suite for MIA Independent Study.

Runs multiple configurations of the Shokri-style MIA to study how
three factors affect membership inference:

  Factor 1: Training set size  (2500 / 5000 / 10000)
  Factor 2: Number of shadows  (4 / 8)
  Factor 3: Regularization     (none / weight_decay=1e-4)

The baseline (Exp 0) is the original run: train_size=2500, shadows=4, no regularization.
Experiments 1-5 vary one factor at a time relative to baseline.

Output: data/experiment_results.json — array of per-experiment result dicts.

Run (from repo root):
    python scripts/run_experiments.py
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
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class AttackMLP(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_classes, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def train_model(model, loader, epochs, lr, device, weight_decay=0.0):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=weight_decay)
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            F.cross_entropy(model(x), y).backward()
            optimizer.step()


@torch.no_grad()
def accuracy(model, loader, device):
    model.eval()
    total, correct = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        total += x.size(0)
    return correct / max(total, 1)


@torch.no_grad()
def softmax_scores(model, loader, device):
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        p = F.softmax(model(x.to(device)), dim=1).cpu().numpy()
        probs.append(p)
        labels.append(y.numpy())
    return np.concatenate(probs), np.concatenate(labels)


def loader_from_indices(dataset, indices, batch_size, shuffle):
    return DataLoader(Subset(dataset, indices.tolist()),
                      batch_size=batch_size, shuffle=shuffle, num_workers=0)


def train_and_collect(dataset, member_idx, nonmember_idx, epochs, lr,
                      batch_size, device, weight_decay, tag):
    model = SmallCNN().to(device)
    train_loader = loader_from_indices(dataset, member_idx, batch_size, True)
    train_model(model, train_loader, epochs, lr, device,
                weight_decay=weight_decay)

    eval_mem = loader_from_indices(dataset, member_idx, 256, False)
    eval_non = loader_from_indices(dataset, nonmember_idx, 256, False)
    train_acc = accuracy(model, eval_mem, device)
    test_acc = accuracy(model, eval_non, device)
    print(f"    [{tag}] train={train_acc:.4f}  test={test_acc:.4f}  "
          f"gap={train_acc - test_acc:.4f}")

    mem_probs, mem_lbls = softmax_scores(model, eval_mem, device)
    non_probs, non_lbls = softmax_scores(model, eval_non, device)
    return {
        "train_acc": train_acc, "test_acc": test_acc,
        "mem_probs": mem_probs, "mem_labels": mem_lbls,
        "non_probs": non_probs, "non_labels": non_lbls,
    }


def train_attack_classifiers(shadow_results, num_classes, device, epochs=30):
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
def attack_scores_fn(attack_models, probs, labels, device):
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
    idx = max(np.searchsorted(fpr, target_fpr, side="right") - 1, 0)
    return float(tpr[idx])


# ---------------------------------------------------------------------------
# Single experiment run
# ---------------------------------------------------------------------------

def run_one_experiment(exp_id, name, train_ds, train_size, num_shadows,
                       epochs, lr, batch_size, weight_decay, seed, device):
    print(f"\n{'='*60}")
    print(f"  Experiment {exp_id}: {name}")
    print(f"  train_size={train_size}, shadows={num_shadows}, "
          f"weight_decay={weight_decay}")
    print(f"{'='*60}")
    t0 = time.time()

    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    n = len(train_ds)
    perm = rng.permutation(n)
    needed = (1 + num_shadows) * 2 * train_size
    if needed > n:
        raise RuntimeError(f"Need {needed} samples, have {n}")

    chunks = [perm[i * train_size:(i + 1) * train_size]
              for i in range(2 * (1 + num_shadows))]
    target_mem, target_non = chunks[0], chunks[1]
    shadows = [(chunks[2 + 2*i], chunks[3 + 2*i])
               for i in range(num_shadows)]

    # Target
    print(f"  Training target model...")
    target = train_and_collect(train_ds, target_mem, target_non,
                                epochs, lr, batch_size, device,
                                weight_decay, tag="target")

    # Shadows
    print(f"  Training {num_shadows} shadow models...")
    shadow_results = []
    for i, (mi, ni) in enumerate(shadows):
        shadow_results.append(
            train_and_collect(train_ds, mi, ni, epochs, lr, batch_size,
                              device, weight_decay,
                              tag=f"shadow-{i+1}"))

    # Attack
    print(f"  Training attack classifiers...")
    attack_models = train_attack_classifiers(shadow_results, 10, device)

    mem_scores = attack_scores_fn(attack_models, target["mem_probs"],
                                  target["mem_labels"], device)
    non_scores = attack_scores_fn(attack_models, target["non_probs"],
                                  target["non_labels"], device)

    y_true = np.concatenate([np.ones_like(mem_scores),
                             np.zeros_like(non_scores)])
    y_score = np.concatenate([mem_scores, non_scores])

    pred = (y_score >= 0.5).astype(int)
    attack_acc = float((pred == y_true).mean())
    auc = float(roc_auc_score(y_true, y_score))
    tpr1 = tpr_at_fpr(y_true, y_score, 0.01)
    tpr01 = tpr_at_fpr(y_true, y_score, 0.001)

    wall = time.time() - t0
    result = {
        "exp_id": exp_id,
        "name": name,
        "config": {
            "train_size": train_size,
            "num_shadows": num_shadows,
            "epochs": epochs,
            "weight_decay": weight_decay,
            "seed": seed,
        },
        "wall_seconds": round(wall, 1),
        "target": {
            "train_acc": round(target["train_acc"], 4),
            "test_acc": round(target["test_acc"], 4),
            "gen_gap": round(target["train_acc"] - target["test_acc"], 4),
        },
        "shadow_avg": {
            "train_acc": round(np.mean([s["train_acc"] for s in shadow_results]), 4),
            "test_acc": round(np.mean([s["test_acc"] for s in shadow_results]), 4),
        },
        "attack": {
            "accuracy": round(attack_acc, 4),
            "auc": round(auc, 4),
            "tpr_at_fpr_1pct": round(tpr1, 4),
            "tpr_at_fpr_0.1pct": round(tpr01, 4),
        },
    }

    print(f"\n  RESULT: acc={attack_acc:.4f}  AUC={auc:.4f}  "
          f"TPR@1%={tpr1:.4f}  TPR@0.1%={tpr01:.4f}  "
          f"gap={target['train_acc']-target['test_acc']:.4f}  "
          f"wall={wall:.0f}s")
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--out", default="data/experiment_results.json")
    args = parser.parse_args()

    device = torch.device("cpu")
    print("[loading CIFAR-10...]")
    train_ds = build_cifar10(args.data)

    # Define experiment configurations
    experiments = [
        # Exp 0: BASELINE (same as original run_mia.py)
        {"exp_id": 0, "name": "Baseline (original)",
         "train_size": 2500, "num_shadows": 4, "epochs": 15,
         "weight_decay": 0.0, "seed": 42},

        # Exp 1: Increase training size to 5000
        {"exp_id": 1, "name": "Larger train set (5k)",
         "train_size": 5000, "num_shadows": 4, "epochs": 15,
         "weight_decay": 0.0, "seed": 42},

        # Exp 2: More shadow models (8 instead of 4)
        {"exp_id": 2, "name": "More shadows (8)",
         "train_size": 2500, "num_shadows": 8, "epochs": 15,
         "weight_decay": 0.0, "seed": 42},

        # Exp 3: Regularization (weight decay=1e-4)
        {"exp_id": 3, "name": "With regularization (wd=1e-4)",
         "train_size": 2500, "num_shadows": 4, "epochs": 15,
         "weight_decay": 1e-4, "seed": 42},

        # Exp 4: Stronger regularization (weight decay=1e-3)
        {"exp_id": 4, "name": "Stronger regularization (wd=1e-3)",
         "train_size": 2500, "num_shadows": 4, "epochs": 15,
         "weight_decay": 1e-3, "seed": 42},

        # Exp 5: Combined — larger data + more shadows + regularization
        {"exp_id": 5, "name": "Combined (5k + 8 shadows + wd)",
         "train_size": 2500, "num_shadows": 8, "epochs": 15,
         "weight_decay": 1e-4, "seed": 42},
    ]

    all_results = []
    total_t0 = time.time()

    for exp in experiments:
        result = run_one_experiment(
            exp_id=exp["exp_id"],
            name=exp["name"],
            train_ds=train_ds,
            train_size=exp["train_size"],
            num_shadows=exp["num_shadows"],
            epochs=exp["epochs"],
            lr=1e-3,
            batch_size=128,
            weight_decay=exp["weight_decay"],
            seed=exp["seed"],
            device=device,
        )
        all_results.append(result)

    total_wall = time.time() - total_t0

    output = {
        "total_wall_seconds": round(total_wall, 1),
        "num_experiments": len(all_results),
        "experiments": all_results,
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*60}")
    print(f"ALL EXPERIMENTS COMPLETE  ({total_wall:.0f}s total)")
    print(f"Saved to {args.out}")
    print(f"{'='*60}")

    # Summary table
    print(f"\n{'Exp':<4} {'Name':<32} {'Gap':<7} {'Acc':<7} "
          f"{'AUC':<7} {'TPR@1%':<8} {'Wall':<6}")
    print("-" * 72)
    for r in all_results:
        print(f"{r['exp_id']:<4} {r['name']:<32} "
              f"{r['target']['gen_gap']:<7.4f} "
              f"{r['attack']['accuracy']:<7.4f} "
              f"{r['attack']['auc']:<7.4f} "
              f"{r['attack']['tpr_at_fpr_1pct']:<8.4f} "
              f"{r['wall_seconds']:<6.0f}")


def build_cifar10(data_dir):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])
    return datasets.CIFAR10(data_dir, train=True, download=True,
                            transform=transform)


if __name__ == "__main__":
    main()
