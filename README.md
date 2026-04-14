# MIA Environment — Membership Inference Attack Research

Reproducible environment for an Independent Study on
**Membership Inference Attacks (MIA) against machine learning models**.

This repository contains everything needed to reproduce the Shokri et al.
(2017) shadow-model attack on CIFAR-10, run a systematic preliminary
experiment varying three factors, and review 14 peer-reviewed papers on MIA.

## Contents

```
mia-env/
├── Dockerfile                  # PyTorch 2.2 + CUDA base image
├── docker-compose.yml          # JupyterLab on port 8888
├── requirements.txt            # Python dependencies
├── README.md
├── scripts/
│   ├── smoke_test.py           # Verifies all library imports
│   ├── train_cifar10.py        # Standalone CIFAR-10 target model training
│   ├── run_mia.py              # Full MIA pipeline — Shokri 2017 reproduction
│   └── run_experiments.py      # Preliminary experiment suite (6 configs)
├── notebooks/                  # Mounted into the container for JupyterLab
├── data/
│   ├── results_mia.json        # Baseline experiment results
│   ├── experiment_results.json # Preliminary experiment results (6 configs)
│   └── cifar-10-batches-py/    # Downloaded CIFAR-10 dataset
└── docs/
    └── literature_review.md    # Verified 14-paper literature review (~4,700 words)
```

## Results Summary

### Baseline Reproduction (Shokri et al., 2017)

| Metric | Value |
|--------|-------|
| Target train accuracy | 92.68% |
| Target test accuracy | 49.92% |
| Generalization gap | 42.76% |
| Attack accuracy | 71.20% |
| Attack AUC | 0.7686 |
| TPR at 1% FPR | 5.80% |
| TPR at 0.1% FPR | 0.52% |

### Preliminary Experiments (6 configurations)

| Exp | Config | Gen. Gap | AUC | TPR@1%FPR |
|-----|--------|----------|-----|-----------|
| 0 | Baseline (2.5k, 4 shadows, no reg) | 42.76% | 0.769 | 5.80% |
| 1 | Larger train set (5k) | 37.46% | 0.740 | 3.50% |
| 2 | More shadows (8) | 42.76% | 0.780 | 4.24% |
| 3 | Weight decay 1e-4 | 42.40% | 0.768 | 4.88% |
| 4 | Weight decay 1e-3 | 40.64% | 0.755 | 6.12% |
| 5 | Combined (8 shadows + wd=1e-4) | 42.40% | 0.778 | 4.00% |

**Key finding:** Regularization reduces aggregate attack metrics (AUC) but
does *not* reduce tail-risk leakage (TPR at low FPR). Exp 4 has the lowest
AUC but the *highest* TPR@1%FPR, confirming Carlini et al.'s (2022)
observation that aggregate metrics overstate defense effectiveness.

## Requirements

- Docker Engine 24+ and Docker Compose v2 (or Python 3.10+ for local runs)
- ~6 GB free disk space for the Docker image
- Optional: NVIDIA GPU + Container Toolkit (uncomment `deploy:` block in
  `docker-compose.yml`)

## Quick Start

### Option 1: Docker (recommended)

```bash
# Build and start
docker compose build
docker compose up -d

# Open JupyterLab at http://localhost:8888

# Run smoke test
docker compose exec mia python scripts/smoke_test.py

# Run baseline MIA reproduction (~3 min on CPU)
docker compose exec mia python scripts/run_mia.py --out data/results_mia.json

# Run all 6 experiments (~24 min on CPU)
docker compose exec mia python scripts/run_experiments.py --out data/experiment_results.json

# Stop
docker compose down
```

### Option 2: Local Python

```bash
pip install -r requirements.txt
python scripts/smoke_test.py
python scripts/run_mia.py --out data/results_mia.json
python scripts/run_experiments.py --out data/experiment_results.json
```

## Scripts

### `run_mia.py` — Baseline MIA Reproduction

Implements the Shokri et al. (2017) shadow-model attack end-to-end:

1. Trains a SmallCNN target model on a random CIFAR-10 subset
2. Trains N shadow models on disjoint subsets
3. Collects softmax confidence vectors (member vs. non-member)
4. Trains per-class attack MLPs on shadow data
5. Evaluates attack accuracy, AUC, and TPR at low FPR

```bash
python scripts/run_mia.py \
    --train-size 2500 \
    --num-shadows 4 \
    --epochs 15 \
    --seed 42 \
    --out data/results_mia.json
```

### `run_experiments.py` — Preliminary Experiment Suite

Runs 6 configurations varying training set size, number of shadow models,
and weight decay regularization. Each experiment is self-contained with
its own seed reset for reproducibility.

```bash
python scripts/run_experiments.py --out data/experiment_results.json
```

### `smoke_test.py` — Environment Verification

Imports all core libraries (PyTorch, torchvision, scikit-learn, Opacus,
ML Privacy Meter), runs a test matrix multiplication, and prints versions.

### `train_cifar10.py` — Standalone Target Training

Trains and saves a CIFAR-10 model with member indices for later MIA
evaluation.

## Literature Review

A comprehensive, verified literature review of 14 peer-reviewed papers is
available at [`docs/literature_review.md`](docs/literature_review.md). All
citations were independently verified against arXiv/publisher abstract
pages. Topics covered:

- Foundational MIA attacks (Shokri 2017, Yeom 2018, Long 2018)
- Advanced attacks with relaxed assumptions (Salem 2019, Carlini/LiRA 2022)
- Label-only and white-box attacks (Choquette-Choo 2021, Nasr 2019)
- Defenses: DP-SGD (Abadi 2016), MemGuard (Jia 2019), Adversarial Reg. (Nasr 2018)
- LLM-era extensions (Carlini 2021, Mattern 2023)

## Notes on Reproducibility

- Base image tag (`pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime`) is
  pinned for deterministic builds.
- All scripts use `--seed 42` by default for reproducible results.
- `train_cifar10.py` saves member indices alongside model weights for
  correct member/non-member labeling.
- The `data/` directory is host-mounted so CIFAR-10 downloads persist
  across container rebuilds.

## Project Status

| Phase | Status |
|-------|--------|
| Literature review (14 papers) | Done |
| Thesis topic selection | Done |
| Docker environment | Done |
| Baseline reproduction (Shokri 2017) | Done |
| Preliminary experiments (6 configs) | Done |
| Final report | Done |

## Author

**Osamah Alsumaitti**
M.Sc. in Cybersecurity, Hamad Bin Khalifa University (HBKU)
osal89659@hbku.edu.qa

## License

Research code for an Independent Study course. Not for production use.
