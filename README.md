# MIA Environment — Reproducible Docker Setup

Reproducible environment for the Independent Study project on
**Membership Inference Attacks (MIA) against machine learning models**.

It provides a single Docker image with PyTorch, torchvision, scikit-learn,
Opacus (differential privacy), and ML Privacy Meter, orchestrated by
`docker-compose`.

## Contents

```
mia-env/
├── Dockerfile              # PyTorch + MIA tooling image
├── docker-compose.yml      # One-service stack (JupyterLab on :8888)
├── requirements.txt        # Python dependencies
├── scripts/
│   ├── smoke_test.py       # Verifies all libraries import correctly
│   └── train_cifar10.py    # Trains a small CIFAR-10 target model
├── notebooks/              # Empty, mounted into the container
└── data/                   # Datasets + saved models (mounted)
```

## Requirements

- Docker Engine 24+ and Docker Compose v2
- ~6 GB free disk space for the image
- Optional: an NVIDIA GPU + NVIDIA Container Toolkit for faster training
  (uncomment the `deploy:` block in `docker-compose.yml`)

## Quick Start

```bash
# 1. Build the image
docker compose build

# 2. Start JupyterLab in the background
docker compose up -d

# 3. Open http://localhost:8888 in a browser
#    (no token — only bind to localhost on shared machines)

# 4. Run the smoke test to verify everything is installed
docker compose exec mia python scripts/smoke_test.py

# 5. Train the target CIFAR-10 model (downloads CIFAR-10 on first run)
docker compose exec mia python scripts/train_cifar10.py \
    --epochs 20 --train-size 10000 --out /workspace/data/target.pt

# 6. Stop the stack
docker compose down
```

## What the smoke test checks

`scripts/smoke_test.py` imports the core libraries and prints their
versions, runs a small matrix multiplication on PyTorch, and verifies that
`opacus` and `privacy_meter` are available. It is the first thing to run
after building the image.

## Notes on reproducibility

- The base image tag (`pytorch/pytorch:2.2.0-...`) is pinned to a fixed
  version so rebuilds give the same library versions.
- `train_cifar10.py` fixes a random seed (default `42`) and saves both the
  model weights and the **member indices** of the training subset. These
  indices are required later to label the "in training" vs. "not in
  training" examples for the membership inference attack.
- The `data/` directory is mounted from the host, so CIFAR-10 is only
  downloaded once across rebuilds.

## Roadmap

1. **Week 5–6 (current):** environment build + smoke test. ✅
2. **Week 7–9:** reproduce the baseline shadow-model attack of Shokri et
   al. (2017) using ML Privacy Meter on the saved CIFAR-10 target model.
3. **Week 10–11:** small preliminary experiment varying training set size
   and regularization strength, and comparing attack success rate (AUC and
   TPR at low FPR, following Carlini et al., 2022) against the baseline.
4. **Week 12:** write the final short report.

## License

Research code for an Independent Study course. Not for production use.
