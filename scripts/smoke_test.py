"""
Quick smoke test to verify the environment is correctly installed.

Checks:
    1. Python, NumPy, PyTorch import and report versions.
    2. Torch can build a small tensor operation (CPU or GPU).
    3. torchvision, scikit-learn, opacus, and privacy_meter import.

Run inside the container:
    python scripts/smoke_test.py
"""

import sys


def main() -> None:
    print(f"[python] {sys.version.split()[0]}")

    import numpy as np
    print(f"[numpy]  {np.__version__}")

    import torch
    print(f"[torch]  {torch.__version__}  cuda={torch.cuda.is_available()}")

    x = torch.randn(1024, 1024)
    y = x @ x.T
    print(f"[torch]  matmul ok, result shape: {tuple(y.shape)}")

    import torchvision
    print(f"[torchvision] {torchvision.__version__}")

    import sklearn
    print(f"[sklearn] {sklearn.__version__}")

    try:
        import opacus
        print(f"[opacus]  {opacus.__version__}")
    except Exception as exc:  # noqa: BLE001
        print(f"[opacus]  NOT AVAILABLE: {exc}")

    try:
        import privacy_meter
        version = getattr(privacy_meter, "__version__", "unknown")
        print(f"[privacy_meter] {version}")
    except Exception as exc:  # noqa: BLE001
        print(f"[privacy_meter] NOT AVAILABLE: {exc}")

    print("[ok] smoke test finished")


if __name__ == "__main__":
    main()
