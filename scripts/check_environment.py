import platform
import sys

import numpy as np
import torch


def main():
    print("=" * 70)
    print("SCALE-Former Environment Check")
    print("=" * 70)

    print(f"Python version : {sys.version.split()[0]}")
    print(f"Platform       : {platform.platform()}")
    print(f"NumPy version  : {np.__version__}")
    print(f"PyTorch version: {torch.__version__}")

    cuda_available = torch.cuda.is_available()

    print(f"CUDA available : {cuda_available}")
    print(f"CUDA runtime   : {torch.version.cuda}")

    if not cuda_available:
        raise RuntimeError(
            "CUDA is not available. SCALE-Former should not be trained "
            "until the CUDA environment is working."
        )

    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)

    print(f"GPU            : {torch.cuda.get_device_name(device)}")
    print(
        f"GPU memory     : "
        f"{properties.total_memory / (1024 ** 3):.2f} GB"
    )

    # Real GPU computation test.
    x = torch.randn(1024, 1024, device=device)
    y = torch.randn(1024, 1024, device=device)

    z = x @ y

    print(f"GPU test tensor: {tuple(z.shape)}")
    print(f"GPU test device: {z.device}")

    assert z.is_cuda
    assert z.shape == (1024, 1024)

    del x, y, z

    torch.cuda.empty_cache()

    print("GPU computation: PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()