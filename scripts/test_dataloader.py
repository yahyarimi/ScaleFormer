import torch
from torch.utils.data import DataLoader

from src.data.dichasus import (
    DichasusDataset,
)


DATA_ROOT = "data/dichasus_005x"

NORMALIZATION_FILE = (
    "results/csv/"
    "training_normalization.json"
)


def main():
    print("=" * 72)
    print("DICHASUS PYTORCH DATALOADER TEST")
    print("=" * 72)

    dataset = DichasusDataset(
        data_root=DATA_ROOT,
        split="train",
        normalization_path=(
            NORMALIZATION_FILE
        ),
        density=0.10,
        pattern="comb",
        seed=42,
        shuffle_buffer=128,
    )

    print(
        f"Expected training samples : "
        f"{len(dataset):,}"
    )

    assert len(dataset) == 15940

    loader = DataLoader(
        dataset,
        batch_size=2,
        num_workers=0,
        pin_memory=True,
    )

    batch = next(
        iter(loader)
    )

    observed = batch[
        "observed"
    ]

    target = batch[
        "target"
    ]

    mask = batch[
        "mask"
    ]

    position = batch[
        "position"
    ]

    snr = batch[
        "snr"
    ]

    print("\n--- Batch shapes ---")

    print(
        f"Observed : "
        f"{tuple(observed.shape)}"
    )

    print(
        f"Target   : "
        f"{tuple(target.shape)}"
    )

    print(
        f"Mask     : "
        f"{tuple(mask.shape)}"
    )

    print(
        f"Position : "
        f"{tuple(position.shape)}"
    )

    print(
        f"SNR      : "
        f"{tuple(snr.shape)}"
    )

    assert observed.shape == (
        2,
        2,
        32,
        1024,
    )

    assert target.shape == (
        2,
        2,
        32,
        1024,
    )

    assert mask.shape == (
        2,
        1,
        32,
        1024,
    )

    assert position.shape == (
        2,
        2,
    )

    assert snr.shape == (
        2,
        32,
    )

    print("\n--- Observation integrity ---")

    mask_2channel = mask.expand(
        -1,
        2,
        -1,
        -1,
    )

    missing = (
        mask_2channel == 0
    )

    measured = (
        mask_2channel == 1
    )

    max_missing_value = (
        observed[missing]
        .abs()
        .max()
        .item()
    )

    max_measured_error = (
        (
            observed[measured]
            - target[measured]
        )
        .abs()
        .max()
        .item()
    )

    print(
        f"Max missing-position value : "
        f"{max_missing_value:.8e}"
    )

    print(
        f"Max measured-position error: "
        f"{max_measured_error:.8e}"
    )

    assert max_missing_value == 0.0
    assert max_measured_error == 0.0

    observed_subcarriers = int(
        mask[
            0,
            0,
            0,
        ]
        .sum()
        .item()
    )

    actual_density = (
        observed_subcarriers
        / 1024
    )

    print(
        f"Observed subcarriers       : "
        f"{observed_subcarriers}"
    )

    print(
        f"Actual density             : "
        f"{actual_density:.6f}"
    )

    assert observed_subcarriers == 102

    print("\n--- Normalized target statistics ---")

    print(
        f"Batch mean : "
        f"{target.mean().item():.6f}"
    )

    print(
        f"Batch std  : "
        f"{target.std().item():.6f}"
    )

    print("\n--- GPU transfer test ---")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA unavailable."
        )

    device = torch.device(
        "cuda"
    )

    observed_gpu = observed.to(
        device,
        non_blocking=True,
    )

    target_gpu = target.to(
        device,
        non_blocking=True,
    )

    mask_gpu = mask.to(
        device,
        non_blocking=True,
    )

    print(
        f"Observed GPU device : "
        f"{observed_gpu.device}"
    )

    print(
        f"Target GPU device   : "
        f"{target_gpu.device}"
    )

    print(
        f"Mask GPU device     : "
        f"{mask_gpu.device}"
    )

    allocated_mb = (
        torch.cuda.memory_allocated()
        / (1024 ** 2)
    )

    print(
        f"GPU memory allocated: "
        f"{allocated_mb:.2f} MiB"
    )

    del (
        observed_gpu,
        target_gpu,
        mask_gpu,
    )

    torch.cuda.empty_cache()

    print(
        "\nPYTORCH DATALOADER TEST: PASSED"
    )

    print("=" * 72)


if __name__ == "__main__":
    main()