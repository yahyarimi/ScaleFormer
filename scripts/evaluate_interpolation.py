from __future__ import annotations

import csv
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.dichasus import (
    CSINormalizer,
    DichasusDataset,
)
from src.metrics.nmse import (
    AggregateNMSE,
)
from src.models.interpolation import (
    frequency_linear_interpolate,
)


DATA_ROOT = (
    "data/dichasus_005x"
)

NORMALIZATION_FILE = (
    "results/csv/"
    "training_normalization.json"
)

OUTPUT_FILE = Path(
    "results/csv/"
    "validation_interpolation.csv"
)

DENSITY = 0.10
PATTERN = "comb"
BATCH_SIZE = 16


def main():
    print("=" * 72)
    print("REAL DICHASUS INTERPOLATION BASELINE")
    print("=" * 72)

    dataset = DichasusDataset(
        data_root=DATA_ROOT,
        split="val",
        normalization_path=(
            NORMALIZATION_FILE
        ),
        density=DENSITY,
        pattern=PATTERN,
        seed=42,
        shuffle_buffer=0,
    )

    assert len(dataset) == 5220

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=0,
        pin_memory=False,
    )

    normalizer = CSINormalizer(
        NORMALIZATION_FILE
    )

    full_metric = AggregateNMSE(
        missing_only=False
    )

    missing_metric = AggregateNMSE(
        missing_only=True
    )

    num_samples = 0

    start = time.perf_counter()

    for batch_index, batch in enumerate(
        loader
    ):
        observed = batch[
            "observed"
        ]

        target = batch[
            "target"
        ]

        mask = batch[
            "mask"
        ]

        prediction = (
            frequency_linear_interpolate(
                observed,
                mask,
            )
        )

        # Publication metrics are computed after
        # restoring original measured CSI scale.
        prediction_physical = (
            normalizer.denormalize_tensor(
                prediction
            )
        )

        target_physical = (
            normalizer.denormalize_tensor(
                target
            )
        )

        full_metric.update(
            prediction=prediction_physical,
            target=target_physical,
            mask=mask,
        )

        missing_metric.update(
            prediction=prediction_physical,
            target=target_physical,
            mask=mask,
        )

        num_samples += (
            target.shape[0]
        )

        if (
            batch_index + 1
        ) % 50 == 0:
            print(
                f"Processed "
                f"{num_samples:,} / "
                f"{len(dataset):,} "
                f"measurements..."
            )

    elapsed = (
        time.perf_counter()
        - start
    )

    full_linear = (
        full_metric.compute_linear()
    )

    full_db = (
        full_metric.compute_db()
    )

    missing_linear = (
        missing_metric.compute_linear()
    )

    missing_db = (
        missing_metric.compute_db()
    )

    seconds_per_sample = (
        elapsed
        / num_samples
    )

    print("\n--- Validation results ---")

    print(
        f"Dataset run              : 0056"
    )

    print(
        f"Measurements evaluated   : "
        f"{num_samples:,}"
    )

    print(
        f"Requested density        : "
        f"{DENSITY:.4f}"
    )

    print(
        f"Pattern                  : "
        f"{PATTERN}"
    )

    print(
        f"Full-grid NMSE linear    : "
        f"{full_linear:.8e}"
    )

    print(
        f"Full-grid NMSE dB        : "
        f"{full_db:.4f} dB"
    )

    print(
        f"Missing-only NMSE linear : "
        f"{missing_linear:.8e}"
    )

    print(
        f"Missing-only NMSE dB     : "
        f"{missing_db:.4f} dB"
    )

    print(
        f"Evaluation time          : "
        f"{elapsed:.3f} s"
    )

    print(
        f"Time / sample            : "
        f"{seconds_per_sample * 1000:.4f} ms"
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_FILE.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "model",
                "split",
                "run",
                "pattern",
                "requested_density",
                "num_samples",
                "full_nmse_linear",
                "full_nmse_db",
                "missing_nmse_linear",
                "missing_nmse_db",
                "total_time_seconds",
                "milliseconds_per_sample",
            ],
        )

        writer.writeheader()

        writer.writerow(
            {
                "model": (
                    "linear_frequency_interpolation"
                ),
                "split": "validation",
                "run": "0056",
                "pattern": PATTERN,
                "requested_density": DENSITY,
                "num_samples": num_samples,
                "full_nmse_linear": full_linear,
                "full_nmse_db": full_db,
                "missing_nmse_linear": (
                    missing_linear
                ),
                "missing_nmse_db": (
                    missing_db
                ),
                "total_time_seconds": elapsed,
                "milliseconds_per_sample": (
                    seconds_per_sample
                    * 1000
                ),
            }
        )

    print(
        f"\nResults saved to:"
        f"\n{OUTPUT_FILE}"
    )

    print("\nINTERPOLATION EVALUATION: PASSED")
    print("=" * 72)


if __name__ == "__main__":
    main()