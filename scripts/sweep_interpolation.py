from __future__ import annotations

import csv
import time
from pathlib import Path

from torch.utils.data import DataLoader

from src.data.dichasus import (
    CSINormalizer,
    DichasusDataset,
    NUM_SUBCARRIERS,
)
from src.metrics.nmse import AggregateNMSE
from src.models.interpolation import (
    frequency_linear_interpolate,
)


DATA_ROOT = "data/dichasus_005x"

NORMALIZATION_FILE = (
    "results/csv/training_normalization.json"
)

OUTPUT_FILE = Path(
    "results/csv/"
    "validation_interpolation_sweep.csv"
)

DENSITIES = [
    0.01,
    0.02,
    0.05,
    0.10,
    0.20,
]

PATTERNS = [
    "comb",
    "random",
    "block",
]

BATCH_SIZE = 16
SEED = 42


FIELDNAMES = [
    "model",
    "split",
    "run",
    "pattern",
    "requested_density",
    "observed_subcarriers",
    "actual_density",
    "num_samples",
    "full_nmse_linear",
    "full_nmse_db",
    "missing_nmse_linear",
    "missing_nmse_db",
    "total_time_seconds",
    "milliseconds_per_sample",
]


def evaluate_condition(
    density: float,
    pattern: str,
    normalizer: CSINormalizer,
) -> dict:

    dataset = DichasusDataset(
        data_root=DATA_ROOT,
        split="val",
        normalization_path=NORMALIZATION_FILE,
        density=density,
        pattern=pattern,
        seed=SEED,
        shuffle_buffer=0,
    )

    if len(dataset) != 5220:
        raise RuntimeError(
            f"Expected 5220 validation samples, "
            f"found {len(dataset)}."
        )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=0,
        pin_memory=False,
    )

    full_metric = AggregateNMSE(
        missing_only=False
    )

    missing_metric = AggregateNMSE(
        missing_only=True
    )

    num_samples = 0

    observed_subcarriers = max(
        1,
        min(
            NUM_SUBCARRIERS,
            int(
                round(
                    density
                    * NUM_SUBCARRIERS
                )
            ),
        ),
    )

    actual_density = (
        observed_subcarriers
        / NUM_SUBCARRIERS
    )

    start = time.perf_counter()

    for batch in loader:

        observed = batch["observed"]
        target = batch["target"]
        mask = batch["mask"]

        prediction = (
            frequency_linear_interpolate(
                observed,
                mask,
            )
        )

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

        num_samples += target.shape[0]

    elapsed = (
        time.perf_counter()
        - start
    )

    return {
        "model": (
            "linear_frequency_interpolation"
        ),
        "split": "validation",
        "run": "0056",
        "pattern": pattern,
        "requested_density": density,
        "observed_subcarriers": (
            observed_subcarriers
        ),
        "actual_density": actual_density,
        "num_samples": num_samples,
        "full_nmse_linear": (
            full_metric.compute_linear()
        ),
        "full_nmse_db": (
            full_metric.compute_db()
        ),
        "missing_nmse_linear": (
            missing_metric.compute_linear()
        ),
        "missing_nmse_db": (
            missing_metric.compute_db()
        ),
        "total_time_seconds": elapsed,
        "milliseconds_per_sample": (
            elapsed
            / num_samples
            * 1000.0
        ),
    }


def save_rows(
    rows: list[dict],
) -> None:

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
            fieldnames=FIELDNAMES,
        )

        writer.writeheader()
        writer.writerows(rows)


def main():

    print("=" * 80)
    print(
        "REAL DICHASUS INTERPOLATION "
        "DIFFICULTY SWEEP"
    )
    print("=" * 80)

    normalizer = CSINormalizer(
        NORMALIZATION_FILE
    )

    rows = []

    total_conditions = (
        len(DENSITIES)
        * len(PATTERNS)
    )

    condition_index = 0

    for pattern in PATTERNS:

        for density in DENSITIES:

            condition_index += 1

            print(
                "\n"
                + "-" * 80
            )

            print(
                f"Condition "
                f"{condition_index}/"
                f"{total_conditions}"
            )

            print(
                f"Pattern : {pattern}"
            )

            print(
                f"Density : "
                f"{density:.2%}"
            )

            row = evaluate_condition(
                density=density,
                pattern=pattern,
                normalizer=normalizer,
            )

            rows.append(
                row
            )

            # Save after every condition so results
            # survive an interruption.
            save_rows(
                rows
            )

            print(
                f"Observed subcarriers : "
                f"{row['observed_subcarriers']}"
            )

            print(
                f"Actual density       : "
                f"{row['actual_density']:.6f}"
            )

            print(
                f"Full NMSE            : "
                f"{row['full_nmse_db']:.4f} dB"
            )

            print(
                f"Missing-only NMSE    : "
                f"{row['missing_nmse_db']:.4f} dB"
            )

            print(
                f"Time                 : "
                f"{row['total_time_seconds']:.2f} s"
            )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "SUMMARY"
    )

    print(
        "=" * 80
    )

    print(
        f"{'Pattern':<10}"
        f"{'Density':>10}"
        f"{'Pilots':>10}"
        f"{'Full NMSE':>15}"
        f"{'Missing NMSE':>18}"
    )

    print(
        "-" * 63
    )

    for row in rows:

        print(
            f"{row['pattern']:<10}"
            f"{row['actual_density'] * 100:>9.2f}%"
            f"{row['observed_subcarriers']:>10d}"
            f"{row['full_nmse_db']:>12.4f} dB"
            f"{row['missing_nmse_db']:>15.4f} dB"
        )

    print(
        f"\nResults saved to:"
        f"\n{OUTPUT_FILE}"
    )

    print(
        "\nINTERPOLATION SWEEP: PASSED"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()