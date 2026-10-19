from __future__ import annotations

import csv
import json
import statistics
import time
from pathlib import Path

import torch
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
from src.models.scaleformer import ScaleFormer


DATA_ROOT = "data/dichasus_005x"

NORMALIZATION_FILE = (
    "results/csv/"
    "training_normalization.json"
)

RESULT_ROOT = Path(
    "results/csv"
)

SEEDS = [
    7,
    42,
    123,
]

CHECKPOINTS = {
    seed: Path(
        "checkpoints"
    )
    / (
        f"scaleformer_seed{seed}"
        f"_bs4_best.pt"
    )
    for seed in SEEDS
}

# Evaluation masks are deliberately fixed across
# all trained model seeds.
EVAL_MASK_SEED = 42

BATCH_SIZE = 8

EXPECTED_VALIDATION_SAMPLES = 5220


CONDITIONS = [
    {
        "pattern": "random",
        "density": 0.01,
        "label": "random_1pct",
    },
    {
        "pattern": "random",
        "density": 0.02,
        "label": "random_2pct",
    },
    {
        "pattern": "random",
        "density": 0.05,
        "label": "random_5pct",
    },
    {
        "pattern": "random",
        "density": 0.10,
        "label": "random_10pct",
    },
    {
        "pattern": "random",
        "density": 0.20,
        "label": "random_20pct",
    },
    {
        "pattern": "comb",
        "density": 0.05,
        "label": "comb_5pct",
    },
    {
        "pattern": "block",
        "density": 0.05,
        "label": "block_5pct",
    },
]


PER_SEED_OUTPUT = (
    RESULT_ROOT
    / "scaleformer_robustness_per_seed.csv"
)

SUMMARY_OUTPUT = (
    RESULT_ROOT
    / "scaleformer_robustness_summary.csv"
)

JSON_OUTPUT = (
    RESULT_ROOT
    / "scaleformer_robustness_summary.json"
)


def load_model(
    seed: int,
    device: torch.device,
):
    checkpoint_path = (
        CHECKPOINTS[
            seed
        ]
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            checkpoint_path
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    architecture = checkpoint[
        "architecture"
    ]

    model = ScaleFormer(
        dim=architecture[
            "dim"
        ],
        num_heads=architecture[
            "heads"
        ],
        num_latents=architecture[
            "latents"
        ],
        num_local_blocks=architecture[
            "local_blocks"
        ],
        num_global_blocks=architecture[
            "global_blocks"
        ],
        num_decoder_blocks=architecture[
            "decoder_blocks"
        ],
        local_kernel_size=7,
        decoder_chunk_size=architecture[
            "decoder_chunk_size"
        ],
        dropout=0.0,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model = model.to(
        device
    )

    model.eval()

    return (
        model,
        checkpoint,
    )


def evaluate_condition(
    condition: dict,
    models: dict,
    checkpoints: dict,
    normalizer: CSINormalizer,
    device: torch.device,
):

    pattern = condition[
        "pattern"
    ]

    density = float(
        condition[
            "density"
        ]
    )

    label = condition[
        "label"
    ]

    print(
        "\n"
        + "=" * 80
    )

    print(
        f"CONDITION: {label}"
    )

    print(
        "=" * 80
    )

    dataset = DichasusDataset(
        data_root=DATA_ROOT,
        split="val",
        normalization_path=(
            NORMALIZATION_FILE
        ),
        density=density,
        pattern=pattern,
        seed=EVAL_MASK_SEED,
        shuffle_buffer=0,
    )

    if (
        len(dataset)
        != EXPECTED_VALIDATION_SAMPLES
    ):
        raise RuntimeError(
            "Unexpected validation "
            "sample count."
        )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=0,
        pin_memory=True,
    )

    interpolation_full = (
        AggregateNMSE(
            missing_only=False
        )
    )

    interpolation_missing = (
        AggregateNMSE(
            missing_only=True
        )
    )

    model_metrics = {}

    for seed in SEEDS:

        model_metrics[
            seed
        ] = {
            "full": AggregateNMSE(
                missing_only=False
            ),
            "missing": AggregateNMSE(
                missing_only=True
            ),
            "maximum_observed_error": (
                0.0
            ),
        }

    num_samples = 0

    observed_subcarriers = None

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()

    with torch.no_grad():

        for batch_index, batch in enumerate(
            loader,
            start=1,
        ):

            observed = batch[
                "observed"
            ].to(
                device,
                non_blocking=True,
            )

            target = batch[
                "target"
            ].to(
                device,
                non_blocking=True,
            )

            mask = batch[
                "mask"
            ].to(
                device,
                non_blocking=True,
            )

            current_observed = int(
                mask[
                    0,
                    0,
                    0,
                ]
                .sum()
                .item()
            )

            if (
                observed_subcarriers
                is None
            ):
                observed_subcarriers = (
                    current_observed
                )

            if (
                current_observed
                != observed_subcarriers
            ):
                raise RuntimeError(
                    "Observation count changed "
                    "within condition."
                )

            interpolated = (
                frequency_linear_interpolate(
                    observed,
                    mask,
                )
            )

            # Move reference tensors to CPU once
            # per batch for all metrics.
            interpolation_cpu = (
                interpolated
                .detach()
                .float()
                .cpu()
            )

            target_cpu = (
                target
                .detach()
                .float()
                .cpu()
            )

            mask_cpu = (
                mask
                .detach()
                .float()
                .cpu()
            )

            interpolation_physical = (
                normalizer.denormalize_tensor(
                    interpolation_cpu
                )
            )

            target_physical = (
                normalizer.denormalize_tensor(
                    target_cpu
                )
            )

            interpolation_full.update(
                prediction=(
                    interpolation_physical
                ),
                target=(
                    target_physical
                ),
                mask=mask_cpu,
            )

            interpolation_missing.update(
                prediction=(
                    interpolation_physical
                ),
                target=(
                    target_physical
                ),
                mask=mask_cpu,
            )

            expanded_mask = (
                mask.expand(
                    -1,
                    2,
                    -1,
                    -1,
                ).bool()
            )

            # Evaluate all independently trained
            # checkpoints on exactly the same batch.
            for seed in SEEDS:

                model = models[
                    seed
                ]

                with torch.amp.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                    enabled=True,
                ):
                    prediction = model(
                        interpolated,
                        mask,
                    )

                observed_error = (
                    prediction[
                        expanded_mask
                    ]
                    - target[
                        expanded_mask
                    ]
                ).abs().max().item()

                model_metrics[
                    seed
                ][
                    "maximum_observed_error"
                ] = max(
                    model_metrics[
                        seed
                    ][
                        "maximum_observed_error"
                    ],
                    observed_error,
                )

                prediction_cpu = (
                    prediction
                    .detach()
                    .float()
                    .cpu()
                )

                prediction_physical = (
                    normalizer
                    .denormalize_tensor(
                        prediction_cpu
                    )
                )

                model_metrics[
                    seed
                ][
                    "full"
                ].update(
                    prediction=(
                        prediction_physical
                    ),
                    target=(
                        target_physical
                    ),
                    mask=mask_cpu,
                )

                model_metrics[
                    seed
                ][
                    "missing"
                ].update(
                    prediction=(
                        prediction_physical
                    ),
                    target=(
                        target_physical
                    ),
                    mask=mask_cpu,
                )

            num_samples += (
                target.shape[0]
            )

            if (
                batch_index % 100
                == 0
            ):
                print(
                    f"Processed "
                    f"{num_samples:,}/"
                    f"{len(dataset):,}"
                )

    elapsed = (
        time.perf_counter()
        - start
    )

    peak_memory_mb = (
        torch.cuda.max_memory_allocated()
        / (1024 ** 2)
    )

    interpolation_full_db = (
        interpolation_full.compute_db()
    )

    interpolation_missing_db = (
        interpolation_missing.compute_db()
    )

    actual_density = (
        observed_subcarriers
        / NUM_SUBCARRIERS
    )

    per_seed_rows = []

    for seed in SEEDS:

        full_db = (
            model_metrics[
                seed
            ][
                "full"
            ].compute_db()
        )

        missing_db = (
            model_metrics[
                seed
            ][
                "missing"
            ].compute_db()
        )

        max_observed_error = (
            model_metrics[
                seed
            ][
                "maximum_observed_error"
            ]
        )

        if max_observed_error > 1e-6:
            raise RuntimeError(
                f"Observed-value preservation "
                f"failed for seed {seed}."
            )

        # The 5%-random condition should reproduce
        # the original checkpoint validation result.
        if (
            pattern == "random"
            and abs(
                density - 0.05
            ) < 1e-12
        ):

            saved_full = float(
                checkpoints[
                    seed
                ][
                    "val_full_nmse_db"
                ]
            )

            saved_missing = float(
                checkpoints[
                    seed
                ][
                    "val_missing_nmse_db"
                ]
            )

            if abs(
                full_db
                - saved_full
            ) > 0.02:
                raise RuntimeError(
                    f"Seed {seed} full NMSE "
                    "did not reproduce."
                )

            if abs(
                missing_db
                - saved_missing
            ) > 0.02:
                raise RuntimeError(
                    f"Seed {seed} missing NMSE "
                    "did not reproduce."
                )

        row = {
            "condition": label,
            "pattern": pattern,
            "requested_density": (
                density
            ),
            "observed_subcarriers": (
                observed_subcarriers
            ),
            "actual_density": (
                actual_density
            ),
            "seed": seed,
            "num_samples": (
                num_samples
            ),
            "interpolation_full_nmse_db": (
                interpolation_full_db
            ),
            "interpolation_missing_nmse_db": (
                interpolation_missing_db
            ),
            "scaleformer_full_nmse_db": (
                full_db
            ),
            "scaleformer_missing_nmse_db": (
                missing_db
            ),
            "gain_vs_interpolation_full_db": (
                interpolation_full_db
                - full_db
            ),
            "gain_vs_interpolation_missing_db": (
                interpolation_missing_db
                - missing_db
            ),
            "maximum_observed_error": (
                max_observed_error
            ),
        }

        per_seed_rows.append(
            row
        )

        print(
            f"Seed {seed:>3} | "
            f"full "
            f"{full_db:.4f} dB | "
            f"missing "
            f"{missing_db:.4f} dB"
        )

    full_values = [
        row[
            "scaleformer_full_nmse_db"
        ]
        for row in per_seed_rows
    ]

    missing_values = [
        row[
            "scaleformer_missing_nmse_db"
        ]
        for row in per_seed_rows
    ]

    full_mean = statistics.mean(
        full_values
    )

    full_std = statistics.stdev(
        full_values
    )

    missing_mean = statistics.mean(
        missing_values
    )

    missing_std = statistics.stdev(
        missing_values
    )

    distribution = (
        "in_distribution"
        if (
            pattern == "random"
            and abs(
                density - 0.05
            ) < 1e-12
        )
        else "robustness_shift"
    )

    summary_row = {
        "condition": label,
        "distribution": distribution,
        "pattern": pattern,
        "requested_density": density,
        "observed_subcarriers": (
            observed_subcarriers
        ),
        "actual_density": (
            actual_density
        ),
        "num_samples": num_samples,
        "interpolation_full_nmse_db": (
            interpolation_full_db
        ),
        "interpolation_missing_nmse_db": (
            interpolation_missing_db
        ),
        "scaleformer_full_nmse_db_mean": (
            full_mean
        ),
        "scaleformer_full_nmse_db_std": (
            full_std
        ),
        "scaleformer_missing_nmse_db_mean": (
            missing_mean
        ),
        "scaleformer_missing_nmse_db_std": (
            missing_std
        ),
        "mean_gain_vs_interpolation_full_db": (
            interpolation_full_db
            - full_mean
        ),
        "mean_gain_vs_interpolation_missing_db": (
            interpolation_missing_db
            - missing_mean
        ),
        "evaluation_seconds": (
            elapsed
        ),
        "peak_gpu_memory_mb": (
            peak_memory_mb
        ),
    }

    print(
        f"\nInterpolation missing : "
        f"{interpolation_missing_db:.4f} dB"
    )

    print(
        f"SCALE-Former missing  : "
        f"{missing_mean:.4f} "
        f"± {missing_std:.4f} dB"
    )

    print(
        f"Mean gain             : "
        f"{interpolation_missing_db - missing_mean:.4f} dB"
    )

    print(
        f"Condition time        : "
        f"{elapsed:.2f} s"
    )

    return (
        per_seed_rows,
        summary_row,
    )


def save_csv(
    rows: list[dict],
    path: Path,
):

    if not rows:
        return

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            rows
        )


def main():

    print("=" * 80)
    print(
        "FROZEN SCALE-FORMER "
        "ROBUSTNESS EVALUATION"
    )
    print("=" * 80)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required."
        )

    device = torch.device(
        "cuda"
    )

    print(
        f"GPU: "
        f"{torch.cuda.get_device_name(0)}"
    )

    print(
        f"Validation mask seed: "
        f"{EVAL_MASK_SEED}"
    )

    print(
        f"Frozen model seeds: "
        f"{SEEDS}"
    )

    models = {}
    checkpoints = {}

    for seed in SEEDS:

        model, checkpoint = (
            load_model(
                seed=seed,
                device=device,
            )
        )

        models[
            seed
        ] = model

        checkpoints[
            seed
        ] = checkpoint

        print(
            f"Loaded seed {seed}: "
            f"epoch "
            f"{checkpoint['epoch']}"
        )

    normalizer = CSINormalizer(
        NORMALIZATION_FILE
    )

    all_per_seed_rows = []
    all_summary_rows = []

    overall_start = (
        time.perf_counter()
    )

    for condition in CONDITIONS:

        (
            per_seed_rows,
            summary_row,
        ) = evaluate_condition(
            condition=condition,
            models=models,
            checkpoints=checkpoints,
            normalizer=normalizer,
            device=device,
        )

        all_per_seed_rows.extend(
            per_seed_rows
        )

        all_summary_rows.append(
            summary_row
        )

        # Save incrementally so completed conditions
        # survive interruption.
        save_csv(
            all_per_seed_rows,
            PER_SEED_OUTPUT,
        )

        save_csv(
            all_summary_rows,
            SUMMARY_OUTPUT,
        )

        with JSON_OUTPUT.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                all_summary_rows,
                file,
                indent=4,
            )

    total_seconds = (
        time.perf_counter()
        - overall_start
    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "ROBUSTNESS SUMMARY"
    )

    print(
        "=" * 100
    )

    print(
        f"{'Condition':<16}"
        f"{'Pilots':>8}"
        f"{'Interp miss':>16}"
        f"{'SCALE-Former miss':>25}"
        f"{'Gain':>12}"
    )

    print(
        "-" * 100
    )

    for row in all_summary_rows:

        mean = row[
            "scaleformer_missing_nmse_db_mean"
        ]

        std = row[
            "scaleformer_missing_nmse_db_std"
        ]

        print(
            f"{row['condition']:<16}"
            f"{row['observed_subcarriers']:>8d}"
            f"{row['interpolation_missing_nmse_db']:>13.4f} dB"
            f"{mean:>13.4f} ± "
            f"{std:<7.4f} dB"
            f"{row['mean_gain_vs_interpolation_missing_db']:>9.4f} dB"
        )

    print(
        f"\nTotal robustness "
        f"evaluation time: "
        f"{total_seconds / 60:.2f} min"
    )

    print(
        f"\nPer-seed results:"
        f"\n{PER_SEED_OUTPUT}"
    )

    print(
        f"\nSummary:"
        f"\n{SUMMARY_OUTPUT}"
    )

    print(
        f"\nJSON summary:"
        f"\n{JSON_OUTPUT}"
    )

    print(
        "\nFROZEN SCALE-FORMER "
        "ROBUSTNESS EVALUATION: PASSED"
    )

    print("=" * 100)


if __name__ == "__main__":
    main()