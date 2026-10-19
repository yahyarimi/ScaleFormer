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
    seed: (
        Path("checkpoints")
        / (
            f"scaleformer_seed{seed}"
            f"_bs4_best.pt"
        )
    )
    for seed in SEEDS
}

EVAL_MASK_SEED = 42

BATCH_SIZE = 8

EXPECTED_SAMPLES = 5220

FULL_NUM_ANTENNAS = 32
FULL_NUM_SUBCARRIERS = 1024

DENSITY = 0.05


GRID_CONDITIONS = [
    (8, 256),
    (8, 512),
    (8, 1024),

    (16, 256),
    (16, 512),
    (16, 1024),

    (32, 256),
    (32, 512),
    (32, 1024),
]


PER_SEED_OUTPUT = (
    RESULT_ROOT
    / "scaleformer_scalability_per_seed.csv"
)

SUMMARY_OUTPUT = (
    RESULT_ROOT
    / "scaleformer_scalability_summary.csv"
)

JSON_OUTPUT = (
    RESULT_ROOT
    / "scaleformer_scalability_summary.json"
)


def evenly_spaced_indices(
    total: int,
    requested: int,
) -> torch.Tensor:

    if requested <= 0:
        raise ValueError(
            "requested must be positive."
        )

    if requested > total:
        raise ValueError(
            "requested cannot exceed total."
        )

    if requested == total:
        return torch.arange(
            total,
            dtype=torch.long,
        )

    indices = torch.linspace(
        0,
        total - 1,
        steps=requested,
        dtype=torch.float64,
    )

    indices = (
        indices.round()
        .to(torch.long)
    )

    if (
        torch.unique(indices).numel()
        != requested
    ):
        raise RuntimeError(
            "Index generation produced "
            "duplicate positions."
        )

    return indices


def make_reduced_random_mask(
    sample_indices: torch.Tensor,
    num_antennas: int,
    num_subcarriers: int,
    density: float,
) -> torch.Tensor:
    """
    Create an exact-count random frequency mask.

    The same frequency positions are exposed across
    all antennas for a given CSI measurement.

    The mask for a given sample/F pair is deterministic.
    It does not depend on B, so all antenna-count
    experiments at the same F see the same frequency
    observation pattern.
    """

    batch_size = int(
        sample_indices.shape[0]
    )

    observed_count = max(
        1,
        min(
            num_subcarriers,
            int(
                round(
                    density
                    * num_subcarriers
                )
            ),
        ),
    )

    mask = torch.zeros(
        batch_size,
        1,
        num_antennas,
        num_subcarriers,
        dtype=torch.float32,
    )

    for batch_index in range(
        batch_size
    ):

        sample_index = int(
            sample_indices[
                batch_index
            ].item()
        )

        # Stable condition-specific seed.
        seed = (
            EVAL_MASK_SEED
            + sample_index * 1_000_003
            + num_subcarriers * 9_176
        )

        generator = (
            torch.Generator(
                device="cpu"
            )
        )

        generator.manual_seed(
            seed
        )

        positions = torch.randperm(
            num_subcarriers,
            generator=generator,
        )[
            :observed_count
        ]

        mask[
            batch_index,
            0,
            :,
            positions,
        ] = 1.0

    return mask


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


def evaluate_grid(
    num_antennas: int,
    num_subcarriers: int,
    models: dict,
    checkpoints: dict,
    normalizer: CSINormalizer,
    device: torch.device,
):

    label = (
        f"B{num_antennas}_"
        f"F{num_subcarriers}"
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        f"GRID CONDITION: {label}"
    )

    print(
        "=" * 80
    )

    antenna_indices = (
        evenly_spaced_indices(
            FULL_NUM_ANTENNAS,
            num_antennas,
        )
    )

    frequency_indices = (
        evenly_spaced_indices(
            FULL_NUM_SUBCARRIERS,
            num_subcarriers,
        )
    )

    print(
        f"Antennas            : "
        f"{num_antennas}/"
        f"{FULL_NUM_ANTENNAS}"
    )

    print(
        f"Frequency bins      : "
        f"{num_subcarriers}/"
        f"{FULL_NUM_SUBCARRIERS}"
    )

    print(
        f"Antenna indices     : "
        f"{antenna_indices.tolist()}"
    )

    # Use the standard validation dataset so the
    # full 32 x 1024 condition exactly preserves
    # our existing 5%-random evaluation protocol.
    dataset = DichasusDataset(
        data_root=DATA_ROOT,
        split="val",
        normalization_path=(
            NORMALIZATION_FILE
        ),
        density=DENSITY,
        pattern="random",
        seed=EVAL_MASK_SEED,
        shuffle_buffer=0,
    )

    if len(dataset) != EXPECTED_SAMPLES:
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

    metrics = {}

    for seed in SEEDS:

        metrics[
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

            full_target = batch[
                "target"
            ]

            sample_indices = batch[
                "sample_index"
            ]

            # Reduce antenna dimension first.
            target = (
                full_target
                .index_select(
                    2,
                    antenna_indices,
                )
            )

            # Then reduce frequency dimension.
            target = (
                target
                .index_select(
                    3,
                    frequency_indices,
                )
            )

            if (
                num_subcarriers
                == FULL_NUM_SUBCARRIERS
            ):
                # Reuse the exact original 5%-random
                # validation masks for F=1024.
                mask = batch[
                    "mask"
                ]

                mask = (
                    mask
                    .index_select(
                        2,
                        antenna_indices,
                    )
                )

                mask = (
                    mask
                    .index_select(
                        3,
                        frequency_indices,
                    )
                )

            else:
                # For a genuinely different frequency
                # resolution, generate exactly 5%
                # observations on that reduced grid.
                mask = (
                    make_reduced_random_mask(
                        sample_indices=(
                            sample_indices
                        ),
                        num_antennas=(
                            num_antennas
                        ),
                        num_subcarriers=(
                            num_subcarriers
                        ),
                        density=DENSITY,
                    )
                )

            current_observed = int(
                mask[
                    0,
                    0,
                    0,
                ].sum().item()
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
                    "Observed frequency count "
                    "changed unexpectedly."
                )

            # Construct sparse normalized CSI.
            observed = (
                target
                * mask
            )

            target = target.to(
                device,
                non_blocking=True,
            )

            observed = observed.to(
                device,
                non_blocking=True,
            )

            mask = mask.to(
                device,
                non_blocking=True,
            )

            interpolated = (
                frequency_linear_interpolate(
                    observed,
                    mask,
                )
            )

            target_cpu = (
                target
                .detach()
                .float()
                .cpu()
            )

            interpolation_cpu = (
                interpolated
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

            target_physical = (
                normalizer
                .denormalize_tensor(
                    target_cpu
                )
            )

            interpolation_physical = (
                normalizer
                .denormalize_tensor(
                    interpolation_cpu
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

                metrics[
                    seed
                ][
                    "maximum_observed_error"
                ] = max(
                    metrics[
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

                metrics[
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

                metrics[
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
        torch.cuda
        .max_memory_allocated()
        / (1024 ** 2)
    )

    interpolation_full_db = (
        interpolation_full.compute_db()
    )

    interpolation_missing_db = (
        interpolation_missing.compute_db()
    )

    per_seed_rows = []

    for seed in SEEDS:

        full_db = (
            metrics[
                seed
            ][
                "full"
            ].compute_db()
        )

        missing_db = (
            metrics[
                seed
            ][
                "missing"
            ].compute_db()
        )

        max_observed_error = (
            metrics[
                seed
            ][
                "maximum_observed_error"
            ]
        )

        if max_observed_error > 1e-6:
            raise RuntimeError(
                f"Observed values changed "
                f"for seed {seed}."
            )

        # The original grid must reproduce the
        # already independently verified result.
        if (
            num_antennas
            == FULL_NUM_ANTENNAS
            and num_subcarriers
            == FULL_NUM_SUBCARRIERS
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
                    f"Seed {seed}: "
                    "full-grid control "
                    "did not reproduce."
                )

            if abs(
                missing_db
                - saved_missing
            ) > 0.02:
                raise RuntimeError(
                    f"Seed {seed}: "
                    "missing-region control "
                    "did not reproduce."
                )

        row = {
            "condition": label,
            "num_antennas": (
                num_antennas
            ),
            "num_subcarriers": (
                num_subcarriers
            ),
            "requested_density": (
                DENSITY
            ),
            "observed_subcarriers": (
                observed_subcarriers
            ),
            "actual_density": (
                observed_subcarriers
                / num_subcarriers
            ),
            "observed_complex_coefficients": (
                num_antennas
                * observed_subcarriers
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

    summary_row = {
        "condition": label,
        "num_antennas": (
            num_antennas
        ),
        "num_subcarriers": (
            num_subcarriers
        ),
        "requested_density": (
            DENSITY
        ),
        "observed_subcarriers": (
            observed_subcarriers
        ),
        "actual_density": (
            observed_subcarriers
            / num_subcarriers
        ),
        "observed_complex_coefficients": (
            num_antennas
            * observed_subcarriers
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
        "mean_gain_vs_interpolation_missing_db": (
            interpolation_missing_db
            - missing_mean
        ),
        "peak_gpu_memory_mb": (
            peak_memory_mb
        ),
        "evaluation_seconds": (
            elapsed
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
        f"Gain                  : "
        f"{interpolation_missing_db - missing_mean:.4f} dB"
    )

    print(
        f"Peak GPU memory       : "
        f"{peak_memory_mb:.2f} MiB"
    )

    print(
        f"Evaluation time       : "
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

    print("=" * 90)
    print(
        "FROZEN SCALE-FORMER "
        "GRID SCALABILITY EVALUATION"
    )
    print("=" * 90)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required."
        )

    device = torch.device(
        "cuda"
    )

    print(
        f"GPU                 : "
        f"{torch.cuda.get_device_name(0)}"
    )

    print(
        f"Frozen model seeds  : "
        f"{SEEDS}"
    )

    print(
        f"Observation density : "
        f"{DENSITY:.2%}"
    )

    print(
        "Retraining           : NO"
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

    for (
        num_antennas,
        num_subcarriers,
    ) in GRID_CONDITIONS:

        (
            per_seed_rows,
            summary_row,
        ) = evaluate_grid(
            num_antennas=(
                num_antennas
            ),
            num_subcarriers=(
                num_subcarriers
            ),
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

        # Incremental saving.
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
        + "=" * 110
    )

    print(
        "GRID SCALABILITY SUMMARY"
    )

    print(
        "=" * 110
    )

    print(
        f"{'Grid':<14}"
        f"{'Observed':>10}"
        f"{'Interp miss':>16}"
        f"{'SCALE-Former miss':>27}"
        f"{'Gain':>13}"
        f"{'GPU MiB':>12}"
    )

    print(
        "-" * 110
    )

    for row in all_summary_rows:

        mean = row[
            "scaleformer_missing_nmse_db_mean"
        ]

        std = row[
            "scaleformer_missing_nmse_db_std"
        ]

        print(
            f"{row['condition']:<14}"
            f"{row['observed_complex_coefficients']:>10d}"
            f"{row['interpolation_missing_nmse_db']:>13.4f} dB"
            f"{mean:>13.4f} ± "
            f"{std:<7.4f} dB"
            f"{row['mean_gain_vs_interpolation_missing_db']:>9.4f} dB"
            f"{row['peak_gpu_memory_mb']:>11.2f}"
        )

    print(
        f"\nTotal evaluation time: "
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
        "GRID SCALABILITY EVALUATION: PASSED"
    )

    print("=" * 110)


if __name__ == "__main__":
    main()