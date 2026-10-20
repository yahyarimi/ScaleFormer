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
from src.models.frequency_transformer_baseline import (
    FrequencyTransformerBaseline,
)
from src.models.interpolation import (
    frequency_linear_interpolate,
)


# ============================================================
# Frozen final-test protocol
# ============================================================

DATA_ROOT = "data/dichasus_005x"

NORMALIZATION_FILE = (
    "results/csv/"
    "training_normalization.json"
)

RESULT_ROOT = Path(
    "results/csv"
)

TEST_RUN = "0057"

EXPECTED_TEST_SAMPLES = 5_001

MASK_PATTERN = "random"
MASK_DENSITY = 0.05
MASK_SEED = 42

EXPECTED_OBSERVED_SUBCARRIERS = 51

BATCH_SIZE = 8

SEEDS = [
    7,
    42,
    123,
]

EXPECTED_PARAMETERS = 108_928

CHECKPOINTS = {
    seed: (
        Path("checkpoints")
        / (
            "frequency_transformer_"
            f"seed{seed}_bs4_best.pt"
        )
    )
    for seed in SEEDS
}

PER_SEED_OUTPUT = (
    RESULT_ROOT
    / (
        "frequency_transformer_"
        "final_test_0057_per_seed.csv"
    )
)

SUMMARY_OUTPUT = (
    RESULT_ROOT
    / (
        "frequency_transformer_"
        "final_test_0057_summary.csv"
    )
)

JSON_OUTPUT = (
    RESULT_ROOT
    / (
        "frequency_transformer_"
        "final_test_0057_summary.json"
    )
)


# ============================================================
# Previously frozen 0057 reference results.
#
# These values are NOT recalculated for model selection.
# They come from the already-completed frozen final test and
# are used only for comparison after the Transformer result
# is computed.
# ============================================================

FROZEN_INTERPOLATION_FULL_DB = -25.7545
FROZEN_INTERPOLATION_MISSING_DB = -25.5322

FROZEN_CNN_FULL_MEAN_DB = -26.2864
FROZEN_CNN_FULL_STD_DB = 0.0374

FROZEN_CNN_MISSING_MEAN_DB = -26.0641
FROZEN_CNN_MISSING_STD_DB = 0.0374

FROZEN_SCALEFORMER_FULL_MEAN_DB = -30.4055
FROZEN_SCALEFORMER_FULL_STD_DB = 0.0772

FROZEN_SCALEFORMER_MISSING_MEAN_DB = -30.1832
FROZEN_SCALEFORMER_MISSING_STD_DB = 0.0772

INTERPOLATION_SANITY_TOLERANCE_DB = 1e-3


def load_transformer(
    seed: int,
    device: torch.device,
) -> FrequencyTransformerBaseline:

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

    if (
        int(
            checkpoint[
                "seed"
            ]
        )
        != seed
    ):

        raise RuntimeError(
            f"Checkpoint seed mismatch for "
            f"seed {seed}."
        )

    architecture = (
        checkpoint[
            "architecture"
        ]
    )

    expected_architecture = {
        "num_antennas": 32,
        "dim": 64,
        "heads": 4,
        "layers": 3,
        "ff_dim": 128,
        "dropout": 0.0,
    }

    for (
        key,
        expected_value,
    ) in expected_architecture.items():

        actual_value = (
            architecture[
                key
            ]
        )

        if actual_value != expected_value:

            raise RuntimeError(
                "Frozen Transformer architecture "
                f"mismatch for '{key}': "
                f"got {actual_value}, "
                f"expected {expected_value}."
            )

    model = FrequencyTransformerBaseline(
        num_antennas=int(
            architecture[
                "num_antennas"
            ]
        ),
        dim=int(
            architecture[
                "dim"
            ]
        ),
        num_heads=int(
            architecture[
                "heads"
            ]
        ),
        num_layers=int(
            architecture[
                "layers"
            ]
        ),
        ff_dim=int(
            architecture[
                "ff_dim"
            ]
        ),
        dropout=float(
            architecture[
                "dropout"
            ]
        ),
    )

    parameters = sum(
        parameter.numel()
        for parameter
        in model.parameters()
        if parameter.requires_grad
    )

    if (
        parameters
        != EXPECTED_PARAMETERS
    ):

        raise RuntimeError(
            "Unexpected Transformer "
            f"parameter count: "
            f"{parameters:,}; expected "
            f"{EXPECTED_PARAMETERS:,}."
        )

    if (
        int(
            checkpoint[
                "num_parameters"
            ]
        )
        != parameters
    ):

        raise RuntimeError(
            "Checkpoint parameter count "
            "does not match reconstructed "
            "Transformer."
        )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ],
        strict=True,
    )

    model = model.to(
        device
    )

    model.eval()

    return model


def main():

    print(
        "=" * 110
    )

    print(
        "FROZEN FULL-ATTENTION TRANSFORMER "
        "FINAL TEST ON DICHASUS 0057"
    )

    print(
        "=" * 110
    )

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA is required."
        )

    device = torch.device(
        "cuda"
    )

    print(
        f"GPU                    : "
        f"{torch.cuda.get_device_name(0)}"
    )

    print(
        f"Test run               : "
        f"{TEST_RUN}"
    )

    print(
        f"Expected test samples  : "
        f"{EXPECTED_TEST_SAMPLES:,}"
    )

    print(
        f"Observation pattern    : "
        f"{MASK_PATTERN}"
    )

    print(
        f"Observation density    : "
        f"{MASK_DENSITY:.1%}"
    )

    print(
        f"Mask seed              : "
        f"{MASK_SEED}"
    )

    print(
        f"Training seeds         : "
        f"{SEEDS}"
    )

    print(
        f"Expected parameters    : "
        f"{EXPECTED_PARAMETERS:,}"
    )

    print(
        "Retraining             : NO"
    )

    print(
        "Hyperparameter tuning  : NO"
    )

    normalizer = CSINormalizer(
        NORMALIZATION_FILE
    )

    dataset = DichasusDataset(
        data_root=DATA_ROOT,
        split="test",
        normalization_path=(
            NORMALIZATION_FILE
        ),
        density=MASK_DENSITY,
        pattern=MASK_PATTERN,
        seed=MASK_SEED,
        shuffle_buffer=0,
    )

    if (
        len(dataset)
        != EXPECTED_TEST_SAMPLES
    ):

        raise RuntimeError(
            "Unexpected test sample count: "
            f"{len(dataset):,}; expected "
            f"{EXPECTED_TEST_SAMPLES:,}."
        )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=0,
        pin_memory=True,
    )

    models = {}

    for seed in SEEDS:

        models[
            seed
        ] = load_transformer(
            seed,
            device,
        )

        print(
            f"Loaded Transformer seed "
            f"{seed}"
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
            "max_observed_error": 0.0,
        }

    observed_subcarriers = None

    total_samples = 0

    start = time.perf_counter()

    with torch.inference_mode():

        for (
            batch_index,
            batch,
        ) in enumerate(
            loader,
            start=1,
        ):

            observed = (
                batch[
                    "observed"
                ]
                .to(
                    device,
                    non_blocking=True,
                )
            )

            target = (
                batch[
                    "target"
                ]
                .to(
                    device,
                    non_blocking=True,
                )
            )

            mask = (
                batch[
                    "mask"
                ]
                .to(
                    device,
                    non_blocking=True,
                )
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
                    "Observed subcarrier "
                    "count changed across batches."
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

            mask_cpu = (
                mask
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
                )
                .bool()
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
                    "max_observed_error"
                ] = max(
                    metrics[
                        seed
                    ][
                        "max_observed_error"
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

            total_samples += (
                target.shape[
                    0
                ]
            )

            if (
                batch_index
                % 100
                == 0
            ):

                print(
                    f"Processed "
                    f"{total_samples:,}/"
                    f"{len(dataset):,}"
                )

    elapsed = (
        time.perf_counter()
        - start
    )

    if (
        total_samples
        != EXPECTED_TEST_SAMPLES
    ):

        raise RuntimeError(
            "Processed sample count "
            "does not match expected "
            "test size."
        )

    if (
        observed_subcarriers
        != EXPECTED_OBSERVED_SUBCARRIERS
    ):

        raise RuntimeError(
            "Unexpected observed "
            "subcarrier count: "
            f"{observed_subcarriers}; "
            "expected "
            f"{EXPECTED_OBSERVED_SUBCARRIERS}."
        )

    interpolation_full_db = (
        interpolation_full.compute_db()
    )

    interpolation_missing_db = (
        interpolation_missing.compute_db()
    )

    if (
        abs(
            interpolation_full_db
            - FROZEN_INTERPOLATION_FULL_DB
        )
        > INTERPOLATION_SANITY_TOLERANCE_DB
    ):

        raise RuntimeError(
            "Interpolation full-NMSE sanity "
            "check failed. This indicates that "
            "the final-test protocol does not "
            "match the previously frozen 0057 "
            "evaluation."
        )

    if (
        abs(
            interpolation_missing_db
            - FROZEN_INTERPOLATION_MISSING_DB
        )
        > INTERPOLATION_SANITY_TOLERANCE_DB
    ):

        raise RuntimeError(
            "Interpolation missing-NMSE sanity "
            "check failed. This indicates that "
            "the final-test protocol does not "
            "match the previously frozen 0057 "
            "evaluation."
        )

    per_seed_rows = []

    for seed in SEEDS:

        full_db = (
            metrics[
                seed
            ][
                "full"
            ]
            .compute_db()
        )

        missing_db = (
            metrics[
                seed
            ][
                "missing"
            ]
            .compute_db()
        )

        max_observed_error = (
            metrics[
                seed
            ][
                "max_observed_error"
            ]
        )

        if (
            max_observed_error
            > 1e-6
        ):

            raise RuntimeError(
                f"Transformer seed {seed} "
                "changed observed CSI values. "
                "Maximum observed error: "
                f"{max_observed_error:.8e}"
            )

        per_seed_rows.append(
            {
                "seed": seed,
                "full_nmse_db": (
                    full_db
                ),
                "missing_nmse_db": (
                    missing_db
                ),
                "max_observed_error": (
                    max_observed_error
                ),
            }
        )

    full_values = [
        row[
            "full_nmse_db"
        ]
        for row in per_seed_rows
    ]

    missing_values = [
        row[
            "missing_nmse_db"
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

    gain_vs_interpolation = (
        FROZEN_INTERPOLATION_MISSING_DB
        - missing_mean
    )

    gain_vs_cnn = (
        FROZEN_CNN_MISSING_MEAN_DB
        - missing_mean
    )

    scaleformer_advantage = (
        missing_mean
        - FROZEN_SCALEFORMER_MISSING_MEAN_DB
    )

    scaleformer_linear_ratio = (
        10.0
        ** (
            scaleformer_advantage
            / 10.0
        )
    )

    scaleformer_reduction_percent = (
        1.0
        - (
            1.0
            / scaleformer_linear_ratio
        )
    ) * 100.0

    summary_row = {
        "method": (
            "Full-Attention Transformer"
        ),
        "num_training_seeds": 3,
        "num_parameters": (
            EXPECTED_PARAMETERS
        ),
        "full_nmse_db_mean": (
            full_mean
        ),
        "full_nmse_db_std": (
            full_std
        ),
        "missing_nmse_db_mean": (
            missing_mean
        ),
        "missing_nmse_db_std": (
            missing_std
        ),
        "gain_vs_interpolation_missing_db": (
            gain_vs_interpolation
        ),
        "gain_vs_cnn_missing_db": (
            gain_vs_cnn
        ),
        "scaleformer_advantage_missing_db": (
            scaleformer_advantage
        ),
        "scaleformer_linear_nmse_reduction_percent": (
            scaleformer_reduction_percent
        ),
    }

    RESULT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    with PER_SEED_OUTPUT.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(
                per_seed_rows[
                    0
                ].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            per_seed_rows
        )

    with SUMMARY_OUTPUT.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(
                summary_row.keys()
            ),
        )

        writer.writeheader()

        writer.writerow(
            summary_row
        )

    payload = {
        "protocol": {
            "test_run": TEST_RUN,
            "num_test_samples": (
                total_samples
            ),
            "pattern": MASK_PATTERN,
            "requested_density": (
                MASK_DENSITY
            ),
            "observed_subcarriers": (
                observed_subcarriers
            ),
            "mask_seed": MASK_SEED,
            "training_seeds": (
                SEEDS
            ),
            "batch_size": (
                BATCH_SIZE
            ),
            "normalization": (
                "training-set statistics only"
            ),
            "retraining": False,
            "hyperparameter_tuning": False,
            "checkpoint_selection": (
                "best validation missing-region "
                "NMSE on DICHASUS 0056"
            ),
        },
        "interpolation_sanity_check": {
            "recomputed_full_nmse_db": (
                interpolation_full_db
            ),
            "recomputed_missing_nmse_db": (
                interpolation_missing_db
            ),
            "frozen_full_nmse_db": (
                FROZEN_INTERPOLATION_FULL_DB
            ),
            "frozen_missing_nmse_db": (
                FROZEN_INTERPOLATION_MISSING_DB
            ),
            "tolerance_db": (
                INTERPOLATION_SANITY_TOLERANCE_DB
            ),
            "passed": True,
        },
        "per_seed": (
            per_seed_rows
        ),
        "summary": (
            summary_row
        ),
        "frozen_reference_results": {
            "linear_interpolation": {
                "full_nmse_db": (
                    FROZEN_INTERPOLATION_FULL_DB
                ),
                "missing_nmse_db": (
                    FROZEN_INTERPOLATION_MISSING_DB
                ),
            },
            "residual_frequency_cnn": {
                "full_nmse_db_mean": (
                    FROZEN_CNN_FULL_MEAN_DB
                ),
                "full_nmse_db_std": (
                    FROZEN_CNN_FULL_STD_DB
                ),
                "missing_nmse_db_mean": (
                    FROZEN_CNN_MISSING_MEAN_DB
                ),
                "missing_nmse_db_std": (
                    FROZEN_CNN_MISSING_STD_DB
                ),
            },
            "scaleformer": {
                "full_nmse_db_mean": (
                    FROZEN_SCALEFORMER_FULL_MEAN_DB
                ),
                "full_nmse_db_std": (
                    FROZEN_SCALEFORMER_FULL_STD_DB
                ),
                "missing_nmse_db_mean": (
                    FROZEN_SCALEFORMER_MISSING_MEAN_DB
                ),
                "missing_nmse_db_std": (
                    FROZEN_SCALEFORMER_MISSING_STD_DB
                ),
            },
        },
        "evaluation_seconds": (
            elapsed
        ),
    }

    with JSON_OUTPUT.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            payload,
            file,
            indent=4,
        )

    print(
        "\n"
        + "=" * 110
    )

    print(
        "FULL-ATTENTION TRANSFORMER "
        "FINAL TEST 0057 SUMMARY"
    )

    print(
        "=" * 110
    )

    for row in per_seed_rows:

        print(
            f"Seed {row['seed']:>3}: "
            f"full="
            f"{row['full_nmse_db']:.4f} dB | "
            f"missing="
            f"{row['missing_nmse_db']:.4f} dB | "
            f"observed error="
            f"{row['max_observed_error']:.3e}"
        )

    print(
        "-" * 110
    )

    print(
        "Transformer full NMSE    : "
        f"{full_mean:.4f} +/- "
        f"{full_std:.4f} dB"
    )

    print(
        "Transformer missing NMSE : "
        f"{missing_mean:.4f} +/- "
        f"{missing_std:.4f} dB"
    )

    print(
        "Gain vs interpolation    : "
        f"{gain_vs_interpolation:.4f} dB"
    )

    print(
        "Gain vs CNN              : "
        f"{gain_vs_cnn:.4f} dB"
    )

    print(
        "SCALE-Former advantage   : "
        f"{scaleformer_advantage:.4f} dB"
    )

    print(
        "SCALE-Former linear-NMSE "
        "reduction vs Transformer: "
        f"{scaleformer_reduction_percent:.2f}%"
    )

    print(
        "\nInterpolation sanity check:"
    )

    print(
        f"  full    : "
        f"{interpolation_full_db:.4f} dB"
    )

    print(
        f"  missing : "
        f"{interpolation_missing_db:.4f} dB"
    )

    print(
        "  status  : PASSED"
    )

    print(
        f"\nEvaluation time:"
        f"\n{elapsed:.2f} s"
    )

    print(
        f"\nPer-seed CSV:"
        f"\n{PER_SEED_OUTPUT}"
    )

    print(
        f"\nSummary CSV:"
        f"\n{SUMMARY_OUTPUT}"
    )

    print(
        f"\nJSON:"
        f"\n{JSON_OUTPUT}"
    )

    print(
        "\nFINAL FROZEN TRANSFORMER "
        "TEST EVALUATION: PASSED"
    )

    print(
        "=" * 110
    )


if __name__ == "__main__":
    main()
