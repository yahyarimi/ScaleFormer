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
from src.models.cnn_baseline import ResidualFrequencyCNN
from src.models.interpolation import frequency_linear_interpolate
from src.models.scaleformer import ScaleFormer


# ============================================================
# Frozen final-test protocol
# ============================================================

DATA_ROOT = "data/dichasus_005x"

NORMALIZATION_FILE = (
    "results/csv/"
    "training_normalization.json"
)

RESULT_ROOT = Path("results/csv")

TEST_RUN = "0057"

EXPECTED_TEST_SAMPLES = 5001

MASK_PATTERN = "random"
MASK_DENSITY = 0.05
MASK_SEED = 42

EXPECTED_OBSERVED_SUBCARRIERS = 51

BATCH_SIZE = 8

SEEDS = [7, 42, 123]


CNN_CHECKPOINTS = {
    seed: (
        Path("checkpoints")
        / f"cnn_seed{seed}_best.pt"
    )
    for seed in SEEDS
}


SCALEFORMER_CHECKPOINTS = {
    seed: (
        Path("checkpoints")
        / (
            f"scaleformer_seed{seed}"
            f"_bs4_best.pt"
        )
    )
    for seed in SEEDS
}


PER_SEED_OUTPUT = (
    RESULT_ROOT
    / "final_test_0057_per_seed.csv"
)

SUMMARY_OUTPUT = (
    RESULT_ROOT
    / "final_test_0057_summary.csv"
)

JSON_OUTPUT = (
    RESULT_ROOT
    / "final_test_0057_summary.json"
)


# ============================================================
# Model loading
# ============================================================

def load_cnn(
    seed: int,
    device: torch.device,
):

    checkpoint_path = (
        CNN_CHECKPOINTS[
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

    model = ResidualFrequencyCNN(
        num_antennas=32,
        hidden_dim=64,
        num_blocks=4,
        kernel_size=7,
    )

    params = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    if params != 60_160:
        raise RuntimeError(
            f"Unexpected CNN parameter "
            f"count: {params:,}"
        )

    if int(
        checkpoint[
            "num_parameters"
        ]
    ) != params:
        raise RuntimeError(
            "CNN checkpoint parameter "
            "count mismatch."
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


def load_scaleformer(
    seed: int,
    device: torch.device,
):

    checkpoint_path = (
        SCALEFORMER_CHECKPOINTS[
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
        local_kernel_size=(
            architecture.get(
                "local_kernel_size",
                7,
            )
        ),
        decoder_chunk_size=architecture[
            "decoder_chunk_size"
        ],
        dropout=0.0,
        variant=architecture.get(
            "variant",
            "full",
        ),
    )

    params = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    if params != 100_034:
        raise RuntimeError(
            "Unexpected SCALE-Former "
            f"parameter count: {params:,}"
        )

    if int(
        checkpoint[
            "num_parameters"
        ]
    ) != params:
        raise RuntimeError(
            "SCALE-Former checkpoint "
            "parameter count mismatch."
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


# ============================================================
# Main evaluation
# ============================================================

def main():

    print("=" * 100)

    print(
        "FINAL UNTOUCHED TEST EVALUATION"
    )

    print("=" * 100)

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

    if len(dataset) != EXPECTED_TEST_SAMPLES:
        raise RuntimeError(
            "Unexpected test sample count: "
            f"{len(dataset):,}"
        )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=0,
        pin_memory=True,
    )

    # --------------------------------------------------------
    # Load all frozen checkpoints.
    # --------------------------------------------------------

    cnn_models = {}

    scaleformer_models = {}

    for seed in SEEDS:

        cnn_models[
            seed
        ] = load_cnn(
            seed,
            device,
        )

        print(
            f"Loaded CNN seed "
            f"{seed}"
        )

    for seed in SEEDS:

        scaleformer_models[
            seed
        ] = load_scaleformer(
            seed,
            device,
        )

        print(
            f"Loaded SCALE-Former "
            f"seed {seed}"
        )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

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

    cnn_metrics = {}

    scaleformer_metrics = {}

    for seed in SEEDS:

        cnn_metrics[
            seed
        ] = {
            "full": AggregateNMSE(
                missing_only=False
            ),
            "missing": AggregateNMSE(
                missing_only=True
            ),
            "max_observed_error": (
                0.0
            ),
        }

        scaleformer_metrics[
            seed
        ] = {
            "full": AggregateNMSE(
                missing_only=False
            ),
            "missing": AggregateNMSE(
                missing_only=True
            ),
            "max_observed_error": (
                0.0
            ),
        }

    observed_subcarriers = None

    total_samples = 0

    start = time.perf_counter()

    # --------------------------------------------------------
    # One single pass through untouched run 0057.
    # --------------------------------------------------------

    with torch.inference_mode():

        for (
            batch_index,
            batch,
        ) in enumerate(
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
                    "Observed subcarrier "
                    "count changed."
                )

            interpolated = (
                frequency_linear_interpolate(
                    observed,
                    mask,
                )
            )

            # ------------------------------------------------
            # CPU copies for denormalized metrics.
            # ------------------------------------------------

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

            # ------------------------------------------------
            # CNN checkpoints
            # ------------------------------------------------

            for seed in SEEDS:

                model = cnn_models[
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

                cnn_metrics[
                    seed
                ][
                    "max_observed_error"
                ] = max(
                    cnn_metrics[
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

                cnn_metrics[
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

                cnn_metrics[
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

            # ------------------------------------------------
            # SCALE-Former checkpoints
            # ------------------------------------------------

            for seed in SEEDS:

                model = (
                    scaleformer_models[
                        seed
                    ]
                )

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

                scaleformer_metrics[
                    seed
                ][
                    "max_observed_error"
                ] = max(
                    scaleformer_metrics[
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

                scaleformer_metrics[
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

                scaleformer_metrics[
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
                target.shape[0]
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
            "Final processed sample "
            "count mismatch."
        )

    if (
        observed_subcarriers
        != EXPECTED_OBSERVED_SUBCARRIERS
    ):

        raise RuntimeError(
            "Unexpected final observed "
            "subcarrier count."
        )

    # --------------------------------------------------------
    # Aggregate results.
    # --------------------------------------------------------

    interpolation_full_db = (
        interpolation_full.compute_db()
    )

    interpolation_missing_db = (
        interpolation_missing.compute_db()
    )

    per_seed_rows = []

    for seed in SEEDS:

        cnn_full = (
            cnn_metrics[
                seed
            ][
                "full"
            ].compute_db()
        )

        cnn_missing = (
            cnn_metrics[
                seed
            ][
                "missing"
            ].compute_db()
        )

        sf_full = (
            scaleformer_metrics[
                seed
            ][
                "full"
            ].compute_db()
        )

        sf_missing = (
            scaleformer_metrics[
                seed
            ][
                "missing"
            ].compute_db()
        )

        cnn_observed_error = (
            cnn_metrics[
                seed
            ][
                "max_observed_error"
            ]
        )

        sf_observed_error = (
            scaleformer_metrics[
                seed
            ][
                "max_observed_error"
            ]
        )

        if (
            cnn_observed_error
            > 1e-6
        ):

            raise RuntimeError(
                f"CNN seed {seed} "
                "changed observed values."
            )

        if (
            sf_observed_error
            > 1e-6
        ):

            raise RuntimeError(
                f"SCALE-Former seed "
                f"{seed} changed "
                "observed values."
            )

        per_seed_rows.append(
            {
                "seed": seed,
                "cnn_full_nmse_db": (
                    cnn_full
                ),
                "cnn_missing_nmse_db": (
                    cnn_missing
                ),
                "scaleformer_full_nmse_db": (
                    sf_full
                ),
                "scaleformer_missing_nmse_db": (
                    sf_missing
                ),
                "scaleformer_gain_vs_cnn_missing_db": (
                    cnn_missing
                    - sf_missing
                ),
                "cnn_max_observed_error": (
                    cnn_observed_error
                ),
                "scaleformer_max_observed_error": (
                    sf_observed_error
                ),
            }
        )

    cnn_full_values = [
        row[
            "cnn_full_nmse_db"
        ]
        for row in per_seed_rows
    ]

    cnn_missing_values = [
        row[
            "cnn_missing_nmse_db"
        ]
        for row in per_seed_rows
    ]

    sf_full_values = [
        row[
            "scaleformer_full_nmse_db"
        ]
        for row in per_seed_rows
    ]

    sf_missing_values = [
        row[
            "scaleformer_missing_nmse_db"
        ]
        for row in per_seed_rows
    ]

    cnn_full_mean = statistics.mean(
        cnn_full_values
    )

    cnn_full_std = statistics.stdev(
        cnn_full_values
    )

    cnn_missing_mean = statistics.mean(
        cnn_missing_values
    )

    cnn_missing_std = statistics.stdev(
        cnn_missing_values
    )

    sf_full_mean = statistics.mean(
        sf_full_values
    )

    sf_full_std = statistics.stdev(
        sf_full_values
    )

    sf_missing_mean = statistics.mean(
        sf_missing_values
    )

    sf_missing_std = statistics.stdev(
        sf_missing_values
    )

    sf_vs_cnn = (
        cnn_missing_mean
        - sf_missing_mean
    )

    sf_vs_interp = (
        interpolation_missing_db
        - sf_missing_mean
    )

    cnn_vs_interp = (
        interpolation_missing_db
        - cnn_missing_mean
    )

    # Convert dB improvement to linear-error reduction.
    sf_vs_cnn_ratio = (
        10 ** (
            sf_vs_cnn
            / 10
        )
    )

    sf_vs_cnn_reduction = (
        1.0
        - (
            1.0
            / sf_vs_cnn_ratio
        )
    ) * 100.0

    summary_rows = [
        {
            "method": (
                "Linear interpolation"
            ),
            "num_training_seeds": 0,
            "full_nmse_db_mean": (
                interpolation_full_db
            ),
            "full_nmse_db_std": None,
            "missing_nmse_db_mean": (
                interpolation_missing_db
            ),
            "missing_nmse_db_std": None,
            "gain_vs_interpolation_missing_db": (
                0.0
            ),
        },
        {
            "method": (
                "Residual Frequency CNN"
            ),
            "num_training_seeds": 3,
            "full_nmse_db_mean": (
                cnn_full_mean
            ),
            "full_nmse_db_std": (
                cnn_full_std
            ),
            "missing_nmse_db_mean": (
                cnn_missing_mean
            ),
            "missing_nmse_db_std": (
                cnn_missing_std
            ),
            "gain_vs_interpolation_missing_db": (
                cnn_vs_interp
            ),
        },
        {
            "method": (
                "SCALE-Former"
            ),
            "num_training_seeds": 3,
            "full_nmse_db_mean": (
                sf_full_mean
            ),
            "full_nmse_db_std": (
                sf_full_std
            ),
            "missing_nmse_db_mean": (
                sf_missing_mean
            ),
            "missing_nmse_db_std": (
                sf_missing_std
            ),
            "gain_vs_interpolation_missing_db": (
                sf_vs_interp
            ),
        },
    ]

    # --------------------------------------------------------
    # Save.
    # --------------------------------------------------------

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
                summary_rows[
                    0
                ].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            summary_rows
        )

    payload = {
        "protocol": {
            "test_run": (
                TEST_RUN
            ),
            "num_test_samples": (
                total_samples
            ),
            "pattern": (
                MASK_PATTERN
            ),
            "requested_density": (
                MASK_DENSITY
            ),
            "observed_subcarriers": (
                observed_subcarriers
            ),
            "mask_seed": (
                MASK_SEED
            ),
            "training_seeds": (
                SEEDS
            ),
            "normalization": (
                "training-set statistics only"
            ),
            "retraining": False,
            "hyperparameter_tuning": (
                False
            ),
        },
        "per_seed": (
            per_seed_rows
        ),
        "summary": (
            summary_rows
        ),
        "scaleformer_vs_cnn": {
            "missing_nmse_gain_db": (
                sf_vs_cnn
            ),
            "linear_nmse_ratio": (
                sf_vs_cnn_ratio
            ),
            "linear_nmse_reduction_percent": (
                sf_vs_cnn_reduction
            ),
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

    # --------------------------------------------------------
    # Report.
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 110
    )

    print(
        "FINAL TEST 0057 SUMMARY"
    )

    print(
        "=" * 110
    )

    print(
        f"{'Method':<28}"
        f"{'Full NMSE':>28}"
        f"{'Missing NMSE':>28}"
        f"{'Gain vs interp':>20}"
    )

    print(
        "-" * 110
    )

    print(
        f"{'Linear interpolation':<28}"
        f"{interpolation_full_db:>24.4f} dB"
        f"{interpolation_missing_db:>24.4f} dB"
        f"{0.0:>16.4f} dB"
    )

    print(
        f"{'Residual Frequency CNN':<28}"
        f"{cnn_full_mean:>13.4f} ± "
        f"{cnn_full_std:<7.4f} dB"
        f"{cnn_missing_mean:>13.4f} ± "
        f"{cnn_missing_std:<7.4f} dB"
        f"{cnn_vs_interp:>16.4f} dB"
    )

    print(
        f"{'SCALE-Former':<28}"
        f"{sf_full_mean:>13.4f} ± "
        f"{sf_full_std:<7.4f} dB"
        f"{sf_missing_mean:>13.4f} ± "
        f"{sf_missing_std:<7.4f} dB"
        f"{sf_vs_interp:>16.4f} dB"
    )

    print(
        "\nSCALE-Former vs CNN "
        f"(missing NMSE): "
        f"{sf_vs_cnn:.4f} dB"
    )

    print(
        "Equivalent linear-NMSE "
        "reduction vs CNN: "
        f"{sf_vs_cnn_reduction:.2f}%"
    )

    print(
        f"\nEvaluation time: "
        f"{elapsed:.2f} s"
    )

    print(
        f"\nPer-seed:"
        f"\n{PER_SEED_OUTPUT}"
    )

    print(
        f"\nSummary:"
        f"\n{SUMMARY_OUTPUT}"
    )

    print(
        f"\nJSON:"
        f"\n{JSON_OUTPUT}"
    )

    print(
        "\nFINAL UNTOUCHED TEST "
        "EVALUATION: PASSED"
    )

    print("=" * 110)


if __name__ == "__main__":
    main()