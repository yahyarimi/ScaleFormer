from __future__ import annotations

import argparse
import json
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

EXPECTED_SAMPLES = 5220
EXPECTED_OBSERVED_SUBCARRIERS = 51

INTERPOLATION_FULL_DB = -25.8060
INTERPOLATION_MISSING_DB = -25.5839


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=(
            "checkpoints/"
            "scaleformer_seed42_bs4_best.pt"
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required."
        )

    device = torch.device(
        "cuda"
    )

    checkpoint_path = Path(
        args.checkpoint
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            checkpoint_path
        )

    print("=" * 80)
    print(
        "INDEPENDENT SCALE-FORMER "
        "CHECKPOINT VERIFICATION"
    )
    print("=" * 80)

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    architecture = checkpoint[
        "architecture"
    ]

    print(
        f"Checkpoint           : "
        f"{checkpoint_path}"
    )

    print(
        f"Saved epoch          : "
        f"{checkpoint['epoch']}"
    )

    print(
        f"Saved seed           : "
        f"{checkpoint['seed']}"
    )

    print(
        f"Saved parameters     : "
        f"{checkpoint['num_parameters']:,}"
    )

    print(
        f"Embedding dim        : "
        f"{architecture['dim']}"
    )

    print(
        f"Attention heads      : "
        f"{architecture['heads']}"
    )

    print(
        f"Latent tokens        : "
        f"{architecture['latents']}"
    )

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

    actual_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    if actual_parameters != checkpoint[
        "num_parameters"
    ]:
        raise RuntimeError(
            "Parameter-count mismatch: "
            f"{actual_parameters} vs "
            f"{checkpoint['num_parameters']}."
        )

    normalizer = CSINormalizer(
        NORMALIZATION_FILE
    )

    dataset = DichasusDataset(
        data_root=DATA_ROOT,
        split="val",
        normalization_path=(
            NORMALIZATION_FILE
        ),
        density=0.05,
        pattern="random",
        seed=42,
        shuffle_buffer=0,
    )

    if len(dataset) != EXPECTED_SAMPLES:
        raise RuntimeError(
            f"Expected "
            f"{EXPECTED_SAMPLES} samples, "
            f"found {len(dataset)}."
        )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=True,
    )

    scaleformer_full = AggregateNMSE(
        missing_only=False
    )

    scaleformer_missing = AggregateNMSE(
        missing_only=True
    )

    interpolation_full = AggregateNMSE(
        missing_only=False
    )

    interpolation_missing = AggregateNMSE(
        missing_only=True
    )

    num_samples = 0

    maximum_observed_error = 0.0

    observed_subcarrier_count = None

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

            current_count = int(
                mask[
                    0,
                    0,
                    0,
                ].sum().item()
            )

            if observed_subcarrier_count is None:
                observed_subcarrier_count = (
                    current_count
                )

            if (
                current_count
                != observed_subcarrier_count
            ):
                raise RuntimeError(
                    "Observed subcarrier count "
                    "changed unexpectedly."
                )

            interpolated = (
                frequency_linear_interpolate(
                    observed,
                    mask,
                )
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

            expanded_mask = mask.expand(
                -1,
                2,
                -1,
                -1,
            ).bool()

            observed_error = (
                prediction[
                    expanded_mask
                ]
                - target[
                    expanded_mask
                ]
            ).abs().max().item()

            maximum_observed_error = max(
                maximum_observed_error,
                observed_error,
            )

            prediction_cpu = (
                prediction
                .float()
                .cpu()
            )

            interpolation_cpu = (
                interpolated
                .float()
                .cpu()
            )

            target_cpu = (
                target
                .float()
                .cpu()
            )

            mask_cpu = (
                mask
                .float()
                .cpu()
            )

            prediction_physical = (
                normalizer.denormalize_tensor(
                    prediction_cpu
                )
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

            scaleformer_full.update(
                prediction=prediction_physical,
                target=target_physical,
                mask=mask_cpu,
            )

            scaleformer_missing.update(
                prediction=prediction_physical,
                target=target_physical,
                mask=mask_cpu,
            )

            interpolation_full.update(
                prediction=interpolation_physical,
                target=target_physical,
                mask=mask_cpu,
            )

            interpolation_missing.update(
                prediction=interpolation_physical,
                target=target_physical,
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

    scaleformer_full_db = (
        scaleformer_full.compute_db()
    )

    scaleformer_missing_db = (
        scaleformer_missing.compute_db()
    )

    interpolation_full_db = (
        interpolation_full.compute_db()
    )

    interpolation_missing_db = (
        interpolation_missing.compute_db()
    )

    peak_memory_mb = (
        torch.cuda.max_memory_allocated()
        / (1024 ** 2)
    )

    saved_full_db = float(
        checkpoint[
            "val_full_nmse_db"
        ]
    )

    saved_missing_db = float(
        checkpoint[
            "val_missing_nmse_db"
        ]
    )

    full_difference = abs(
        scaleformer_full_db
        - saved_full_db
    )

    missing_difference = abs(
        scaleformer_missing_db
        - saved_missing_db
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "VERIFICATION RESULTS"
    )

    print(
        "=" * 80
    )

    print(
        f"Validation run            : 0056"
    )

    print(
        f"Measurements              : "
        f"{num_samples:,}"
    )

    print(
        f"Observed subcarriers      : "
        f"{observed_subcarrier_count}"
    )

    print(
        f"Requested density         : 5%"
    )

    print(
        f"Pattern                   : random"
    )

    print(
        f"\nInterpolation full NMSE   : "
        f"{interpolation_full_db:.4f} dB"
    )

    print(
        f"Interpolation missing NMSE: "
        f"{interpolation_missing_db:.4f} dB"
    )

    print(
        f"\nSCALE-Former full NMSE    : "
        f"{scaleformer_full_db:.4f} dB"
    )

    print(
        f"SCALE-Former missing NMSE : "
        f"{scaleformer_missing_db:.4f} dB"
    )

    print(
        f"\nSaved checkpoint full     : "
        f"{saved_full_db:.4f} dB"
    )

    print(
        f"Saved checkpoint missing  : "
        f"{saved_missing_db:.4f} dB"
    )

    print(
        f"Full metric difference    : "
        f"{full_difference:.8f} dB"
    )

    print(
        f"Missing metric difference : "
        f"{missing_difference:.8f} dB"
    )

    print(
        f"\nMax observed-value error  : "
        f"{maximum_observed_error:.8e}"
    )

    print(
        f"Peak inference GPU memory : "
        f"{peak_memory_mb:.2f} MiB"
    )

    print(
        f"Evaluation time           : "
        f"{elapsed:.2f} s"
    )

    if (
        num_samples
        != EXPECTED_SAMPLES
    ):
        raise RuntimeError(
            "Validation sample-count failure."
        )

    if (
        observed_subcarrier_count
        != EXPECTED_OBSERVED_SUBCARRIERS
    ):
        raise RuntimeError(
            "Unexpected observation count."
        )

    if maximum_observed_error > 1e-6:
        raise RuntimeError(
            "Observed CSI was not preserved."
        )

    if full_difference > 0.02:
        raise RuntimeError(
            "Independent full NMSE differs "
            "too much from training-time result."
        )

    if missing_difference > 0.02:
        raise RuntimeError(
            "Independent missing NMSE differs "
            "too much from training-time result."
        )

    # This also independently verifies our
    # interpolation reference to a small tolerance.
    if abs(
        interpolation_full_db
        - INTERPOLATION_FULL_DB
    ) > 0.02:
        raise RuntimeError(
            "Interpolation full-grid benchmark "
            "did not reproduce."
        )

    if abs(
        interpolation_missing_db
        - INTERPOLATION_MISSING_DB
    ) > 0.02:
        raise RuntimeError(
            "Interpolation missing benchmark "
            "did not reproduce."
        )

    result = {
        "checkpoint": str(
            checkpoint_path
        ),
        "validation_run": "0056",
        "pattern": "random",
        "requested_density": 0.05,
        "observed_subcarriers": (
            observed_subcarrier_count
        ),
        "num_samples": num_samples,
        "interpolation_full_nmse_db": (
            interpolation_full_db
        ),
        "interpolation_missing_nmse_db": (
            interpolation_missing_db
        ),
        "scaleformer_full_nmse_db": (
            scaleformer_full_db
        ),
        "scaleformer_missing_nmse_db": (
            scaleformer_missing_db
        ),
        "saved_full_nmse_db": (
            saved_full_db
        ),
        "saved_missing_nmse_db": (
            saved_missing_db
        ),
        "maximum_observed_error": (
            maximum_observed_error
        ),
        "peak_inference_gpu_memory_mb": (
            peak_memory_mb
        ),
        "evaluation_seconds": elapsed,
    }

    output_path = Path(
        "results/csv/"
        f"{checkpoint_path.stem}_verification.json"
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result,
            file,
            indent=4,
        )

    print(
        f"\nSaved verification:"
        f"\n{output_path}"
    )

    print(
        "\nINDEPENDENT CHECKPOINT "
        "VERIFICATION: PASSED"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()