from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
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
from src.models.frequency_transformer_baseline import (
    FrequencyTransformerBaseline,
)

# ============================================================
# Frozen full-attention Transformer baseline architecture
#
# Architecture was fixed before validation performance was
# observed. Do not tune these values using run 0056.
# ============================================================

TRANSFORMER_NUM_ANTENNAS = 32
TRANSFORMER_DIM = 64
TRANSFORMER_HEADS = 4
TRANSFORMER_LAYERS = 3
TRANSFORMER_FF_DIM = 128
TRANSFORMER_DROPOUT = 0.0

EXPECTED_TRANSFORMER_PARAMS = 108_928

DATA_ROOT = "data/dichasus_005x"

NORMALIZATION_FILE = (
    "results/csv/"
    "training_normalization.json"
)

VAL_MASK_SEED = 42

# Frozen validation interpolation benchmark:
# DICHASUS-0056, 5% random observations.
INTERPOLATION_FULL_DB = -25.8060
INTERPOLATION_MISSING_DB = -25.5839


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def missing_region_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:

    missing = (
        1.0 - mask
    ).expand(
        -1,
        2,
        -1,
        -1,
    )

    squared_error = (
        prediction - target
    ).square()

    numerator = (
        squared_error
        * missing
    ).sum()

    denominator = (
        missing.sum()
        .clamp_min(1.0)
    )

    return numerator / denominator


@torch.no_grad()
def validate(
    model: FrequencyTransformerBaseline,
    loader: DataLoader,
    normalizer: CSINormalizer,
    device: torch.device,
    use_amp: bool,
) -> dict:

    model.eval()

    full_metric = AggregateNMSE(
        missing_only=False
    )

    missing_metric = AggregateNMSE(
        missing_only=True
    )

    normalized_loss_sum = 0.0
    num_batches = 0
    num_samples = 0

    for batch in loader:

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

        interpolated = (
            frequency_linear_interpolate(
                observed,
                mask,
            )
        )

        with torch.amp.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=use_amp,
        ):
            prediction = model(
                interpolated,
                mask,
            )

            loss = missing_region_mse(
                prediction,
                target,
                mask,
            )

        normalized_loss_sum += float(
            loss.detach().cpu().item()
        )

        num_batches += 1
        num_samples += target.shape[0]

        # Publication metrics are always calculated
        # after restoring the original measured scale.
        prediction_cpu = (
            prediction
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

        prediction_physical = (
            normalizer.denormalize_tensor(
                prediction_cpu
            )
        )

        target_physical = (
            normalizer.denormalize_tensor(
                target_cpu
            )
        )

        full_metric.update(
            prediction=prediction_physical,
            target=target_physical,
            mask=mask_cpu,
        )

        missing_metric.update(
            prediction=prediction_physical,
            target=target_physical,
            mask=mask_cpu,
        )

    return {
        "normalized_missing_mse": (
            normalized_loss_sum
            / max(num_batches, 1)
        ),
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
        "num_samples": num_samples,
    }


def save_history(
    history: list[dict],
    output_file: Path,
) -> None:

    if not history:
        return

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_file.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(
                history[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(history)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=5e-4,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=5,
    )

    args = parser.parse_args()

    print("=" * 80)
    print(
        "TRAIN FULL-ATTENTION FREQUENCY "
        "TRANSFORMER ON REAL DICHASUS CSI"
    )
    print("=" * 80)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required."
        )

    set_seed(
        args.seed
    )

    device = torch.device(
        "cuda"
    )

    use_amp = True

    print(
        f"Seed                : {args.seed}"
    )

    print(
        f"GPU                 : "
        f"{torch.cuda.get_device_name(0)}"
    )

    print(
        f"Batch size          : "
        f"{args.batch_size}"
    )

    print(
        f"Maximum epochs      : "
        f"{args.epochs}"
    )

    print(
        f"Learning rate       : "
        f"{args.lr}"
    )

    print(
        f"Transformer dim     : "
        f"{TRANSFORMER_DIM}"
    )

    print(
        f"Attention heads     : "
        f"{TRANSFORMER_HEADS}"
    )

    print(
        f"Transformer layers  : "
        f"{TRANSFORMER_LAYERS}"
    )

    print(
        f"FFN dimension       : "
        f"{TRANSFORMER_FF_DIM}"
    )

    print(
        f"Dropout             : "
        f"{TRANSFORMER_DROPOUT}"
    )

    print(
        "Training pattern    : random"
    )

    print(
        "Training density    : 5%"
    )

    print(
        "Validation run      : 0056"
    )

    train_dataset = DichasusDataset(
        data_root=DATA_ROOT,
        split="train",
        normalization_path=(
            NORMALIZATION_FILE
        ),
        density=0.05,
        pattern="random",
        seed=args.seed,
        shuffle_buffer=512,
    )

    val_dataset = DichasusDataset(
        data_root=DATA_ROOT,
        split="val",
        normalization_path=(
            NORMALIZATION_FILE
        ),
        density=0.05,
        pattern="random",
        seed=VAL_MASK_SEED,
        shuffle_buffer=0,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=True,
    )

    model = FrequencyTransformerBaseline(
        num_antennas=(
            TRANSFORMER_NUM_ANTENNAS
        ),
        dim=(
            TRANSFORMER_DIM
        ),
        num_heads=(
            TRANSFORMER_HEADS
        ),
        num_layers=(
            TRANSFORMER_LAYERS
        ),
        ff_dim=(
            TRANSFORMER_FF_DIM
        ),
        dropout=(
            TRANSFORMER_DROPOUT
        ),
    ).to(device)

    num_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        f"Trainable params    : "
        f"{num_parameters:,}"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=(
            args.weight_decay
        ),
    )

    scheduler = (
        torch.optim.lr_scheduler
        .CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=1e-5,
        )
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp,
    )

    checkpoint_dir = Path(
        "checkpoints"
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    """checkpoint_path = (
        checkpoint_dir
        / (
            f"scaleformer_seed"
            f"{args.seed}_best.pt"
        )
    )"""

    run_name = (
    f"scaleformer_seed{args.seed}"
    f"_bs{args.batch_size}"
    )

    checkpoint_path = (
        checkpoint_dir
        / f"{run_name}_best.pt"
    )


    """history_file = Path(
        "results/csv/"
        f"scaleformer_seed"
        f"{args.seed}_training.csv"
    )

    summary_file = Path(
        "results/csv/"
        f"scaleformer_seed"
        f"{args.seed}_summary.json"
    )"""

    history_file = Path(
    "results/csv/"
    f"{run_name}_training.csv"
    )

    summary_file = Path(
        "results/csv/"
        f"{run_name}_summary.json"
    )

    history = []

    best_missing_nmse = float(
        "inf"
    )

    epochs_without_improvement = 0

    experiment_start = time.time()

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        train_dataset.set_epoch(
            epoch - 1
        )

        model.train()

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        epoch_start = time.time()

        training_loss_sum = 0.0
        training_batches = 0
        training_samples = 0

        for batch_index, batch in enumerate(
            train_loader,
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

            interpolated = (
                frequency_linear_interpolate(
                    observed,
                    mask,
                )
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            with torch.amp.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=use_amp,
            ):
                prediction = model(
                    interpolated,
                    mask,
                )

                loss = missing_region_mse(
                    prediction,
                    target,
                    mask,
                )

            scaler.scale(
                loss
            ).backward()

            scaler.unscale_(
                optimizer
            )

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            scaler.step(
                optimizer
            )

            scaler.update()

            training_loss_sum += float(
                loss.detach().cpu().item()
            )

            training_batches += 1
            training_samples += (
                target.shape[0]
            )

            if (
                batch_index % 500
                == 0
            ):
                print(
                    f"Epoch {epoch:02d} | "
                    f"batch {batch_index:04d} | "
                    f"samples "
                    f"{training_samples:,}/"
                    f"{len(train_dataset):,} | "
                    f"loss "
                    f"{training_loss_sum / training_batches:.6f}"
                )

        train_loss = (
            training_loss_sum
            / max(
                training_batches,
                1,
            )
        )

        validation = validate(
            model=model,
            loader=val_loader,
            normalizer=(
                train_dataset.normalizer
            ),
            device=device,
            use_amp=use_amp,
        )

        peak_memory_mb = (
            torch.cuda.max_memory_allocated()
            / (1024 ** 2)
        )

        epoch_time = (
            time.time()
            - epoch_start
        )

        current_lr = (
            optimizer.param_groups[
                0
            ]["lr"]
        )

        row = {
            "epoch": epoch,
            "train_missing_mse": (
                train_loss
            ),
            "val_missing_mse_normalized": (
                validation[
                    "normalized_missing_mse"
                ]
            ),
            "val_full_nmse_db": (
                validation[
                    "full_nmse_db"
                ]
            ),
            "val_missing_nmse_db": (
                validation[
                    "missing_nmse_db"
                ]
            ),
            "learning_rate": (
                current_lr
            ),
            "peak_gpu_memory_mb": (
                peak_memory_mb
            ),
            "epoch_seconds": (
                epoch_time
            ),
        }

        history.append(
            row
        )

        save_history(
            history,
            history_file,
        )

        print(
            "\n"
            f"Epoch {epoch:02d} complete\n"
            f"  Train missing MSE : "
            f"{train_loss:.6f}\n"
            f"  Val full NMSE     : "
            f"{validation['full_nmse_db']:.4f} dB\n"
            f"  Val missing NMSE  : "
            f"{validation['missing_nmse_db']:.4f} dB\n"
            f"  Interp full       : "
            f"{INTERPOLATION_FULL_DB:.4f} dB\n"
            f"  Interp missing    : "
            f"{INTERPOLATION_MISSING_DB:.4f} dB\n"
            f"  Peak GPU memory   : "
            f"{peak_memory_mb:.1f} MiB\n"
            f"  Epoch time        : "
            f"{epoch_time:.1f} s\n"
        )

        current_missing = validation[
            "missing_nmse_db"
        ]

        if (
            current_missing
            < best_missing_nmse
        ):
            best_missing_nmse = (
                current_missing
            )

            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict": (
                        model.state_dict()
                    ),
                    "optimizer_state_dict": (
                        optimizer.state_dict()
                    ),
                    "epoch": epoch,
                    "seed": args.seed,
                    "num_parameters": (
                        num_parameters
                    ),
                    "architecture": {
                        "dim": args.dim,
                        "heads": (
                            args.heads
                        ),
                        "latents": (
                            args.latents
                        ),
                        "local_blocks": (
                            args.local_blocks
                        ),
                        "global_blocks": (
                            args.global_blocks
                        ),
                        "decoder_blocks": (
                            args.decoder_blocks
                        ),
                        "decoder_chunk_size": (
                            args.decoder_chunk_size
                        ),
                    },
                    "val_full_nmse_db": (
                        validation[
                            "full_nmse_db"
                        ]
                    ),
                    "val_missing_nmse_db": (
                        validation[
                            "missing_nmse_db"
                        ]
                    ),
                },
                checkpoint_path,
            )

            print(
                "  New best checkpoint saved."
            )

        else:
            epochs_without_improvement += 1

        scheduler.step()

        if (
            epochs_without_improvement
            >= args.patience
        ):
            print(
                "\nEarly stopping triggered."
            )
            break

    total_time = (
        time.time()
        - experiment_start
    )

    best_checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    best_full = float(
        best_checkpoint[
            "val_full_nmse_db"
        ]
    )

    best_missing = float(
        best_checkpoint[
            "val_missing_nmse_db"
        ]
    )

    summary = {
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "maximum_epochs": args.epochs,
        "patience": args.patience,
        "model": (
            "FrequencyTransformerBaseline"
        ),
        "seed": args.seed,
        "training_runs": [
            "0053",
            "0054",
            "0055",
        ],
        "validation_run": "0056",
        "pattern": "random",
        "requested_density": 0.05,
        "observed_subcarriers": 51,
        "num_parameters": (
            num_parameters
        ),
        "architecture": (
            best_checkpoint[
                "architecture"
            ]
        ),
        "best_epoch": (
            best_checkpoint[
                "epoch"
            ]
        ),
        "best_val_full_nmse_db": (
            best_full
        ),
        "best_val_missing_nmse_db": (
            best_missing
        ),
        "interpolation_full_nmse_db": (
            INTERPOLATION_FULL_DB
        ),
        "interpolation_missing_nmse_db": (
            INTERPOLATION_MISSING_DB
        ),
        "improvement_vs_interpolation_full_db": (
            INTERPOLATION_FULL_DB
            - best_full
        ),
        "improvement_vs_interpolation_missing_db": (
            INTERPOLATION_MISSING_DB
            - best_missing
        ),
        "cnn_seed42_full_nmse_db": (
            -26.3391
        ),
        "cnn_seed42_missing_nmse_db": (
            -26.1170
        ),
        "training_seconds": (
            total_time
        ),
    }

    summary_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with summary_file.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=4,
        )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "SCALE-FORMER TRAINING COMPLETE"
    )

    print(
        "=" * 80
    )

    print(
        f"Best epoch                : "
        f"{summary['best_epoch']}"
    )

    print(
        f"Best full NMSE            : "
        f"{best_full:.4f} dB"
    )

    print(
        f"Best missing NMSE         : "
        f"{best_missing:.4f} dB"
    )

    print(
        f"Interpolation full        : "
        f"{INTERPOLATION_FULL_DB:.4f} dB"
    )

    print(
        f"Interpolation missing     : "
        f"{INTERPOLATION_MISSING_DB:.4f} dB"
    )

    print(
        f"Improvement vs interp full: "
        f"{INTERPOLATION_FULL_DB - best_full:.4f} dB"
    )

    print(
        f"Improvement vs interp miss: "
        f"{INTERPOLATION_MISSING_DB - best_missing:.4f} dB"
    )

    print(
        f"CNN full benchmark        : "
        f"-26.3391 dB"
    )

    print(
        f"CNN missing benchmark     : "
        f"-26.1170 dB"
    )

    print(
        f"Difference vs CNN missing : "
        f"{-26.1170 - best_missing:.4f} dB"
    )

    print(
        f"Total training time       : "
        f"{total_time / 60:.2f} min"
    )

    print(
        f"\nHistory:"
        f"\n{history_file}"
    )

    print(
        f"\nSummary:"
        f"\n{summary_file}"
    )

    print(
        f"\nBest checkpoint:"
        f"\n{checkpoint_path}"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()