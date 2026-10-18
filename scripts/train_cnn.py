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
from src.models.cnn_baseline import (
    ResidualFrequencyCNN,
)
from src.models.interpolation import (
    frequency_linear_interpolate,
)


DATA_ROOT = "data/dichasus_005x"

NORMALIZATION_FILE = (
    "results/csv/"
    "training_normalization.json"
)

VAL_MASK_SEED = 42

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

    denominator = missing.sum().clamp_min(
        1.0
    )

    return (
        numerator
        / denominator
    )


@torch.no_grad()
def validate(
    model,
    loader,
    normalizer,
    device,
    use_amp,
):
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

        # Publication metrics are calculated in
        # original measured CSI scale.
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
    history,
    output_file,
):
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
        default=20,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
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

    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--num-blocks",
        type=int,
        default=4,
    )

    args = parser.parse_args()

    print("=" * 80)
    print(
        "TRAIN COMPACT RESIDUAL FREQUENCY CNN"
    )
    print("=" * 80)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for this experiment."
        )

    set_seed(
        args.seed
    )

    device = torch.device(
        "cuda"
    )

    use_amp = True

    print(
        f"Seed             : {args.seed}"
    )

    print(
        f"GPU              : "
        f"{torch.cuda.get_device_name(0)}"
    )

    print(
        f"Batch size       : "
        f"{args.batch_size}"
    )

    print(
        f"Maximum epochs   : "
        f"{args.epochs}"
    )

    print(
        f"Learning rate    : "
        f"{args.lr}"
    )

    print(
        "Training pattern : random"
    )

    print(
        "Training density : 5%"
    )

    print(
        "Validation run   : 0056"
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

    model = ResidualFrequencyCNN(
        num_antennas=32,
        hidden_dim=args.hidden_dim,
        num_blocks=args.num_blocks,
        kernel_size=7,
    ).to(
        device
    )

    num_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        f"Trainable params : "
        f"{num_parameters:,}"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
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

    checkpoint_path = (
        checkpoint_dir
        / f"cnn_seed{args.seed}_best.pt"
    )

    history_file = Path(
        "results/csv/"
        f"cnn_seed{args.seed}_training.csv"
    )

    summary_file = Path(
        "results/csv/"
        f"cnn_seed{args.seed}_summary.json"
    )

    history = []

    best_missing_nmse = float(
        "inf"
    )

    best_epoch = 0
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
                batch_index % 250
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
            / training_batches
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
            "train_missing_mse": train_loss,
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
            "learning_rate": current_lr,
            "peak_gpu_memory_mb": (
                peak_memory_mb
            ),
            "epoch_seconds": epoch_time,
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

            best_epoch = epoch

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

    summary = {
        "model": (
            "ResidualFrequencyCNN"
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
        "best_epoch": (
            best_checkpoint["epoch"]
        ),
        "best_val_full_nmse_db": (
            best_checkpoint[
                "val_full_nmse_db"
            ]
        ),
        "best_val_missing_nmse_db": (
            best_checkpoint[
                "val_missing_nmse_db"
            ]
        ),
        "interpolation_full_nmse_db": (
            INTERPOLATION_FULL_DB
        ),
        "interpolation_missing_nmse_db": (
            INTERPOLATION_MISSING_DB
        ),
        "improvement_full_db": (
            INTERPOLATION_FULL_DB
            - best_checkpoint[
                "val_full_nmse_db"
            ]
        ),
        "improvement_missing_db": (
            INTERPOLATION_MISSING_DB
            - best_checkpoint[
                "val_missing_nmse_db"
            ]
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
        "CNN TRAINING COMPLETE"
    )

    print(
        "=" * 80
    )

    print(
        f"Best epoch              : "
        f"{summary['best_epoch']}"
    )

    print(
        f"Best full NMSE          : "
        f"{summary['best_val_full_nmse_db']:.4f} dB"
    )

    print(
        f"Best missing NMSE       : "
        f"{summary['best_val_missing_nmse_db']:.4f} dB"
    )

    print(
        f"Interpolation full      : "
        f"{INTERPOLATION_FULL_DB:.4f} dB"
    )

    print(
        f"Interpolation missing   : "
        f"{INTERPOLATION_MISSING_DB:.4f} dB"
    )

    print(
        f"Full-grid improvement   : "
        f"{summary['improvement_full_db']:.4f} dB"
    )

    print(
        f"Missing-region improvement: "
        f"{summary['improvement_missing_db']:.4f} dB"
    )

    print(
        f"Total training time     : "
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