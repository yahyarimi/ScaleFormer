from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

import torch

from scripts.benchmark_complexity import (
    cleanup_cuda,
    load_cnn,
    load_scaleformer,
    load_validation_example,
)

from src.models.frequency_transformer_baseline import (
    FrequencyTransformerBaseline,
)
from src.models.interpolation import (
    frequency_linear_interpolate,
)


RESULT_ROOT = Path("results/csv")

CSV_OUTPUT = (
    RESULT_ROOT
    / "latency_benchmark.csv"
)

JSON_OUTPUT = (
    RESULT_ROOT
    / "latency_benchmark.json"
)

WARMUP_PAIRS = 100
MEASURED_PAIRS = 500

FREQUENCY_TRANSFORMER_CHECKPOINT = Path(
    "checkpoints/"
    "frequency_transformer_seed42_bs4_best.pt"
)

EXPECTED_FREQUENCY_TRANSFORMER_PARAMS = 108_928


def load_frequency_transformer(
    device: torch.device,
):
    """
    Load the frozen seed-42 full-attention Transformer checkpoint
    for controlled inference benchmarking.

    Seed 42 is used only as a representative frozen checkpoint for
    architecture/efficiency benchmarking. Parameter count and
    architecture metadata are verified before inference.
    """

    if not FREQUENCY_TRANSFORMER_CHECKPOINT.exists():
        raise FileNotFoundError(
            FREQUENCY_TRANSFORMER_CHECKPOINT
        )

    checkpoint = torch.load(
        FREQUENCY_TRANSFORMER_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    architecture = checkpoint[
        "architecture"
    ]

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
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    if (
        parameters
        != EXPECTED_FREQUENCY_TRANSFORMER_PARAMS
    ):
        raise RuntimeError(
            "Unexpected full-attention Transformer "
            f"parameter count: {parameters:,}; "
            "expected "
            f"{EXPECTED_FREQUENCY_TRANSFORMER_PARAMS:,}."
        )

    checkpoint_parameters = int(
        checkpoint[
            "num_parameters"
        ]
    )

    if checkpoint_parameters != parameters:
        raise RuntimeError(
            "Full-attention Transformer checkpoint "
            "parameter count does not match the "
            "reconstructed model."
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

    return model, parameters


def percentile(
    values: list[float],
    fraction: float,
) -> float:

    ordered = sorted(
        values
    )

    index = int(
        fraction
        * (
            len(ordered)
            - 1
        )
    )

    return ordered[
        index
    ]


def summarize(
    values: list[float],
) -> dict:

    return {
        "mean_ms": statistics.mean(
            values
        ),
        "std_ms": statistics.stdev(
            values
        ),
        "median_ms": statistics.median(
            values
        ),
        "min_ms": min(
            values
        ),
        "max_ms": max(
            values
        ),
        "p95_ms": percentile(
            values,
            0.95,
        ),
    }


def time_single_call(
    function,
) -> float:
    """
    Measure one GPU operation using CUDA events.

    Synchronizing each measurement avoids a long
    asynchronous timing queue and makes the paired
    measurements easier to interpret on a Windows
    laptop GPU.
    """

    start = torch.cuda.Event(
        enable_timing=True
    )

    end = torch.cuda.Event(
        enable_timing=True
    )

    start.record()

    output = function()

    end.record()

    end.synchronize()

    elapsed_ms = start.elapsed_time(
        end
    )

    del output

    return elapsed_ms


def benchmark_interpolation(
    observed: torch.Tensor,
    mask: torch.Tensor,
):

    print(
        "\n"
        + "=" * 80
    )

    print(
        "INTERPOLATION LATENCY"
    )

    print(
        "=" * 80
    )

    def interpolation():

        return (
            frequency_linear_interpolate(
                observed,
                mask,
            )
        )

    with torch.inference_mode():

        for _ in range(
            WARMUP_PAIRS
        ):

            _ = interpolation()

        torch.cuda.synchronize()

        times = []

        for _ in range(
            MEASURED_PAIRS
        ):

            times.append(
                time_single_call(
                    interpolation
                )
            )

    stats = summarize(
        times
    )

    print(
        f"Mean   : "
        f"{stats['mean_ms']:.4f} "
        f"+/- {stats['std_ms']:.4f} ms"
    )

    print(
        f"Median : "
        f"{stats['median_ms']:.4f} ms"
    )

    print(
        f"P95    : "
        f"{stats['p95_ms']:.4f} ms"
    )

    return {
        "method": (
            "Linear interpolation"
        ),
        "parameters": 0,
        "model_only_mean_ms": None,
        "model_only_std_ms": None,
        "model_only_median_ms": None,
        "model_only_p95_ms": None,
        "end_to_end_mean_ms": (
            stats[
                "mean_ms"
            ]
        ),
        "end_to_end_std_ms": (
            stats[
                "std_ms"
            ]
        ),
        "end_to_end_median_ms": (
            stats[
                "median_ms"
            ]
        ),
        "end_to_end_p95_ms": (
            stats[
                "p95_ms"
            ]
        ),
        "median_e2e_overhead_ms": None,
    }


def benchmark_learned_model(
    name: str,
    model: torch.nn.Module,
    parameters: int,
    observed: torch.Tensor,
    mask: torch.Tensor,
):

    print(
        "\n"
        + "=" * 80
    )

    print(
        f"{name.upper()} LATENCY"
    )

    print(
        "=" * 80
    )

    interpolated = (
        frequency_linear_interpolate(
            observed,
            mask,
        )
    )

    torch.cuda.synchronize()

    probe = (
        frequency_linear_interpolate(
            observed,
            mask,
        )
    )

    max_difference = (
        probe
        .sub(
            interpolated
        )
        .abs()
        .max()
        .item()
    )

    print(
        f"Interpolation check : "
        f"max diff "
        f"{max_difference:.8e}"
    )

    print(
        f"Interpolation dtype : "
        f"{interpolated.dtype}"
    )

    print(
        f"Interpolation shape : "
        f"{tuple(interpolated.shape)}"
    )

    print(
        f"Contiguous          : "
        f"{interpolated.is_contiguous()}"
    )

    del probe

    def model_only():

        with torch.amp.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=True,
        ):

            return model(
                interpolated,
                mask,
            )

    def end_to_end():

        current_interpolated = (
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

            return model(
                current_interpolated,
                mask,
            )

    with torch.inference_mode():

        for index in range(
            WARMUP_PAIRS
        ):

            if index % 2 == 0:

                _ = model_only()
                _ = end_to_end()

            else:

                _ = end_to_end()
                _ = model_only()

        torch.cuda.synchronize()

        model_times = []
        e2e_times = []

        for index in range(
            MEASURED_PAIRS
        ):

            if index % 2 == 0:

                model_ms = (
                    time_single_call(
                        model_only
                    )
                )

                e2e_ms = (
                    time_single_call(
                        end_to_end
                    )
                )

            else:

                e2e_ms = (
                    time_single_call(
                        end_to_end
                    )
                )

                model_ms = (
                    time_single_call(
                        model_only
                    )
                )

            model_times.append(
                model_ms
            )

            e2e_times.append(
                e2e_ms
            )

    model_stats = summarize(
        model_times
    )

    e2e_stats = summarize(
        e2e_times
    )

    print()

    print(
        f"Parameters          : "
        f"{parameters:,}"
    )

    print(
        f"Model-only mean     : "
        f"{model_stats['mean_ms']:.4f} "
        f"+/- "
        f"{model_stats['std_ms']:.4f} ms"
    )

    print(
        f"Model-only median   : "
        f"{model_stats['median_ms']:.4f} ms"
    )

    print(
        f"Model-only P95      : "
        f"{model_stats['p95_ms']:.4f} ms"
    )

    print(
        f"End-to-end mean     : "
        f"{e2e_stats['mean_ms']:.4f} "
        f"+/- "
        f"{e2e_stats['std_ms']:.4f} ms"
    )

    print(
        f"End-to-end median   : "
        f"{e2e_stats['median_ms']:.4f} ms"
    )

    print(
        f"End-to-end P95      : "
        f"{e2e_stats['p95_ms']:.4f} ms"
    )

    difference = (
        e2e_stats[
            "median_ms"
        ]
        - model_stats[
            "median_ms"
        ]
    )

    print(
        f"Median E2E overhead : "
        f"{difference:.4f} ms"
    )

    if (
        e2e_stats[
            "median_ms"
        ]
        <= model_stats[
            "median_ms"
        ]
    ):

        print(
            "WARNING: end-to-end median "
            "is not greater than "
            "model-only median."
        )

    return {
        "method": name,
        "parameters": (
            parameters
        ),
        "model_only_mean_ms": (
            model_stats[
                "mean_ms"
            ]
        ),
        "model_only_std_ms": (
            model_stats[
                "std_ms"
            ]
        ),
        "model_only_median_ms": (
            model_stats[
                "median_ms"
            ]
        ),
        "model_only_p95_ms": (
            model_stats[
                "p95_ms"
            ]
        ),
        "end_to_end_mean_ms": (
            e2e_stats[
                "mean_ms"
            ]
        ),
        "end_to_end_std_ms": (
            e2e_stats[
                "std_ms"
            ]
        ),
        "end_to_end_median_ms": (
            e2e_stats[
                "median_ms"
            ]
        ),
        "end_to_end_p95_ms": (
            e2e_stats[
                "p95_ms"
            ]
        ),
        "median_e2e_overhead_ms": (
            difference
        ),
    }


def save_results(
    rows: list[dict],
):

    RESULT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    with CSV_OUTPUT.open(
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

    payload = {
        "protocol": {
            "batch_size": 1,
            "grid": [
                32,
                1024,
            ],
            "observation_density": (
                0.05
            ),
            "observed_subcarriers": (
                51
            ),
            "warmup_pairs": (
                WARMUP_PAIRS
            ),
            "measured_pairs": (
                MEASURED_PAIRS
            ),
            "timing": (
                "per-call CUDA events "
                "with synchronization"
            ),
            "pairing": (
                "alternating model-only "
                "and end-to-end order"
            ),
            "learned_precision": (
                "AMP FP16"
            ),
            "interpolation_precision": (
                "FP32"
            ),
            "representative_checkpoint_seed": (
                42
            ),
        },
        "results": rows,
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


def main():

    print(
        "=" * 90
    )

    print(
        "PAIRED / INTERLEAVED "
        "CUDA LATENCY BENCHMARK"
    )

    print(
        "=" * 90
    )

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA is required."
        )

    device = torch.device(
        "cuda"
    )

    print(
        f"GPU            : "
        f"{torch.cuda.get_device_name(0)}"
    )

    print(
        f"Warm-up pairs  : "
        f"{WARMUP_PAIRS}"
    )

    print(
        f"Measured pairs : "
        f"{MEASURED_PAIRS}"
    )

    observed, mask, _ = (
        load_validation_example(
            device
        )
    )

    rows = []

    cleanup_cuda()

    rows.append(
        benchmark_interpolation(
            observed,
            mask,
        )
    )

    cleanup_cuda()

    cnn, cnn_params = (
        load_cnn(
            device
        )
    )

    rows.append(
        benchmark_learned_model(
            name=(
                "Residual Frequency CNN"
            ),
            model=cnn,
            parameters=cnn_params,
            observed=observed,
            mask=mask,
        )
    )

    del cnn

    cleanup_cuda()

    (
        transformer,
        transformer_params,
    ) = load_frequency_transformer(
        device
    )

    rows.append(
        benchmark_learned_model(
            name=(
                "Full-Attention Transformer"
            ),
            model=transformer,
            parameters=transformer_params,
            observed=observed,
            mask=mask,
        )
    )

    del transformer

    cleanup_cuda()

    scaleformer, sf_params = (
        load_scaleformer(
            device
        )
    )

    rows.append(
        benchmark_learned_model(
            name="SCALE-Former",
            model=scaleformer,
            parameters=sf_params,
            observed=observed,
            mask=mask,
        )
    )

    del scaleformer

    cleanup_cuda()

    save_results(
        rows
    )

    print(
        "\n"
        + "=" * 120
    )

    print(
        "SINGLE-RUN LATENCY SUMMARY"
    )

    print(
        "=" * 120
    )

    print(
        f"{'Method':<30}"
        f"{'Params':>12}"
        f"{'Model median':>20}"
        f"{'E2E median':>20}"
        f"{'E2E P95':>16}"
    )

    print(
        "-" * 120
    )

    for row in rows:

        if (
            row[
                "model_only_median_ms"
            ]
            is None
        ):

            model_text = "-"

        else:

            model_text = (
                f"{row['model_only_median_ms']:.4f} ms"
            )

        print(
            f"{row['method']:<30}"
            f"{row['parameters']:>12,}"
            f"{model_text:>20}"
            f"{row['end_to_end_median_ms']:>17.4f} ms"
            f"{row['end_to_end_p95_ms']:>13.4f} ms"
        )

    print()

    print(
        "NOTE: publication latency should use the "
        "repeated benchmark summary, not this "
        "single-run output."
    )

    print(
        f"\nCSV:"
        f"\n{CSV_OUTPUT}"
    )

    print(
        f"\nJSON:"
        f"\n{JSON_OUTPUT}"
    )

    print(
        "=" * 120
    )


if __name__ == "__main__":
    main()
