from __future__ import annotations

import csv
import gc
import json
import statistics
from pathlib import Path

import torch

from scripts.benchmark_complexity import (
    cleanup_cuda,
    load_cnn,
    load_scaleformer,
    load_validation_example,
    measure_peak_memory,
)

from src.models.interpolation import (
    frequency_linear_interpolate,
)


RESULT_ROOT = Path("results/csv")

CSV_OUTPUT = (
    RESULT_ROOT
    / "memory_benchmark.csv"
)

JSON_OUTPUT = (
    RESULT_ROOT
    / "memory_benchmark.json"
)

REPETITIONS = 3


def measure_repeated(
    function,
):

    peaks = []
    incremental_peaks = []

    for repetition in range(
        1,
        REPETITIONS + 1,
    ):

        result = measure_peak_memory(
            function
        )

        peaks.append(
            result[
                "peak_allocated_mib"
            ]
        )

        incremental_peaks.append(
            result[
                "incremental_peak_mib"
            ]
        )

        print(
            f"  repetition {repetition}: "
            f"peak={peaks[-1]:.4f} MiB | "
            f"incremental="
            f"{incremental_peaks[-1]:.4f} MiB"
        )

    return {
        "peak_mean_mib": (
            statistics.mean(
                peaks
            )
        ),
        "peak_std_mib": (
            statistics.stdev(
                peaks
            )
            if len(peaks) > 1
            else 0.0
        ),
        "peak_max_mib": max(
            peaks
        ),
        "incremental_peak_mean_mib": (
            statistics.mean(
                incremental_peaks
            )
        ),
    }


def main():

    print("=" * 90)
    print(
        "CONTROLLED END-TO-END "
        "GPU MEMORY BENCHMARK"
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
        f"GPU         : "
        f"{torch.cuda.get_device_name(0)}"
    )

    print(
        f"Repetitions : "
        f"{REPETITIONS}"
    )

    print(
        "Metric      : "
        "torch.cuda.max_memory_allocated"
    )

    observed, mask, observed_count = (
        load_validation_example(
            device
        )
    )

    print(
        f"Grid        : "
        f"{tuple(observed.shape)}"
    )

    print(
        f"Observed    : "
        f"{observed_count} frequencies"
    )

    rows = []

    # ========================================================
    # Interpolation
    # ========================================================

    cleanup_cuda()

    print(
        "\nLINEAR INTERPOLATION"
    )

    def interpolation():

        return (
            frequency_linear_interpolate(
                observed,
                mask,
            )
        )

    result = measure_repeated(
        interpolation
    )

    rows.append(
        {
            "method": (
                "Linear interpolation"
            ),
            "parameters": 0,
            **result,
        }
    )

    # ========================================================
    # CNN
    # ========================================================

    cleanup_cuda()

    cnn, cnn_params = (
        load_cnn(
            device
        )
    )

    print(
        "\nRESIDUAL FREQUENCY CNN"
    )

    def cnn_end_to_end():

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

            return cnn(
                interpolated,
                mask,
            )

    result = measure_repeated(
        cnn_end_to_end
    )

    rows.append(
        {
            "method": (
                "Residual Frequency CNN"
            ),
            "parameters": (
                cnn_params
            ),
            **result,
        }
    )

    del cnn
    gc.collect()
    cleanup_cuda()

    # ========================================================
    # SCALE-Former
    # ========================================================

    (
        scaleformer,
        sf_params,
    ) = load_scaleformer(
        device
    )

    print(
        "\nSCALE-FORMER"
    )

    def sf_end_to_end():

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

            return scaleformer(
                interpolated,
                mask,
            )

    result = measure_repeated(
        sf_end_to_end
    )

    rows.append(
        {
            "method": (
                "SCALE-Former"
            ),
            "parameters": (
                sf_params
            ),
            **result,
        }
    )

    del scaleformer
    gc.collect()
    cleanup_cuda()

    # ========================================================
    # Save
    # ========================================================

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
            "grid_antennas": 32,
            "grid_subcarriers": 1024,
            "observed_subcarriers": (
                observed_count
            ),
            "observation_density": (
                0.05
            ),
            "learned_precision": (
                "AMP FP16"
            ),
            "interpolation_precision": (
                "FP32"
            ),
            "repetitions": (
                REPETITIONS
            ),
            "metric": (
                "torch.cuda."
                "max_memory_allocated"
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

    print(
        "\n"
        + "=" * 100
    )

    print(
        "GPU MEMORY SUMMARY"
    )

    print(
        "=" * 100
    )

    print(
        f"{'Method':<28}"
        f"{'Params':>12}"
        f"{'Peak mean':>18}"
        f"{'SD':>14}"
        f"{'Peak max':>16}"
    )

    print(
        "-" * 100
    )

    for row in rows:

        print(
            f"{row['method']:<28}"
            f"{row['parameters']:>12,}"
            f"{row['peak_mean_mib']:>14.4f} MiB"
            f"{row['peak_std_mib']:>10.4f}"
            f"{row['peak_max_mib']:>12.4f} MiB"
        )

    print(
        f"\nCSV:"
        f"\n{CSV_OUTPUT}"
    )

    print(
        f"\nJSON:"
        f"\n{JSON_OUTPUT}"
    )

    print("=" * 100)


if __name__ == "__main__":
    main()