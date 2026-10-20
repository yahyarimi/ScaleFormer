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
)

from scripts.benchmark_latency import (
    benchmark_interpolation,
    benchmark_learned_model,
    load_frequency_transformer,
)


RESULT_ROOT = Path("results/csv")

NUM_ROUNDS = 5

ROUND_CSV = (
    RESULT_ROOT
    / "latency_repeated_rounds.csv"
)

SUMMARY_CSV = (
    RESULT_ROOT
    / "latency_repeated_summary.csv"
)

SUMMARY_JSON = (
    RESULT_ROOT
    / "latency_repeated_summary.json"
)


# Predeclared learned-model order for each round.
# With three learned models and five rounds, exact position
# balance is impossible, but this schedule distributes first,
# middle, and last positions without keeping one model fixed.
ROUND_MODEL_ORDERS = [
    [
        "cnn",
        "transformer",
        "scaleformer",
    ],
    [
        "transformer",
        "scaleformer",
        "cnn",
    ],
    [
        "scaleformer",
        "cnn",
        "transformer",
    ],
    [
        "scaleformer",
        "transformer",
        "cnn",
    ],
    [
        "transformer",
        "cnn",
        "scaleformer",
    ],
]


def aggregate(
    rows: list[dict],
):

    methods = sorted(
        {
            row["method"]
            for row in rows
        }
    )

    summary_rows = []

    for method in methods:

        method_rows = [
            row
            for row in rows
            if row["method"] == method
        ]

        e2e_medians = [
            row[
                "end_to_end_median_ms"
            ]
            for row in method_rows
        ]

        e2e_p95 = [
            row[
                "end_to_end_p95_ms"
            ]
            for row in method_rows
        ]

        model_values = [
            row[
                "model_only_median_ms"
            ]
            for row in method_rows
            if row[
                "model_only_median_ms"
            ] is not None
        ]

        row = {
            "method": method,
            "rounds": len(
                method_rows
            ),
            "parameters": (
                method_rows[
                    0
                ][
                    "parameters"
                ]
            ),
            "e2e_median_mean_ms": (
                statistics.mean(
                    e2e_medians
                )
            ),
            "e2e_median_std_ms": (
                statistics.stdev(
                    e2e_medians
                )
            ),
            "e2e_median_min_ms": (
                min(
                    e2e_medians
                )
            ),
            "e2e_median_max_ms": (
                max(
                    e2e_medians
                )
            ),
            "e2e_p95_mean_ms": (
                statistics.mean(
                    e2e_p95
                )
            ),
            "e2e_p95_std_ms": (
                statistics.stdev(
                    e2e_p95
                )
            ),
            "model_median_mean_ms": (
                statistics.mean(
                    model_values
                )
                if model_values
                else None
            ),
            "model_median_std_ms": (
                statistics.stdev(
                    model_values
                )
                if len(
                    model_values
                ) > 1
                else None
            ),
        }

        summary_rows.append(
            row
        )

    return summary_rows


def save_csv(
    rows: list[dict],
    path: Path,
):

    if not rows:
        return

    fieldnames = []

    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(
                    key
                )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


def load_named_model(
    model_name: str,
    device: torch.device,
):

    if model_name == "cnn":

        model, params = (
            load_cnn(
                device
            )
        )

        display_name = (
            "Residual Frequency CNN"
        )

    elif model_name == "transformer":

        model, params = (
            load_frequency_transformer(
                device
            )
        )

        display_name = (
            "Full-Attention Transformer"
        )

    elif model_name == "scaleformer":

        model, params = (
            load_scaleformer(
                device
            )
        )

        display_name = (
            "SCALE-Former"
        )

    else:

        raise ValueError(
            f"Unknown model name: {model_name}"
        )

    return (
        model,
        params,
        display_name,
    )


def main():

    print(
        "=" * 100
    )

    print(
        "REPEATED LATENCY BENCHMARK"
    )

    print(
        "=" * 100
    )

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA is required."
        )

    device = torch.device(
        "cuda"
    )

    print(
        f"GPU    : "
        f"{torch.cuda.get_device_name(0)}"
    )

    print(
        f"Rounds : "
        f"{NUM_ROUNDS}"
    )

    observed, mask, observed_count = (
        load_validation_example(
            device
        )
    )

    print(
        f"Grid   : "
        f"{tuple(observed.shape)}"
    )

    print(
        f"Observed frequencies: "
        f"{observed_count}"
    )

    all_rows = []

    for round_index in range(
        1,
        NUM_ROUNDS + 1,
    ):

        print(
            "\n"
            + "#" * 100
        )

        print(
            f"ROUND "
            f"{round_index}/"
            f"{NUM_ROUNDS}"
        )

        print(
            "#" * 100
        )

        cleanup_cuda()

        interpolation_row = (
            benchmark_interpolation(
                observed,
                mask,
            )
        )

        interpolation_row[
            "round"
        ] = round_index

        interpolation_row[
            "order_position"
        ] = 0

        all_rows.append(
            interpolation_row
        )

        model_order = (
            ROUND_MODEL_ORDERS[
                round_index - 1
            ]
        )

        print(
            "Learned-model order: "
            + " -> ".join(
                model_order
            )
        )

        for (
            order_position,
            model_name,
        ) in enumerate(
            model_order,
            start=1,
        ):

            cleanup_cuda()

            (
                model,
                params,
                display_name,
            ) = load_named_model(
                model_name,
                device,
            )

            row = (
                benchmark_learned_model(
                    name=display_name,
                    model=model,
                    parameters=params,
                    observed=observed,
                    mask=mask,
                )
            )

            row[
                "round"
            ] = round_index

            row[
                "order_position"
            ] = order_position

            all_rows.append(
                row
            )

            del model

            gc.collect()

            cleanup_cuda()

        save_csv(
            all_rows,
            ROUND_CSV,
        )

    summary_rows = aggregate(
        all_rows
    )

    save_csv(
        summary_rows,
        SUMMARY_CSV,
    )

    payload = {
        "num_rounds": (
            NUM_ROUNDS
        ),
        "gpu": (
            torch.cuda
            .get_device_name(
                0
            )
        ),
        "protocol": {
            "batch_size": 1,
            "grid_antennas": 32,
            "grid_subcarriers": 1024,
            "observed_subcarriers": (
                observed_count
            ),
            "observation_density": 0.05,
            "warmup_pairs_per_round": 100,
            "measured_pairs_per_round": 500,
            "learned_precision": (
                "AMP FP16"
            ),
            "interpolation_precision": (
                "FP32"
            ),
            "representative_checkpoint_seed": 42,
            "learned_model_orders": (
                ROUND_MODEL_ORDERS
            ),
        },
        "results": (
            summary_rows
        ),
    }

    with SUMMARY_JSON.open(
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
        + "=" * 120
    )

    print(
        "REPEATED LATENCY SUMMARY"
    )

    print(
        "=" * 120
    )

    print(
        f"{'Method':<30}"
        f"{'Params':>12}"
        f"{'E2E median mean':>22}"
        f"{'Across-run SD':>18}"
        f"{'Min':>13}"
        f"{'Max':>13}"
    )

    print(
        "-" * 120
    )

    for row in summary_rows:

        print(
            f"{row['method']:<30}"
            f"{row['parameters']:>12,}"
            f"{row['e2e_median_mean_ms']:>18.4f} ms"
            f"{row['e2e_median_std_ms']:>14.4f} ms"
            f"{row['e2e_median_min_ms']:>9.4f} ms"
            f"{row['e2e_median_max_ms']:>9.4f} ms"
        )

    print()

    print(
        f"Round results:"
        f"\n{ROUND_CSV}"
    )

    print(
        f"\nSummary:"
        f"\n{SUMMARY_CSV}"
    )

    print(
        f"\nJSON:"
        f"\n{SUMMARY_JSON}"
    )

    print(
        "=" * 120
    )


if __name__ == "__main__":
    main()
