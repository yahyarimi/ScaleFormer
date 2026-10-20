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

    observed, mask, _ = (
        load_validation_example(
            device
        )
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

        # ----------------------------------------------------
        # Interpolation
        # ----------------------------------------------------

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

        all_rows.append(
            interpolation_row
        )

        # Alternate learned-model order across rounds
        # to reduce systematic thermal / clock bias.
        if (
            round_index
            % 2
            == 1
        ):

            model_order = [
                "cnn",
                "scaleformer",
            ]

        else:

            model_order = [
                "scaleformer",
                "cnn",
            ]

        for model_name in model_order:

            cleanup_cuda()

            if model_name == "cnn":

                model, params = (
                    load_cnn(
                        device
                    )
                )

                name = (
                    "Residual Frequency CNN"
                )

            else:

                model, params = (
                    load_scaleformer(
                        device
                    )
                )

                name = (
                    "SCALE-Former"
                )

            row = (
                benchmark_learned_model(
                    name=name,
                    model=model,
                    parameters=params,
                    observed=observed,
                    mask=mask,
                )
            )

            row[
                "round"
            ] = round_index

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
        + "=" * 110
    )

    print(
        "REPEATED LATENCY SUMMARY"
    )

    print(
        "=" * 110
    )

    print(
        f"{'Method':<28}"
        f"{'Params':>12}"
        f"{'E2E median mean':>22}"
        f"{'Across-run SD':>18}"
        f"{'Min':>13}"
        f"{'Max':>13}"
    )

    print(
        "-" * 110
    )

    for row in summary_rows:

        print(
            f"{row['method']:<28}"
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
        "=" * 110
    )


if __name__ == "__main__":
    main()