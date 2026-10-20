from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path


RESULT_ROOT = Path("results/csv")

SEEDS = [7, 42, 123]

VARIANTS = [
    "full",
    "no_global",
    "no_local",
    "no_coordinates",
]

DISPLAY_NAMES = {
    "full": "Full SCALE-Former",
    "no_global": "No Global",
    "no_local": "No Local",
    "no_coordinates": "No Coordinates",
}

INTERPOLATION_MISSING_DB = -25.5839


def summary_path(
    variant: str,
    seed: int,
) -> Path:

    if variant == "full":

        return RESULT_ROOT / (
            f"scaleformer_seed{seed}"
            f"_bs4_summary.json"
        )

    return RESULT_ROOT / (
        f"scaleformer_{variant}"
        f"_seed{seed}"
        f"_bs4_summary.json"
    )


def load_variant(
    variant: str,
):

    rows = []

    for seed in SEEDS:

        path = summary_path(
            variant,
            seed,
        )

        if not path.exists():
            raise FileNotFoundError(
                path
            )

        with path.open(
            "r",
            encoding="utf-8",
        ) as file:

            data = json.load(
                file
            )

        rows.append(
            {
                "variant": variant,
                "seed": seed,
                "parameters": int(
                    data[
                        "num_parameters"
                    ]
                ),
                "best_epoch": int(
                    data[
                        "best_epoch"
                    ]
                ),
                "full_nmse_db": float(
                    data[
                        "best_val_full_nmse_db"
                    ]
                ),
                "missing_nmse_db": float(
                    data[
                        "best_val_missing_nmse_db"
                    ]
                ),
            }
        )

    return rows


def main():

    all_rows = []

    variant_summaries = {}

    for variant in VARIANTS:

        rows = load_variant(
            variant
        )

        all_rows.extend(
            rows
        )

        full_values = [
            row[
                "full_nmse_db"
            ]
            for row in rows
        ]

        missing_values = [
            row[
                "missing_nmse_db"
            ]
            for row in rows
        ]

        parameter_values = {
            row[
                "parameters"
            ]
            for row in rows
        }

        if len(parameter_values) != 1:
            raise RuntimeError(
                f"Parameter count changed "
                f"across seeds for {variant}."
            )

        variant_summaries[
            variant
        ] = {
            "variant": variant,
            "display_name": (
                DISPLAY_NAMES[
                    variant
                ]
            ),
            "parameters": (
                parameter_values.pop()
            ),
            "num_seeds": len(
                SEEDS
            ),
            "seeds": SEEDS,
            "full_nmse_db_mean": (
                statistics.mean(
                    full_values
                )
            ),
            "full_nmse_db_std": (
                statistics.stdev(
                    full_values
                )
            ),
            "missing_nmse_db_mean": (
                statistics.mean(
                    missing_values
                )
            ),
            "missing_nmse_db_std": (
                statistics.stdev(
                    missing_values
                )
            ),
        }

    full_mean = (
        variant_summaries[
            "full"
        ][
            "missing_nmse_db_mean"
        ]
    )

    summary_rows = []

    for variant in VARIANTS:

        summary = (
            variant_summaries[
                variant
            ]
        )

        missing_mean = summary[
            "missing_nmse_db_mean"
        ]

        degradation = (
            missing_mean
            - full_mean
        )

        gain_vs_interp = (
            INTERPOLATION_MISSING_DB
            - missing_mean
        )

        row = {
            **summary,
            "degradation_vs_full_db": (
                degradation
            ),
            "gain_vs_interpolation_db": (
                gain_vs_interp
            ),
        }

        summary_rows.append(
            row
        )

    per_seed_output = (
        RESULT_ROOT
        / "scaleformer_ablation_per_seed.csv"
    )

    summary_output = (
        RESULT_ROOT
        / "scaleformer_ablation_summary.csv"
    )

    json_output = (
        RESULT_ROOT
        / "scaleformer_ablation_summary.json"
    )

    with per_seed_output.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(
                all_rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            all_rows
        )

    with summary_output.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(
                summary_rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            summary_rows
        )

    with json_output.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary_rows,
            file,
            indent=4,
        )

    print("=" * 100)
    print(
        "SCALE-FORMER THREE-SEED "
        "ABLATION SUMMARY"
    )
    print("=" * 100)

    print(
        f"{'Variant':<22}"
        f"{'Params':>12}"
        f"{'Missing NMSE':>28}"
        f"{'Drop vs Full':>18}"
    )

    print(
        "-" * 100
    )

    for row in summary_rows:

        mean = row[
            "missing_nmse_db_mean"
        ]

        std = row[
            "missing_nmse_db_std"
        ]

        degradation = row[
            "degradation_vs_full_db"
        ]

        if row[
            "variant"
        ] == "full":

            degradation_text = "—"

        else:

            degradation_text = (
                f"{degradation:.4f} dB"
            )

        print(
            f"{row['display_name']:<22}"
            f"{row['parameters']:>12,}"
            f"{mean:>13.4f} ± "
            f"{std:<7.4f} dB"
            f"{degradation_text:>18}"
        )

    print()

    print(
        f"Saved:"
        f"\n{per_seed_output}"
        f"\n{summary_output}"
        f"\n{json_output}"
    )

    print("=" * 100)


if __name__ == "__main__":
    main()