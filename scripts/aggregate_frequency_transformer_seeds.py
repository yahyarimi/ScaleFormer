from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path


SEEDS = [7, 42, 123]

RESULT_DIR = Path(
    "results/csv"
)

SUMMARY_FILES = {
    seed: (
        RESULT_DIR
        / (
            "frequency_transformer_"
            f"seed{seed}_bs4_summary.json"
        )
    )
    for seed in SEEDS
}

OUTPUT_CSV = (
    RESULT_DIR
    / "frequency_transformer_validation_3seed.csv"
)

OUTPUT_SUMMARY_CSV = (
    RESULT_DIR
    / "frequency_transformer_validation_3seed_summary.csv"
)

OUTPUT_JSON = (
    RESULT_DIR
    / "frequency_transformer_validation_3seed_summary.json"
)


INTERPOLATION_FULL_DB = -25.8060
INTERPOLATION_MISSING_DB = -25.5839

CNN_FULL_MEAN_DB = -26.3596
CNN_MISSING_MEAN_DB = -26.1375

SCALEFORMER_FULL_MEAN_DB = -30.7100
SCALEFORMER_MISSING_MEAN_DB = -30.4879


def main():

    rows = []

    for seed in SEEDS:

        path = SUMMARY_FILES[
            seed
        ]

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

        if (
            data[
                "model"
            ]
            != "FrequencyTransformerBaseline"
        ):
            raise RuntimeError(
                f"Unexpected model in {path}"
            )

        if (
            int(
                data[
                    "num_parameters"
                ]
            )
            != 108_928
        ):
            raise RuntimeError(
                "Unexpected parameter count "
                f"for seed {seed}."
            )

        row = {
            "seed": seed,
            "best_epoch": int(
                data[
                    "best_epoch"
                ]
            ),
            "num_parameters": int(
                data[
                    "num_parameters"
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
            "training_seconds": float(
                data[
                    "training_seconds"
                ]
            ),
        }

        rows.append(
            row
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

    gain_vs_interp_full = (
        INTERPOLATION_FULL_DB
        - full_mean
    )

    gain_vs_interp_missing = (
        INTERPOLATION_MISSING_DB
        - missing_mean
    )

    gain_vs_cnn_full = (
        CNN_FULL_MEAN_DB
        - full_mean
    )

    gain_vs_cnn_missing = (
        CNN_MISSING_MEAN_DB
        - missing_mean
    )

    scaleformer_advantage_full = (
        full_mean
        - SCALEFORMER_FULL_MEAN_DB
    )

    scaleformer_advantage_missing = (
        missing_mean
        - SCALEFORMER_MISSING_MEAN_DB
    )

    linear_ratio = (
        10.0
        ** (
            scaleformer_advantage_missing
            / 10.0
        )
    )

    scaleformer_reduction_percent = (
        1.0
        - 1.0 / linear_ratio
    ) * 100.0

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_CSV.open(
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

    summary_row = {
        "method": (
            "Full-attention Frequency Transformer"
        ),
        "num_training_seeds": len(
            SEEDS
        ),
        "num_parameters": 108_928,
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
        "gain_vs_interpolation_full_db": (
            gain_vs_interp_full
        ),
        "gain_vs_interpolation_missing_db": (
            gain_vs_interp_missing
        ),
        "gain_vs_cnn_full_db": (
            gain_vs_cnn_full
        ),
        "gain_vs_cnn_missing_db": (
            gain_vs_cnn_missing
        ),
        "scaleformer_advantage_full_db": (
            scaleformer_advantage_full
        ),
        "scaleformer_advantage_missing_db": (
            scaleformer_advantage_missing
        ),
        "scaleformer_linear_nmse_reduction_percent": (
            scaleformer_reduction_percent
        ),
    }

    with OUTPUT_SUMMARY_CSV.open(
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
        "seeds": SEEDS,
        "per_seed": rows,
        "summary": summary_row,
    }

    with OUTPUT_JSON.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            payload,
            file,
            indent=4,
        )

    print("=" * 100)
    print(
        "FULL-ATTENTION TRANSFORMER "
        "THREE-SEED VALIDATION"
    )
    print("=" * 100)

    for row in rows:

        print(
            f"Seed {row['seed']:>3}: "
            f"full={row['full_nmse_db']:.4f} dB | "
            f"missing={row['missing_nmse_db']:.4f} dB | "
            f"best epoch={row['best_epoch']}"
        )

    print("-" * 100)

    print(
        "Full NMSE    : "
        f"{full_mean:.4f} ± "
        f"{full_std:.4f} dB"
    )

    print(
        "Missing NMSE : "
        f"{missing_mean:.4f} ± "
        f"{missing_std:.4f} dB"
    )

    print(
        "Gain vs interpolation "
        "(missing): "
        f"{gain_vs_interp_missing:.4f} dB"
    )

    print(
        "Gain vs CNN "
        "(missing): "
        f"{gain_vs_cnn_missing:.4f} dB"
    )

    print(
        "SCALE-Former advantage "
        "(missing): "
        f"{scaleformer_advantage_missing:.4f} dB"
    )

    print(
        "SCALE-Former linear NMSE "
        "reduction vs Transformer: "
        f"{scaleformer_reduction_percent:.2f}%"
    )

    print(
        f"\nPer-seed CSV:\n"
        f"{OUTPUT_CSV}"
    )

    print(
        f"\nSummary CSV:\n"
        f"{OUTPUT_SUMMARY_CSV}"
    )

    print(
        f"\nSummary JSON:\n"
        f"{OUTPUT_JSON}"
    )

    print("=" * 100)


if __name__ == "__main__":
    main()