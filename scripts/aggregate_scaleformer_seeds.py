from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path


SEEDS = [7, 42, 123]

RESULT_ROOT = Path(
    "results/csv"
)

OUTPUT_CSV = (
    RESULT_ROOT
    / "scaleformer_validation_3seed.csv"
)

OUTPUT_JSON = (
    RESULT_ROOT
    / "scaleformer_validation_3seed_summary.json"
)

INTERPOLATION_FULL_DB = -25.8060
INTERPOLATION_MISSING_DB = -25.5839

# Current CNN result. This comparison remains
# provisional until CNN seeds 7 and 123 are trained.
CNN_SEED42_FULL_DB = -26.3391
CNN_SEED42_MISSING_DB = -26.1170


def main():

    rows = []

    full_values = []
    missing_values = []

    for seed in SEEDS:

        path = RESULT_ROOT / (
            f"scaleformer_seed{seed}"
            f"_bs4_summary.json"
        )

        if not path.exists():
            raise FileNotFoundError(
                path
            )

        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            result = json.load(
                file
            )

        full_db = float(
            result[
                "best_val_full_nmse_db"
            ]
        )

        missing_db = float(
            result[
                "best_val_missing_nmse_db"
            ]
        )

        best_epoch = int(
            result[
                "best_epoch"
            ]
        )

        full_values.append(
            full_db
        )

        missing_values.append(
            missing_db
        )

        rows.append(
            {
                "seed": seed,
                "best_epoch": best_epoch,
                "full_nmse_db": full_db,
                "missing_nmse_db": (
                    missing_db
                ),
            }
        )

    full_mean = statistics.mean(
        full_values
    )

    missing_mean = statistics.mean(
        missing_values
    )

    full_std = statistics.stdev(
        full_values
    )

    missing_std = statistics.stdev(
        missing_values
    )

    summary = {
        "model": "ScaleFormer",
        "split": "validation",
        "run": "0056",
        "pattern": "random",
        "requested_density": 0.05,
        "observed_subcarriers": 51,
        "seeds": SEEDS,
        "num_seeds": len(SEEDS),
        "full_nmse_db_mean": (
            full_mean
        ),
        "full_nmse_db_sample_std": (
            full_std
        ),
        "missing_nmse_db_mean": (
            missing_mean
        ),
        "missing_nmse_db_sample_std": (
            missing_std
        ),
        "interpolation_full_nmse_db": (
            INTERPOLATION_FULL_DB
        ),
        "interpolation_missing_nmse_db": (
            INTERPOLATION_MISSING_DB
        ),
        "mean_gain_vs_interpolation_full_db": (
            INTERPOLATION_FULL_DB
            - full_mean
        ),
        "mean_gain_vs_interpolation_missing_db": (
            INTERPOLATION_MISSING_DB
            - missing_mean
        ),
        "cnn_seed42_full_nmse_db": (
            CNN_SEED42_FULL_DB
        ),
        "cnn_seed42_missing_nmse_db": (
            CNN_SEED42_MISSING_DB
        ),
        "provisional_mean_gain_vs_cnn_full_db": (
            CNN_SEED42_FULL_DB
            - full_mean
        ),
        "provisional_mean_gain_vs_cnn_missing_db": (
            CNN_SEED42_MISSING_DB
            - missing_mean
        ),
    }

    with OUTPUT_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "seed",
                "best_epoch",
                "full_nmse_db",
                "missing_nmse_db",
            ],
        )

        writer.writeheader()
        writer.writerows(
            rows
        )

    with OUTPUT_JSON.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
        )

    print("=" * 72)
    print(
        "SCALE-FORMER THREE-SEED "
        "VALIDATION SUMMARY"
    )
    print("=" * 72)

    for row in rows:
        print(
            f"Seed {row['seed']:>3}: "
            f"full "
            f"{row['full_nmse_db']:.4f} dB | "
            f"missing "
            f"{row['missing_nmse_db']:.4f} dB"
        )

    print()

    print(
        f"Full NMSE    : "
        f"{full_mean:.4f} "
        f"± {full_std:.4f} dB"
    )

    print(
        f"Missing NMSE : "
        f"{missing_mean:.4f} "
        f"± {missing_std:.4f} dB"
    )

    print(
        f"Gain vs interpolation "
        f"(missing): "
        f"{INTERPOLATION_MISSING_DB - missing_mean:.4f} dB"
    )

    print(
        f"Provisional gain vs CNN "
        f"(missing): "
        f"{CNN_SEED42_MISSING_DB - missing_mean:.4f} dB"
    )

    print(
        f"\nSaved:"
        f"\n{OUTPUT_CSV}"
        f"\n{OUTPUT_JSON}"
    )

    print("=" * 72)


if __name__ == "__main__":
    main()