from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any


RESULT_ROOT = Path("results/csv")
PAPER_ROOT = Path("results/paper")

PRIMARY_OUTPUT = PAPER_ROOT / "primary_results.csv"
EFFICIENCY_OUTPUT = PAPER_ROOT / "efficiency_results.csv"
MASTER_OUTPUT = PAPER_ROOT / "master_results.json"
MANIFEST_OUTPUT = PAPER_ROOT / "source_manifest.json"
AUDIT_OUTPUT = PAPER_ROOT / "paper_numbers_audit.txt"

COPY_MAP = {
    "robustness_results.csv": (
        RESULT_ROOT / "scaleformer_robustness_summary.csv"
    ),
    "robustness_results.json": (
        RESULT_ROOT / "scaleformer_robustness_summary.json"
    ),
    "scalability_results.csv": (
        RESULT_ROOT / "scaleformer_scalability_summary.csv"
    ),
    "scalability_results.json": (
        RESULT_ROOT / "scaleformer_scalability_summary.json"
    ),
    "ablation_results.csv": (
        RESULT_ROOT / "scaleformer_ablation_summary.csv"
    ),
    "ablation_results.json": (
        RESULT_ROOT / "scaleformer_ablation_summary.json"
    ),
    "interpolation_sweep.csv": (
        RESULT_ROOT / "validation_interpolation_sweep.csv"
    ),
    "training_normalization.json": (
        RESULT_ROOT / "training_normalization.json"
    ),
}

AUTHORITATIVE_SOURCES = {
    # Primary validation interpolation must come from the
    # frozen sweep row for DICHASUS-0056, random 5% observations.
    # Do NOT use validation_interpolation.csv here: that file is
    # a separate comb-10% experiment.
    "validation_interpolation": (
        RESULT_ROOT / "validation_interpolation_sweep.csv"
    ),
    "cnn_validation": (
        RESULT_ROOT / "cnn_validation_3seed_summary.json"
    ),
    "transformer_validation": (
        RESULT_ROOT
        / "frequency_transformer_validation_3seed_summary.json"
    ),
    "scaleformer_validation": (
        RESULT_ROOT / "scaleformer_validation_3seed_summary.json"
    ),
    "primary_final_test": (
        RESULT_ROOT / "final_test_0057_summary.json"
    ),
    "transformer_final_test": (
        RESULT_ROOT
        / "frequency_transformer_final_test_0057_summary.json"
    ),
    "robustness": (
        RESULT_ROOT / "scaleformer_robustness_summary.json"
    ),
    "scalability": (
        RESULT_ROOT / "scaleformer_scalability_summary.json"
    ),
    "ablation": (
        RESULT_ROOT / "scaleformer_ablation_summary.json"
    ),
    "macs": (
        RESULT_ROOT / "model_macs.json"
    ),
    "memory": (
        RESULT_ROOT / "memory_benchmark.json"
    ),
    "latency": (
        RESULT_ROOT / "latency_repeated_summary.json"
    ),
    "normalization": (
        RESULT_ROOT / "training_normalization.json"
    ),
}


PRIMARY_ROWS = [
    {
        "stage": "validation_0056",
        "method": "Linear interpolation",
        "parameters": 0,
        "full_nmse_db_mean": -25.8060,
        "full_nmse_db_std": "",
        "missing_nmse_db_mean": -25.5839,
        "missing_nmse_db_std": "",
        "source_key": "validation_interpolation",
    },
    {
        "stage": "validation_0056",
        "method": "Residual Frequency CNN",
        "parameters": 60_160,
        "full_nmse_db_mean": -26.3596,
        "full_nmse_db_std": 0.0360,
        "missing_nmse_db_mean": -26.1375,
        "missing_nmse_db_std": 0.0360,
        "source_key": "cnn_validation",
    },
    {
        "stage": "validation_0056",
        "method": "Full-Attention Transformer",
        "parameters": 108_928,
        "full_nmse_db_mean": -28.8937,
        "full_nmse_db_std": 0.0631,
        "missing_nmse_db_mean": -28.6716,
        "missing_nmse_db_std": 0.0631,
        "source_key": "transformer_validation",
    },
    {
        "stage": "validation_0056",
        "method": "SCALE-Former",
        "parameters": 100_034,
        "full_nmse_db_mean": -30.7100,
        "full_nmse_db_std": 0.0837,
        "missing_nmse_db_mean": -30.4879,
        "missing_nmse_db_std": 0.0837,
        "source_key": "scaleformer_validation",
    },
    {
        "stage": "test_0057",
        "method": "Linear interpolation",
        "parameters": 0,
        "full_nmse_db_mean": -25.7545,
        "full_nmse_db_std": "",
        "missing_nmse_db_mean": -25.5322,
        "missing_nmse_db_std": "",
        "source_key": "primary_final_test",
    },
    {
        "stage": "test_0057",
        "method": "Residual Frequency CNN",
        "parameters": 60_160,
        "full_nmse_db_mean": -26.2864,
        "full_nmse_db_std": 0.0374,
        "missing_nmse_db_mean": -26.0641,
        "missing_nmse_db_std": 0.0374,
        "source_key": "primary_final_test",
    },
    {
        "stage": "test_0057",
        "method": "Full-Attention Transformer",
        "parameters": 108_928,
        "full_nmse_db_mean": -28.6934,
        "full_nmse_db_std": 0.0489,
        "missing_nmse_db_mean": -28.4711,
        "missing_nmse_db_std": 0.0489,
        "source_key": "transformer_final_test",
    },
    {
        "stage": "test_0057",
        "method": "SCALE-Former",
        "parameters": 100_034,
        "full_nmse_db_mean": -30.4055,
        "full_nmse_db_std": 0.0772,
        "missing_nmse_db_mean": -30.1832,
        "missing_nmse_db_std": 0.0772,
        "source_key": "primary_final_test",
    },
]


EFFICIENCY_ROWS = [
    {
        "method": "Linear interpolation",
        "parameters": 0,
        "core_mmacs": "",
        "approximate_gflops": "",
        "peak_gpu_memory_mib": 1.7222,
        "e2e_latency_mean_ms": 1.2866,
        "e2e_latency_sd_ms": 0.5674,
    },
    {
        "method": "Residual Frequency CNN",
        "parameters": 60_160,
        "core_mmacs": 60.8829,
        "approximate_gflops": 0.1218,
        "peak_gpu_memory_mib": 2.0762,
        "e2e_latency_mean_ms": 3.0208,
        "e2e_latency_sd_ms": 1.3494,
    },
    {
        "method": "Full-Attention Transformer",
        "parameters": 108_928,
        "core_mmacs": 511.7706,
        "approximate_gflops": 1.0235,
        "peak_gpu_memory_mib": 12.2124,
        "e2e_latency_mean_ms": 3.0617,
        "e2e_latency_sd_ms": 0.2088,
    },
    {
        "method": "SCALE-Former",
        "parameters": 100_034,
        "core_mmacs": 1256.1285,
        "approximate_gflops": 2.5123,
        "peak_gpu_memory_mib": 40.1797,
        "e2e_latency_mean_ms": 12.3771,
        "e2e_latency_sd_ms": 0.1562,
    },
]


EXPECTED_ABLATION_VALUES = [
    100_034,
    -30.4879,
    0.0837,
    23_090,
    -26.5498,
    0.0382,
    3.9382,
    82_178,
    -25.9119,
    0.0876,
    4.5761,
    97_538,
    -27.2467,
    0.2109,
    3.2412,
]

EXPECTED_ROBUSTNESS_VALUES = [
    -11.3937,
    -18.9724,
    -30.4879,
    -34.4184,
    -34.8579,
    -34.6292,
    0.4178,
    1.2607,
    2.7281,
    4.9041,
    3.6117,
    1.0673,
    2.5098,
    -0.1553,
]

EXPECTED_SCALABILITY_VALUES = [
    -13.8151,
    -22.1988,
    -30.4499,
    -13.3912,
    -21.6723,
    -30.0387,
    -13.8507,
    -22.1483,
    -30.4879,
]


def load_json(path: Path) -> Any:
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            chunk = file.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def collect_numbers(value: Any) -> list[float]:
    numbers = []

    if isinstance(
        value,
        bool,
    ):
        return numbers

    if isinstance(
        value,
        (int, float),
    ):
        numbers.append(
            float(value)
        )

    elif isinstance(
        value,
        dict,
    ):
        for item in value.values():
            numbers.extend(
                collect_numbers(
                    item
                )
            )

    elif isinstance(
        value,
        list,
    ):
        for item in value:
            numbers.extend(
                collect_numbers(
                    item
                )
            )

    return numbers


def file_numbers(path: Path) -> list[float]:
    suffix = path.suffix.lower()

    if suffix == ".json":
        return collect_numbers(
            load_json(path)
        )

    if suffix == ".csv":
        numbers = []

        with path.open(
            "r",
            encoding="utf-8",
            newline="",
        ) as file:

            reader = csv.reader(
                file
            )

            for row in reader:
                for cell in row:
                    try:
                        numbers.append(
                            float(cell)
                        )
                    except ValueError:
                        pass

        return numbers

    raise ValueError(
        f"Unsupported source type: {path}"
    )


def assert_number_present(
    source_key: str,
    expected: float,
    tolerance: float = 5e-4,
) -> None:

    path = AUTHORITATIVE_SOURCES[
        source_key
    ]

    numbers = file_numbers(
        path
    )

    if not any(
        math.isclose(
            number,
            float(expected),
            rel_tol=0.0,
            abs_tol=tolerance,
        )
        for number in numbers
    ):
        raise RuntimeError(
            f"Audit failed: {expected} was not "
            f"found in {path} within tolerance "
            f"{tolerance}."
        )


def audit_primary_interpolation_row() -> dict:
    """
    Recover and verify the exact primary interpolation row:
    DICHASUS-0056, random pattern, requested density 0.05,
    51 observed subcarriers, 5,220 samples.
    """

    path = AUTHORITATIVE_SOURCES[
        "validation_interpolation"
    ]

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:

        reader = csv.DictReader(
            file
        )

        matches = []

        for row in reader:

            if (
                row.get("run") == "0056"
                and row.get("pattern") == "random"
                and math.isclose(
                    float(
                        row[
                            "requested_density"
                        ]
                    ),
                    0.05,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                and int(
                    row[
                        "observed_subcarriers"
                    ]
                ) == 51
                and int(
                    row[
                        "num_samples"
                    ]
                ) == 5_220
            ):

                matches.append(
                    row
                )

    if len(matches) != 1:

        raise RuntimeError(
            "Expected exactly one primary interpolation "
            "row for 0056/random/5%/51 observed/5220 "
            f"samples in {path}, found {len(matches)}."
        )

    row = matches[
        0
    ]

    full_db = float(
        row[
            "full_nmse_db"
        ]
    )

    missing_db = float(
        row[
            "missing_nmse_db"
        ]
    )

    if not math.isclose(
        full_db,
        -25.80597845163497,
        rel_tol=0.0,
        abs_tol=5e-10,
    ):

        raise RuntimeError(
            "Primary interpolation full NMSE does not "
            "match the frozen 0056 random-5% result: "
            f"{full_db}."
        )

    if not math.isclose(
        missing_db,
        -25.583870695451377,
        rel_tol=0.0,
        abs_tol=5e-10,
    ):

        raise RuntimeError(
            "Primary interpolation missing NMSE does not "
            "match the frozen 0056 random-5% result: "
            f"{missing_db}."
        )

    return {
        "full_nmse_db": full_db,
        "missing_nmse_db": missing_db,
    }


def write_csv(
    path: Path,
    rows: list[dict],
) -> None:

    if not rows:
        raise RuntimeError(
            f"No rows for {path}."
        )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(
                rows[
                    0
                ].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            rows
        )


def linear_nmse_reduction_percent(
    baseline_db: float,
    proposed_db: float,
) -> float:

    ratio = 10.0 ** (
        (
            proposed_db
            - baseline_db
        )
        / 10.0
    )

    return (
        1.0 - ratio
    ) * 100.0


def main():

    print("=" * 100)
    print(
        "BUILD SCALE-FORMER PAPER RESULTS PACKAGE"
    )
    print("=" * 100)

    missing_files = [
        str(path)
        for path in (
            list(
                AUTHORITATIVE_SOURCES.values()
            )
            + list(
                COPY_MAP.values()
            )
        )
        if not path.exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Required authoritative source files "
            "are missing:\n"
            + "\n".join(
                missing_files
            )
        )

    PAPER_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    audit_lines = []

    audit_lines.append(
        "SCALE-FORMER PAPER NUMBERS AUDIT"
    )

    audit_lines.append(
        "=" * 80
    )

    audit_lines.append(
        ""
    )

    primary_interpolation = (
        audit_primary_interpolation_row()
    )

    audit_lines.append(
        "Primary interpolation protocol row:"
    )

    audit_lines.append(
        "PASS  0056 | random | 5% | "
        "51 observed subcarriers | 5,220 samples | "
        f"full={primary_interpolation['full_nmse_db']:.12f} dB | "
        f"missing={primary_interpolation['missing_nmse_db']:.12f} dB"
    )

    audit_lines.append("")
    audit_lines.append(
        "Primary validation/test table:"
    )

    for row in PRIMARY_ROWS:

        source_key = row[
            "source_key"
        ]

        for key in [
            "full_nmse_db_mean",
            "missing_nmse_db_mean",
        ]:

            assert_number_present(
                source_key,
                float(
                    row[
                        key
                    ]
                ),
            )

        for key in [
            "full_nmse_db_std",
            "missing_nmse_db_std",
        ]:

            value = row[
                key
            ]

            if value != "":

                assert_number_present(
                    source_key,
                    float(value),
                )

        audit_lines.append(
            "PASS  "
            f"{row['stage']} | "
            f"{row['method']}"
        )

    write_csv(
        PRIMARY_OUTPUT,
        PRIMARY_ROWS,
    )

    audit_lines.append("")
    audit_lines.append(
        "Efficiency table:"
    )

    for row in EFFICIENCY_ROWS:

        method = row[
            "method"
        ]

        if method != "Linear interpolation":

            assert_number_present(
                "macs",
                float(
                    row[
                        "core_mmacs"
                    ]
                ),
            )

            assert_number_present(
                "macs",
                float(
                    row[
                        "approximate_gflops"
                    ]
                ),
            )

        assert_number_present(
            "memory",
            float(
                row[
                    "peak_gpu_memory_mib"
                ]
            ),
        )

        assert_number_present(
            "latency",
            float(
                row[
                    "e2e_latency_mean_ms"
                ]
            ),
        )

        assert_number_present(
            "latency",
            float(
                row[
                    "e2e_latency_sd_ms"
                ]
            ),
        )

        audit_lines.append(
            "PASS  "
            f"{method}"
        )

    write_csv(
        EFFICIENCY_OUTPUT,
        EFFICIENCY_ROWS,
    )

    audit_lines.append("")
    audit_lines.append(
        "Ablation headline values:"
    )

    for value in EXPECTED_ABLATION_VALUES:
        assert_number_present(
            "ablation",
            value,
        )

    audit_lines.append(
        "PASS  ablation summary values"
    )

    audit_lines.append("")
    audit_lines.append(
        "Robustness headline values:"
    )

    # The block-stress difference may be stored as +0.1553
    # or -0.1553 depending on which direction a source labels
    # the comparison. Require its magnitude rather than sign.
    for value in EXPECTED_ROBUSTNESS_VALUES:

        if math.isclose(
            abs(value),
            0.1553,
            rel_tol=0.0,
            abs_tol=1e-8,
        ):

            try:
                assert_number_present(
                    "robustness",
                    value,
                )

            except RuntimeError:
                assert_number_present(
                    "robustness",
                    -value,
                )

        else:

            assert_number_present(
                "robustness",
                value,
            )

    audit_lines.append(
        "PASS  robustness summary values"
    )

    audit_lines.append("")
    audit_lines.append(
        "Scalability headline values:"
    )

    for value in EXPECTED_SCALABILITY_VALUES:
        assert_number_present(
            "scalability",
            value,
        )

    audit_lines.append(
        "PASS  scalability summary values"
    )

    for (
        destination_name,
        source,
    ) in COPY_MAP.items():

        shutil.copy2(
            source,
            PAPER_ROOT
            / destination_name,
        )

    test_transformer = next(
        row
        for row in PRIMARY_ROWS
        if (
            row[
                "stage"
            ]
            == "test_0057"
            and row[
                "method"
            ]
            == "Full-Attention Transformer"
        )
    )

    test_scaleformer = next(
        row
        for row in PRIMARY_ROWS
        if (
            row[
                "stage"
            ]
            == "test_0057"
            and row[
                "method"
            ]
            == "SCALE-Former"
        )
    )

    test_cnn = next(
        row
        for row in PRIMARY_ROWS
        if (
            row[
                "stage"
            ]
            == "test_0057"
            and row[
                "method"
            ]
            == "Residual Frequency CNN"
        )
    )

    test_interpolation = next(
        row
        for row in PRIMARY_ROWS
        if (
            row[
                "stage"
            ]
            == "test_0057"
            and row[
                "method"
            ]
            == "Linear interpolation"
        )
    )

    sf_missing = float(
        test_scaleformer[
            "missing_nmse_db_mean"
        ]
    )

    transformer_missing = float(
        test_transformer[
            "missing_nmse_db_mean"
        ]
    )

    cnn_missing = float(
        test_cnn[
            "missing_nmse_db_mean"
        ]
    )

    interpolation_missing = float(
        test_interpolation[
            "missing_nmse_db_mean"
        ]
    )

    sf_vs_transformer_db = (
        transformer_missing
        - sf_missing
    )

    sf_vs_cnn_db = (
        cnn_missing
        - sf_missing
    )

    sf_vs_interpolation_db = (
        interpolation_missing
        - sf_missing
    )

    transformer_eff = next(
        row
        for row in EFFICIENCY_ROWS
        if row[
            "method"
        ]
        == "Full-Attention Transformer"
    )

    sf_eff = next(
        row
        for row in EFFICIENCY_ROWS
        if row[
            "method"
        ]
        == "SCALE-Former"
    )

    headline_claims = {
        "test_0057": {
            "scaleformer_vs_full_transformer_missing_nmse_gain_db": (
                sf_vs_transformer_db
            ),
            "scaleformer_vs_full_transformer_linear_nmse_reduction_percent": (
                linear_nmse_reduction_percent(
                    transformer_missing,
                    sf_missing,
                )
            ),
            "scaleformer_vs_cnn_missing_nmse_gain_db": (
                sf_vs_cnn_db
            ),
            "scaleformer_vs_cnn_linear_nmse_reduction_percent": (
                linear_nmse_reduction_percent(
                    cnn_missing,
                    sf_missing,
                )
            ),
            "scaleformer_vs_interpolation_missing_nmse_gain_db": (
                sf_vs_interpolation_db
            ),
            "scaleformer_vs_interpolation_linear_nmse_reduction_percent": (
                linear_nmse_reduction_percent(
                    interpolation_missing,
                    sf_missing,
                )
            ),
        },
        "efficiency_tradeoff_vs_full_transformer": {
            "scaleformer_parameter_ratio": (
                sf_eff[
                    "parameters"
                ]
                / transformer_eff[
                    "parameters"
                ]
            ),
            "transformer_parameter_excess_percent_vs_scaleformer": (
                (
                    transformer_eff[
                        "parameters"
                    ]
                    / sf_eff[
                        "parameters"
                    ]
                )
                - 1.0
            ) * 100.0,
            "scaleformer_mmac_ratio": (
                sf_eff[
                    "core_mmacs"
                ]
                / transformer_eff[
                    "core_mmacs"
                ]
            ),
            "scaleformer_peak_memory_ratio": (
                sf_eff[
                    "peak_gpu_memory_mib"
                ]
                / transformer_eff[
                    "peak_gpu_memory_mib"
                ]
            ),
            "scaleformer_e2e_latency_ratio": (
                sf_eff[
                    "e2e_latency_mean_ms"
                ]
                / transformer_eff[
                    "e2e_latency_mean_ms"
                ]
            ),
        },
    }

    master = {
        "protocol": {
            "dataset": (
                "DICHASUS-005x real measured CSI"
            ),
            "training_runs": [
                "0053",
                "0054",
                "0055",
            ],
            "validation_run": "0056",
            "final_test_run": "0057",
            "primary_pattern": "random",
            "primary_density": 0.05,
            "primary_observed_subcarriers": 51,
            "validation_and_test_mask_seed": 42,
            "learned_model_seeds": [
                7,
                42,
                123,
            ],
            "metric_scale": (
                "denormalized measured CSI"
            ),
            "primary_metric": (
                "dataset-level missing-region NMSE"
            ),
        },
        "primary_results": PRIMARY_ROWS,
        "efficiency_results": EFFICIENCY_ROWS,
        "headline_claims": (
            headline_claims
        ),
        "detailed_result_files": {
            key: str(
                PAPER_ROOT
                / name
            )
            for name, key in [
                (
                    "robustness_results.csv",
                    "robustness_csv",
                ),
                (
                    "robustness_results.json",
                    "robustness_json",
                ),
                (
                    "scalability_results.csv",
                    "scalability_csv",
                ),
                (
                    "scalability_results.json",
                    "scalability_json",
                ),
                (
                    "ablation_results.csv",
                    "ablation_csv",
                ),
                (
                    "ablation_results.json",
                    "ablation_json",
                ),
                (
                    "interpolation_sweep.csv",
                    "interpolation_sweep_csv",
                ),
            ]
        },
        "reporting_guardrails": [
            (
                "Use dense measured reference CSI; "
                "do not call it physical ground truth."
            ),
            (
                "The sparse masks are controlled "
                "observation patterns, not DICHASUS "
                "operational pilot placements."
            ),
            (
                "Do not claim SCALE-Former is lighter "
                "or faster than the full-attention "
                "Transformer or CNN."
            ),
            (
                "Report repeated five-round E2E latency "
                "as the authoritative latency result; "
                "single-run latency is diagnostic."
            ),
            (
                "Block masking is an out-of-distribution "
                "stress case and SCALE-Former does not "
                "outperform interpolation there."
            ),
            (
                "Scalability results demonstrate transfer "
                "across tested antenna counts and frequency "
                "resolutions without retraining, not arbitrary "
                "bandwidth/carrier or new-environment generalization."
            ),
            (
                "No BER claim should be made without a "
                "proper communications link simulation."
            ),
        ],
    }

    with MASTER_OUTPUT.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            master,
            file,
            indent=4,
        )

    manifest_entries = []

    for (
        key,
        source,
    ) in AUTHORITATIVE_SOURCES.items():

        manifest_entries.append(
            {
                "key": key,
                "path": str(source),
                "bytes": source.stat().st_size,
                "sha256": sha256(
                    source
                ),
            }
        )

    manifest = {
        "authoritative_sources": (
            manifest_entries
        ),
        "excluded_from_headline_tables": [
            (
                "results/csv/validation_interpolation.csv "
                "(separate 0056 comb-10% interpolation run; "
                "not the primary random-5% condition)"
            ),
            (
                "results/csv/latency_benchmark.csv "
                "(single-run diagnostic latency)"
            ),
            (
                "results/csv/latency_benchmark.json "
                "(single-run diagnostic latency)"
            ),
            (
                "results/csv/scaleformer_seed42_bs4_smoke_summary.json"
            ),
            (
                "results/csv/scaleformer_seed42_bs4_smoke_training.csv"
            ),
            (
                "per-epoch training CSVs are provenance/supporting "
                "artifacts rather than headline paper tables"
            ),
        ],
        "repository_housekeeping_review": [
            (
                "scripts/train_frequency_transformer1.py appears "
                "to be a duplicate/alternate trainer and should "
                "be manually reviewed before deletion; this script "
                "does not modify source files."
            ),
        ],
    }

    with MANIFEST_OUTPUT.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            manifest,
            file,
            indent=4,
        )

    audit_lines.append("")
    audit_lines.append(
        "Headline claims computed from audited source values:"
    )

    audit_lines.append(
        "PASS  SCALE-Former vs Transformer test missing NMSE "
        f"gain = {sf_vs_transformer_db:.4f} dB"
    )

    audit_lines.append(
        "PASS  SCALE-Former vs Transformer linear NMSE "
        "reduction = "
        f"{headline_claims['test_0057']['scaleformer_vs_full_transformer_linear_nmse_reduction_percent']:.2f}%"
    )

    audit_lines.append(
        "PASS  SCALE-Former vs CNN test missing NMSE "
        f"gain = {sf_vs_cnn_db:.4f} dB"
    )

    audit_lines.append(
        "PASS  SCALE-Former vs interpolation test missing NMSE "
        f"gain = {sf_vs_interpolation_db:.4f} dB"
    )

    audit_lines.append("")
    audit_lines.append(
        "Efficiency trade-off vs full-attention Transformer:"
    )

    audit_lines.append(
        "INFO  Transformer has "
        f"{headline_claims['efficiency_tradeoff_vs_full_transformer']['transformer_parameter_excess_percent_vs_scaleformer']:.2f}% "
        "more trainable parameters than SCALE-Former."
    )

    audit_lines.append(
        "INFO  SCALE-Former core-MAC ratio = "
        f"{headline_claims['efficiency_tradeoff_vs_full_transformer']['scaleformer_mmac_ratio']:.3f}x"
    )

    audit_lines.append(
        "INFO  SCALE-Former peak-memory ratio = "
        f"{headline_claims['efficiency_tradeoff_vs_full_transformer']['scaleformer_peak_memory_ratio']:.3f}x"
    )

    audit_lines.append(
        "INFO  SCALE-Former E2E-latency ratio = "
        f"{headline_claims['efficiency_tradeoff_vs_full_transformer']['scaleformer_e2e_latency_ratio']:.3f}x"
    )

    audit_lines.append("")
    audit_lines.append(
        "Excluded/superseded headline artifacts:"
    )

    audit_lines.append(
        "INFO  validation_interpolation.csv is a separate comb-10% run "
        "and is not the primary random-5% interpolation source."
    )

    audit_lines.append(
        "INFO  latency_benchmark.csv/json are single-run diagnostic outputs."
    )

    audit_lines.append(
        "INFO  scaleformer_seed42_bs4_smoke_* are smoke-test outputs."
    )

    audit_lines.append("")
    audit_lines.append(
        "Repository housekeeping:"
    )

    audit_lines.append(
        "REVIEW scripts/train_frequency_transformer1.py manually; "
        "it appears to be an alternate/duplicate trainer."
    )

    audit_lines.append("")
    audit_lines.append(
        "FINAL STATUS: ALL PAPER-NUMBER AUDITS PASSED"
    )

    AUDIT_OUTPUT.write_text(
        "\n".join(
            audit_lines
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print(
        "Created:"
    )

    for output in [
        PRIMARY_OUTPUT,
        EFFICIENCY_OUTPUT,
        PAPER_ROOT / "robustness_results.csv",
        PAPER_ROOT / "scalability_results.csv",
        PAPER_ROOT / "ablation_results.csv",
        MASTER_OUTPUT,
        MANIFEST_OUTPUT,
        AUDIT_OUTPUT,
    ]:

        print(
            f"  {output}"
        )

    print()
    print(
        "Headline test result:"
    )

    print(
        "  SCALE-Former vs Full-Attention Transformer "
        f"= {sf_vs_transformer_db:.4f} dB"
    )

    print(
        "  Linear NMSE reduction "
        f"= "
        f"{headline_claims['test_0057']['scaleformer_vs_full_transformer_linear_nmse_reduction_percent']:.2f}%"
    )

    print()
    print(
        "ALL PAPER-NUMBER AUDITS PASSED"
    )

    print("=" * 100)


if __name__ == "__main__":
    main()
