from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METHOD_ORDER = [
    "Linear interpolation",
    "Residual Frequency CNN",
    "Full-Attention Transformer",
    "SCALE-Former",
]

METHOD_SHORT = {
    "Linear interpolation": "Interpolation",
    "Residual Frequency CNN": "CNN",
    "Full-Attention Transformer": "Full Transformer",
    "SCALE-Former": "SCALE-Former",
}

INPUT_FILES = {
    "primary": "primary_results.csv",
    "efficiency": "efficiency_results.csv",
    "robustness": "robustness_results.csv",
    "scalability": "scalability_results.csv",
    "ablation": "ablation_results.csv",
}


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        return list(
            csv.DictReader(file)
        )


def as_float(
    row: dict[str, str],
    key: str,
) -> float:

    value = row.get(
        key,
        "",
    )

    if value in (
        "",
        None,
    ):
        return float(
            "nan"
        )

    return float(
        value
    )


def require_columns(
    rows: list[dict[str, str]],
    required: set[str],
    name: str,
) -> None:

    if not rows:
        raise RuntimeError(
            f"{name} is empty."
        )

    available = set(
        rows[
            0
        ].keys()
    )

    missing = sorted(
        required
        - available
    )

    if missing:
        raise RuntimeError(
            f"{name} is missing columns: "
            + ", ".join(
                missing
            )
        )


def file_sha256(
    path: Path,
) -> str:

    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as file:

        while True:

            chunk = file.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


def save_figure(
    fig,
    output_dir: Path,
    stem: str,
) -> list[str]:

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    png = (
        output_dir
        / f"{stem}.png"
    )

    pdf = (
        output_dir
        / f"{stem}.pdf"
    )

    fig.savefig(
        png,
        dpi=600,
        bbox_inches="tight",
    )

    fig.savefig(
        pdf,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    return [
        str(
            png
        ),
        str(
            pdf
        ),
    ]


def fig_primary_test(
    rows: list[dict[str, str]],
    output_dir: Path,
) -> list[str]:

    test = {
        row[
            "method"
        ]: row
        for row in rows
        if row[
            "stage"
        ] == "test_0057"
    }

    for method in METHOD_ORDER:
        if method not in test:
            raise RuntimeError(
                "Primary-results table "
                f"is missing {method}."
            )

    fig, ax = plt.subplots(
        figsize=(
            7.2,
            4.7,
        ),
        constrained_layout=True,
    )

    markers = [
        "o",
        "s",
        "^",
        "D",
    ]

    means = []

    for (
        index,
        method,
        marker,
    ) in zip(
        range(
            len(
                METHOD_ORDER
            )
        ),
        METHOD_ORDER,
        markers,
    ):

        row = test[
            method
        ]

        mean = as_float(
            row,
            "missing_nmse_db_mean",
        )

        std = as_float(
            row,
            "missing_nmse_db_std",
        )

        means.append(
            mean
        )

        ax.errorbar(
            [index],
            [mean],
            yerr=(
                None
                if math.isnan(
                    std
                )
                else std
            ),
            fmt=marker,
            markersize=8,
            capsize=4,
            linewidth=1.4,
        )

        value_text = (
            f"{mean:.2f}"
            if math.isnan(
                std
            )
            else (
                f"{mean:.2f}\n"
                f"± {std:.2f}"
            )
        )

        ax.annotate(
            value_text,
            xy=(
                index,
                mean,
            ),
            xytext=(
                0,
                9,
            ),
            textcoords=(
                "offset points"
            ),
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_xticks(
        range(
            len(
                METHOD_ORDER
            )
        ),
        [
            METHOD_SHORT[
                method
            ]
            for method
            in METHOD_ORDER
        ],
    )

    ax.set_ylabel(
        "Missing-region NMSE (dB)"
    )

    ax.set_title(
        "Held-out DICHASUS-0057 reconstruction performance"
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    ax.text(
        0.01,
        0.02,
        "Lower is better",
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
    )

    span = max(
        max(
            means
        )
        - min(
            means
        ),
        1.0,
    )

    ax.set_ylim(
        min(
            means
        )
        - 0.12
        * span,
        max(
            means
        )
        + 0.25
        * span,
    )

    return save_figure(
        fig,
        output_dir,
        "fig02_primary_test_comparison",
    )


def fig_random_density(
    rows: list[dict[str, str]],
    output_dir: Path,
) -> list[str]:

    random_rows = [
        row
        for row in rows
        if row[
            "pattern"
        ] == "random"
    ]

    random_rows.sort(
        key=lambda row: as_float(
            row,
            "requested_density",
        )
    )

    densities = np.array(
        [
            100.0
            * as_float(
                row,
                "requested_density",
            )
            for row in random_rows
        ]
    )

    interpolation = np.array(
        [
            as_float(
                row,
                "interpolation_missing_nmse_db",
            )
            for row
            in random_rows
        ]
    )

    scaleformer = np.array(
        [
            as_float(
                row,
                "scaleformer_missing_nmse_db_mean",
            )
            for row
            in random_rows
        ]
    )

    sf_std = np.array(
        [
            as_float(
                row,
                "scaleformer_missing_nmse_db_std",
            )
            for row
            in random_rows
        ]
    )

    fig, ax = plt.subplots(
        figsize=(
            7.2,
            4.7,
        ),
        constrained_layout=True,
    )

    ax.plot(
        densities,
        interpolation,
        marker="o",
        linewidth=1.6,
        label="Interpolation",
    )

    ax.errorbar(
        densities,
        scaleformer,
        yerr=sf_std,
        marker="s",
        linewidth=1.6,
        capsize=4,
        label="SCALE-Former",
    )

    ax.set_xlabel(
        "Observed frequency coefficients (%)"
    )

    ax.set_ylabel(
        "Missing-region NMSE (dB)"
    )

    ax.set_title(
        "Robustness to random observation density"
    )

    ax.set_xticks(
        densities
    )

    ax.grid(
        alpha=0.25,
    )

    ax.legend(
        frameon=False,
    )

    ax.text(
        0.01,
        0.02,
        "Lower is better",
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
    )

    return save_figure(
        fig,
        output_dir,
        "fig03a_random_density_robustness",
    )



def fig_mask_patterns(
    rows: list[dict[str, str]],
    output_dir: Path,
) -> list[str]:

    order = [
        "random_5pct",
        "comb_5pct",
        "block_5pct",
    ]

    labels = [
        "Random",
        "Comb",
        "Block",
    ]

    by_condition = {
        row["condition"]: row
        for row in rows
    }

    for condition in order:
        if condition not in by_condition:
            raise RuntimeError(
                f"Robustness table is missing {condition}."
            )

    x = np.arange(
        len(order),
        dtype=float,
    )

    gains = np.array(
        [
            as_float(
                by_condition[condition],
                "mean_gain_vs_interpolation_missing_db",
            )
            for condition in order
        ]
    )

    gain_std = np.array(
        [
            as_float(
                by_condition[condition],
                "scaleformer_missing_nmse_db_std",
            )
            for condition in order
        ]
    )

    fig, ax = plt.subplots(
        figsize=(7.2, 4.7),
        constrained_layout=True,
    )

    markers = ["o", "s", "^"]

    for index, marker in zip(
        x,
        markers,
    ):

        i = int(index)
        gain = gains[i]

        ax.errorbar(
            [index],
            [gain],
            yerr=[gain_std[i]],
            fmt=marker,
            markersize=9,
            capsize=4,
            linewidth=1.4,
        )

        ax.annotate(
            f"{gain:+.2f} dB",
            xy=(index, gain),
            xytext=(0, 10 if gain >= 0 else -18),
            textcoords="offset points",
            ha="center",
            va="bottom" if gain >= 0 else "top",
            fontsize=8,
        )

    ax.axhline(
        0.0,
        linewidth=1.0,
        linestyle="--",
        alpha=0.7,
    )

    ax.set_xticks(
        x,
        labels,
    )

    ax.set_ylabel(
        "Gain over interpolation (dB)"
    )

    ax.set_title(
        "Observation-pattern stress test at 5% density"
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    ax.text(
        0.01,
        0.02,
        "Positive is better; block masking is an OOD stress case",
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
    )

    ax.set_ylim(
        min(-0.8, float(gains.min() - 0.8)),
        max(5.6, float(gains.max() + 0.8)),
    )

    return save_figure(
        fig,
        output_dir,
        "fig03b_mask_pattern_robustness",
    )


def fig_scalability(
    rows: list[dict[str, str]],
    output_dir: Path,
) -> list[str]:

    antennas = sorted(
        {
            int(row["num_antennas"])
            for row in rows
        }
    )

    frequencies = sorted(
        {
            int(row["num_subcarriers"])
            for row in rows
        }
    )

    lookup = {
        (
            int(row["num_antennas"]),
            int(row["num_subcarriers"]),
        ): row
        for row in rows
    }

    matrix = np.empty(
        (len(antennas), len(frequencies)),
        dtype=float,
    )

    std_matrix = np.empty_like(matrix)

    for i, antenna_count in enumerate(antennas):
        for j, frequency_count in enumerate(frequencies):
            key = (antenna_count, frequency_count)

            if key not in lookup:
                raise RuntimeError(
                    "Incomplete scalability grid at "
                    f"B={antenna_count}, F={frequency_count}."
                )

            row = lookup[key]

            matrix[i, j] = as_float(
                row,
                "scaleformer_missing_nmse_db_mean",
            )

            std_matrix[i, j] = as_float(
                row,
                "scaleformer_missing_nmse_db_std",
            )

    fig, ax = plt.subplots(
        figsize=(7.2, 4.9),
        constrained_layout=True,
    )

    image = ax.imshow(
        matrix,
        aspect="auto",
    )

    colorbar = fig.colorbar(
        image,
        ax=ax,
    )

    colorbar.set_label(
        "Missing-region NMSE (dB)"
    )

    ax.set_xticks(
        np.arange(len(frequencies)),
        [str(frequency) for frequency in frequencies],
    )

    ax.set_yticks(
        np.arange(len(antennas)),
        [str(antenna) for antenna in antennas],
    )

    ax.set_xlabel(
        "Frequency coefficients F"
    )

    ax.set_ylabel(
        "Antennas B"
    )

    ax.set_title(
        "Transfer across tested B-F grids without retraining"
    )

    for i in range(len(antennas)):
        for j in range(len(frequencies)):
            ax.text(
                j,
                i,
                (
                    f"{matrix[i, j]:.2f}\n"
                    f"± {std_matrix[i, j]:.2f}"
                ),
                ha="center",
                va="center",
                fontsize=8,
            )

    ax.text(
        0.01,
        -0.16,
        "Random 5% observations; lower is better. "
        "Only the tested B-F combinations are claimed.",
        transform=ax.transAxes,
        fontsize=8,
        va="top",
    )

    return save_figure(
        fig,
        output_dir,
        "fig04_scalability",
    )


def fig_ablation(
    rows: list[dict[str, str]],
    output_dir: Path,
) -> list[str]:

    order = [
        "full",
        "no_global",
        "no_local",
        "no_coordinates",
    ]

    by_variant = {
        row["variant"]: row
        for row in rows
    }

    for variant in order:
        if variant not in by_variant:
            raise RuntimeError(
                f"Ablation table is missing {variant}."
            )

    display_labels = {
        "full": "Full\n(100k params)",
        "no_global": "No global\n(23k params)",
        "no_local": "No learned local\nencoder (82k)",
        "no_coordinates": "No coordinates\n(98k params)",
    }

    x = np.arange(
        len(order),
        dtype=float,
    )

    markers = ["D", "o", "s", "^"]
    means = []

    fig, ax = plt.subplots(
        figsize=(7.4, 5.0),
        constrained_layout=True,
    )

    for index, variant, marker in zip(
        x,
        order,
        markers,
    ):

        row = by_variant[variant]
        mean = as_float(row, "missing_nmse_db_mean")
        std = as_float(row, "missing_nmse_db_std")
        drop = as_float(row, "degradation_vs_full_db")

        means.append(mean)

        ax.errorbar(
            [index],
            [mean],
            yerr=std,
            fmt=marker,
            markersize=8,
            capsize=4,
            linewidth=1.4,
        )

        annotation = (
            f"{mean:.2f}"
            if variant == "full"
            else f"{mean:.2f}\ndrop {drop:.2f}"
        )

        ax.annotate(
            annotation,
            xy=(index, mean),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )

    ax.set_xticks(
        x,
        [
            display_labels[variant]
            for variant in order
        ],
    )

    ax.set_ylabel(
        "Missing-region NMSE (dB)"
    )

    ax.set_title(
        "SCALE-Former ablation study"
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    ax.text(
        0.01,
        0.02,
        "Lower is better",
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
    )

    ax.text(
        0.99,
        0.02,
        "No-global result is capacity-confounded",
        transform=ax.transAxes,
        fontsize=8,
        ha="right",
        va="bottom",
    )

    span = max(
        max(means) - min(means),
        1.0,
    )

    ax.set_ylim(
        min(means) - 0.12 * span,
        max(means) + 0.30 * span,
    )

    return save_figure(
        fig,
        output_dir,
        "fig05_ablation",
    )


def fig_accuracy_efficiency(
    primary_rows: list[dict[str, str]],
    efficiency_rows: list[dict[str, str]],
    output_dir: Path,
) -> list[str]:

    test = {
        row["method"]: row
        for row in primary_rows
        if row["stage"] == "test_0057"
    }

    efficiency = {
        row["method"]: row
        for row in efficiency_rows
    }

    fig, ax = plt.subplots(
        figsize=(7.5, 4.9),
        constrained_layout=True,
    )

    markers = ["o", "s", "^", "D"]

    annotation_offsets = {
        "Linear interpolation": (8, 7, "left"),
        "Residual Frequency CNN": (8, 7, "left"),
        "Full-Attention Transformer": (8, 7, "left"),
        "SCALE-Former": (-8, 7, "right"),
    }

    latency_values = []

    for method, marker in zip(
        METHOD_ORDER,
        markers,
    ):

        if method not in test or method not in efficiency:
            raise RuntimeError(
                f"Accuracy-efficiency inputs are missing {method}."
            )

        test_row = test[method]
        efficiency_row = efficiency[method]

        latency = as_float(
            efficiency_row,
            "e2e_latency_mean_ms",
        )

        latency_std = as_float(
            efficiency_row,
            "e2e_latency_sd_ms",
        )

        nmse = as_float(
            test_row,
            "missing_nmse_db_mean",
        )

        nmse_std = as_float(
            test_row,
            "missing_nmse_db_std",
        )

        latency_values.append(latency)

        ax.errorbar(
            [latency],
            [nmse],
            xerr=latency_std,
            yerr=None if math.isnan(nmse_std) else nmse_std,
            fmt=marker,
            markersize=8,
            capsize=4,
            linewidth=1.3,
        )

        parameters = int(
            float(
                efficiency_row["parameters"]
            )
        )

        label = METHOD_SHORT[method]

        if parameters > 0:
            label = (
                f"{label} "
                f"({parameters / 1000:.0f}k params)"
            )

        dx, dy, alignment = annotation_offsets[method]

        ax.annotate(
            label,
            xy=(latency, nmse),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8,
            ha=alignment,
        )

    ax.set_xlabel(
        "End-to-end latency (ms)"
    )

    ax.set_ylabel(
        "Held-out missing-region NMSE (dB)"
    )

    ax.set_title(
        "Held-out accuracy versus inference latency"
    )

    ax.grid(
        alpha=0.25,
    )

    ax.text(
        0.01,
        0.02,
        "Lower-left is preferable; latency is mean ± SD "
        "across five per-round medians",
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
    )

    left = min(latency_values)
    right = max(latency_values)
    span = max(right - left, 1.0)

    ax.set_xlim(
        max(0.0, left - 0.10 * span),
        right + 0.18 * span,
    )

    return save_figure(
        fig,
        output_dir,
        "fig06_accuracy_efficiency",
    )

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--paper-root",
        type=Path,
        default=Path(
            "results/paper"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "results/paper/figures"
        ),
    )

    args = parser.parse_args()

    source_paths = {
        key: (
            args.paper_root
            / filename
        )
        for (
            key,
            filename,
        ) in INPUT_FILES.items()
    }

    missing = [
        str(
            path
        )
        for path in source_paths.values()
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing audited figure inputs:\n"
            + "\n".join(
                missing
            )
        )

    primary = load_csv(
        source_paths[
            "primary"
        ]
    )

    efficiency = load_csv(
        source_paths[
            "efficiency"
        ]
    )

    robustness = load_csv(
        source_paths[
            "robustness"
        ]
    )

    scalability = load_csv(
        source_paths[
            "scalability"
        ]
    )

    ablation = load_csv(
        source_paths[
            "ablation"
        ]
    )

    require_columns(
        primary,
        {
            "stage",
            "method",
            "missing_nmse_db_mean",
            "missing_nmse_db_std",
        },
        "primary_results.csv",
    )

    require_columns(
        efficiency,
        {
            "method",
            "parameters",
            "e2e_latency_mean_ms",
            "e2e_latency_sd_ms",
        },
        "efficiency_results.csv",
    )

    require_columns(
        robustness,
        {
            "condition",
            "pattern",
            "requested_density",
            "interpolation_missing_nmse_db",
            "scaleformer_missing_nmse_db_mean",
            "scaleformer_missing_nmse_db_std",
            "mean_gain_vs_interpolation_missing_db",
        },
        "robustness_results.csv",
    )

    require_columns(
        scalability,
        {
            "num_antennas",
            "num_subcarriers",
            "scaleformer_missing_nmse_db_mean",
            "scaleformer_missing_nmse_db_std",
        },
        "scalability_results.csv",
    )

    require_columns(
        ablation,
        {
            "variant",
            "display_name",
            "missing_nmse_db_mean",
            "missing_nmse_db_std",
            "degradation_vs_full_db",
        },
        "ablation_results.csv",
    )

    output_dir = args.output_dir

    generated = []

    generated.extend(
        fig_primary_test(
            primary,
            output_dir,
        )
    )

    generated.extend(
        fig_random_density(
            robustness,
            output_dir,
        )
    )

    generated.extend(
        fig_mask_patterns(
            robustness,
            output_dir,
        )
    )

    generated.extend(
        fig_scalability(
            scalability,
            output_dir,
        )
    )

    generated.extend(
        fig_ablation(
            ablation,
            output_dir,
        )
    )

    generated.extend(
        fig_accuracy_efficiency(
            primary,
            efficiency,
            output_dir,
        )
    )

    manifest = {
        "sources": [
            {
                "key": key,
                "path": str(
                    path
                ),
                "sha256": file_sha256(
                    path
                ),
            }
            for (
                key,
                path,
            ) in source_paths.items()
        ],
        "generated_files": (
            generated
        ),
        "notes": [
            (
                "All figures are generated from "
                "audited results/paper CSV files."
            ),
            (
                "PNG outputs are saved at 600 dpi; "
                "PDF outputs are vector graphics."
            ),
            (
                "The block-mask result is an "
                "out-of-distribution stress test."
            ),
            (
                "The accuracy-efficiency plot uses "
                "the authoritative five-round E2E "
                "latency summary."
            ),
        ],
    }

    with (
        output_dir
        / "figure_manifest.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            manifest,
            file,
            indent=4,
        )

    print(
        "=" * 100
    )

    print(
        "SCALE-FORMER PAPER FIGURES BUILT"
    )

    print(
        "=" * 100
    )

    for path in generated:
        print(
            path
        )

    print(
        "\nManifest:"
    )

    print(
        output_dir
        / "figure_manifest.json"
    )

    print(
        "\nAll figures use only "
        "audited results/paper inputs."
    )

    print(
        "=" * 100
    )


if __name__ == "__main__":
    main()
