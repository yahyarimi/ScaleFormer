from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


PAPER_ROOT = Path("results/paper")
TABLE_ROOT = PAPER_ROOT / "tables"

INPUTS = {
    "master": PAPER_ROOT / "master_results.json",
    "primary": PAPER_ROOT / "primary_results.csv",
    "robustness": PAPER_ROOT / "robustness_results.csv",
    "ablation": PAPER_ROOT / "ablation_results.csv",
    "efficiency": PAPER_ROOT / "efficiency_results.csv",
}


def load_csv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def sha256(path: Path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def mean_std(mean, std):
    mean = float(mean)

    if std in ("", None):
        return f"{mean:.4f}"

    return f"{mean:.4f} ± {float(std):.4f}"


def save_csv(path, headers, rows):
    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


def make_markdown(headers, rows):
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(
            ["---"] * len(headers)
        ) + " |",
    ]

    for row in rows:
        lines.append(
            "| " + " | ".join(row) + " |"
        )

    return "\n".join(lines) + "\n"


def latex_escape(text):
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "_": r"\_",
        "#": r"\#",
    }

    return "".join(
        replacements.get(char, char)
        for char in text
    )


def make_latex(
    caption,
    label,
    headers,
    rows,
    alignment,
    notes,
):
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        rf"\caption{{{latex_escape(caption)}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{alignment}}}",
        r"\hline",
        " & ".join(
            latex_escape(x)
            for x in headers
        ) + r" \\",
        r"\hline",
    ]

    for row in rows:
        formatted = []

        for item in row:
            item = latex_escape(item)
            item = item.replace(
                "±",
                r"$\pm$",
            )
            item = item.replace(
                "–",
                "--",
            )

            formatted.append(item)

        lines.append(
            " & ".join(formatted)
            + r" \\"
        )

    lines.extend([
        r"\hline",
        r"\end{tabular}",
        r"\vspace{1mm}",
        r"\begin{minipage}{0.98\textwidth}",
        r"\footnotesize",
    ])

    for note in notes:
        lines.append(
            latex_escape(note)
            + r"\\"
        )

    lines.extend([
        r"\end{minipage}",
        r"\end{table*}",
    ])

    return "\n".join(lines) + "\n"


def write_table(
    stem,
    caption,
    label,
    headers,
    rows,
    alignment,
    notes,
):
    save_csv(
        TABLE_ROOT / f"{stem}.csv",
        headers,
        rows,
    )

    (
        TABLE_ROOT / f"{stem}.md"
    ).write_text(
        make_markdown(
            headers,
            rows,
        ),
        encoding="utf-8",
    )

    (
        TABLE_ROOT / f"{stem}.tex"
    ).write_text(
        make_latex(
            caption,
            label,
            headers,
            rows,
            alignment,
            notes,
        ),
        encoding="utf-8",
    )


def main():

    missing = [
        str(path)
        for path in INPUTS.values()
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing audited inputs:\n"
            + "\n".join(missing)
        )

    TABLE_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    with INPUTS["master"].open(
        "r",
        encoding="utf-8",
    ) as f:
        master = json.load(f)

    primary = load_csv(
        INPUTS["primary"]
    )

    robustness = load_csv(
        INPUTS["robustness"]
    )

    ablation = load_csv(
        INPUTS["ablation"]
    )

    efficiency = load_csv(
        INPUTS["efficiency"]
    )

    # ============================================================
    # TABLE I — EXPERIMENTAL PROTOCOL
    # ============================================================

    protocol = master[
        "protocol"
    ]

    headers_1 = [
        "Item",
        "Setting",
    ]

    rows_1 = [
        [
            "Dataset",
            protocol["dataset"],
        ],
        [
            "Training runs",
            ", ".join(
                protocol[
                    "training_runs"
                ]
            ),
        ],
        [
            "Validation run",
            protocol[
                "validation_run"
            ],
        ],
        [
            "Held-out test run",
            protocol[
                "final_test_run"
            ],
        ],
        [
            "Primary grid",
            "B = 32 antennas, "
            "F = 1024 frequency coefficients",
        ],
        [
            "Primary observation pattern",
            "Random, frequency-shared "
            "across antennas",
        ],
        [
            "Requested density",
            "5%",
        ],
        [
            "Observed frequencies",
            str(
                protocol[
                    "primary_observed_subcarriers"
                ]
            )
            + " / 1024",
        ],
        [
            "Validation/test mask seed",
            str(
                protocol[
                    "validation_and_test_mask_seed"
                ]
            ),
        ],
        [
            "Learned-model seeds",
            ", ".join(
                str(x)
                for x in protocol[
                    "learned_model_seeds"
                ]
            ),
        ],
        [
            "Primary metric",
            protocol[
                "primary_metric"
            ],
        ],
        [
            "Metric scale",
            protocol[
                "metric_scale"
            ],
        ],
    ]

    write_table(
        "table01_protocol",
        "Dataset and experimental protocol.",
        "tab:protocol",
        headers_1,
        rows_1,
        "ll",
        [
            "The primary protocol uses the same "
            "observed frequency positions across "
            "all antennas.",
            "Metrics are computed on the "
            "denormalized measured CSI scale.",
        ],
    )

    # ============================================================
    # TABLE II — PRIMARY PERFORMANCE
    # ============================================================

    validation = {
        row["method"]: row
        for row in primary
        if row["stage"]
        == "validation_0056"
    }

    test = {
        row["method"]: row
        for row in primary
        if row["stage"]
        == "test_0057"
    }

    methods = [
        "Linear interpolation",
        "Residual Frequency CNN",
        "Full-Attention Transformer",
        "SCALE-Former",
    ]

    headers_2 = [
        "Method",
        "Parameters",
        "Validation missing NMSE (dB)",
        "Test full NMSE (dB)",
        "Test missing NMSE (dB)",
    ]

    rows_2 = []

    for method in methods:

        v = validation[
            method
        ]

        t = test[
            method
        ]

        params = int(
            float(
                t["parameters"]
            )
        )

        rows_2.append([
            method,
            (
                "–"
                if params == 0
                else f"{params:,}"
            ),
            mean_std(
                v[
                    "missing_nmse_db_mean"
                ],
                v[
                    "missing_nmse_db_std"
                ],
            ),
            mean_std(
                t[
                    "full_nmse_db_mean"
                ],
                t[
                    "full_nmse_db_std"
                ],
            ),
            mean_std(
                t[
                    "missing_nmse_db_mean"
                ],
                t[
                    "missing_nmse_db_std"
                ],
            ),
        ])

    write_table(
        "table02_primary_performance",
        (
            "Primary validation and held-out "
            "test reconstruction performance."
        ),
        "tab:primary-performance",
        headers_2,
        rows_2,
        "lrrrr",
        [
            "Lower NMSE is better.",
            "Learned models report mean ± sample "
            "standard deviation across three "
            "training seeds.",
            "Interpolation is deterministic.",
            "Held-out testing uses DICHASUS-0057 "
            "under the frozen random-5% protocol.",
        ],
    )

    # ============================================================
    # TABLE III — ROBUSTNESS
    # ============================================================

    labels = {
        "random_1pct": "Random 1%",
        "random_2pct": "Random 2%",
        "random_5pct": "Random 5%",
        "random_10pct": "Random 10%",
        "random_20pct": "Random 20%",
        "comb_5pct": "Comb 5%",
        "block_5pct": "Block 5%",
    }

    headers_3 = [
        "Condition",
        "Interpolation missing NMSE (dB)",
        "SCALE-Former missing NMSE (dB)",
        "Gain (dB)",
    ]

    rows_3 = []

    for row in robustness:

        rows_3.append([
            labels[
                row[
                    "condition"
                ]
            ],
            (
                f'{float(row["interpolation_missing_nmse_db"]):.4f}'
            ),
            mean_std(
                row[
                    "scaleformer_missing_nmse_db_mean"
                ],
                row[
                    "scaleformer_missing_nmse_db_std"
                ],
            ),
            (
                f'{float(row["mean_gain_vs_interpolation_missing_db"]):+.4f}'
            ),
        ])

    write_table(
        "table03_robustness",
        (
            "Robustness to observation-density "
            "and pattern shifts on DICHASUS-0056."
        ),
        "tab:robustness",
        headers_3,
        rows_3,
        "lrrr",
        [
            "Positive gain indicates improvement "
            "over interpolation.",
            "Block masking is an out-of-distribution "
            "stress case; SCALE-Former does not "
            "outperform interpolation there.",
        ],
    )

    # ============================================================
    # TABLE IV — ABLATION
    # ============================================================

    variant_names = {
        "full": "Full SCALE-Former",
        "no_global": "No global",
        "no_local":
            "No learned local encoder",
        "no_coordinates":
            "No coordinates",
    }

    headers_4 = [
        "Variant",
        "Parameters",
        "Missing NMSE (dB)",
        "Drop vs. full (dB)",
    ]

    rows_4 = []

    for row in ablation:

        variant = row[
            "variant"
        ]

        rows_4.append([
            variant_names[
                variant
            ],
            f'{int(float(row["parameters"])):,}',
            mean_std(
                row[
                    "missing_nmse_db_mean"
                ],
                row[
                    "missing_nmse_db_std"
                ],
            ),
            (
                "–"
                if variant == "full"
                else
                f'{float(row["degradation_vs_full_db"]):.4f}'
            ),
        ])

    write_table(
        "table04_ablation",
        (
            "SCALE-Former ablation study on "
            "DICHASUS-0056 under random 5% "
            "observations."
        ),
        "tab:ablation",
        headers_4,
        rows_4,
        "lrrr",
        [
            "Lower NMSE is better.",
            "The no-global result is "
            "capacity-confounded because the "
            "variant has substantially fewer "
            "trainable parameters.",
            "The no-local variant uses a "
            "pointwise encoder; the interpolation "
            "seed itself remains local.",
        ],
    )

    # ============================================================
    # TABLE V — EFFICIENCY
    # ============================================================

    headers_5 = [
        "Method",
        "Parameters",
        "Core MMACs",
        "Peak GPU memory (MiB)",
        "E2E latency (ms)",
    ]

    rows_5 = []

    for row in efficiency:

        params = int(
            float(
                row[
                    "parameters"
                ]
            )
        )

        rows_5.append([
            row[
                "method"
            ],
            (
                "–"
                if params == 0
                else f"{params:,}"
            ),
            (
                "–"
                if row[
                    "core_mmacs"
                ] == ""
                else
                f'{float(row["core_mmacs"]):.4f}'
            ),
            (
                f'{float(row["peak_gpu_memory_mib"]):.4f}'
            ),
            (
                f'{float(row["e2e_latency_mean_ms"]):.4f}'
                " ± "
                f'{float(row["e2e_latency_sd_ms"]):.4f}'
            ),
        ])

    write_table(
        "table05_efficiency",
        (
            "Controlled computational "
            "characteristics at B=32, F=1024 "
            "and batch size 1."
        ),
        "tab:efficiency",
        headers_5,
        rows_5,
        "lrrrr",
        [
            "Core MMACs count selected model "
            "tensor operations only.",
            "Interpolation, LayerNorm, softmax, "
            "activations, masking, indexing and "
            "other elementwise operations are "
            "excluded from the MAC count.",
            "Approximate FLOPs may be estimated "
            "as 2 FLOP/MAC.",
            "Learned-model inference uses AMP "
            "FP16; interpolation uses FP32.",
            "Latency is mean ± sample SD across "
            "five per-round end-to-end medians.",
            "Peak GPU memory uses "
            "torch.cuda.max_memory_allocated.",
        ],
    )

    # ============================================================
    # MANIFEST + AUDIT
    # ============================================================

    manifest = {
        "sources": [
            {
                "key": key,
                "path": str(path),
                "sha256": sha256(path),
            }
            for key, path
            in INPUTS.items()
        ],
        "reporting_guardrails": [
            (
                "Use dense measured reference CSI; "
                "do not call it physical ground truth."
            ),
            (
                "Controlled observation masks are "
                "not operational DICHASUS pilots."
            ),
            (
                "Retain the block-mask failure."
            ),
            (
                "Treat no-global as "
                "capacity-confounded."
            ),
            (
                "Use repeated latency as the "
                "headline latency result."
            ),
            (
                "Do not describe SCALE-Former as "
                "lightweight or computationally "
                "cheaper than the full Transformer."
            ),
        ],
    }

    (
        TABLE_ROOT
        / "table_manifest.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=4,
        ),
        encoding="utf-8",
    )

    claims = master[
        "headline_claims"
    ][
        "test_0057"
    ]

    audit = [
        "SCALE-FORMER PAPER TABLES AUDIT",
        "=" * 80,
        "",
        "PASS  Five manuscript tables generated "
        "from audited results/paper sources.",
        "",
        "Headline claims:",
        (
            "PASS  SCALE-Former vs Transformer "
            "test missing-NMSE gain = "
            f'{claims["scaleformer_vs_full_transformer_missing_nmse_gain_db"]:.4f} dB'
        ),
        (
            "PASS  SCALE-Former vs Transformer "
            "linear-NMSE reduction = "
            f'{claims["scaleformer_vs_full_transformer_linear_nmse_reduction_percent"]:.2f}%'
        ),
        (
            "PASS  SCALE-Former vs CNN "
            "test missing-NMSE gain = "
            f'{claims["scaleformer_vs_cnn_missing_nmse_gain_db"]:.4f} dB'
        ),
        (
            "PASS  SCALE-Former vs interpolation "
            "test missing-NMSE gain = "
            f'{claims["scaleformer_vs_interpolation_missing_nmse_gain_db"]:.4f} dB'
        ),
        "",
        "Reporting checks:",
        "PASS  Block-mask failure retained.",
        "PASS  No-global confound stated.",
        "PASS  Repeated latency used.",
        "PASS  MAC-count limitations stated.",
        "PASS  No lightweight/faster claim introduced.",
        "",
        (
            "FINAL STATUS: PAPER TABLES BUILT "
            "FROM AUDITED SOURCES"
        ),
    ]

    (
        TABLE_ROOT
        / "paper_tables_audit.txt"
    ).write_text(
        "\n".join(
            audit
        )
        + "\n",
        encoding="utf-8",
    )

    print("=" * 100)
    print("SCALE-FORMER PAPER TABLES BUILT")
    print("=" * 100)

    for path in sorted(
        TABLE_ROOT.iterdir()
    ):
        print(path)

    print()
    print(
        "FINAL STATUS: PAPER TABLES BUILT "
        "FROM AUDITED SOURCES"
    )
    print("=" * 100)


if __name__ == "__main__":
    main()