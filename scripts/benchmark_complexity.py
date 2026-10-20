from __future__ import annotations

import csv
import gc
import json
import platform
import statistics
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.dichasus import DichasusDataset
from src.models.cnn_baseline import ResidualFrequencyCNN
from src.models.interpolation import frequency_linear_interpolate
from src.models.scaleformer import ScaleFormer


# ============================================================
# Configuration
# ============================================================

DATA_ROOT = "data/dichasus_005x"

NORMALIZATION_FILE = (
    "results/csv/"
    "training_normalization.json"
)

RESULT_ROOT = Path(
    "results/csv"
)

CNN_CHECKPOINT = Path(
    "checkpoints/"
    "cnn_seed42_best.pt"
)

SCALEFORMER_CHECKPOINT = Path(
    "checkpoints/"
    "scaleformer_seed42_bs4_best.pt"
)

VALIDATION_DENSITY = 0.05
VALIDATION_PATTERN = "random"
VALIDATION_MASK_SEED = 42

BATCH_SIZE = 1

WARMUP_ITERATIONS = 50
MEASURED_ITERATIONS = 200

EXPECTED_ANTENNAS = 32
EXPECTED_SUBCARRIERS = 1024
EXPECTED_OBSERVED = 51


CSV_OUTPUT = (
    RESULT_ROOT
    / "complexity_benchmark.csv"
)

JSON_OUTPUT = (
    RESULT_ROOT
    / "complexity_benchmark.json"
)


# ============================================================
# Utilities
# ============================================================

def cleanup_cuda():

    gc.collect()

    if torch.cuda.is_available():

        torch.cuda.empty_cache()

        torch.cuda.synchronize()


def parameter_count(
    model: torch.nn.Module,
) -> int:

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def tensor_mib(
    number_of_bytes: int,
) -> float:

    return (
        number_of_bytes
        / (1024 ** 2)
    )


def benchmark_cuda_latency(
    function,
    warmup_iterations: int,
    measured_iterations: int,
):

    # --------------------------------------------------------
    # Warm-up
    # --------------------------------------------------------

    with torch.inference_mode():

        for _ in range(
            warmup_iterations
        ):

            _ = function()

    torch.cuda.synchronize()

    # --------------------------------------------------------
    # CUDA-event timing
    # --------------------------------------------------------

    starts = [
        torch.cuda.Event(
            enable_timing=True
        )
        for _ in range(
            measured_iterations
        )
    ]

    ends = [
        torch.cuda.Event(
            enable_timing=True
        )
        for _ in range(
            measured_iterations
        )
    ]

    with torch.inference_mode():

        output = None

        for index in range(
            measured_iterations
        ):

            starts[
                index
            ].record()

            output = function()

            ends[
                index
            ].record()

    torch.cuda.synchronize()

    times_ms = [
        starts[index].elapsed_time(
            ends[index]
        )
        for index in range(
            measured_iterations
        )
    ]

    del output

    return {
        "iterations": (
            measured_iterations
        ),
        "mean_ms": (
            statistics.mean(
                times_ms
            )
        ),
        "std_ms": (
            statistics.stdev(
                times_ms
            )
        ),
        "median_ms": (
            statistics.median(
                times_ms
            )
        ),
        "min_ms": min(
            times_ms
        ),
        "max_ms": max(
            times_ms
        ),
        "p95_ms": (
            sorted(
                times_ms
            )[
                int(
                    0.95
                    * (
                        len(
                            times_ms
                        )
                        - 1
                    )
                )
            ]
        ),
    }


def measure_peak_memory(
    function,
):

    # Warm once so CUDA kernels / workspaces
    # have already initialized.
    with torch.inference_mode():

        warm_output = function()

    torch.cuda.synchronize()

    del warm_output

    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()

    starting_allocated = (
        torch.cuda.memory_allocated()
    )

    with torch.inference_mode():

        output = function()

    torch.cuda.synchronize()

    peak_allocated = (
        torch.cuda
        .max_memory_allocated()
    )

    final_allocated = (
        torch.cuda.memory_allocated()
    )

    del output

    torch.cuda.synchronize()

    return {
        "starting_allocated_mib": (
            tensor_mib(
                starting_allocated
            )
        ),
        "peak_allocated_mib": (
            tensor_mib(
                peak_allocated
            )
        ),
        "incremental_peak_mib": (
            tensor_mib(
                peak_allocated
                - starting_allocated
            )
        ),
        "final_allocated_mib": (
            tensor_mib(
                final_allocated
            )
        ),
    }


# ============================================================
# Data
# ============================================================

def load_validation_example(
    device: torch.device,
):

    dataset = DichasusDataset(
        data_root=DATA_ROOT,
        split="val",
        normalization_path=(
            NORMALIZATION_FILE
        ),
        density=(
            VALIDATION_DENSITY
        ),
        pattern=(
            VALIDATION_PATTERN
        ),
        seed=(
            VALIDATION_MASK_SEED
        ),
        shuffle_buffer=0,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=0,
        pin_memory=True,
    )

    batch = next(
        iter(
            loader
        )
    )

    observed = (
        batch[
            "observed"
        ]
        .to(
            device,
            non_blocking=True,
        )
    )

    mask = (
        batch[
            "mask"
        ]
        .to(
            device,
            non_blocking=True,
        )
    )

    if observed.shape != (
        BATCH_SIZE,
        2,
        EXPECTED_ANTENNAS,
        EXPECTED_SUBCARRIERS,
    ):

        raise RuntimeError(
            "Unexpected CSI shape: "
            f"{tuple(observed.shape)}"
        )

    observed_count = int(
        mask[
            0,
            0,
            0,
        ]
        .sum()
        .item()
    )

    if (
        observed_count
        != EXPECTED_OBSERVED
    ):

        raise RuntimeError(
            "Unexpected observed "
            "subcarrier count: "
            f"{observed_count}"
        )

    return (
        observed,
        mask,
        observed_count,
    )


# ============================================================
# Model loading
# ============================================================

def load_cnn(
    device: torch.device,
):

    if not CNN_CHECKPOINT.exists():

        raise FileNotFoundError(
            CNN_CHECKPOINT
        )

    checkpoint = torch.load(
        CNN_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    # The historical checkpoint contains no
    # architecture dictionary. Reconstruct using
    # the frozen/default CNN configuration.
    model = ResidualFrequencyCNN(
        num_antennas=32,
        hidden_dim=64,
        num_blocks=4,
        kernel_size=7,
    )

    params = parameter_count(
        model
    )

    if params != 60_160:

        raise RuntimeError(
            "Unexpected CNN parameter "
            f"count: {params:,}"
        )

    saved_params = checkpoint.get(
        "num_parameters"
    )

    if (
        saved_params is not None
        and int(
            saved_params
        )
        != params
    ):

        raise RuntimeError(
            "CNN checkpoint parameter "
            "metadata mismatch."
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

    return (
        model,
        params,
    )


def load_scaleformer(
    device: torch.device,
):

    if not SCALEFORMER_CHECKPOINT.exists():

        raise FileNotFoundError(
            SCALEFORMER_CHECKPOINT
        )

    checkpoint = torch.load(
        SCALEFORMER_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    architecture = checkpoint[
        "architecture"
    ]

    model = ScaleFormer(
        dim=architecture[
            "dim"
        ],
        num_heads=architecture[
            "heads"
        ],
        num_latents=architecture[
            "latents"
        ],
        num_local_blocks=architecture[
            "local_blocks"
        ],
        num_global_blocks=architecture[
            "global_blocks"
        ],
        num_decoder_blocks=architecture[
            "decoder_blocks"
        ],
        local_kernel_size=(
            architecture.get(
                "local_kernel_size",
                7,
            )
        ),
        decoder_chunk_size=architecture[
            "decoder_chunk_size"
        ],
        dropout=0.0,
        variant=architecture.get(
            "variant",
            "full",
        ),
    )

    params = parameter_count(
        model
    )

    if params != 100_034:

        raise RuntimeError(
            "Unexpected SCALE-Former "
            f"parameter count: {params:,}"
        )

    saved_params = checkpoint.get(
        "num_parameters"
    )

    if (
        saved_params is not None
        and int(
            saved_params
        )
        != params
    ):

        raise RuntimeError(
            "SCALE-Former checkpoint "
            "parameter metadata mismatch."
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

    return (
        model,
        params,
    )


# ============================================================
# Individual benchmarks
# ============================================================

def benchmark_interpolation(
    observed: torch.Tensor,
    mask: torch.Tensor,
):

    print(
        "\n"
        + "=" * 80
    )

    print(
        "BENCHMARK: LINEAR INTERPOLATION"
    )

    print(
        "=" * 80
    )

    def end_to_end():

        return (
            frequency_linear_interpolate(
                observed,
                mask,
            )
        )

    latency = benchmark_cuda_latency(
        function=end_to_end,
        warmup_iterations=(
            WARMUP_ITERATIONS
        ),
        measured_iterations=(
            MEASURED_ITERATIONS
        ),
    )

    memory = measure_peak_memory(
        end_to_end
    )

    row = {
        "method": (
            "Linear interpolation"
        ),
        "parameters": 0,
        "model_only_mean_ms": None,
        "model_only_std_ms": None,
        "model_only_median_ms": None,
        "model_only_p95_ms": None,
        "end_to_end_mean_ms": (
            latency[
                "mean_ms"
            ]
        ),
        "end_to_end_std_ms": (
            latency[
                "std_ms"
            ]
        ),
        "end_to_end_median_ms": (
            latency[
                "median_ms"
            ]
        ),
        "end_to_end_p95_ms": (
            latency[
                "p95_ms"
            ]
        ),
        "model_only_peak_allocated_mib": (
            None
        ),
        "end_to_end_peak_allocated_mib": (
            memory[
                "peak_allocated_mib"
            ]
        ),
        "end_to_end_incremental_peak_mib": (
            memory[
                "incremental_peak_mib"
            ]
        ),
    }

    print(
        f"End-to-end latency : "
        f"{latency['mean_ms']:.4f} "
        f"± {latency['std_ms']:.4f} ms"
    )

    print(
        f"Median latency     : "
        f"{latency['median_ms']:.4f} ms"
    )

    print(
        f"P95 latency        : "
        f"{latency['p95_ms']:.4f} ms"
    )

    print(
        f"Peak allocated     : "
        f"{memory['peak_allocated_mib']:.2f} MiB"
    )

    return row


def benchmark_learned_model(
    method_name: str,
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
        f"BENCHMARK: {method_name.upper()}"
    )

    print(
        "=" * 80
    )

    # Precompute interpolation so model-only
    # timing excludes preprocessing.
    interpolated = (
        frequency_linear_interpolate(
            observed,
            mask,
        )
    )

    torch.cuda.synchronize()

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

    # Sanity check before timing.
    with torch.inference_mode():

        sanity_output = (
            model_only()
        )

    if (
        sanity_output.shape
        != observed.shape
    ):

        raise RuntimeError(
            f"{method_name}: unexpected "
            "output shape."
        )

    del sanity_output

    torch.cuda.synchronize()

    model_latency = (
        benchmark_cuda_latency(
            function=model_only,
            warmup_iterations=(
                WARMUP_ITERATIONS
            ),
            measured_iterations=(
                MEASURED_ITERATIONS
            ),
        )
    )

    model_memory = (
        measure_peak_memory(
            model_only
        )
    )

    # Delete precomputed interpolation so the
    # end-to-end memory measurement does not
    # unfairly keep an extra full CSI tensor alive.
    del interpolated

    torch.cuda.synchronize()

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

    end_to_end_latency = (
        benchmark_cuda_latency(
            function=end_to_end,
            warmup_iterations=(
                WARMUP_ITERATIONS
            ),
            measured_iterations=(
                MEASURED_ITERATIONS
            ),
        )
    )

    end_to_end_memory = (
        measure_peak_memory(
            end_to_end
        )
    )

    row = {
        "method": method_name,
        "parameters": (
            parameters
        ),
        "model_only_mean_ms": (
            model_latency[
                "mean_ms"
            ]
        ),
        "model_only_std_ms": (
            model_latency[
                "std_ms"
            ]
        ),
        "model_only_median_ms": (
            model_latency[
                "median_ms"
            ]
        ),
        "model_only_p95_ms": (
            model_latency[
                "p95_ms"
            ]
        ),
        "end_to_end_mean_ms": (
            end_to_end_latency[
                "mean_ms"
            ]
        ),
        "end_to_end_std_ms": (
            end_to_end_latency[
                "std_ms"
            ]
        ),
        "end_to_end_median_ms": (
            end_to_end_latency[
                "median_ms"
            ]
        ),
        "end_to_end_p95_ms": (
            end_to_end_latency[
                "p95_ms"
            ]
        ),
        "model_only_peak_allocated_mib": (
            model_memory[
                "peak_allocated_mib"
            ]
        ),
        "end_to_end_peak_allocated_mib": (
            end_to_end_memory[
                "peak_allocated_mib"
            ]
        ),
        "end_to_end_incremental_peak_mib": (
            end_to_end_memory[
                "incremental_peak_mib"
            ]
        ),
    }

    print(
        f"Parameters          : "
        f"{parameters:,}"
    )

    print(
        f"Model-only latency  : "
        f"{model_latency['mean_ms']:.4f} "
        f"± {model_latency['std_ms']:.4f} ms"
    )

    print(
        f"Model-only median   : "
        f"{model_latency['median_ms']:.4f} ms"
    )

    print(
        f"Model-only P95      : "
        f"{model_latency['p95_ms']:.4f} ms"
    )

    print(
        f"End-to-end latency  : "
        f"{end_to_end_latency['mean_ms']:.4f} "
        f"± "
        f"{end_to_end_latency['std_ms']:.4f} ms"
    )

    print(
        f"End-to-end median   : "
        f"{end_to_end_latency['median_ms']:.4f} ms"
    )

    print(
        f"End-to-end P95      : "
        f"{end_to_end_latency['p95_ms']:.4f} ms"
    )

    print(
        f"Model-only peak     : "
        f"{model_memory['peak_allocated_mib']:.2f} MiB"
    )

    print(
        f"End-to-end peak     : "
        f"{end_to_end_memory['peak_allocated_mib']:.2f} MiB"
    )

    return row


# ============================================================
# Output
# ============================================================

def save_results(
    rows: list[dict],
    metadata: dict,
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
        "metadata": metadata,
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


# ============================================================
# Main
# ============================================================

def main():

    print(
        "=" * 90
    )

    print(
        "CONTROLLED COMPLEXITY / "
        "INFERENCE BENCHMARK"
    )

    print(
        "=" * 90
    )

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA GPU is required."
        )

    device = torch.device(
        "cuda"
    )

    gpu_name = (
        torch.cuda
        .get_device_name(
            0
        )
    )

    gpu_properties = (
        torch.cuda
        .get_device_properties(
            0
        )
    )

    print(
        f"Python             : "
        f"{platform.python_version()}"
    )

    print(
        f"PyTorch            : "
        f"{torch.__version__}"
    )

    print(
        f"CUDA runtime       : "
        f"{torch.version.cuda}"
    )

    print(
        f"GPU                : "
        f"{gpu_name}"
    )

    print(
        f"GPU memory         : "
        f"{gpu_properties.total_memory / (1024 ** 3):.2f} GiB"
    )

    print(
        f"Batch size         : "
        f"{BATCH_SIZE}"
    )

    print(
        f"Warm-up iterations : "
        f"{WARMUP_ITERATIONS}"
    )

    print(
        f"Timed iterations   : "
        f"{MEASURED_ITERATIONS}"
    )

    print(
        "Learned precision  : "
        "AMP FP16"
    )

    print(
        "Interpolation      : "
        "FP32"
    )

    observed, mask, observed_count = (
        load_validation_example(
            device=device
        )
    )

    print(
        f"CSI grid           : "
        f"{EXPECTED_ANTENNAS} × "
        f"{EXPECTED_SUBCARRIERS}"
    )

    print(
        f"Observed carriers  : "
        f"{observed_count}"
    )

    rows = []

    # --------------------------------------------------------
    # Interpolation
    # --------------------------------------------------------

    cleanup_cuda()

    rows.append(
        benchmark_interpolation(
            observed=observed,
            mask=mask,
        )
    )

    cleanup_cuda()

    # --------------------------------------------------------
    # CNN
    # --------------------------------------------------------

    cnn, cnn_params = (
        load_cnn(
            device=device
        )
    )

    rows.append(
        benchmark_learned_model(
            method_name=(
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

    # --------------------------------------------------------
    # SCALE-Former
    # --------------------------------------------------------

    (
        scaleformer,
        scaleformer_params,
    ) = load_scaleformer(
        device=device
    )

    rows.append(
        benchmark_learned_model(
            method_name=(
                "SCALE-Former"
            ),
            model=scaleformer,
            parameters=(
                scaleformer_params
            ),
            observed=observed,
            mask=mask,
        )
    )

    del scaleformer

    cleanup_cuda()

    metadata = {
        "python_version": (
            platform.python_version()
        ),
        "pytorch_version": (
            torch.__version__
        ),
        "cuda_runtime": (
            torch.version.cuda
        ),
        "gpu": gpu_name,
        "gpu_total_memory_gib": (
            gpu_properties.total_memory
            / (1024 ** 3)
        ),
        "batch_size": BATCH_SIZE,
        "grid_antennas": (
            EXPECTED_ANTENNAS
        ),
        "grid_subcarriers": (
            EXPECTED_SUBCARRIERS
        ),
        "requested_density": (
            VALIDATION_DENSITY
        ),
        "observed_subcarriers": (
            observed_count
        ),
        "mask_pattern": (
            VALIDATION_PATTERN
        ),
        "mask_seed": (
            VALIDATION_MASK_SEED
        ),
        "warmup_iterations": (
            WARMUP_ITERATIONS
        ),
        "measured_iterations": (
            MEASURED_ITERATIONS
        ),
        "learned_model_precision": (
            "AMP FP16"
        ),
        "interpolation_precision": (
            "FP32"
        ),
        "memory_metric": (
            "torch.cuda."
            "max_memory_allocated"
        ),
        "latency_metric": (
            "CUDA events"
        ),
    }

    save_results(
        rows=rows,
        metadata=metadata,
    )

    print(
        "\n"
        + "=" * 120
    )

    print(
        "COMPLEXITY BENCHMARK SUMMARY"
    )

    print(
        "=" * 120
    )

    print(
        f"{'Method':<28}"
        f"{'Params':>12}"
        f"{'Model ms':>18}"
        f"{'E2E ms':>18}"
        f"{'Peak MiB':>16}"
    )

    print(
        "-" * 120
    )

    for row in rows:

        if (
            row[
                "model_only_mean_ms"
            ]
            is None
        ):

            model_text = "—"

        else:

            model_text = (
                f"{row['model_only_mean_ms']:.4f}"
            )

        print(
            f"{row['method']:<28}"
            f"{row['parameters']:>12,}"
            f"{model_text:>18}"
            f"{row['end_to_end_mean_ms']:>18.4f}"
            f"{row['end_to_end_peak_allocated_mib']:>16.2f}"
        )

    print()

    print(
        f"CSV:"
        f"\n{CSV_OUTPUT}"
    )

    print(
        f"\nJSON:"
        f"\n{JSON_OUTPUT}"
    )

    print()

    print(
        "NOTE: FLOP/MAC values are intentionally "
        "not reported here."
    )

    print(
        "They will be counted separately so "
        "attention operations are not silently "
        "under-counted by a generic profiler."
    )

    print(
        "=" * 120
    )


if __name__ == "__main__":
    main()