from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn

from scripts.benchmark_complexity import (
    cleanup_cuda,
    load_cnn,
    load_scaleformer,
    load_validation_example,
)

from scripts.benchmark_latency import (
    load_frequency_transformer,
)

from src.models.interpolation import (
    frequency_linear_interpolate,
)


RESULT_ROOT = Path("results/csv")

CSV_OUTPUT = (
    RESULT_ROOT
    / "model_macs.csv"
)

JSON_OUTPUT = (
    RESULT_ROOT
    / "model_macs.json"
)


class MACCounter:

    def __init__(self):

        self.total_macs = 0

        self.breakdown = defaultdict(
            int
        )

        self.handles = []

        self.attention_calls = 0

        self.mha_out_proj_ids = set()

    def add(
        self,
        category: str,
        macs: int,
    ):

        macs = int(
            macs
        )

        self.breakdown[
            category
        ] += macs

        self.total_macs += macs

    def linear_hook(
        self,
        module,
        args,
        kwargs,
        output,
    ):

        if (
            id(module)
            in self.mha_out_proj_ids
        ):
            return

        if not isinstance(
            output,
            torch.Tensor,
        ):
            return

        macs = (
            output.numel()
            * module.in_features
        )

        self.add(
            "linear",
            macs,
        )

    def conv1d_hook(
        self,
        module,
        args,
        kwargs,
        output,
    ):

        if not isinstance(
            output,
            torch.Tensor,
        ):
            return

        kernel = (
            module.kernel_size[
                0
            ]
        )

        input_channels_per_group = (
            module.in_channels
            // module.groups
        )

        macs = (
            output.numel()
            * input_channels_per_group
            * kernel
        )

        self.add(
            "conv1d",
            macs,
        )

    def conv2d_hook(
        self,
        module,
        args,
        kwargs,
        output,
    ):

        if not isinstance(
            output,
            torch.Tensor,
        ):
            return

        kernel_h = (
            module.kernel_size[
                0
            ]
        )

        kernel_w = (
            module.kernel_size[
                1
            ]
        )

        input_channels_per_group = (
            module.in_channels
            // module.groups
        )

        macs = (
            output.numel()
            * input_channels_per_group
            * kernel_h
            * kernel_w
        )

        self.add(
            "conv2d",
            macs,
        )

    @staticmethod
    def sequence_information(
        tensor: torch.Tensor,
        batch_first: bool,
    ):

        if tensor.ndim != 3:
            raise RuntimeError(
                "Expected a 3D attention tensor, "
                f"got shape {tuple(tensor.shape)}."
            )

        if batch_first:

            batch = tensor.shape[
                0
            ]

            length = tensor.shape[
                1
            ]

            features = tensor.shape[
                2
            ]

        else:

            length = tensor.shape[
                0
            ]

            batch = tensor.shape[
                1
            ]

            features = tensor.shape[
                2
            ]

        return (
            int(batch),
            int(length),
            int(features),
        )

    def mha_hook(
        self,
        module,
        args,
        kwargs,
        output,
    ):

        self.attention_calls += 1

        if len(args) >= 3:

            query = args[
                0
            ]

            key = args[
                1
            ]

            value = args[
                2
            ]

        else:

            query = kwargs.get(
                "query"
            )

            key = kwargs.get(
                "key"
            )

            value = kwargs.get(
                "value"
            )

        if (
            query is None
            or key is None
            or value is None
        ):

            raise RuntimeError(
                "Could not recover Q/K/V tensors "
                "from MultiheadAttention."
            )

        (
            batch_q,
            length_q,
            q_features,
        ) = self.sequence_information(
            query,
            module.batch_first,
        )

        (
            batch_k,
            length_k,
            k_features,
        ) = self.sequence_information(
            key,
            module.batch_first,
        )

        (
            batch_v,
            length_v,
            v_features,
        ) = self.sequence_information(
            value,
            module.batch_first,
        )

        if not (
            batch_q
            == batch_k
            == batch_v
        ):

            raise RuntimeError(
                "Attention batch sizes differ."
            )

        if length_k != length_v:

            raise RuntimeError(
                "Key/value sequence lengths differ."
            )

        batch = batch_q

        embed_dim = int(
            module.embed_dim
        )

        q_projection = (
            batch
            * length_q
            * q_features
            * embed_dim
        )

        k_projection = (
            batch
            * length_k
            * k_features
            * embed_dim
        )

        v_projection = (
            batch
            * length_v
            * v_features
            * embed_dim
        )

        qk_macs = (
            batch
            * length_q
            * length_k
            * embed_dim
        )

        av_macs = (
            batch
            * length_q
            * length_k
            * embed_dim
        )

        output_projection = (
            batch
            * length_q
            * embed_dim
            * embed_dim
        )

        self.add(
            "attention_q_projection",
            q_projection,
        )

        self.add(
            "attention_k_projection",
            k_projection,
        )

        self.add(
            "attention_v_projection",
            v_projection,
        )

        self.add(
            "attention_qk",
            qk_macs,
        )

        self.add(
            "attention_av",
            av_macs,
        )

        self.add(
            "attention_output_projection",
            output_projection,
        )

    def register(
        self,
        model: nn.Module,
    ):

        for module in model.modules():

            if isinstance(
                module,
                nn.MultiheadAttention,
            ):

                self.mha_out_proj_ids.add(
                    id(
                        module.out_proj
                    )
                )

        for module in model.modules():

            if isinstance(
                module,
                nn.MultiheadAttention,
            ):

                handle = (
                    module.register_forward_hook(
                        self.mha_hook,
                        with_kwargs=True,
                    )
                )

                self.handles.append(
                    handle
                )

            elif isinstance(
                module,
                nn.Linear,
            ):

                handle = (
                    module.register_forward_hook(
                        self.linear_hook,
                        with_kwargs=True,
                    )
                )

                self.handles.append(
                    handle
                )

            elif isinstance(
                module,
                nn.Conv1d,
            ):

                handle = (
                    module.register_forward_hook(
                        self.conv1d_hook,
                        with_kwargs=True,
                    )
                )

                self.handles.append(
                    handle
                )

            elif isinstance(
                module,
                nn.Conv2d,
            ):

                handle = (
                    module.register_forward_hook(
                        self.conv2d_hook,
                        with_kwargs=True,
                    )
                )

                self.handles.append(
                    handle
                )

    def remove(
        self,
    ):

        for handle in self.handles:

            handle.remove()

        self.handles.clear()


def count_model(
    name: str,
    model: nn.Module,
    parameters: int,
    interpolated: torch.Tensor,
    mask: torch.Tensor,
):

    print(
        "\n"
        + "=" * 80
    )

    print(
        f"COUNTING: {name.upper()}"
    )

    print(
        "=" * 80
    )

    counter = MACCounter()

    counter.register(
        model
    )

    output = None

    try:

        with torch.inference_mode():

            with torch.amp.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=True,
            ):

                output = model(
                    interpolated,
                    mask,
                )

        torch.cuda.synchronize()

    finally:

        counter.remove()

    if output is None:

        raise RuntimeError(
            f"{name}: no output produced."
        )

    if (
        output.shape
        != interpolated.shape
    ):

        raise RuntimeError(
            f"{name}: unexpected output shape."
        )

    total_macs = (
        counter.total_macs
    )

    approximate_flops = (
        2
        * total_macs
    )

    print(
        f"Parameters          : "
        f"{parameters:,}"
    )

    print(
        f"Attention calls     : "
        f"{counter.attention_calls}"
    )

    for (
        category,
        value,
    ) in sorted(
        counter.breakdown.items()
    ):

        print(
            f"{category:<30}: "
            f"{value / 1e6:>12.4f} MMAC"
        )

    print(
        "-" * 80
    )

    print(
        f"TOTAL               : "
        f"{total_macs / 1e6:.4f} MMAC"
    )

    print(
        f"Approx. FLOPs       : "
        f"{approximate_flops / 1e9:.4f} GFLOPs"
    )

    return {
        "method": name,
        "parameters": parameters,
        "total_macs": (
            total_macs
        ),
        "mmacs": (
            total_macs
            / 1e6
        ),
        "approximate_flops": (
            approximate_flops
        ),
        "approximate_gflops": (
            approximate_flops
            / 1e9
        ),
        "attention_calls": (
            counter.attention_calls
        ),
        "breakdown": dict(
            counter.breakdown
        ),
    }


def main():

    print(
        "=" * 90
    )

    print(
        "ANALYTICAL / FORWARD-SHAPE "
        "MODEL MAC COUNTER"
    )

    print(
        "=" * 90
    )

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA is required."
        )

    device = torch.device(
        "cuda"
    )

    observed, mask, observed_count = (
        load_validation_example(
            device
        )
    )

    interpolated = (
        frequency_linear_interpolate(
            observed,
            mask,
        )
    )

    torch.cuda.synchronize()

    print(
        f"GPU                 : "
        f"{torch.cuda.get_device_name(0)}"
    )

    print(
        f"Grid                : "
        f"{tuple(interpolated.shape)}"
    )

    print(
        f"Observed frequencies: "
        f"{observed_count}"
    )

    results = []

    cleanup_cuda()

    cnn, cnn_params = (
        load_cnn(
            device
        )
    )

    results.append(
        count_model(
            name=(
                "Residual Frequency CNN"
            ),
            model=cnn,
            parameters=cnn_params,
            interpolated=interpolated,
            mask=mask,
        )
    )

    del cnn

    cleanup_cuda()

    (
        transformer,
        transformer_params,
    ) = load_frequency_transformer(
        device
    )

    transformer_result = (
        count_model(
            name=(
                "Full-Attention Transformer"
            ),
            model=transformer,
            parameters=transformer_params,
            interpolated=interpolated,
            mask=mask,
        )
    )

    if (
        transformer_result[
            "attention_calls"
        ]
        != 3
    ):

        raise RuntimeError(
            "Expected exactly 3 self-attention "
            "calls for the 3-layer full-attention "
            "Transformer, but observed "
            f"{transformer_result['attention_calls']}."
        )

    results.append(
        transformer_result
    )

    del transformer

    cleanup_cuda()

    (
        scaleformer,
        sf_params,
    ) = load_scaleformer(
        device
    )

    results.append(
        count_model(
            name="SCALE-Former",
            model=scaleformer,
            parameters=sf_params,
            interpolated=interpolated,
            mask=mask,
        )
    )

    del scaleformer

    cleanup_cuda()

    RESULT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    flat_rows = []

    for result in results:

        flat_rows.append(
            {
                "method": (
                    result[
                        "method"
                    ]
                ),
                "parameters": (
                    result[
                        "parameters"
                    ]
                ),
                "mmacs": (
                    result[
                        "mmacs"
                    ]
                ),
                "approximate_gflops": (
                    result[
                        "approximate_gflops"
                    ]
                ),
                "attention_calls": (
                    result[
                        "attention_calls"
                    ]
                ),
            }
        )

    with CSV_OUTPUT.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(
                flat_rows[
                    0
                ].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            flat_rows
        )

    payload = {
        "scope": (
            "Model-only MACs at batch size 1 "
            "for a 32x1024 CSI grid. "
            "Linear/Conv MACs and explicit "
            "MultiheadAttention Q/K/V projection, "
            "QK^T, attention-V, and output-projection "
            "MACs are counted. Interpolation, "
            "LayerNorm, softmax, activation functions, "
            "mask/index operations and other "
            "elementwise operations are excluded."
        ),
        "mac_convention": (
            "One multiply-accumulate = one MAC."
        ),
        "flop_convention": (
            "Approximate FLOPs = 2 x MACs."
        ),
        "representative_checkpoint_seed": (
            42
        ),
        "results": results,
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
        + "=" * 110
    )

    print(
        "MODEL MAC SUMMARY"
    )

    print(
        "=" * 110
    )

    print(
        f"{'Method':<30}"
        f"{'Params':>12}"
        f"{'MMACs':>18}"
        f"{'Approx GFLOPs':>20}"
    )

    print(
        "-" * 110
    )

    for result in results:

        print(
            f"{result['method']:<30}"
            f"{result['parameters']:>12,}"
            f"{result['mmacs']:>18.4f}"
            f"{result['approximate_gflops']:>20.4f}"
        )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "These are model-only core tensor MACs. "
        "Interpolation preprocessing is not included."
    )

    print(
        "Approximate FLOPs use the explicit "
        "2-FLOPs-per-MAC convention."
    )

    print(
        f"\nCSV:"
        f"\n{CSV_OUTPUT}"
    )

    print(
        f"\nJSON:"
        f"\n{JSON_OUTPUT}"
    )

    print(
        "=" * 110
    )


if __name__ == "__main__":
    main()
