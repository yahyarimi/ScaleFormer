from __future__ import annotations

import torch
from torch import nn


class ResidualFrequencyBlock(nn.Module):
    """
    Lightweight local frequency-processing block.

    Depthwise convolution models local frequency structure.
    Pointwise convolutions mix information across the
    antenna/feature representation.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 7,
    ):
        super().__init__()

        if kernel_size % 2 == 0:
            raise ValueError(
                "kernel_size must be odd."
            )

        padding = kernel_size // 2

        self.block = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                kernel_size=kernel_size,
                padding=padding,
                groups=channels,
                bias=False,
            ),
            nn.GroupNorm(
                num_groups=8,
                num_channels=channels,
            ),
            nn.GELU(),

            nn.Conv1d(
                channels,
                channels,
                kernel_size=1,
                bias=False,
            ),
            nn.GELU(),

            nn.Conv1d(
                channels,
                channels,
                kernel_size=1,
                bias=False,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return x + self.block(x)


class ResidualFrequencyCNN(nn.Module):
    """
    Compact CNN baseline for sparse CSI reconstruction.

    Input
    -----
    interpolated:
        [N, 2, B, F]

    mask:
        [N, 1, B, F]

    Output
    ------
    reconstructed:
        [N, 2, B, F]

    The model predicts only a residual correction to
    frequency-linear interpolation. Observed locations
    are preserved exactly.
    """

    def __init__(
        self,
        num_antennas: int = 32,
        hidden_dim: int = 64,
        num_blocks: int = 4,
        kernel_size: int = 7,
    ):
        super().__init__()

        if num_antennas <= 0:
            raise ValueError(
                "num_antennas must be positive."
            )

        self.num_antennas = int(
            num_antennas
        )

        # Each frequency token contains real and
        # imaginary CSI for all B antennas, plus
        # one observation-mask value.
        input_channels = (
            2 * self.num_antennas
            + 1
        )

        output_channels = (
            2 * self.num_antennas
        )

        self.stem = nn.Sequential(
            nn.Conv1d(
                input_channels,
                hidden_dim,
                kernel_size=5,
                padding=2,
                bias=False,
            ),
            nn.GroupNorm(
                num_groups=8,
                num_channels=hidden_dim,
            ),
            nn.GELU(),
        )

        self.blocks = nn.Sequential(
            *[
                ResidualFrequencyBlock(
                    channels=hidden_dim,
                    kernel_size=kernel_size,
                )
                for _ in range(
                    num_blocks
                )
            ]
        )

        self.head = nn.Conv1d(
            hidden_dim,
            output_channels,
            kernel_size=1,
        )

        # Start exactly from the interpolation baseline.
        # The network initially predicts zero residual and
        # learns only corrections at unobserved frequencies.
        nn.init.zeros_(
            self.head.weight
        )

        if self.head.bias is not None:
            nn.init.zeros_(
                self.head.bias
            )

    def forward(
        self,
        interpolated: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:

        if interpolated.ndim != 4:
            raise ValueError(
                "interpolated must have shape "
                "[N, 2, B, F]."
            )

        if interpolated.shape[1] != 2:
            raise ValueError(
                "CSI must contain real and "
                "imaginary channels."
            )

        batch_size = interpolated.shape[0]
        num_antennas = interpolated.shape[2]
        num_subcarriers = interpolated.shape[3]

        if num_antennas != self.num_antennas:
            raise ValueError(
                f"Model expects {self.num_antennas} "
                f"antennas but received "
                f"{num_antennas}."
            )

        if mask.shape != (
            batch_size,
            1,
            num_antennas,
            num_subcarriers,
        ):
            raise ValueError(
                "mask must have shape "
                "[N, 1, B, F]."
            )

        # The primary protocol has identical
        # observed frequencies for all antennas.
        frequency_mask = mask[
            :,
            :,
            0,
            :,
        ]

        antenna_masks = mask[
            :,
            0,
            :,
            :,
        ]

        reference_mask = frequency_mask[
            :,
            0,
            :,
        ].unsqueeze(1)

        if not torch.equal(
            antenna_masks > 0.5,
            reference_mask.expand(
                -1,
                num_antennas,
                -1,
            ) > 0.5,
        ):
            raise ValueError(
                "All antennas must share the "
                "same frequency observation mask."
            )

        # [N, 2, B, F]
        # ->
        # [N, 2B, F]
        interpolation_features = (
            interpolated.reshape(
                batch_size,
                2 * num_antennas,
                num_subcarriers,
            )
        )

        # [N, 2B+1, F]
        x = torch.cat(
            [
                interpolation_features,
                frequency_mask,
            ],
            dim=1,
        )

        x = self.stem(x)
        x = self.blocks(x)

        residual = self.head(x)

        # Never modify directly observed values.
        missing_frequency_mask = (
            1.0
            - frequency_mask
        )

        residual = (
            residual
            * missing_frequency_mask
        )

        prediction = (
            interpolation_features
            + residual
        )

        prediction = prediction.reshape(
            batch_size,
            2,
            num_antennas,
            num_subcarriers,
        )

        return prediction