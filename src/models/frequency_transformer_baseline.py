from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalFrequencyEncoding(nn.Module):
    """
    Parameter-free sinusoidal positional encoding along
    the frequency dimension.

    The encoding is generated dynamically, so the module
    is not tied to a learned 1024-position embedding table.
    """

    def __init__(
        self,
        dim: int,
    ) -> None:

        super().__init__()

        if dim <= 0:
            raise ValueError(
                "dim must be positive."
            )

        if dim % 2 != 0:
            raise ValueError(
                "Sinusoidal encoding requires "
                "an even embedding dimension."
            )

        self.dim = dim

    def forward(
        self,
        length: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:

        if length <= 0:
            raise ValueError(
                "length must be positive."
            )

        # Construct in FP32 for numerical stability,
        # then cast to the model/input dtype.
        positions = torch.arange(
            length,
            device=device,
            dtype=torch.float32,
        ).unsqueeze(1)

        dimension_indices = torch.arange(
            0,
            self.dim,
            2,
            device=device,
            dtype=torch.float32,
        )

        inverse_frequencies = torch.exp(
            dimension_indices
            * (
                -math.log(10000.0)
                / self.dim
            )
        )

        angles = (
            positions
            * inverse_frequencies.unsqueeze(0)
        )

        encoding = torch.zeros(
            length,
            self.dim,
            device=device,
            dtype=torch.float32,
        )

        encoding[
            :,
            0::2,
        ] = torch.sin(
            angles
        )

        encoding[
            :,
            1::2,
        ] = torch.cos(
            angles
        )

        return (
            encoding
            .to(dtype=dtype)
            .unsqueeze(0)
        )


class FrequencyTransformerBaseline(nn.Module):
    """
    Conventional full-self-attention Transformer baseline
    for sparse measured CSI reconstruction.

    Input
    -----
    interpolated:
        [N, 2, B, F]

        Linearly interpolated normalized CSI.

    mask:
        [N, 1, B, F]

        Frequency-shared observation mask.

    Architecture
    ------------
    Each frequency location becomes one Transformer token.

    At B=32, each token contains:
        32 real values
        32 imaginary values
        1 observation-mask value

    Therefore:
        input feature dimension = 2*B + 1 = 65.

    Full self-attention is applied across all F frequency
    tokens.

    The network predicts a residual correction and applies
    that correction only at missing locations. Observed CSI
    values are therefore preserved exactly.

    This baseline intentionally has a fixed antenna feature
    dimension. Unlike SCALE-Former, it is not intended as
    an antenna-count-scalable architecture.
    """

    def __init__(
        self,
        num_antennas: int = 32,
        dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 3,
        ff_dim: int = 128,
        dropout: float = 0.0,
    ) -> None:

        super().__init__()

        if num_antennas <= 0:
            raise ValueError(
                "num_antennas must be positive."
            )

        if dim <= 0:
            raise ValueError(
                "dim must be positive."
            )

        if num_heads <= 0:
            raise ValueError(
                "num_heads must be positive."
            )

        if dim % num_heads != 0:
            raise ValueError(
                "dim must be divisible "
                "by num_heads."
            )

        if num_layers <= 0:
            raise ValueError(
                "num_layers must be positive."
            )

        if ff_dim <= 0:
            raise ValueError(
                "ff_dim must be positive."
            )

        self.num_antennas = (
            num_antennas
        )

        self.dim = dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.ff_dim = ff_dim

        input_dim = (
            2 * num_antennas
            + 1
        )

        output_dim = (
            2 * num_antennas
        )

        self.input_projection = (
            nn.Linear(
                input_dim,
                dim,
            )
        )

        self.position_encoding = (
            SinusoidalFrequencyEncoding(
                dim=dim,
            )
        )

        encoder_layer = (
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=num_heads,
                dim_feedforward=ff_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
        )

        self.encoder = (
            nn.TransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=num_layers,
                norm=nn.LayerNorm(
                    dim
                ),
                enable_nested_tensor=False,
            )
        )

        self.residual_head = (
            nn.Linear(
                dim,
                output_dim,
            )
        )

        # Start exactly from the interpolation baseline.
        nn.init.zeros_(
            self.residual_head.weight
        )

        nn.init.zeros_(
            self.residual_head.bias
        )

    def _validate_inputs(
        self,
        interpolated: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:

        if interpolated.ndim != 4:
            raise ValueError(
                "interpolated must have shape "
                "[N, 2, B, F]."
            )

        if mask.ndim != 4:
            raise ValueError(
                "mask must have shape "
                "[N, 1, B, F]."
            )

        if (
            interpolated.shape[1]
            != 2
        ):
            raise ValueError(
                "interpolated channel dimension "
                "must equal 2."
            )

        if (
            mask.shape[1]
            != 1
        ):
            raise ValueError(
                "mask channel dimension "
                "must equal 1."
            )

        batch_size = (
            interpolated.shape[0]
        )

        num_antennas = (
            interpolated.shape[2]
        )

        num_frequencies = (
            interpolated.shape[3]
        )

        expected_mask_shape = (
            batch_size,
            1,
            num_antennas,
            num_frequencies,
        )

        if (
            tuple(mask.shape)
            != expected_mask_shape
        ):
            raise ValueError(
                "Mask shape does not match "
                "the CSI grid."
            )

        if (
            num_antennas
            != self.num_antennas
        ):
            raise ValueError(
                "FrequencyTransformerBaseline "
                f"was configured for "
                f"{self.num_antennas} antennas, "
                f"but received {num_antennas}."
            )

        # The experimental observation mask is frequency
        # shared across antennas. Reject antenna-dependent
        # masks rather than silently treating them as shared.
        reference_mask = (
            mask[
                :,
                :,
                0:1,
                :,
            ]
        )

        if not torch.equal(
            mask,
            reference_mask.expand_as(
                mask
            ),
        ):
            raise ValueError(
                "FrequencyTransformerBaseline "
                "requires a frequency-shared "
                "observation mask."
            )

    def forward(
        self,
        interpolated: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:

        self._validate_inputs(
            interpolated,
            mask,
        )

        batch_size = (
            interpolated.shape[0]
        )

        num_antennas = (
            interpolated.shape[2]
        )

        num_frequencies = (
            interpolated.shape[3]
        )

        # ----------------------------------------------------
        # CSI tokenization
        #
        # [N, 2, B, F]
        # -> [N, F, B, 2]
        # -> [N, F, 2B]
        # ----------------------------------------------------

        csi_features = (
            interpolated
            .permute(
                0,
                3,
                2,
                1,
            )
            .contiguous()
            .view(
                batch_size,
                num_frequencies,
                2 * num_antennas,
            )
        )

        # Since the mask is shared across antennas,
        # one scalar per frequency token is sufficient.
        frequency_mask = (
            mask[
                :,
                0,
                0,
                :,
            ]
            .unsqueeze(-1)
            .to(
                dtype=(
                    interpolated.dtype
                )
            )
        )

        token_features = (
            torch.cat(
                [
                    csi_features,
                    frequency_mask,
                ],
                dim=-1,
            )
        )

        tokens = (
            self.input_projection(
                token_features
            )
        )

        position_encoding = (
            self.position_encoding(
                length=(
                    num_frequencies
                ),
                device=tokens.device,
                dtype=tokens.dtype,
            )
        )

        tokens = (
            tokens
            + position_encoding
        )

        encoded = (
            self.encoder(
                tokens
            )
        )

        residual = (
            self.residual_head(
                encoded
            )
        )

        # ----------------------------------------------------
        # [N, F, 2B]
        # -> [N, F, B, 2]
        # -> [N, 2, B, F]
        # ----------------------------------------------------

        residual = (
            residual
            .view(
                batch_size,
                num_frequencies,
                num_antennas,
                2,
            )
            .permute(
                0,
                3,
                2,
                1,
            )
            .contiguous()
        )

        missing_mask = (
            1.0
            - mask.to(
                dtype=(
                    interpolated.dtype
                )
            )
        )

        reconstruction = (
            interpolated
            + residual
            * missing_mask
        )

        return reconstruction