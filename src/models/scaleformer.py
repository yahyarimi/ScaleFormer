from __future__ import annotations

import torch
from torch import nn


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        expansion: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()

        hidden_dim = dim * expansion

        self.net = nn.Sequential(
            nn.Linear(
                dim,
                hidden_dim,
            ),
            nn.GELU(),
            nn.Dropout(
                dropout
            ),
            nn.Linear(
                hidden_dim,
                dim,
            ),
            nn.Dropout(
                dropout
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.net(x)


class LocalFrequencyBlock(nn.Module):
    """
    Shared local frequency block.

    The same block is applied independently to each
    receive antenna, so the parameter count does not
    depend on the number of antennas.
    """

    def __init__(
        self,
        dim: int,
        kernel_size: int = 7,
    ):
        super().__init__()

        if kernel_size % 2 == 0:
            raise ValueError(
                "kernel_size must be odd."
            )

        padding = kernel_size // 2

        self.depthwise = nn.Conv1d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=padding,
            groups=dim,
            bias=False,
        )

        self.norm = nn.GroupNorm(
            num_groups=4,
            num_channels=dim,
        )

        self.pointwise = nn.Sequential(
            nn.Conv1d(
                dim,
                2 * dim,
                kernel_size=1,
            ),
            nn.GELU(),
            nn.Conv1d(
                2 * dim,
                dim,
                kernel_size=1,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        residual = x

        x = self.depthwise(x)
        x = self.norm(x)
        x = self.pointwise(x)

        return residual + x


class SharedLocalFrequencyEncoder(nn.Module):
    """
    Local encoder shared across all antennas.

    Input:
        interpolated: [N, 2, B, F]
        mask:         [N, 1, B, F]

    Output:
        features:     [N, B, F, D]
    """

    def __init__(
        self,
        dim: int = 48,
        num_blocks: int = 2,
        kernel_size: int = 7,
    ):
        super().__init__()

        self.dim = dim

        self.stem = nn.Sequential(
            nn.Conv1d(
                3,
                dim,
                kernel_size=5,
                padding=2,
                bias=False,
            ),
            nn.GroupNorm(
                num_groups=4,
                num_channels=dim,
            ),
            nn.GELU(),
        )

        self.blocks = nn.Sequential(
            *[
                LocalFrequencyBlock(
                    dim=dim,
                    kernel_size=kernel_size,
                )
                for _ in range(
                    num_blocks
                )
            ]
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

        # [N, 2, B, F]
        # ->
        # [N, B, 2, F]
        csi = interpolated.permute(
            0,
            2,
            1,
            3,
        )

        mask_per_antenna = mask.permute(
            0,
            2,
            1,
            3,
        )

        # [N, B, 3, F]
        x = torch.cat(
            [
                csi,
                mask_per_antenna,
            ],
            dim=2,
        )

        # Process every antenna using the
        # exact same convolutional weights.
        x = x.reshape(
            batch_size * num_antennas,
            3,
            num_subcarriers,
        )

        x = self.stem(x)
        x = self.blocks(x)

        # [NB, D, F]
        # ->
        # [N, B, F, D]
        x = x.reshape(
            batch_size,
            num_antennas,
            self.dim,
            num_subcarriers,
        )

        x = x.permute(
            0,
            1,
            3,
            2,
        )

        return x

class SharedPointwiseEncoder(nn.Module):
    """
    Pointwise ablation encoder.

    Each antenna-frequency location is encoded
    independently from:
        real CSI,
        imaginary CSI,
        observation mask.

    There is no convolution and therefore no local
    frequency receptive field.
    """

    def __init__(
        self,
        dim: int = 48,
    ):
        super().__init__()

        self.dim = dim

        self.net = nn.Sequential(
            nn.Linear(
                3,
                dim,
            ),
            nn.GELU(),
            nn.Linear(
                dim,
                dim,
            ),
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

        batch_size = interpolated.shape[0]
        num_antennas = interpolated.shape[2]
        num_subcarriers = interpolated.shape[3]

        if mask.shape != (
            batch_size,
            1,
            num_antennas,
            num_subcarriers,
        ):
            raise ValueError(
                "mask and CSI dimensions do not match."
            )

        # [N,2,B,F] -> [N,B,F,2]
        csi = interpolated.permute(
            0,
            2,
            3,
            1,
        )

        # [N,1,B,F] -> [N,B,F,1]
        mask_features = mask.permute(
            0,
            2,
            3,
            1,
        )

        x = torch.cat(
            [
                csi,
                mask_features,
            ],
            dim=-1,
        )

        # [N,B,F,D]
        return self.net(x)

class CoordinateEncoder(nn.Module):
    """
    Encode normalized antenna-frequency coordinates.

    Coordinates are generated dynamically from the
    requested B x F grid.
    """

    def __init__(
        self,
        dim: int,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(
                2,
                dim,
            ),
            nn.GELU(),
            nn.Linear(
                dim,
                dim,
            ),
        )

    def forward(
        self,
        num_antennas: int,
        num_subcarriers: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:

        if num_antennas == 1:
            antenna_coordinates = torch.zeros(
                1,
                device=device,
                dtype=dtype,
            )
        else:
            antenna_coordinates = torch.linspace(
                -1.0,
                1.0,
                steps=num_antennas,
                device=device,
                dtype=dtype,
            )

        if num_subcarriers == 1:
            frequency_coordinates = torch.zeros(
                1,
                device=device,
                dtype=dtype,
            )
        else:
            frequency_coordinates = torch.linspace(
                -1.0,
                1.0,
                steps=num_subcarriers,
                device=device,
                dtype=dtype,
            )

        antenna_grid, frequency_grid = (
            torch.meshgrid(
                antenna_coordinates,
                frequency_coordinates,
                indexing="ij",
            )
        )

        coordinates = torch.stack(
            [
                antenna_grid,
                frequency_grid,
            ],
            dim=-1,
        )

        # [B, F, 2]
        encoded = self.net(
            coordinates
        )

        # [B, F, D]
        return encoded


class LatentGlobalBlock(nn.Module):
    """
    Compress sparse observed CSI tokens into a small
    latent global representation.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.latent_norm_cross = (
            nn.LayerNorm(dim)
        )

        self.observed_norm = (
            nn.LayerNorm(dim)
        )

        self.cross_attention = (
            nn.MultiheadAttention(
                embed_dim=dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
        )

        self.latent_norm_self = (
            nn.LayerNorm(dim)
        )

        self.self_attention = (
            nn.MultiheadAttention(
                embed_dim=dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
        )

        self.ff_norm = (
            nn.LayerNorm(dim)
        )

        self.ff = FeedForward(
            dim=dim,
            expansion=2,
            dropout=dropout,
        )

    def forward(
        self,
        latents: torch.Tensor,
        observed_tokens: torch.Tensor,
    ) -> torch.Tensor:

        query = self.latent_norm_cross(
            latents
        )

        memory = self.observed_norm(
            observed_tokens
        )

        cross_output, _ = (
            self.cross_attention(
                query=query,
                key=memory,
                value=memory,
                need_weights=False,
            )
        )

        latents = (
            latents
            + cross_output
        )

        normalized = self.latent_norm_self(
            latents
        )

        self_output, _ = (
            self.self_attention(
                query=normalized,
                key=normalized,
                value=normalized,
                need_weights=False,
            )
        )

        latents = (
            latents
            + self_output
        )

        latents = (
            latents
            + self.ff(
                self.ff_norm(
                    latents
                )
            )
        )

        return latents


class CoordinateDecoderBlock(nn.Module):
    """
    Dynamic coordinate query -> latent global memory.

    The number of query points may change without
    changing learned parameters.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.query_norm = (
            nn.LayerNorm(dim)
        )

        self.latent_norm = (
            nn.LayerNorm(dim)
        )

        self.cross_attention = (
            nn.MultiheadAttention(
                embed_dim=dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
        )

        self.ff_norm = (
            nn.LayerNorm(dim)
        )

        self.ff = FeedForward(
            dim=dim,
            expansion=2,
            dropout=dropout,
        )

    def forward(
        self,
        query_tokens: torch.Tensor,
        latents: torch.Tensor,
    ) -> torch.Tensor:

        q = self.query_norm(
            query_tokens
        )

        memory = self.latent_norm(
            latents
        )

        output, _ = (
            self.cross_attention(
                query=q,
                key=memory,
                value=memory,
                need_weights=False,
            )
        )

        query_tokens = (
            query_tokens
            + output
        )

        query_tokens = (
            query_tokens
            + self.ff(
                self.ff_norm(
                    query_tokens
                )
            )
        )

        return query_tokens


class ScaleFormer(nn.Module):
    """
    SCALE-Former for sparse measured CSI reconstruction.

    Key properties
    --------------
    1. Shared local frequency encoding.
    2. Sparse observed CSI tokenization.
    3. Small latent global bottleneck.
    4. Dynamic antenna-frequency coordinate queries.
    5. Chunked decoding for bounded GPU memory.
    6. Exact preservation of measured locations.
    """

    def __init__(
        self,
        dim: int = 48,
        num_heads: int = 4,
        num_latents: int = 16,
        num_local_blocks: int = 2,
        num_global_blocks: int = 2,
        num_decoder_blocks: int = 1,
        local_kernel_size: int = 7,
        decoder_chunk_size: int = 4096,
        dropout: float = 0.0,
        variant: str = "full",
    ):
        super().__init__()

        valid_variants = {
            "full",
            "no_global",
            "no_local",
            "no_coordinates",
        }

        if variant not in valid_variants:
            raise ValueError(
                f"Unknown SCALE-Former variant: {variant}. "
                f"Expected one of {sorted(valid_variants)}."
            )

        self.variant = variant

        self.use_local = (
            variant != "no_local"
        )

        self.use_global = (
            variant != "no_global"
        )

        self.use_coordinates = (
            variant != "no_coordinates"
        )

        if dim % num_heads != 0:
            raise ValueError(
                "dim must be divisible by num_heads."
            )

        if num_latents <= 0:
            raise ValueError(
                "num_latents must be positive."
            )

        if decoder_chunk_size <= 0:
            raise ValueError(
                "decoder_chunk_size must be positive."
            )

        self.dim = int(dim)

        self.num_latents = int(
            num_latents
        )

        self.decoder_chunk_size = int(
            decoder_chunk_size
        )

        if self.use_local:
            self.local_encoder = (
                SharedLocalFrequencyEncoder(
                    dim=dim,
                    num_blocks=num_local_blocks,
                    kernel_size=local_kernel_size,
                )
            )

            self.pointwise_encoder = None

        else:
            self.local_encoder = None

            self.pointwise_encoder = (
                SharedPointwiseEncoder(
                    dim=dim,
                )
            )

        if self.use_coordinates:
            self.coordinate_encoder = (
                CoordinateEncoder(
                    dim=dim
                )
            )
        else:
            self.coordinate_encoder = None

        if self.use_global:

            self.latent_tokens = nn.Parameter(
                torch.randn(
                    1,
                    num_latents,
                    dim,
                )
                * 0.02
            )

            self.global_blocks = nn.ModuleList(
                [
                    LatentGlobalBlock(
                        dim=dim,
                        num_heads=num_heads,
                        dropout=dropout,
                    )
                    for _ in range(
                        num_global_blocks
                    )
                ]
            )

            self.decoder_blocks = nn.ModuleList(
                [
                    CoordinateDecoderBlock(
                        dim=dim,
                        num_heads=num_heads,
                        dropout=dropout,
                    )
                    for _ in range(
                        num_decoder_blocks
                    )
                ]
            )

        else:

            self.register_parameter(
                "latent_tokens",
                None,
            )

            self.global_blocks = nn.ModuleList()
            self.decoder_blocks = nn.ModuleList()

        self.residual_head = nn.Sequential(
            nn.LayerNorm(
                dim
            ),
            nn.Linear(
                dim,
                2,
            ),
        )

        # Start exactly from interpolation.
        final_linear = (
            self.residual_head[-1]
        )

        nn.init.zeros_(
            final_linear.weight
        )

        if final_linear.bias is not None:
            nn.init.zeros_(
                final_linear.bias
            )

    @staticmethod
    def _validate_mask(
        mask: torch.Tensor,
    ) -> None:

        if mask.ndim != 4:
            raise ValueError(
                "mask must have shape "
                "[N, 1, B, F]."
            )

        if mask.shape[1] != 1:
            raise ValueError(
                "mask channel dimension must be 1."
            )

        # Current primary experiment requires
        # identical frequency pilots for all antennas.
        antenna_masks = (
            mask[:, 0] > 0.5
        )

        reference = antenna_masks[
            :,
            0:1,
            :,
        ]

        if not torch.equal(
            antenna_masks,
            reference.expand_as(
                antenna_masks
            ),
        ):
            raise ValueError(
                "SCALE-Former currently expects the "
                "same observed frequency positions "
                "across all antennas."
            )

    def _extract_observed_tokens(
        self,
        features: torch.Tensor,
        coordinates: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Select only measured antenna-frequency positions.

        features:
            [N, B, F, D]

        coordinates:
            [B, F, D]

        mask:
            [N, 1, B, F]

        Returns:
            [N, K, D]
        """
        batch_size = features.shape[0]
        num_antennas = features.shape[1]
        num_subcarriers = features.shape[2]

        coordinates = (
            coordinates
            .unsqueeze(0)
            .expand(
                batch_size,
                -1,
                -1,
                -1,
            )
        )

        tokens = (
            features
            + coordinates
        )

        tokens = tokens.reshape(
            batch_size,
            num_antennas
            * num_subcarriers,
            self.dim,
        )

        selector = (
            mask[:, 0]
            .reshape(
                batch_size,
                num_antennas
                * num_subcarriers,
            )
            > 0.5
        )

        counts = selector.sum(
            dim=1
        )

        if not torch.all(
            counts == counts[0]
        ):
            raise ValueError(
                "All samples within a batch must "
                "contain the same number of "
                "observed coefficients."
            )

        selected = []

        for batch_index in range(
            batch_size
        ):
            selected.append(
                tokens[
                    batch_index,
                    selector[
                        batch_index
                    ],
                ]
            )

        return torch.stack(
            selected,
            dim=0,
        )

    def _decode_chunks(
        self,
        query_tokens: torch.Tensor,
        latents: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decode large grids in bounded-memory chunks.
        """
        total_queries = (
            query_tokens.shape[1]
        )

        outputs = []

        for start in range(
            0,
            total_queries,
            self.decoder_chunk_size,
        ):
            end = min(
                start
                + self.decoder_chunk_size,
                total_queries,
            )

            chunk = query_tokens[
                :,
                start:end,
                :,
            ]

            for block in self.decoder_blocks:
                chunk = block(
                    chunk,
                    latents,
                )

            residual = (
                self.residual_head(
                    chunk
                )
            )

            outputs.append(
                residual
            )

        return torch.cat(
            outputs,
            dim=1,
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

        if mask.shape != (
            batch_size,
            1,
            num_antennas,
            num_subcarriers,
        ):
            raise ValueError(
                "mask and CSI dimensions do not match."
            )

        self._validate_mask(
            mask
        )

        if self.use_local:

            base_features = (
                self.local_encoder(
                    interpolated,
                    mask,
                )
            )

        else:

            base_features = (
                self.pointwise_encoder(
                    interpolated,
                    mask,
                )
            )

        if self.use_coordinates:

            coordinates = (
                self.coordinate_encoder(
                    num_antennas=num_antennas,
                    num_subcarriers=num_subcarriers,
                    device=interpolated.device,
                    dtype=interpolated.dtype,
                )
            )

        else:

            coordinates = torch.zeros(
                num_antennas,
                num_subcarriers,
                self.dim,
                device=interpolated.device,
                dtype=interpolated.dtype,
            )

        if self.use_global:

            observed_tokens = (
                self._extract_observed_tokens(
                    features=base_features,
                    coordinates=coordinates,
                    mask=mask,
                )
            )

            latents = (
                self.latent_tokens
                .expand(
                    batch_size,
                    -1,
                    -1,
                )
            )

            for block in self.global_blocks:

                latents = block(
                    latents,
                    observed_tokens,
                )

            queries = (
                base_features
                + coordinates.unsqueeze(0)
            )

            queries = queries.reshape(
                batch_size,
                num_antennas
                * num_subcarriers,
                self.dim,
            )

            residual = self._decode_chunks(
                query_tokens=queries,
                latents=latents,
            )

        else:

            # Local-only ablation.
            #
            # No latent global pathway and no decoder
            # cross-attention. Each grid feature is mapped
            # directly to a residual correction.
            queries = (
                base_features
                + coordinates.unsqueeze(0)
            )

            queries = queries.reshape(
                batch_size,
                num_antennas
                * num_subcarriers,
                self.dim,
            )

            residual = (
                self.residual_head(
                    queries
                )
            )

        # [N, BF, 2]
        # ->
        # [N, 2, B, F]
        residual = residual.reshape(
            batch_size,
            num_antennas,
            num_subcarriers,
            2,
        )

        residual = residual.permute(
            0,
            3,
            1,
            2,
        )

        # Corrections are allowed only at
        # unobserved locations.
        missing_mask = (
            1.0
            - mask
        )

        prediction = (
            interpolated
            + residual
            * missing_mask
        )

        return prediction