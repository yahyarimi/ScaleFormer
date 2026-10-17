from __future__ import annotations

import torch


def _validate_inputs(
    csi: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    if csi.ndim != 4:
        raise ValueError(
            "CSI must have shape [N, 2, B, F]."
        )

    if csi.shape[1] != 2:
        raise ValueError(
            "CSI must contain real and imaginary channels."
        )

    if mask.ndim != 4:
        raise ValueError(
            "Mask must have shape [N, 1, B, F]."
        )

    if mask.shape[0] != csi.shape[0]:
        raise ValueError(
            "CSI and mask batch sizes must match."
        )

    if mask.shape[1] != 1:
        raise ValueError(
            "Mask channel dimension must equal 1."
        )

    if tuple(mask.shape[2:]) != tuple(
        csi.shape[2:]
    ):
        raise ValueError(
            "CSI and mask antenna/frequency "
            "dimensions must match."
        )


def frequency_linear_interpolate(
    observed: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Perform linear interpolation along the frequency axis.

    Parameters
    ----------
    observed:
        Sparse normalized CSI tensor with shape
        [N, 2, B, F].

    mask:
        Observation mask with shape
        [N, 1, B, F].

    Returns
    -------
    reconstructed:
        Interpolated CSI with shape
        [N, 2, B, F].

    Notes
    -----
    The interpolation operates independently for each
    sample, antenna, and real/imaginary component.

    Outside the range covered by observed frequencies,
    nearest-observation extrapolation is used.
    """
    _validate_inputs(
        observed,
        mask,
    )

    device = observed.device
    dtype = observed.dtype

    batch_size = observed.shape[0]
    num_antennas = observed.shape[2]
    num_subcarriers = observed.shape[3]

    output = torch.empty_like(
        observed
    )

    frequency_positions = torch.arange(
        num_subcarriers,
        device=device,
        dtype=torch.long,
    )

    for batch_index in range(batch_size):

        # The primary frequency-pilot protocol uses
        # identical frequency locations across antennas.
        reference_mask = mask[
            batch_index,
            0,
            0,
        ] > 0.5

        observed_indices = torch.where(
            reference_mask
        )[0]

        if observed_indices.numel() == 0:
            raise ValueError(
                "At least one frequency observation "
                "is required."
            )

        # Verify that every antenna uses the same
        # frequency observation pattern.
        antenna_masks = mask[
            batch_index,
            0,
        ] > 0.5

        if not torch.all(
            antenna_masks
            == reference_mask.unsqueeze(0)
        ):
            raise ValueError(
                "frequency_linear_interpolate requires "
                "the same frequency mask for all antennas."
            )

        if observed_indices.numel() == 1:
            only_index = observed_indices[0]

            value = observed[
                batch_index,
                :,
                :,
                only_index,
            ]

            output[
                batch_index
            ] = value.unsqueeze(
                -1
            ).expand(
                -1,
                -1,
                num_subcarriers,
            )

            continue

        # Find the observed frequency immediately to
        # the right of every target frequency.
        right_positions = torch.searchsorted(
            observed_indices,
            frequency_positions,
        )

        right_positions = torch.clamp(
            right_positions,
            0,
            observed_indices.numel() - 1,
        )

        left_positions = torch.clamp(
            right_positions - 1,
            0,
            observed_indices.numel() - 1,
        )

        left_indices = observed_indices[
            left_positions
        ]

        right_indices = observed_indices[
            right_positions
        ]

        # Frequencies before the first observation use
        # the first pilot; frequencies after the last
        # observation use the last pilot.
        before_first = (
            frequency_positions
            <= observed_indices[0]
        )

        after_last = (
            frequency_positions
            >= observed_indices[-1]
        )

        left_indices[
            before_first
        ] = observed_indices[0]

        right_indices[
            before_first
        ] = observed_indices[0]

        left_indices[
            after_last
        ] = observed_indices[-1]

        right_indices[
            after_last
        ] = observed_indices[-1]

        left_values = observed[
            batch_index,
            :,
            :,
            left_indices,
        ]

        right_values = observed[
            batch_index,
            :,
            :,
            right_indices,
        ]

        left_float = left_indices.to(
            dtype=dtype
        )

        right_float = right_indices.to(
            dtype=dtype
        )

        target_float = frequency_positions.to(
            dtype=dtype
        )

        denominator = (
            right_float
            - left_float
        )

        safe_denominator = torch.where(
            denominator == 0,
            torch.ones_like(
                denominator
            ),
            denominator,
        )

        alpha = (
            target_float
            - left_float
        ) / safe_denominator

        alpha = torch.where(
            denominator == 0,
            torch.zeros_like(
                alpha
            ),
            alpha,
        )

        alpha = alpha.view(
            1,
            1,
            num_subcarriers,
        )

        reconstructed = (
            left_values
            + alpha
            * (
                right_values
                - left_values
            )
        )

        # Ensure observed CSI values are preserved
        # exactly rather than only numerically.
        reconstructed[
            :,
            :,
            reference_mask,
        ] = observed[
            batch_index,
            :,
            :,
            reference_mask,
        ]

        output[
            batch_index
        ] = reconstructed

    return output