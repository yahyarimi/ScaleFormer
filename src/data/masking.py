from __future__ import annotations

import torch


SUPPORTED_PATTERNS = {
    "comb",
    "random",
    "block",
}


def _validate_inputs(
    num_antennas: int,
    num_subcarriers: int,
    density: float,
    pattern: str,
) -> None:
    if num_antennas <= 0:
        raise ValueError(
            "num_antennas must be positive."
        )

    if num_subcarriers <= 0:
        raise ValueError(
            "num_subcarriers must be positive."
        )

    if not (0.0 < density <= 1.0):
        raise ValueError(
            "density must satisfy 0 < density <= 1."
        )

    if pattern not in SUPPORTED_PATTERNS:
        raise ValueError(
            f"Unsupported pattern '{pattern}'. "
            f"Supported patterns: "
            f"{sorted(SUPPORTED_PATTERNS)}"
        )


def _num_observed(
    num_subcarriers: int,
    density: float,
) -> int:
    """
    Convert the requested observation density into
    an integer number of observed subcarriers.
    """
    return max(
        1,
        min(
            num_subcarriers,
            int(round(
                density * num_subcarriers
            )),
        ),
    )


def _comb_indices(
    num_subcarriers: int,
    num_observed: int,
) -> torch.Tensor:
    """
    Return approximately equally spaced frequency
    indices without duplicates.
    """
    if num_observed == num_subcarriers:
        return torch.arange(
            num_subcarriers,
            dtype=torch.long,
        )

    # floor(i * F / K) generates K unique,
    # approximately equally spaced indices for K <= F.
    indices = torch.floor(
        torch.arange(
            num_observed,
            dtype=torch.float64,
        )
        * num_subcarriers
        / num_observed
    ).to(torch.long)

    return indices


def _random_indices(
    num_subcarriers: int,
    num_observed: int,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(
        device="cpu"
    )

    generator.manual_seed(seed)

    permutation = torch.randperm(
        num_subcarriers,
        generator=generator,
    )

    indices = permutation[
        :num_observed
    ]

    return torch.sort(indices).values


def _block_indices(
    num_subcarriers: int,
    num_observed: int,
    seed: int,
) -> torch.Tensor:
    if num_observed == num_subcarriers:
        return torch.arange(
            num_subcarriers,
            dtype=torch.long,
        )

    generator = torch.Generator(
        device="cpu"
    )

    generator.manual_seed(seed)

    max_start = (
        num_subcarriers
        - num_observed
    )

    start = int(
        torch.randint(
            low=0,
            high=max_start + 1,
            size=(1,),
            generator=generator,
        ).item()
    )

    return torch.arange(
        start,
        start + num_observed,
        dtype=torch.long,
    )


def make_frequency_mask(
    num_antennas: int,
    num_subcarriers: int,
    density: float,
    pattern: str = "comb",
    seed: int = 0,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Create a frequency-domain observation mask.

    The same selected frequency coefficients are
    observed across all receive antennas.

    Returns
    -------
    mask:
        Tensor with shape [B, F].
    """
    _validate_inputs(
        num_antennas=num_antennas,
        num_subcarriers=num_subcarriers,
        density=density,
        pattern=pattern,
    )

    num_observed = _num_observed(
        num_subcarriers,
        density,
    )

    if pattern == "comb":
        indices = _comb_indices(
            num_subcarriers,
            num_observed,
        )

    elif pattern == "random":
        indices = _random_indices(
            num_subcarriers,
            num_observed,
            seed,
        )

    elif pattern == "block":
        indices = _block_indices(
            num_subcarriers,
            num_observed,
            seed,
        )

    else:
        raise RuntimeError(
            "Pattern validation failed."
        )

    one_dimensional_mask = torch.zeros(
        num_subcarriers,
        dtype=dtype,
    )

    one_dimensional_mask[
        indices
    ] = 1.0

    mask = one_dimensional_mask.unsqueeze(
        0
    ).expand(
        num_antennas,
        -1,
    ).clone()

    return mask


def apply_observation_mask(
    csi: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Apply a [B, F] observation mask to CSI.

    Supported CSI shapes:
        [2, B, F]
        [N, 2, B, F]

    The two CSI channels correspond to the
    real and imaginary components.
    """
    if mask.ndim != 2:
        raise ValueError(
            "mask must have shape [B, F]."
        )

    if csi.ndim == 3:
        # [2, B, F]
        if csi.shape[0] != 2:
            raise ValueError(
                "3-D CSI must have shape "
                "[2, B, F]."
            )

        if tuple(csi.shape[1:]) != tuple(
            mask.shape
        ):
            raise ValueError(
                "CSI and mask dimensions "
                "do not match."
            )

        return csi * mask.unsqueeze(0)

    if csi.ndim == 4:
        # [N, 2, B, F]
        if csi.shape[1] != 2:
            raise ValueError(
                "4-D CSI must have shape "
                "[N, 2, B, F]."
            )

        if tuple(csi.shape[2:]) != tuple(
            mask.shape
        ):
            raise ValueError(
                "CSI and mask dimensions "
                "do not match."
            )

        return (
            csi
            * mask.unsqueeze(0).unsqueeze(0)
        )

    raise ValueError(
        "CSI must have shape [2, B, F] "
        "or [N, 2, B, F]."
    )