from __future__ import annotations

import math

import torch


def _validate_csi_pair(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> None:
    if prediction.shape != target.shape:
        raise ValueError(
            "Prediction and target shapes must match. "
            f"Got prediction={tuple(prediction.shape)} "
            f"and target={tuple(target.shape)}."
        )

    if prediction.ndim not in (3, 4):
        raise ValueError(
            "CSI tensors must have shape "
            "[2, B, F] or [N, 2, B, F]."
        )

    complex_axis = 0 if prediction.ndim == 3 else 1

    if prediction.shape[complex_axis] != 2:
        raise ValueError(
            "CSI must contain exactly two channels: "
            "real and imaginary."
        )


def _squared_complex_energy(
    tensor: torch.Tensor,
) -> torch.Tensor:
    """
    Sum squared real and imaginary components.

    For a complex coefficient h = a + jb,

        |h|^2 = a^2 + b^2.

    Summing the two-channel real representation
    therefore gives the complex Frobenius energy.
    """
    return torch.sum(
        tensor.to(torch.float64) ** 2
    )


def nmse_sums(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    missing_only: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Return numerator and denominator required for
    aggregate complex NMSE.

    Parameters
    ----------
    prediction:
        Physical-scale CSI with shape [2, B, F]
        or [N, 2, B, F].

    target:
        Physical-scale reference CSI with the same
        shape as prediction.

    mask:
        Observation mask. Expected shape:
            [1, B, F] for one sample, or
            [N, 1, B, F] for a batch.

        A value of 1 indicates an observed frequency
        coefficient.

    missing_only:
        If True, calculate error/energy only on
        unobserved locations.

    Returns
    -------
    numerator:
        Sum of complex squared reconstruction errors.

    denominator:
        Sum of complex squared reference-channel energy.
    """
    _validate_csi_pair(
        prediction,
        target,
    )

    prediction = prediction.to(
        torch.float64
    )

    target = target.to(
        torch.float64
    )

    error = prediction - target

    if missing_only:
        if mask is None:
            raise ValueError(
                "mask is required when "
                "missing_only=True."
            )

        if prediction.ndim == 3:
            if mask.shape != (
                1,
                prediction.shape[1],
                prediction.shape[2],
            ):
                raise ValueError(
                    "For one CSI sample, mask must "
                    "have shape [1, B, F]."
                )

            missing_mask = (
                1.0
                - mask.to(
                    dtype=torch.float64,
                    device=prediction.device,
                )
            )

            missing_mask = missing_mask.expand(
                2,
                -1,
                -1,
            )

        else:
            if mask.shape != (
                prediction.shape[0],
                1,
                prediction.shape[2],
                prediction.shape[3],
            ):
                raise ValueError(
                    "For batched CSI, mask must have "
                    "shape [N, 1, B, F]."
                )

            missing_mask = (
                1.0
                - mask.to(
                    dtype=torch.float64,
                    device=prediction.device,
                )
            )

            missing_mask = missing_mask.expand(
                -1,
                2,
                -1,
                -1,
            )

        error = error * missing_mask
        target = target * missing_mask

    numerator = _squared_complex_energy(
        error
    )

    denominator = _squared_complex_energy(
        target
    )

    return numerator, denominator


def nmse_linear(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    missing_only: bool = False,
    eps: float = 1e-15,
) -> torch.Tensor:
    """
    Compute aggregate complex NMSE in linear scale.
    """
    numerator, denominator = nmse_sums(
        prediction=prediction,
        target=target,
        mask=mask,
        missing_only=missing_only,
    )

    return numerator / denominator.clamp_min(
        eps
    )


def nmse_db(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    missing_only: bool = False,
    eps: float = 1e-15,
) -> torch.Tensor:
    """
    Compute aggregate complex NMSE in decibels.
    """
    value = nmse_linear(
        prediction=prediction,
        target=target,
        mask=mask,
        missing_only=missing_only,
        eps=eps,
    )

    return 10.0 * torch.log10(
        value.clamp_min(eps)
    )


class AggregateNMSE:
    """
    Accumulate NMSE numerator and denominator over
    many mini-batches.

    This implements:

        sum_n ||H_hat_n - H_n||^2
        --------------------------------
        sum_n ||H_n||^2

    rather than averaging mini-batch NMSE values.
    """

    def __init__(
        self,
        missing_only: bool = False,
    ):
        self.missing_only = bool(
            missing_only
        )

        self.reset()

    def reset(self) -> None:
        self.numerator = 0.0
        self.denominator = 0.0

    @torch.no_grad()
    def update(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> None:
        numerator, denominator = nmse_sums(
            prediction=prediction,
            target=target,
            mask=mask,
            missing_only=self.missing_only,
        )

        self.numerator += float(
            numerator.detach().cpu().item()
        )

        self.denominator += float(
            denominator.detach().cpu().item()
        )

    def compute_linear(
        self,
        eps: float = 1e-15,
    ) -> float:
        denominator = max(
            self.denominator,
            eps,
        )

        return (
            self.numerator
            / denominator
        )

    def compute_db(
        self,
        eps: float = 1e-15,
    ) -> float:
        linear = max(
            self.compute_linear(
                eps=eps
            ),
            eps,
        )

        return 10.0 * math.log10(
            linear
        )