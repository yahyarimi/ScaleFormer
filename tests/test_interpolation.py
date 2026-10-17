import torch

from src.models.interpolation import (
    frequency_linear_interpolate,
)


def test_linear_signal_exact_reconstruction():
    """
    A linear frequency response should be reconstructed
    exactly between observed locations.
    """
    frequencies = torch.arange(
        8,
        dtype=torch.float32,
    )

    real = (
        2.0 * frequencies
        + 1.0
    )

    imag = (
        -0.5 * frequencies
        + 3.0
    )

    target = torch.stack(
        [
            real,
            imag,
        ],
        dim=0,
    )

    target = target.view(
        1,
        2,
        1,
        8,
    )

    mask = torch.zeros(
        1,
        1,
        1,
        8,
    )

    # Include both endpoints to make the linear
    # function exactly reconstructable everywhere.
    mask[
        0,
        0,
        0,
        [0, 2, 4, 6, 7],
    ] = 1.0

    observed = (
        target
        * mask
    )

    reconstruction = (
        frequency_linear_interpolate(
            observed,
            mask,
        )
    )

    assert torch.allclose(
        reconstruction,
        target,
        atol=1e-6,
    )


def test_observed_values_are_preserved():
    torch.manual_seed(
        123
    )

    target = torch.randn(
        2,
        2,
        4,
        16,
    )

    mask = torch.zeros(
        2,
        1,
        4,
        16,
    )

    pilot_indices = [
        0,
        3,
        6,
        9,
        12,
        15,
    ]

    mask[
        :,
        :,
        :,
        pilot_indices,
    ] = 1.0

    observed = (
        target
        * mask
    )

    reconstruction = (
        frequency_linear_interpolate(
            observed,
            mask,
        )
    )

    expanded_mask = mask.expand(
        -1,
        2,
        -1,
        -1,
    ).bool()

    assert torch.equal(
        reconstruction[
            expanded_mask
        ],
        target[
            expanded_mask
        ],
    )


def test_output_shape_matches_input():
    observed = torch.zeros(
        3,
        2,
        32,
        1024,
    )

    mask = torch.zeros(
        3,
        1,
        32,
        1024,
    )

    mask[
        :,
        :,
        :,
        ::10,
    ] = 1.0

    reconstruction = (
        frequency_linear_interpolate(
            observed,
            mask,
        )
    )

    assert reconstruction.shape == (
        3,
        2,
        32,
        1024,
    )


def test_single_observation():
    target = torch.randn(
        1,
        2,
        2,
        8,
    )

    mask = torch.zeros(
        1,
        1,
        2,
        8,
    )

    mask[
        :,
        :,
        :,
        3,
    ] = 1.0

    observed = (
        target
        * mask
    )

    reconstruction = (
        frequency_linear_interpolate(
            observed,
            mask,
        )
    )

    expected = target[
        :,
        :,
        :,
        3,
    ].unsqueeze(
        -1
    ).expand(
        -1,
        -1,
        -1,
        8,
    )

    assert torch.equal(
        reconstruction,
        expected,
    )


def test_mismatched_antenna_masks_rejected():
    observed = torch.zeros(
        1,
        2,
        2,
        8,
    )

    mask = torch.zeros(
        1,
        1,
        2,
        8,
    )

    mask[
        0,
        0,
        0,
        [0, 4],
    ] = 1.0

    mask[
        0,
        0,
        1,
        [1, 5],
    ] = 1.0

    try:
        frequency_linear_interpolate(
            observed,
            mask,
        )

    except ValueError:
        return

    raise AssertionError(
        "Expected ValueError for "
        "antenna-dependent frequency masks."
    )


if __name__ == "__main__":
    tests = [
        test_linear_signal_exact_reconstruction,
        test_observed_values_are_preserved,
        test_output_shape_matches_input,
        test_single_observation,
        test_mismatched_antenna_masks_rejected,
    ]

    for test in tests:
        test()

        print(
            f"PASSED: {test.__name__}"
        )

    print(
        "\nALL INTERPOLATION TESTS PASSED"
    )