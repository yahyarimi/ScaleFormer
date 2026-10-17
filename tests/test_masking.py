import torch

from src.data.masking import (
    apply_observation_mask,
    make_frequency_mask,
)


B = 32
F = 1024


def test_comb_mask_shape():
    mask = make_frequency_mask(
        num_antennas=B,
        num_subcarriers=F,
        density=0.10,
        pattern="comb",
    )

    assert mask.shape == (B, F)


def test_comb_density():
    density = 0.10

    mask = make_frequency_mask(
        num_antennas=B,
        num_subcarriers=F,
        density=density,
        pattern="comb",
    )

    expected = round(
        density * F
    )

    observed = int(
        mask[0].sum().item()
    )

    assert observed == expected


def test_all_antennas_share_frequency_mask():
    mask = make_frequency_mask(
        num_antennas=B,
        num_subcarriers=F,
        density=0.20,
        pattern="comb",
    )

    reference = mask[0]

    for antenna in range(1, B):
        assert torch.equal(
            mask[antenna],
            reference,
        )


def test_random_mask_is_reproducible():
    mask_1 = make_frequency_mask(
        num_antennas=B,
        num_subcarriers=F,
        density=0.12,
        pattern="random",
        seed=123,
    )

    mask_2 = make_frequency_mask(
        num_antennas=B,
        num_subcarriers=F,
        density=0.12,
        pattern="random",
        seed=123,
    )

    assert torch.equal(
        mask_1,
        mask_2,
    )


def test_random_seed_changes_pattern():
    mask_1 = make_frequency_mask(
        num_antennas=B,
        num_subcarriers=F,
        density=0.12,
        pattern="random",
        seed=123,
    )

    mask_2 = make_frequency_mask(
        num_antennas=B,
        num_subcarriers=F,
        density=0.12,
        pattern="random",
        seed=456,
    )

    assert not torch.equal(
        mask_1,
        mask_2,
    )


def test_block_mask_is_contiguous():
    density = 0.10

    mask = make_frequency_mask(
        num_antennas=B,
        num_subcarriers=F,
        density=density,
        pattern="block",
        seed=42,
    )

    indices = torch.where(
        mask[0] > 0
    )[0]

    expected_count = round(
        density * F
    )

    assert len(indices) == expected_count

    if len(indices) > 1:
        differences = (
            indices[1:]
            - indices[:-1]
        )

        assert torch.all(
            differences == 1
        )


def test_density_one_observes_everything():
    mask = make_frequency_mask(
        num_antennas=B,
        num_subcarriers=F,
        density=1.0,
        pattern="comb",
    )

    assert torch.all(
        mask == 1
    )


def test_apply_mask_single_sample():
    csi = torch.ones(
        2,
        B,
        F,
    )

    mask = make_frequency_mask(
        num_antennas=B,
        num_subcarriers=F,
        density=0.10,
        pattern="comb",
    )

    observed = apply_observation_mask(
        csi,
        mask,
    )

    assert observed.shape == (
        2,
        B,
        F,
    )

    expected_nonzero = int(
        mask.sum().item()
        * 2
    )

    actual_nonzero = int(
        torch.count_nonzero(
            observed
        ).item()
    )

    assert (
        actual_nonzero
        == expected_nonzero
    )


def test_invalid_density():
    try:
        make_frequency_mask(
            num_antennas=B,
            num_subcarriers=F,
            density=0.0,
        )

    except ValueError:
        return

    raise AssertionError(
        "Expected ValueError for "
        "density=0."
    )


if __name__ == "__main__":
    tests = [
        test_comb_mask_shape,
        test_comb_density,
        test_all_antennas_share_frequency_mask,
        test_random_mask_is_reproducible,
        test_random_seed_changes_pattern,
        test_block_mask_is_contiguous,
        test_density_one_observes_everything,
        test_apply_mask_single_sample,
        test_invalid_density,
    ]

    for test in tests:
        test()
        print(
            f"PASSED: {test.__name__}"
        )

    print(
        "\nALL MASK TESTS PASSED"
    )