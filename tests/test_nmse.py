import math

import torch

from src.metrics.nmse import (
    AggregateNMSE,
    nmse_db,
    nmse_linear,
)


def make_reference():
    """
    Small deterministic complex CSI tensor.

    Shape:
        [1, 2, 2, 4]

    Channel 0 = real
    Channel 1 = imaginary
    """
    real = torch.tensor(
        [
            [
                [1.0, 2.0, 3.0, 4.0],
                [2.0, 3.0, 4.0, 5.0],
            ]
        ]
    )

    imag = torch.tensor(
        [
            [
                [0.5, -1.0, 1.5, -2.0],
                [1.0, -1.5, 2.0, -2.5],
            ]
        ]
    )

    target = torch.stack(
        [real, imag],
        dim=1,
    )

    return target


def test_perfect_prediction():
    target = make_reference()
    prediction = target.clone()

    value = nmse_linear(
        prediction,
        target,
    )

    assert value.item() == 0.0


def test_zero_prediction_is_zero_db():
    target = make_reference()

    prediction = torch.zeros_like(
        target
    )

    linear = nmse_linear(
        prediction,
        target,
    )

    db = nmse_db(
        prediction,
        target,
    )

    assert torch.allclose(
        linear,
        torch.tensor(
            1.0,
            dtype=linear.dtype,
        ),
        atol=1e-12,
    )

    assert abs(
        db.item()
    ) < 1e-10


def test_half_scale_prediction():
    target = make_reference()

    prediction = (
        0.5 * target
    )

    linear = nmse_linear(
        prediction,
        target,
    )

    db = nmse_db(
        prediction,
        target,
    )

    expected_linear = 0.25

    expected_db = (
        10.0
        * math.log10(
            expected_linear
        )
    )

    assert abs(
        linear.item()
        - expected_linear
    ) < 1e-12

    assert abs(
        db.item()
        - expected_db
    ) < 1e-10


def test_missing_only_nmse():
    target = make_reference()

    prediction = target.clone()

    # Observe frequency indices 0 and 2.
    # Frequencies 1 and 3 are unobserved.
    mask = torch.tensor(
        [
            [
                [
                    [1.0, 0.0, 1.0, 0.0],
                    [1.0, 0.0, 1.0, 0.0],
                ]
            ]
        ]
    )

    # Corrupt only the unobserved locations.
    prediction[:, :, :, 1] = 0.0
    prediction[:, :, :, 3] = 0.0

    missing_nmse = nmse_linear(
        prediction,
        target,
        mask=mask,
        missing_only=True,
    )

    # Since prediction is zero at every missing
    # position, error energy equals target energy.
    assert abs(
        missing_nmse.item()
        - 1.0
    ) < 1e-12


def test_observed_errors_excluded_from_missing_nmse():
    target = make_reference()

    prediction = target.clone()

    mask = torch.tensor(
        [
            [
                [
                    [1.0, 0.0, 1.0, 0.0],
                    [1.0, 0.0, 1.0, 0.0],
                ]
            ]
        ]
    )

    # Introduce very large errors ONLY at observed
    # positions. Missing-region NMSE must remain zero.
    prediction[:, :, :, 0] += 100.0
    prediction[:, :, :, 2] -= 100.0

    missing_nmse = nmse_linear(
        prediction,
        target,
        mask=mask,
        missing_only=True,
    )

    assert missing_nmse.item() == 0.0


def test_aggregate_accumulator():
    target = make_reference()

    prediction_1 = (
        0.5 * target
    )

    prediction_2 = (
        0.5 * target
    )

    metric = AggregateNMSE()

    metric.update(
        prediction_1,
        target,
    )

    metric.update(
        prediction_2,
        target,
    )

    linear = metric.compute_linear()
    db = metric.compute_db()

    assert abs(
        linear - 0.25
    ) < 1e-12

    assert abs(
        db
        - 10.0 * math.log10(0.25)
    ) < 1e-10


def test_batch_and_sequential_aggregation_match():
    target = make_reference()

    prediction_1 = (
        0.25 * target
    )

    prediction_2 = (
        0.75 * target
    )

    targets = torch.cat(
        [
            target,
            target,
        ],
        dim=0,
    )

    predictions = torch.cat(
        [
            prediction_1,
            prediction_2,
        ],
        dim=0,
    )

    batch_value = nmse_linear(
        predictions,
        targets,
    ).item()

    metric = AggregateNMSE()

    metric.update(
        prediction_1,
        target,
    )

    metric.update(
        prediction_2,
        target,
    )

    sequential_value = (
        metric.compute_linear()
    )

    assert abs(
        batch_value
        - sequential_value
    ) < 1e-12


if __name__ == "__main__":
    tests = [
        test_perfect_prediction,
        test_zero_prediction_is_zero_db,
        test_half_scale_prediction,
        test_missing_only_nmse,
        test_observed_errors_excluded_from_missing_nmse,
        test_aggregate_accumulator,
        test_batch_and_sequential_aggregation_match,
    ]

    for test in tests:
        test()

        print(
            f"PASSED: {test.__name__}"
        )

    print(
        "\nALL NMSE TESTS PASSED"
    )