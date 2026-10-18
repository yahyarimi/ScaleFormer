import torch

from src.models.cnn_baseline import (
    ResidualFrequencyCNN,
)


def make_inputs(
    batch_size=2,
    antennas=32,
    frequencies=1024,
):
    torch.manual_seed(123)

    interpolated = torch.randn(
        batch_size,
        2,
        antennas,
        frequencies,
    )

    mask = torch.zeros(
        batch_size,
        1,
        antennas,
        frequencies,
    )

    mask[
        :,
        :,
        :,
        ::20,
    ] = 1.0

    return interpolated, mask


def test_output_shape():
    model = ResidualFrequencyCNN()

    interpolated, mask = (
        make_inputs()
    )

    output = model(
        interpolated,
        mask,
    )

    assert output.shape == (
        2,
        2,
        32,
        1024,
    )


def test_observed_values_preserved():
    model = ResidualFrequencyCNN()

    interpolated, mask = (
        make_inputs()
    )

    output = model(
        interpolated,
        mask,
    )

    expanded_mask = mask.expand(
        -1,
        2,
        -1,
        -1,
    ).bool()

    assert torch.equal(
        output[expanded_mask],
        interpolated[expanded_mask],
    )


def test_gradients_exist():
    model = ResidualFrequencyCNN()

    interpolated, mask = (
        make_inputs(
            batch_size=1,
            frequencies=128,
        )
    )

    output = model(
        interpolated,
        mask,
    )

    loss = output.square().mean()

    loss.backward()

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    assert len(
        trainable_parameters
    ) > 0

    assert any(
        parameter.grad is not None
        for parameter in trainable_parameters
    )


def test_gpu_forward_if_available():
    if not torch.cuda.is_available():
        return

    device = torch.device(
        "cuda"
    )

    model = (
        ResidualFrequencyCNN()
        .to(device)
    )

    interpolated, mask = (
        make_inputs(
            batch_size=2,
            frequencies=256,
        )
    )

    interpolated = (
        interpolated.to(device)
    )

    mask = mask.to(device)

    output = model(
        interpolated,
        mask,
    )

    assert output.is_cuda

    assert output.shape == (
        2,
        2,
        32,
        256,
    )


def test_parameter_count_is_compact():
    model = ResidualFrequencyCNN()

    parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        f"CNN trainable parameters: "
        f"{parameters:,}"
    )

    assert parameters < 250_000

def test_initial_model_equals_interpolation():
    model = ResidualFrequencyCNN()

    interpolated, mask = (
        make_inputs()
    )

    output = model(
        interpolated,
        mask,
    )

    assert torch.equal(
        output,
        interpolated,
    )

if __name__ == "__main__":
    tests = [
        test_output_shape,
        test_observed_values_preserved,
        test_gradients_exist,
        test_gpu_forward_if_available,
        test_parameter_count_is_compact,
        test_initial_model_equals_interpolation,
    ]

    for test in tests:
        test()

        print(
            f"PASSED: {test.__name__}"
        )

    print(
        "\nALL CNN BASELINE TESTS PASSED"
    )