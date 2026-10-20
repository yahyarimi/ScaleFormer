import pytest
import torch

from src.models.frequency_transformer_baseline import (
    FrequencyTransformerBaseline,
)


def make_input(
    batch_size=2,
    antennas=32,
    frequencies=128,
):

    x = torch.randn(
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
        ::10,
    ] = 1.0

    return x, mask


def test_output_shape():

    model = (
        FrequencyTransformerBaseline()
    )

    x, mask = make_input()

    output = model(
        x,
        mask,
    )

    assert (
        output.shape
        == x.shape
    )


def test_zero_initialization_returns_interpolation():

    model = (
        FrequencyTransformerBaseline()
    )

    x, mask = make_input()

    output = model(
        x,
        mask,
    )

    assert torch.allclose(
        output,
        x,
        atol=0.0,
        rtol=0.0,
    )


def test_observed_values_are_preserved():

    model = (
        FrequencyTransformerBaseline()
    )

    # Make the residual head nonzero so this test
    # does not pass merely because of zero initialization.
    torch.nn.init.normal_(
        model.residual_head.weight,
        mean=0.0,
        std=0.01,
    )

    torch.nn.init.normal_(
        model.residual_head.bias,
        mean=0.0,
        std=0.01,
    )

    x, mask = make_input()

    output = model(
        x,
        mask,
    )

    expanded_mask = (
        mask.expand(
            -1,
            2,
            -1,
            -1,
        ).bool()
    )

    assert torch.equal(
        output[
            expanded_mask
        ],
        x[
            expanded_mask
        ],
    )


def test_parameter_count():

    model = (
        FrequencyTransformerBaseline(
            num_antennas=32,
            dim=64,
            num_heads=4,
            num_layers=3,
            ff_dim=128,
            dropout=0.0,
        )
    )

    params = sum(
        parameter.numel()
        for parameter
        in model.parameters()
        if parameter.requires_grad
    )

    assert params == 108_928


def test_variable_frequency_length():

    model = (
        FrequencyTransformerBaseline()
    )

    x, mask = make_input(
        frequencies=256
    )

    output = model(
        x,
        mask,
    )

    assert (
        output.shape
        == x.shape
    )


def test_rejects_wrong_antenna_count():

    model = (
        FrequencyTransformerBaseline(
            num_antennas=32
        )
    )

    x, mask = make_input(
        antennas=16
    )

    with pytest.raises(
        ValueError,
        match="configured for 32 antennas",
    ):

        model(
            x,
            mask,
        )


def test_rejects_antenna_dependent_mask():

    model = (
        FrequencyTransformerBaseline()
    )

    x, mask = make_input()

    mask[
        0,
        0,
        1,
        1,
    ] = 1.0

    with pytest.raises(
        ValueError,
        match="frequency-shared",
    ):

        model(
            x,
            mask,
        )


def test_gradients_reach_residual_head():

    model = (
        FrequencyTransformerBaseline()
    )

    x, mask = make_input(
        batch_size=1,
        frequencies=64,
    )

    output = model(
        x,
        mask,
    )

    loss = (
        output.square().mean()
    )

    loss.backward()

    assert (
        model
        .residual_head
        .weight
        .grad
        is not None
    )

    assert torch.isfinite(
        model
        .residual_head
        .weight
        .grad
    ).all()


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available.",
)
def test_full_grid_cuda_forward():

    device = torch.device(
        "cuda"
    )

    model = (
        FrequencyTransformerBaseline()
        .to(device)
        .eval()
    )

    x = torch.randn(
        1,
        2,
        32,
        1024,
        device=device,
    )

    mask = torch.zeros(
        1,
        1,
        32,
        1024,
        device=device,
    )

    mask[
        :,
        :,
        :,
        ::20,
    ] = 1.0

    with torch.inference_mode():

        with torch.amp.autocast(
            device_type="cuda",
            dtype=torch.float16,
        ):

            output = model(
                x,
                mask,
            )

    torch.cuda.synchronize()

    assert (
        output.shape
        == x.shape
    )

    assert torch.isfinite(
        output
    ).all()