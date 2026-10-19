import torch

from src.models.scaleformer import (
    ScaleFormer,
)


def make_inputs(
    batch_size=2,
    antennas=8,
    frequencies=128,
    pilot_step=10,
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
        ::pilot_step,
    ] = 1.0

    return interpolated, mask


def make_model():
    return ScaleFormer(
        dim=48,
        num_heads=4,
        num_latents=8,
        num_local_blocks=2,
        num_global_blocks=2,
        num_decoder_blocks=1,
        decoder_chunk_size=64,
    )


def test_output_shape():
    model = make_model()

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
        8,
        128,
    )


def test_initial_model_equals_interpolation():
    model = make_model()

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


def test_observed_values_preserved():
    model = make_model()

    interpolated, mask = (
        make_inputs()
    )

    # Make the residual head non-zero so that
    # preservation is tested independently of
    # zero initialization.
    with torch.no_grad():
        model.residual_head[-1].weight.normal_(
            mean=0.0,
            std=0.1,
        )

        model.residual_head[-1].bias.normal_(
            mean=0.0,
            std=0.1,
        )

    output = model(
        interpolated,
        mask,
    )

    observed_mask = mask.expand(
        -1,
        2,
        -1,
        -1,
    ).bool()

    assert torch.equal(
        output[observed_mask],
        interpolated[observed_mask],
    )


def test_dynamic_antenna_count():
    model = make_model()

    input_a, mask_a = make_inputs(
        batch_size=1,
        antennas=4,
        frequencies=64,
        pilot_step=8,
    )

    input_b, mask_b = make_inputs(
        batch_size=1,
        antennas=11,
        frequencies=64,
        pilot_step=8,
    )

    output_a = model(
        input_a,
        mask_a,
    )

    output_b = model(
        input_b,
        mask_b,
    )

    assert output_a.shape == (
        1,
        2,
        4,
        64,
    )

    assert output_b.shape == (
        1,
        2,
        11,
        64,
    )


def test_dynamic_frequency_size():
    model = make_model()

    input_a, mask_a = make_inputs(
        batch_size=1,
        antennas=6,
        frequencies=96,
        pilot_step=8,
    )

    input_b, mask_b = make_inputs(
        batch_size=1,
        antennas=6,
        frequencies=160,
        pilot_step=8,
    )

    output_a = model(
        input_a,
        mask_a,
    )

    output_b = model(
        input_b,
        mask_b,
    )

    assert output_a.shape == (
        1,
        2,
        6,
        96,
    )

    assert output_b.shape == (
        1,
        2,
        6,
        160,
    )


def test_gradients_exist():
    model = make_model()

    interpolated, mask = make_inputs(
        batch_size=1,
        antennas=4,
        frequencies=64,
        pilot_step=8,
    )

    # Give the head a small non-zero value so
    # gradients also reach earlier layers during
    # this structural unit test.
    with torch.no_grad():
        model.residual_head[-1].weight.normal_(
            mean=0.0,
            std=0.01,
        )

    output = model(
        interpolated,
        mask,
    )

    loss = output.square().mean()

    loss.backward()

    parameters_with_gradients = [
        name
        for name, parameter
        in model.named_parameters()
        if (
            parameter.requires_grad
            and parameter.grad is not None
        )
    ]

    assert len(
        parameters_with_gradients
    ) > 0

    assert any(
        "local_encoder"
        in name
        for name in
        parameters_with_gradients
    )

    assert any(
        "global_blocks"
        in name
        for name in
        parameters_with_gradients
    )

    assert any(
        "decoder_blocks"
        in name
        for name in
        parameters_with_gradients
    )


def test_parameter_count():
    model = make_model()

    parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        f"SCALE-Former trainable parameters: "
        f"{parameters:,}"
    )

    assert parameters < 300_000


def test_gpu_full_grid_forward():
    if not torch.cuda.is_available():
        return

    device = torch.device(
        "cuda"
    )

    model = ScaleFormer(
        dim=48,
        num_heads=4,
        num_latents=16,
        num_local_blocks=2,
        num_global_blocks=2,
        num_decoder_blocks=1,
        decoder_chunk_size=4096,
    ).to(device)

    interpolated = torch.randn(
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

    # Approximately 5%.
    mask[
        :,
        :,
        :,
        ::20,
    ] = 1.0

    torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        output = model(
            interpolated,
            mask,
        )

    peak_memory_mb = (
        torch.cuda.max_memory_allocated()
        / (1024 ** 2)
    )

    print(
        f"Full-grid GPU peak memory: "
        f"{peak_memory_mb:.2f} MiB"
    )

    assert output.shape == (
        1,
        2,
        32,
        1024,
    )

    assert output.is_cuda

    assert peak_memory_mb < 3500.0


def test_antenna_dependent_mask_rejected():
    model = make_model()

    interpolated = torch.randn(
        1,
        2,
        4,
        32,
    )

    mask = torch.zeros(
        1,
        1,
        4,
        32,
    )

    mask[
        0,
        0,
        0,
        ::4,
    ] = 1.0

    mask[
        0,
        0,
        1,
        1::4,
    ] = 1.0

    mask[
        0,
        0,
        2,
        ::4,
    ] = 1.0

    mask[
        0,
        0,
        3,
        ::4,
    ] = 1.0

    try:
        model(
            interpolated,
            mask,
        )

    except ValueError:
        return

    raise AssertionError(
        "Expected ValueError for "
        "antenna-dependent masks."
    )

def test_ablation_variants_forward():

    variants = [
        "full",
        "no_global",
        "no_local",
        "no_coordinates",
    ]

    interpolated, mask = make_inputs(
        batch_size=1,
        antennas=6,
        frequencies=96,
        pilot_step=8,
    )

    for variant in variants:

        model = ScaleFormer(
            dim=48,
            num_heads=4,
            num_latents=8,
            num_local_blocks=2,
            num_global_blocks=2,
            num_decoder_blocks=1,
            decoder_chunk_size=64,
            variant=variant,
        )

        output = model(
            interpolated,
            mask,
        )

        assert output.shape == (
            1,
            2,
            6,
            96,
        )


def test_ablation_initializes_to_interpolation():

    variants = [
        "full",
        "no_global",
        "no_local",
        "no_coordinates",
    ]

    interpolated, mask = make_inputs(
        batch_size=1,
        antennas=4,
        frequencies=64,
        pilot_step=8,
    )

    for variant in variants:

        model = ScaleFormer(
            variant=variant,
        )

        output = model(
            interpolated,
            mask,
        )

        assert torch.equal(
            output,
            interpolated,
        )


def test_ablation_parameter_counts():

    variants = [
        "full",
        "no_global",
        "no_local",
        "no_coordinates",
    ]

    print()

    for variant in variants:

        model = ScaleFormer(
            dim=48,
            num_heads=4,
            num_latents=16,
            num_local_blocks=2,
            num_global_blocks=2,
            num_decoder_blocks=1,
            decoder_chunk_size=4096,
            variant=variant,
        )

        parameters = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )

        print(
            f"{variant:>14}: "
            f"{parameters:,} parameters"
        )

        assert parameters > 0


def test_invalid_ablation_rejected():

    try:

        ScaleFormer(
            variant="invalid_variant"
        )

    except ValueError:
        return

    raise AssertionError(
        "Expected invalid variant "
        "to raise ValueError."
    )

if __name__ == "__main__":
    tests = [
        test_ablation_variants_forward,
        test_ablation_initializes_to_interpolation,
        test_ablation_parameter_counts,
        test_invalid_ablation_rejected,
        test_output_shape,
        test_initial_model_equals_interpolation,
        test_observed_values_preserved,
        test_dynamic_antenna_count,
        test_dynamic_frequency_size,
        test_gradients_exist,
        test_parameter_count,
        test_gpu_full_grid_forward,
        test_antenna_dependent_mask_rejected,
    ]

    for test in tests:
        test()

        print(
            f"PASSED: {test.__name__}"
        )

    print(
        "\nALL SCALE-FORMER TESTS PASSED"
    )