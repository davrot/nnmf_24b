import torch
from PositionalEncoding import PositionalEncoding
from SequentialSplit import SequentialSplit
from NNMF2dGrouped import NNMF2dGrouped
from Functional2Layer import Functional2Layer


def add_block(
    network: torch.nn.Sequential,
    embed_dim: int,
    num_heads: int,
    dtype: torch.dtype,
    device: torch.device,
    example_image: torch.Tensor,
    mlp_ratio: int = 4,
    block_id: int = 0,
    iterations: int = 20,
    padding: int = 1,
    kernel_size: tuple[int, int] = (3, 3),
) -> torch.Tensor | None:

    # ###########
    # Attention #
    # ###########

    example_image_a: torch.Tensor = example_image.clone()
    example_image_b: torch.Tensor = example_image.clone()

    attention_a_sequential = torch.nn.Sequential()

    attention_a_sequential.add_module(
        "Attention Layer Norm 1 [Pre-Permute]",
        Functional2Layer(func=torch.permute, dims=(0, 2, 3, 1)),
    )
    example_image_a = attention_a_sequential[-1](example_image_a)

    attention_a_sequential.add_module(
        "Attention Layer Norm 1",
        torch.nn.LayerNorm(
            normalized_shape=example_image_a.shape[-1],
            eps=1e-06,
            bias=True,
            dtype=dtype,
            device=device,
        ),
    )
    example_image_a = attention_a_sequential[-1](example_image_a)

    attention_a_sequential.add_module(
        "Attention Layer Norm 1 [Post-Permute]",
        Functional2Layer(func=torch.permute, dims=(0, 3, 1, 2)),
    )
    example_image_a = attention_a_sequential[-1](example_image_a)

    attention_a_sequential.add_module(
        "Attention Clamp Layer", Functional2Layer(func=torch.clamp, min=1e-6)
    )
    example_image_a = attention_a_sequential[-1](example_image_a)

    backup_image_dim = example_image_a.shape[1]

    attention_a_sequential.add_module(
        "Attention Zero Padding Layer", torch.nn.ZeroPad2d(padding=padding)
    )
    example_image_a = attention_a_sequential[-1](example_image_a)

    # I need the output size
    mock_output_shape = (
        torch.nn.functional.conv2d(
            torch.zeros(
                1,
                1,
                example_image_a.shape[2],
                example_image_a.shape[3],
            ),
            torch.zeros((1, 1, kernel_size[0], kernel_size[1])),
            stride=1,
            padding=0,
            dilation=1,
        )
        .squeeze(0)
        .squeeze(0)
    ).shape

    attention_a_sequential.add_module(
        "Attention Windowing [Part 1]",
        torch.nn.Unfold(
            kernel_size=(kernel_size[-2], kernel_size[-1]),
            dilation=1,
            padding=0,
            stride=1,
        ),
    )
    example_image_a = attention_a_sequential[-1](example_image_a)

    attention_a_sequential.add_module(
        "Attention Windowing [Part 2]",
        torch.nn.Fold(
            output_size=mock_output_shape,
            kernel_size=(1, 1),
            dilation=1,
            padding=0,
            stride=1,
        ),
    )
    example_image_a = attention_a_sequential[-1](example_image_a)

    attention_a_sequential.add_module("Attention NNMFConv2d", torch.nn.ReLU())
    example_image_a = attention_a_sequential[-1](example_image_a)

    attention_a_sequential.add_module(
        "Attention NNMFConv2d",
        NNMF2dGrouped(
            in_channels=example_image_a.shape[1],
            out_channels=embed_dim,
            groups=num_heads,
            device=device,
            dtype=dtype,
            iterations=iterations,
        ),
    )
    example_image_a = attention_a_sequential[-1](example_image_a)

    attention_a_sequential.add_module(
        "Attention Layer Norm 2 [Pre-Permute]",
        Functional2Layer(func=torch.permute, dims=(0, 2, 3, 1)),
    )
    example_image_a = attention_a_sequential[-1](example_image_a)

    attention_a_sequential.add_module(
        "Attention Layer Norm 2",
        torch.nn.LayerNorm(
            normalized_shape=example_image_a.shape[-1],
            eps=1e-06,
            bias=True,
            dtype=dtype,
            device=device,
        ),
    )
    example_image_a = attention_a_sequential[-1](example_image_a)

    attention_a_sequential.add_module(
        "Attention Layer Norm 2 [Post-Permute]",
        Functional2Layer(func=torch.permute, dims=(0, 3, 1, 2)),
    )
    example_image_a = attention_a_sequential[-1](example_image_a)

    attention_a_sequential.add_module(
        "Attention Conv2d Layer ",
        torch.nn.Conv2d(
            in_channels=example_image_a.shape[1],
            out_channels=backup_image_dim,
            kernel_size=1,
            dtype=dtype,
            device=device,
        ),
    )
    example_image_a = attention_a_sequential[-1](example_image_a)

    attention_b_sequential = torch.nn.Sequential()
    attention_b_sequential.add_module(
        "Attention Identity for the skip", torch.nn.Identity()
    )
    example_image_b = attention_b_sequential[-1](example_image_b)

    assert example_image_b.shape == example_image_a.shape

    network.add_module(
        f"Block Number {block_id} [Attention]",
        SequentialSplit(
            torch.nn.Sequential(
                attention_a_sequential,
                attention_b_sequential,
            ),
            combine="SUM",
        ),
    )
    example_image = network[-1](example_image)

    # ######
    # MLP #
    # #####

    example_image_a = example_image.clone()
    example_image_b = example_image.clone()

    mlp_a_sequential = torch.nn.Sequential()

    mlp_a_sequential.add_module(
        "MLP [Pre-Permute]", Functional2Layer(func=torch.permute, dims=(0, 2, 3, 1))
    )
    example_image_a = mlp_a_sequential[-1](example_image_a)

    mlp_a_sequential.add_module(
        "MLP Layer Norm",
        torch.nn.LayerNorm(
            normalized_shape=example_image_a.shape[-1],
            eps=1e-06,
            bias=True,
            dtype=dtype,
            device=device,
        ),
    )
    example_image_a = mlp_a_sequential[-1](example_image_a)

    mlp_a_sequential.add_module(
        "MLP Linear Layer A",
        torch.nn.Linear(
            example_image_a.shape[-1],
            int(example_image_a.shape[-1] * mlp_ratio),
            dtype=dtype,
            device=device,
        ),
    )
    example_image_a = mlp_a_sequential[-1](example_image_a)

    mlp_a_sequential.add_module("MLP GELU", torch.nn.GELU())
    example_image_a = mlp_a_sequential[-1](example_image_a)

    mlp_a_sequential.add_module(
        "MLP Linear Layer B",
        torch.nn.Linear(
            example_image_a.shape[-1],
            int(example_image_a.shape[-1] // mlp_ratio),
            dtype=dtype,
            device=device,
        ),
    )
    example_image_a = mlp_a_sequential[-1](example_image_a)

    mlp_a_sequential.add_module(
        "MLP [Post-Permute]", Functional2Layer(func=torch.permute, dims=(0, 3, 1, 2))
    )
    example_image_a = mlp_a_sequential[-1](example_image_a)

    mlp_b_sequential = torch.nn.Sequential()
    mlp_b_sequential.add_module("MLP Identity for the skip", torch.nn.Identity())

    example_image_b = attention_b_sequential[-1](example_image_b)

    assert example_image_b.shape == example_image_a.shape

    network.add_module(
        f"Block Number {block_id} [MLP]",
        SequentialSplit(
            torch.nn.Sequential(
                mlp_a_sequential,
                mlp_b_sequential,
            ),
            combine="SUM",
        ),
    )
    example_image = network[-1](example_image)

    return example_image


def make_network(
    in_channels: int = 3,
    dims: list[int] = [72, 72, 72],
    embed_dims: list[int] = [192, 192, 192],
    n_classes: int = 10,
    heads: int = 12,
    example_image_shape: list[int] = [1, 3, 28, 28],
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
    iterations: int = 20,
) -> torch.nn.Sequential:

    assert device is not None

    network = torch.nn.Sequential()

    example_image: torch.Tensor = torch.zeros(
        example_image_shape, dtype=dtype, device=device
    )

    network.add_module(
        "Encode Conv2d",
        torch.nn.Conv2d(
            in_channels,
            dims[0],
            kernel_size=4,
            stride=4,
            padding=0,
            dtype=dtype,
            device=device,
        ),
    )
    example_image = network[-1](example_image)

    network.add_module(
        "Encode Offset",
        PositionalEncoding(
            [example_image.shape[-3], example_image.shape[-2], example_image.shape[-1]]
        ).to(device=device),
    )
    example_image = network[-1](example_image)

    network.add_module(
        "Encode Layer Norm [Pre-Permute]",
        Functional2Layer(func=torch.permute, dims=(0, 2, 3, 1)),
    )
    example_image = network[-1](example_image)

    network.add_module(
        "Encode Layer Norm",
        torch.nn.LayerNorm(
            normalized_shape=example_image.shape[-1],
            eps=1e-06,
            bias=True,
            dtype=dtype,
            device=device,
        ),
    )
    example_image = network[-1](example_image)

    network.add_module(
        "Encode Layer Norm [Post-Permute]",
        Functional2Layer(func=torch.permute, dims=(0, 3, 1, 2)),
    )
    example_image = network[-1](example_image)

    for i in range(len(dims)):
        example_image = add_block(
            network=network,
            embed_dim=embed_dims[i],
            num_heads=heads,
            mlp_ratio=2,
            block_id=i,
            example_image=example_image,
            dtype=dtype,
            device=device,
            iterations=iterations,
        )

    network.add_module(
        "Spatial Mean Layer", Functional2Layer(func=torch.mean, dim=(-1, -2))
    )
    example_image = network[-1](example_image)

    network.add_module(
        "Final Linear Layer",
        torch.nn.Linear(example_image.shape[-1], n_classes, dtype=dtype, device=device),
    )
    example_image = network[-1](example_image)

    network.add_module("Final Softmax Layer", torch.nn.Softmax(dim=-1))
    example_image = network[-1](example_image)

    assert example_image.ndim == 2
    assert example_image.shape[0] == example_image_shape[0]
    assert example_image.shape[1] == n_classes

    return network


if __name__ == "__main__":
    network = make_network(device=torch.device("cuda:0"))
    print(network)

    number_of_parameter: int = 0
    for name, param in network.named_parameters():
        print(f"Parameter name: {name}, Shape: {param.shape}")
        number_of_parameter += param.numel()

    print("Number of total parameters:", number_of_parameter)
