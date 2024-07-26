import torch
from non_linear_weigth_function import non_linear_weigth_function


class NNMF2dGrouped(torch.nn.Module):

    in_channels: int
    out_channels: int
    weight: torch.Tensor
    iterations: int
    epsilon: float | None
    init_min: float
    init_max: float
    beta: torch.Tensor | None
    positive_function_type: int
    local_learning: bool
    local_learning_kl: bool
    groups: int

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        groups: int = 1,
        device=None,
        dtype=None,
        iterations: int = 20,
        epsilon: float | None = None,
        init_min: float = 0.0,
        init_max: float = 1.0,
        beta: float | None = None,
        positive_function_type: int = 0,
        local_learning: bool = False,
        local_learning_kl: bool = False,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}

        super().__init__()

        self.positive_function_type = positive_function_type
        self.init_min = init_min
        self.init_max = init_max

        self.groups = groups
        assert (
            in_channels % self.groups == 0
        ), f"Can't divide without rest {in_channels} / {self.groups}"
        self.in_channels = in_channels // self.groups
        assert (
            out_channels % self.groups == 0
        ), f"Can't divide without rest {out_channels} / {self.groups}"
        self.out_channels = out_channels // self.groups

        self.iterations = iterations
        self.local_learning = local_learning
        self.local_learning_kl = local_learning_kl

        self.weight = torch.nn.parameter.Parameter(
            torch.empty(
                (self.groups, self.out_channels, self.in_channels), **factory_kwargs
            )
        )

        if beta is not None:
            self.beta = torch.nn.parameter.Parameter(torch.empty((1), **factory_kwargs))
            self.beta.data[0] = beta
        else:
            self.beta = None

        self.reset_parameters()
        self.functional_nnmf2d_grouped = FunctionalNNMF2dGrouped.apply

        self.epsilon = epsilon

    def extra_repr(self) -> str:
        s: str = f"{self.in_channels}, {self.out_channels}"

        if self.epsilon is not None:
            s += f", epsilon={self.epsilon}"
        s += f", pfunctype={self.positive_function_type}"
        s += f", local_learning={self.local_learning}"
        s += f", groups={self.groups}"

        if self.local_learning:
            s += f", local_learning_kl={self.local_learning_kl}"

        return s

    def reset_parameters(self) -> None:
        torch.nn.init.uniform_(self.weight, a=self.init_min, b=self.init_max)

    def forward(self, input: torch.Tensor) -> torch.Tensor:

        positive_weights = non_linear_weigth_function(
            self.weight, self.beta, self.positive_function_type
        )
        positive_weights = positive_weights / (
            positive_weights.sum(dim=-1, keepdim=True) + 10e-20
        )
        assert self.groups * self.in_channels == input.shape[1]

        input = input.reshape(
            (
                input.shape[0],
                self.groups,
                self.in_channels,
                input.shape[-2],
                input.shape[-1],
            )
        )
        input = input / (input.sum(dim=2, keepdim=True) + 10e-20)

        h_dyn = self.functional_nnmf2d_grouped(
            input,
            positive_weights,
            self.out_channels,
            self.iterations,
            self.epsilon,
            self.local_learning,
            self.local_learning_kl,
        )

        h_dyn = h_dyn.reshape(
            (
                h_dyn.shape[0],
                h_dyn.shape[1] * h_dyn.shape[2],
                h_dyn.shape[3],
                h_dyn.shape[4],
            )
        )
        h_dyn = h_dyn / (h_dyn.sum(dim=1, keepdim=True) + 10e-20)

        return h_dyn


@torch.jit.script
def grouped_linear_einsum_h_weights(h, weights):
    return torch.einsum("bgoxy,goi->bgixy", h, weights)


@torch.jit.script
def grouped_linear_einsum_reconstruction_weights(reconstruction, weights):
    return torch.einsum("bgixy,goi->bgoxy", reconstruction, weights)


@torch.jit.script
def grouped_linear_einsum_h_input(h, reconstruction):
    return torch.einsum("bgoxy,bgixy->goi", h, reconstruction)


class FunctionalNNMF2dGrouped(torch.autograd.Function):

    @staticmethod
    def forward(  # type: ignore
        ctx,
        input: torch.Tensor,
        weight: torch.Tensor,
        out_channels: int,
        iterations: int,
        epsilon: float | None,
        local_learning: bool,
        local_learning_kl: bool,
    ) -> torch.Tensor:

        # Prepare h
        h = torch.full(
            (
                input.shape[0],
                input.shape[1],
                out_channels,
                input.shape[-2],
                input.shape[-1],
            ),
            1.0 / float(out_channels),
            device=input.device,
            dtype=input.dtype,
        )

        for _ in range(0, iterations):

            reconstruction = grouped_linear_einsum_h_weights(h, weight)
            reconstruction += 1e-20

            if epsilon is None:
                h *= grouped_linear_einsum_reconstruction_weights(
                    (input / reconstruction), weight
                )
            else:
                h *= 1 + epsilon * grouped_linear_einsum_reconstruction_weights(
                    (input / reconstruction), weight
                )
            h /= h.sum(2, keepdim=True) + 10e-20

        # ###########################################################
        # Save the necessary data for the backward pass
        # ###########################################################
        ctx.save_for_backward(input, weight, h)
        ctx.local_learning = local_learning
        ctx.local_learning_kl = local_learning_kl

        assert torch.isfinite(h).all()
        return h

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output: torch.Tensor) -> tuple[  # type: ignore
        torch.Tensor,
        torch.Tensor | None,
        None,
        None,
        None,
        None,
        None,
    ]:

        # ##############################################
        # Default values
        # ##############################################
        grad_weight: torch.Tensor | None = None

        # ##############################################
        # Get the variables back
        # ##############################################
        (input, weight, h) = ctx.saved_tensors

        # The back prop gradient
        big_r = grouped_linear_einsum_h_weights(h, weight)

        big_r_div = 1.0 / (big_r + 1e-20)

        factor_x_div_r = input * big_r_div

        grad_input: torch.Tensor = (
            grouped_linear_einsum_h_weights(h * grad_output, weight) * big_r_div
        )

        del big_r_div

        # The weight gradient
        if ctx.local_learning is False:
            del big_r

            grad_weight = -grouped_linear_einsum_h_input(
                h, (factor_x_div_r * grad_input)
            )

            grad_weight += grouped_linear_einsum_h_input(
                (h * grad_output),
                factor_x_div_r,
            )

        else:
            if ctx.local_learning_kl:

                grad_weight = -grouped_linear_einsum_h_input(
                    h,
                    factor_x_div_r,
                )

            else:
                grad_weight = -grouped_linear_einsum_h_input(
                    h,
                    (2 * (input - big_r)),
                )

        assert torch.isfinite(grad_input).all()
        assert torch.isfinite(grad_weight).all()

        return (
            grad_input,
            grad_weight,
            None,
            None,
            None,
            None,
            None,
        )
