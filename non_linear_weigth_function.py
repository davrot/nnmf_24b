import torch


def non_linear_weigth_function(
    weight: torch.Tensor, beta: torch.Tensor | None, positive_function_type: int
) -> torch.Tensor:

    if positive_function_type == 0:
        positive_weights = torch.abs(weight)

    elif positive_function_type == 1:
        assert beta is not None
        positive_weights = weight
        max_value = torch.abs(positive_weights).max()
        if max_value > 80:
            positive_weights = 80.0 * positive_weights / max_value
        positive_weights = torch.exp((torch.tanh(beta) + 1.0) * 0.5 * positive_weights)

    elif positive_function_type == 2:
        assert beta is not None
        positive_weights = (torch.tanh(beta * weight) + 1.0) * 0.5

    else:
        positive_weights = weight

    return positive_weights
