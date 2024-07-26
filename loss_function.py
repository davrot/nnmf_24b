import torch


# loss_mode == 0: "normal" SbS loss function mixture
# loss_mode == 1: cross_entropy
def loss_function(
    h: torch.Tensor,
    labels: torch.Tensor,
    loss_mode: int = 0,
    number_of_output_neurons: int = 10,
    loss_coeffs_mse: float = 0.0,
    loss_coeffs_kldiv: float = 0.0,
) -> torch.Tensor | None:

    assert loss_mode >= 0
    assert loss_mode <= 1

    assert h.ndim == 2

    if loss_mode == 0:

        # Convert label into one hot
        target_one_hot: torch.Tensor = torch.zeros(
            (
                labels.shape[0],
                number_of_output_neurons,
            ),
            device=h.device,
            dtype=h.dtype,
        )

        target_one_hot.scatter_(
            1,
            labels.to(h.device).unsqueeze(1),
            torch.ones(
                (labels.shape[0], 1),
                device=h.device,
                dtype=h.dtype,
            ),
        )

        my_loss: torch.Tensor = ((h - target_one_hot) ** 2).sum(dim=0).mean(
            dim=0
        ) * loss_coeffs_mse

        my_loss = (
            my_loss
            + (
                (target_one_hot * torch.log((target_one_hot + 1e-20) / (h + 1e-20)))
                .sum(dim=0)
                .mean(dim=0)
            )
            * loss_coeffs_kldiv
        )

        my_loss = my_loss / (abs(loss_coeffs_kldiv) + abs(loss_coeffs_mse))

        return my_loss

    elif loss_mode == 1:
        my_loss = torch.nn.functional.cross_entropy(h, labels.to(h.device))
        return my_loss
    else:
        return None
