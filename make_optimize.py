import torch


def make_optimize(
    parameters: list[list[torch.nn.parameter.Parameter]],
    lr_initial: list[float],
    eps=1e-10,
) -> tuple[
    list[torch.optim.Adam | None],
    list[torch.optim.lr_scheduler.ReduceLROnPlateau | None],
]:
    list_optimizer: list[torch.optim.Adam | None] = []
    list_lr_scheduler: list[torch.optim.lr_scheduler.ReduceLROnPlateau | None] = []

    assert len(parameters) == len(lr_initial)

    for i in range(0, len(parameters)):
        if len(parameters[i]) > 0:
            list_optimizer.append(torch.optim.Adam(parameters[i], lr=lr_initial[i]))
        else:
            list_optimizer.append(None)

    for i in range(0, len(list_optimizer)):
        if list_optimizer[i] is not None:
            pass
            list_lr_scheduler.append(
                torch.optim.lr_scheduler.ReduceLROnPlateau(list_optimizer[i], eps=eps)  # type: ignore
            )
        else:
            list_lr_scheduler.append(None)

    return (list_optimizer, list_lr_scheduler)
