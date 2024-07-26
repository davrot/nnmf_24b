import torch


def data_loader(
    pattern: torch.Tensor,
    labels: torch.Tensor,
    worker_init_fn,
    generator,
    batch_size: int = 128,
    shuffle: bool = True,
    torch_device: torch.device = torch.device("cpu"),
) -> torch.utils.data.dataloader.DataLoader:

    assert pattern.ndim >= 3

    pattern_storage: torch.Tensor = pattern.to(torch_device).type(torch.float32)
    if pattern_storage.ndim == 3:
        pattern_storage = pattern_storage.unsqueeze(1)
    pattern_storage /= pattern_storage.max()

    label_storage: torch.Tensor = labels.to(torch_device).type(torch.int64)

    dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(pattern_storage, label_storage),
        batch_size=batch_size,
        shuffle=shuffle,
        worker_init_fn=worker_init_fn,
        generator=generator,
    )

    return dataloader
