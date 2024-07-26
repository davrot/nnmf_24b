import torch
import torchvision  # type: ignore
from data_loader import data_loader

from torchvision.transforms import v2  # type: ignore
import numpy as np


def get_the_data(
    dataset: str,
    batch_size_train: int,
    batch_size_test: int,
    torch_device: torch.device,
    input_dim_x: int,
    input_dim_y: int,
    flip_p: float = 0.5,
    jitter_brightness: float = 0.5,
    jitter_contrast: float = 0.1,
    jitter_saturation: float = 0.1,
    jitter_hue: float = 0.15,
    da_auto_mode: bool = False,
) -> tuple[
    torch.utils.data.dataloader.DataLoader,
    torch.utils.data.dataloader.DataLoader,
    torchvision.transforms.Compose,
    torchvision.transforms.Compose,
]:
    if dataset == "MNIST":
        tv_dataset_train = torchvision.datasets.MNIST(
            root="data", train=True, download=True
        )
        tv_dataset_test = torchvision.datasets.MNIST(
            root="data", train=False, download=True
        )
    elif dataset == "FashionMNIST":
        tv_dataset_train = torchvision.datasets.FashionMNIST(
            root="data", train=True, download=True
        )
        tv_dataset_test = torchvision.datasets.FashionMNIST(
            root="data", train=False, download=True
        )
    elif dataset == "CIFAR10":
        tv_dataset_train = torchvision.datasets.CIFAR10(
            root="data", train=True, download=True
        )
        tv_dataset_test = torchvision.datasets.CIFAR10(
            root="data", train=False, download=True
        )
    else:
        raise NotImplementedError("This dataset is not implemented.")

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        torch.random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(0)

    if dataset == "MNIST" or dataset == "FashionMNIST":

        train_dataloader = data_loader(
            torch_device=torch_device,
            batch_size=batch_size_train,
            pattern=tv_dataset_train.data,
            labels=tv_dataset_train.targets,
            shuffle=True,
            worker_init_fn=seed_worker,
            generator=g,
        )

        test_dataloader = data_loader(
            torch_device=torch_device,
            batch_size=batch_size_test,
            pattern=tv_dataset_test.data,
            labels=tv_dataset_test.targets,
            shuffle=False,
            worker_init_fn=seed_worker,
            generator=g,
        )

        # Data augmentation filter
        test_processing_chain = torchvision.transforms.Compose(
            transforms=[torchvision.transforms.CenterCrop((input_dim_x, input_dim_y))],
        )

        train_processing_chain = torchvision.transforms.Compose(
            transforms=[torchvision.transforms.RandomCrop((input_dim_x, input_dim_y))],
        )
    else:

        train_dataloader = data_loader(
            torch_device=torch_device,
            batch_size=batch_size_train,
            pattern=torch.tensor(tv_dataset_train.data).movedim(-1, 1),
            labels=torch.tensor(tv_dataset_train.targets),
            shuffle=True,
            worker_init_fn=seed_worker,
            generator=g,
        )

        test_dataloader = data_loader(
            torch_device=torch_device,
            batch_size=batch_size_test,
            pattern=torch.tensor(tv_dataset_test.data).movedim(-1, 1),
            labels=torch.tensor(tv_dataset_test.targets),
            shuffle=False,
            worker_init_fn=seed_worker,
            generator=g,
        )

        # Data augmentation filter
        test_processing_chain = torchvision.transforms.Compose(
            transforms=[torchvision.transforms.CenterCrop((input_dim_x, input_dim_y))],
        )

        if da_auto_mode:
            train_processing_chain = torchvision.transforms.Compose(
                transforms=[
                    v2.AutoAugment(
                        policy=torchvision.transforms.AutoAugmentPolicy(
                            v2.AutoAugmentPolicy.CIFAR10
                        )
                    ),
                    torchvision.transforms.CenterCrop((input_dim_x, input_dim_y)),
                ],
            )
        else:
            train_processing_chain = torchvision.transforms.Compose(
                transforms=[
                    torchvision.transforms.RandomCrop((input_dim_x, input_dim_y)),
                    torchvision.transforms.RandomHorizontalFlip(p=flip_p),
                    torchvision.transforms.ColorJitter(
                        brightness=jitter_brightness,
                        contrast=jitter_contrast,
                        saturation=jitter_saturation,
                        hue=jitter_hue,
                    ),
                ],
            )

    return (
        train_dataloader,
        test_dataloader,
        train_processing_chain,
        test_processing_chain,
    )
