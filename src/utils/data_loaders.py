from typing import Dict
from omegaconf import DictConfig
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def cifar10(cfg: DictConfig) -> tuple[DataLoader, DataLoader]:

    use_cuda = cfg.cuda and torch.cuda.is_available()

    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    transform_test = transforms.Compose([transforms.ToTensor()])

    trainset = datasets.CIFAR10(
        root=cfg.dataset.directory,
        train=True,
        download=True,
        transform=transform_train,
    )
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=cfg.dataset.train_batch_size, shuffle=True, num_workers=2
    )

    testset = datasets.CIFAR10(
        root=cfg.dataset.directory,
        train=False,
        download=True,
        transform=transform_test,
    )
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=cfg.dataset.test_batch_size, shuffle=False, num_workers=2
    )

    return train_loader, test_loader
