from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
import torch
from numpy.random import seed
from typing import Callable, Optional
from .namers import model_log_namer
from ..models.resnet import ResNet18
import logging
from torch import nn, optim
from omegaconf import DictConfig
from .data_loaders import cifar10
from ..models.vgg import VGG


def init_seeds(cfg: DictConfig) -> None:
    seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)


def init_logger(cfg: DictConfig) -> logging.Logger:

    file_path = model_log_namer(cfg)

    logger = logging.getLogger(file_path)
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(file_path)
    fh.setLevel(logging.INFO)

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)

    formatter = logging.Formatter(
        fmt='[%(asctime)s] - %(message)s', datefmt='%Y/%m/%d %H:%M:%S',)
    sh.setFormatter(formatter)
    fh.setFormatter(formatter)
    # add the handlers to logger
    logger.addHandler(sh)
    logger.addHandler(fh)

    return logger


def init_loaders(cfg: DictConfig) -> tuple[DataLoader, DataLoader]:

    if cfg.dataset.name == "cifar10":
        train_loader, test_loader = cifar10(cfg)
    else:
        raise NotImplementedError

    return train_loader, test_loader


def init_model(cfg: DictConfig) -> nn.Module:
    model: nn.Module
    num_outputs = 10
    image_size = cfg.dataset.image_size
    if cfg.model == "resnet":
        model = ResNet18()
    elif cfg.model == "vgg":
        model = VGG("VGG16")
    else:
        raise NotImplementedError

    return model


def init_optimizer_scheduler(cfg: DictConfig, model: nn.Module, batches_per_epoch: Optional[int] = None, printer: Callable = print, verbose: bool = True) -> tuple[Optimizer, Optional[optim.lr_scheduler._LRScheduler]]:

    optimizer: Optimizer

    if cfg.train.optimizer == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=cfg.train.lr,
            momentum=cfg.train.momentum,
            weight_decay=cfg.train.wd,
        )
    elif cfg.train.optimizer == "rms":
        optimizer = optim.RMSprop(
            model.parameters(),
            lr=cfg.train.lr,
            weight_decay=cfg.train.wd,
            momentum=cfg.train.momentum,
        )

    elif cfg.train.optimizer == "adam":
        optimizer = optim.Adam(
            model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.wd
        )
    else:
        raise NotImplementedError

    scheduler: Optional[optim.lr_scheduler._LRScheduler]

    if cfg.train.scheduler == "cyc":
        lr_steps = cfg.train.n_epochs * batches_per_epoch
        scheduler = optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=cfg.train.lr_min,
            max_lr=cfg.train.lr_max,
            step_size_up=lr_steps / 2,
            step_size_down=lr_steps / 2,
        )
    elif cfg.train.scheduler == "step":
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=cfg.train.scheduler_steps, gamma=0.1
        )
    elif cfg.train.scheduler == "plat":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.1, patience=10, threshold=0.0001, verbose=True)

    elif cfg.train.scheduler == "mult":

        def lr_fun(epoch: int) -> float:
            if epoch % 3 == 0:
                return 0.962
            else:
                return 1.0

        scheduler = optim.lr_scheduler.MultiplicativeLR(
            optimizer, lr_fun)
    else:
        scheduler = None

    if verbose == True:
        printer(optimizer)
        printer(scheduler)

    return optimizer, scheduler
