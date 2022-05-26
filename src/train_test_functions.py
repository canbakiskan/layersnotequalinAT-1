import torch
from torch import nn
import numpy as np
from tqdm import trange
import numpy.typing as npt
from time import time
from omegaconf import DictConfig
from deepillusion import torchattacks
from .utils.initializers import init_optimizer_scheduler
from copy import copy
from tqdm import tqdm
from deepillusion.torchattacks.analysis._evaluate import whitebox_test
import logging
from typing import Optional

NDArray_uint8 = npt.NDArray[np.uint8]
NDArray_float = npt.NDArray[np.float64]


def train_one_type(model: nn.Module,
                   adversarial: bool,
                   train_loader: torch.utils.data.DataLoader,
                   test_loader: torch.utils.data.DataLoader,
                   cfg: DictConfig,
                   logger: logging.Logger,
                   save_every: Optional[int] = None,
                   save_path: Optional[str] = None) -> None:

    assert (save_every == None) == (save_path == None)

    if adversarial:
        old_cfg = cfg
        cfg = copy(cfg)
        cfg.train = cfg.adv_train

        adversarial_args = {
            "attack": torchattacks.__dict__[cfg.adv_train.attack],
            "attack_args": {
                "data_params": {
                    "x_min": 0.0,
                    "x_max": 1.0,
                },
                "attack_params": {
                    "norm": cfg.adv_train.norm,
                    "eps": cfg.adv_train.eps,
                    "step_size": cfg.adv_train.step_size,
                    "num_steps": cfg.adv_train.n_steps,
                    "random_start": cfg.adv_train.random_start,
                    "num_restarts": cfg.adv_train.n_restarts
                }
            }
        }

    optimizer, scheduler = init_optimizer_scheduler(
        cfg, model, verbose=False, batches_per_epoch=len(train_loader))

    epoch: int

    for epoch in trange(cfg.train.n_epochs):

        start_time = time()

        if adversarial:
            train_loss, train_acc = adversarial_epoch(model,
                                                      train_loader,
                                                      optimizer,
                                                      scheduler,
                                                      adversarial_args=adversarial_args,
                                                      progress_bar=True
                                                      )
        else:
            train_loss, train_acc = natural_epoch(model,
                                                  train_loader,
                                                  optimizer,
                                                  scheduler
                                                  )
        end_time = time()

        np.set_printoptions(precision=2, suppress=True)

        logger.info(
            f'ep: {epoch} \t {end_time - start_time:.0f}sec \t xe: {train_loss:.3f} \t acc: {train_acc:.4f} \t lr: {optimizer.param_groups[0]["lr"]:.2e}')
        test_loss, test_acc = test(model, test_loader)
        logger.info(f"Test loss: {test_loss:.4f} \t acc: {test_acc:.4f}")

        if scheduler and isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(test_loss)

        if save_every is not None and ((epoch+1) % save_every == 0) and save_path is not None:
            torch.save(model.state_dict(), save_path.replace(
                ".pt", f"_{epoch}.pt"))


def natural_epoch(model: nn.Module,
                  train_loader: torch.utils.data.DataLoader,
                  optimizer: torch.optim.Optimizer,
                  scheduler: Optional[torch.optim.lr_scheduler._LRScheduler]) -> tuple[float, float]:

    model.train()

    device = model.parameters().__next__().device
    train_loss = 0.0
    train_correct = 0
    cross_ent = nn.CrossEntropyLoss()

    for data, target in train_loader:

        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        output = model(data)
        loss_xe = cross_ent(output, target)

        loss_xe.backward()
        optimizer.step()

        if scheduler and isinstance(scheduler, torch.optim.lr_scheduler.CyclicLR):
            scheduler.step()

        with torch.no_grad():
            train_loss += loss_xe.item() * len(data)
            pred = output.argmax(dim=1, keepdim=True)
            train_correct += pred.eq(target.view_as(pred)).sum().item()

    train_size = len(train_loader.dataset.data)
    train_loss /= train_size
    train_acc = train_correct/train_size

    if scheduler and not isinstance(scheduler, (torch.optim.lr_scheduler.CyclicLR, torch.optim.lr_scheduler.ReduceLROnPlateau)):
        scheduler.step()

    return train_loss, train_acc


def adversarial_epoch(model, train_loader, optimizer, scheduler=None, adversarial_args=None, progress_bar=False):
    """
    Description: Single epoch,
        if adversarial args are present then adversarial training.
    Input :
        model : Neural Network               (torch.nn.Module)
        train_loader : Data loader           (torch.utils.data.DataLoader)
        optimizer : Optimizer                (torch.nn.optimizer)
        scheduler: Scheduler (Optional)      (torch.optim.lr_scheduler.CyclicLR)
        adversarial_args :
            attack:                          (deepillusion.torchattacks)
            attack_args:
                attack arguments for given attack except "x" and "y_true"
        progress_bar:
    Output:
        train_loss : Train loss              (float)
        train_accuracy : Train accuracy      (float)
    """

    model.train()

    device = model.parameters().__next__().device

    train_loss = 0
    train_correct = 0
    if progress_bar:
        iter_train_loader = tqdm(
            iterable=train_loader,
            desc="Epoch Progress",
            unit="batch",
            leave=False)
    else:
        iter_train_loader = train_loader

    for data, target in iter_train_loader:

        data, target = data.to(device), target.to(device)

        # Adversary
        if adversarial_args and adversarial_args["attack"]:
            adversarial_args["attack_args"]["net"] = model
            adversarial_args["attack_args"]["x"] = data
            adversarial_args["attack_args"]["y_true"] = target
            model.eval()
            perturbs = adversarial_args['attack'](
                **adversarial_args["attack_args"])
            data += perturbs
            model.train()

        optimizer.zero_grad()
        output = model(data)
        cross_ent = nn.CrossEntropyLoss()
        loss = cross_ent(output, target)
        loss.backward()
        optimizer.step()
        if scheduler and isinstance(scheduler, torch.optim.lr_scheduler.CyclicLR):
            scheduler.step()

        with torch.no_grad():
            train_loss += loss.item() * data.size(0)
            pred_adv = output.argmax(dim=1, keepdim=False)
            train_correct += pred_adv.eq(target.view_as(pred_adv)).sum().item()

    if scheduler and not isinstance(scheduler, (torch.optim.lr_scheduler.CyclicLR, torch.optim.lr_scheduler.ReduceLROnPlateau)):
        scheduler.step()

    train_size = len(train_loader.dataset)

    return train_loss/train_size, train_correct/train_size


def test(model: nn.Module, test_loader: torch.utils.data.DataLoader) -> tuple[float, float]:

    model.eval()

    device = model.parameters().__next__().device

    test_loss = 0
    test_correct = 0
    cross_ent = nn.CrossEntropyLoss()
    with torch.no_grad():
        for data, target in test_loader:
            if isinstance(data, list):
                data = data[0]
                target = target[0]

            data, target = data.to(device), target.to(device)

            output = model(data)
            test_loss += cross_ent(output, target).item() * len(data)
            pred = output.argmax(dim=1, keepdim=True)
            test_correct += pred.eq(target.view_as(pred)).sum().item()

    test_size = len(test_loader.dataset.data)

    return test_loss / test_size, test_correct / test_size


def adversarial_test(model, test_loader, adversarial_args=None, verbose=False, progress_bar=False):
    """
    Description: Evaluate model with test dataset,
        if adversarial args are present then adversarially perturbed test set.
    Input :
        model : Neural Network               (torch.nn.Module)
        test_loader : Data loader            (torch.utils.data.DataLoader)
        adversarial_args :                   (dict)
            attack:                          (deepillusion.torchattacks)
            attack_args:                     (dict)
                attack arguments for given attack except "x" and "y_true"
        verbose: Verbosity                   (Bool)
        progress_bar: Progress bar           (Bool)
    Output:
        train_loss : Train loss              (float)
        train_accuracy : Train accuracy      (float)
    """

    return whitebox_test(model, test_loader, adversarial_args, verbose, progress_bar)
