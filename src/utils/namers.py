import os
from omegaconf import DictConfig
import numpy as np
from copy import copy
from numpy import format_float_scientific


def model_params_string(cfg: DictConfig, adv: bool) -> str:

    if adv:
        old_cfg = cfg
        cfg = copy(cfg)
        cfg.train = cfg.adv_train
        model_params_string = "_adv"
    else:
        model_params_string = ""

    model_params_string += f"_{cfg.train.optimizer}"

    model_params_string += f"_{cfg.train.scheduler}"

    if cfg.train.scheduler == "cyc":
        model_params_string += f"_lrm_{format_float_scientific(cfg.train.lr_max, exp_digits=1, precision=2)}"
    else:
        model_params_string += f"_lr_{format_float_scientific(cfg.train.lr, exp_digits=1, precision=2)}"

    if cfg.train.wd != 0:
        model_params_string += f"_wd_{format_float_scientific(cfg.train.wd, exp_digits=1, precision=2)}"

    if cfg.train.optimizer in ["sgd", "rms"] and cfg.train.momentum != 0:
        model_params_string += f"_mo_{format_float_scientific(cfg.train.momentum, exp_digits=1, precision=2)}"

    model_params_string += f"_ep_{cfg.train.n_epochs}"
    if adv:
        model_params_string += adv_training_params_string(cfg)

    return model_params_string


def adv_training_params_string(cfg: DictConfig) -> str:

    adv_training_params_string = f"_{cfg.adv_train.attack}"
    adv_training_params_string += (
        f"_eps_{int(np.round(cfg.adv_train.eps*255))}"
    )
    if "EOT" in cfg.adv_train.attack:
        adv_training_params_string += f"_Ne_{cfg.adv_train.EOT_size}"
    if "PGD" in cfg.adv_train.attack or "CW" in cfg.adv_train.attack:
        adv_training_params_string += f"_Ns_{cfg.adv_train.n_steps}"
        adv_training_params_string += (
            f"_ss_{int(np.round(cfg.adv_train.step_size*255))}"
        )
        adv_training_params_string += f"_Nr_{cfg.adv_train.n_restarts}"
        if cfg.adv_train.n_restarts == 1 and cfg.adv_train.random_start:
            adv_training_params_string += f"_rand"
    if "FGSM" in cfg.adv_train.attack:
        adv_training_params_string += (
            f"_a_{int(np.round(cfg.adv_train.alpha*255))}"
        )

    return adv_training_params_string


def model_ckpt_namer(cfg: DictConfig) -> str:

    dir_path = os.path.join(
        cfg.directory, "checkpoints", cfg.dataset.name)

    os.makedirs(dir_path, exist_ok=True)

    if cfg.train_type in ("all_adv", "all_nat"):

        file_path = os.path.join(dir_path, cfg.model +
                                 model_params_string(cfg, cfg.train_type == "all_adv"))

    elif cfg.train_type == "retrain":
        file_path = os.path.join(
            dir_path,  cfg.model + model_params_string(cfg, False) + model_params_string(cfg, True))

        assert (cfg.single_layer_different is not None) ^ (
            cfg.cutoff_before is not None)

        if cfg.single_layer_different is not None:
            file_path = file_path + \
                f"_{cfg.retrain_code}_single_{cfg.single_layer_different}"
        elif cfg.cutoff_before is not None:
            file_path = file_path + \
                f"_{cfg.retrain_code}_cutoff_before_{cfg.cutoff_before}"

    elif cfg.train_type == "mixed":
        file_path = os.path.join(
            dir_path,  cfg.model + model_params_string(cfg, False) + model_params_string(cfg, True))
        file_path = file_path + \
            f"_{cfg.mixed_code}_mixed_cutoff_before_{cfg.cutoff_before}"
    else:
        raise AssertionError(
            f"Expected one of all_adv, all_nat, retrain, mixed, got {cfg.train_type}")

    file_path += ".pt"

    return file_path


def model_log_namer(cfg: DictConfig) -> str:

    dir_path = os.path.join(
        cfg.directory, "logs", cfg.dataset.name)

    os.makedirs(dir_path, exist_ok=True)

    if cfg.train_type in ["all_adv", "all_nat"]:

        file_path = os.path.join(dir_path, cfg.model +
                                 model_params_string(cfg, cfg.train_type == "all_adv"))

    elif cfg.train_type == "retrain":

        file_path = os.path.join(
            dir_path,  cfg.model + model_params_string(cfg, False) + model_params_string(cfg, True))

        assert (cfg.single_layer_different is not None) ^ (
            cfg.cutoff_before is not None)

        if cfg.single_layer_different is not None:
            file_path = file_path + \
                f"_{cfg.retrain_code}_single_{cfg.single_layer_different}"
        elif cfg.cutoff_before is not None:
            file_path = file_path + \
                f"_{cfg.retrain_code}_cutoff_before_{cfg.cutoff_before}"

    elif cfg.train_type == "mixed":
        file_path = os.path.join(
            dir_path,  cfg.model + model_params_string(cfg, False) + model_params_string(cfg, True))
        file_path = file_path + \
            f"_{cfg.mixed_code}_mixed_cutoff_before_{cfg.cutoff_before}"
    else:
        raise AssertionError(
            f"Expected one of all_adv, all_nat, retrain, mixed, got {cfg.train_type}")

    file_path += ".log"

    return file_path
