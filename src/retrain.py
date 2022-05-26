import logging
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from .utils.initializers import init_model, init_loaders, init_seeds, init_logger
from .models.freeze_reinitialize import Freeze_reinitialize_wrapper
from .train_test_functions import train_one_type
from .utils import saver
from .utils.namers import model_ckpt_namer
from deepillusion import torchattacks
from deepillusion.torchdefenses import adversarial_test
import logging
from copy import deepcopy


def retrain(wrapped_model: Freeze_reinitialize_wrapper,
            train_loader: torch.utils.data.DataLoader,
            test_loader: torch.utils.data.DataLoader,
            cfg: DictConfig,
            loggers: logging.Logger) -> None:
    logger = loggers
    wrapped_model.freeze_reinitialize()

    adversarial = "A2" in cfg.retrain_code

    train_one_type(wrapped_model,
                   adversarial,
                   train_loader,
                   test_loader,
                   cfg,
                   logger)

    adversarial_test_args = {
        "attack": torchattacks.__dict__[cfg.adv_test.attack],
        "attack_args": {
            "data_params": {
                "x_min": 0.0,
                "x_max": 1.0,
            },
            "attack_params": {
                "norm": cfg.adv_test.norm,
                "eps": cfg.adv_test.eps,
                "step_size": cfg.adv_test.step_size,
                "num_steps": cfg.adv_test.n_steps,
                "random_start": cfg.adv_test.random_start,
                "num_restarts": cfg.adv_test.n_restarts
            }
        }
    }
    adv_test_loss, adv_test_acc = adversarial_test(
        wrapped_model, test_loader, adversarial_args=adversarial_test_args, progress_bar=True)
    logger.info(
        f"Adversarial test loss: {adv_test_loss:.4f} \t acc: {adv_test_acc:.4f}")


def freeze_check(model_initial, model_retrained, cfg):

    if cfg.single_layer_different is not None:
        single_layer_frozen = cfg.retrain_code[1] == "1" and cfg.retrain_code[3] == "2"
        for (name, module_in_initial), (_, module_in_retrained) in zip(model_initial.named_modules(), model_retrained.named_modules()):

            if (name == cfg.single_layer_different and single_layer_frozen) or \
                    (name != cfg.single_layer_different and not single_layer_frozen):
                if hasattr(module_in_initial, "weight") and isinstance(module_in_initial.weight, torch.Tensor):
                    assert (module_in_initial.weight -
                            module_in_retrained.weight).abs().max().item() < 1e-4

                if hasattr(module_in_initial, "bias") and isinstance(module_in_initial.bias, torch.Tensor):
                    assert (module_in_initial.bias -
                            module_in_retrained.bias).abs().max().item() < 1e-4

            else:
                if hasattr(module_in_initial, "weight") and isinstance(module_in_initial.weight, torch.Tensor):
                    assert (module_in_initial.weight -
                            module_in_retrained.weight).abs().max().item() > 1e-4

                if hasattr(module_in_initial, "bias") and isinstance(module_in_initial.bias, torch.Tensor):
                    assert (module_in_initial.bias -
                            module_in_retrained.bias).abs().max().item() > 1e-4

    elif cfg.cutoff_before is not None:
        frozen = cfg.retrain_code[1] == "1" and cfg.retrain_code[3] == "2"
        for (name, module_in_initial), (_, module_in_retrained) in zip(model_initial.named_modules(), model_retrained.named_modules()):

            if name == cfg.cutoff_before:
                frozen = not frozen

            if frozen:
                if hasattr(module_in_initial, "weight") and isinstance(module_in_initial.weight, torch.Tensor):
                    assert (module_in_initial.weight -
                            module_in_retrained.weight).abs().max().item() < 1e-4

                if hasattr(module_in_initial, "bias") and isinstance(module_in_initial.bias, torch.Tensor):
                    assert (module_in_initial.bias -
                            module_in_retrained.bias).abs().max().item() < 1e-4

            else:
                if hasattr(module_in_initial, "weight") and isinstance(module_in_initial.weight, torch.Tensor):
                    assert (module_in_initial.weight -
                            module_in_retrained.weight).abs().max().item() > 1e-4

                if hasattr(module_in_initial, "bias") and isinstance(module_in_initial.bias, torch.Tensor):
                    assert (module_in_initial.bias -
                            module_in_retrained.bias).abs().max().item() > 1e-4


def retrain_code_check(retrain_code):
    assert type(retrain_code) == str
    assert len(retrain_code) == 4
    assert (retrain_code[1] == "1" and retrain_code[3] == "2") or (
        retrain_code[1] == "2" and retrain_code[3] == "1")
    assert retrain_code[0] in ("N", "A") and retrain_code[2] in ("N", "A")


@hydra.main(config_path="configs", config_name="cifar10.yaml")
def main(cfg: DictConfig) -> None:

    retrain_code_check(cfg.retrain_code)
    assert (cfg.single_layer_different is not None) ^ (
        cfg.cutoff_before is not None)
    assert cfg.train_type == "retrain"

    logger = init_logger(cfg)

    logger.info(OmegaConf.to_yaml(cfg))

    init_seeds(cfg)

    use_cuda = cfg.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = init_model(cfg).to(device)

    logger.info(model)

    train_loader, test_loader = init_loaders(cfg)

    # do this to load the all adv or all natural trained model
    cfg.train_type = "all_adv" if "A1" in cfg.retrain_code else "all_nat"
    weights = torch.load(model_ckpt_namer(cfg),
                         map_location=torch.device(device),)
    model.load_state_dict(weights)
    cfg.train_type = "retrain"

    model.cpu()
    model_initial = deepcopy(model)
    model.to(device)

    freeze_earlier = cfg.retrain_code[1] == "1"

    if cfg.single_layer_different is not None:

        wrapped_model = Freeze_reinitialize_wrapper(
            model, freeze_earlier, single_layer_different=cfg.single_layer_different)
    elif cfg.cutoff_before is not None:
        wrapped_model = Freeze_reinitialize_wrapper(
            model, freeze_earlier, cutoff_before=cfg.cutoff_before)

    retrain(wrapped_model,
            train_loader,
            test_loader,
            cfg,
            logger)

    freeze_check(model_initial, model.cpu(), cfg)

    ckpt_filepath = model_ckpt_namer(cfg)
    saver(model, ckpt_filepath)
    logger.info(f"Saved to {ckpt_filepath}.")


if __name__ == "__main__":
    main()
