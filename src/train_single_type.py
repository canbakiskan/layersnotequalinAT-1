import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from .utils.initializers import init_model, init_loaders, init_seeds, init_logger
from .train_test_functions import train_one_type
from .utils import saver
from .utils.namers import model_ckpt_namer
from deepillusion.torchdefenses import adversarial_test
from deepillusion import torchattacks


@hydra.main(config_path="configs", config_name="cifar10.yaml")
def main(cfg: DictConfig) -> None:

    assert cfg.train_type in ["all_adv", "all_nat"]
    is_adv = (cfg.train_type == "all_adv")
    logger = init_logger(cfg)

    logger.info(OmegaConf.to_yaml(cfg))

    init_seeds(cfg)

    use_cuda = cfg.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = init_model(cfg).to(device)
    ckpt_filepath = model_ckpt_namer(cfg)
    logger.info(model)

    train_loader, test_loader = init_loaders(cfg)

    train_one_type(model,
                   is_adv,
                   train_loader,
                   test_loader,
                   cfg,
                   logger,
                   5,
                   ckpt_filepath)

    adversarial_args = {
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
        model, test_loader, adversarial_args=adversarial_args, progress_bar=True)
    logger.info(
        f"Adversarial test loss: {adv_test_loss:.4f} \t acc: {adv_test_acc:.4f}")

    saver(model, ckpt_filepath)
    logger.info(
        f"Saved to {ckpt_filepath}.")


if __name__ == "__main__":
    main()
