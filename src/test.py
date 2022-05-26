import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from .utils.initializers import init_model, init_loaders, init_seeds
from .train_test_functions import test
from .utils.namers import model_ckpt_namer
from deepillusion import torchattacks
from deepillusion.torchdefenses import adversarial_test


@hydra.main(config_path="configs", config_name="cifar10.yaml")
def main(cfg: DictConfig) -> None:

    print(OmegaConf.to_yaml(cfg))

    init_seeds(cfg)

    use_cuda = cfg.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = init_model(cfg).to(device)

    model.load_state_dict(
        torch.load(model_ckpt_namer(cfg), map_location=torch.device(device),)
    )
    _, test_loader = init_loaders(cfg)

    test_loss, test_acc = test(model, test_loader)
    print(f"Test loss: {test_loss:.4f} \t acc: {test_acc:.4f}")

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
    test_loss, test_acc = adversarial_test(
        model, test_loader, adversarial_args, progress_bar=True)
    print(f"Adv test loss: {test_loss:.4f} \t acc: {test_acc:.4f}")


if __name__ == "__main__":
    main()
