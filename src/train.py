import hydra
from omegaconf import DictConfig


@hydra.main(config_path="configs", config_name="cifar10.yaml")
def main(cfg: DictConfig) -> None:

    if cfg.train_type in ["all_adv", "all_nat"]:
        from .train_single_type import main
    elif cfg.train_type == "retrain":
        from .retrain import main
    else:
        raise ValueError("Unknown train_type: {}".format(cfg.train_type))

    main(cfg)


if __name__ == "__main__":
    main()
