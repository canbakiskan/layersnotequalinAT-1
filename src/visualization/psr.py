# finds surviving perturbation after each layer of AT and clean trained models

import torch
from torch import nn
import numpy as np
from tqdm import tqdm
import hydra
from omegaconf import DictConfig
from ..utils.initializers import init_seeds, init_model, init_loaders
from typing import OrderedDict
import pandas as pd
from ..utils.namers import model_ckpt_namer
import os
from ..models.layer_output_extractor import SpecificLayerTypeOutputExtractor_wrapper
from deepillusion import torchattacks
import re
from . import plot_settings
import matplotlib.pyplot as plt
import joypy
import matplotlib.colors as colors
from matplotlib.lines import Line2D


def generate_perturbations(model, test_loader, adversarial_args=None, verbose=False, progress_bar=False):
    """
    Description: Evaluate model with test dataset,
            if adversarial args are present then adversarially perturbed test set.
    Input :
            model : Neural Network               (torch.nn.Module)
            test_loader : Data loader            (torch.utils.data.DataLoader)
            adversarial_args :                   (dict)
                    attack:                          (
                        deepillusion.torchattacks)
                    attack_args:                     (dict)
                            attack arguments for given attack except "x" and "y_true"
            verbose: Verbosity                   (Bool)
            progress_bar: Progress bar           (Bool)
    Output:
            train_loss : Train loss              (float)
            train_accuracy : Train accuracy      (float)
    """
    device = model.parameters().__next__().device

    batch_size = test_loader.batch_size
    model.eval()

    test_loss = 0
    test_correct = 0
    if progress_bar:
        iter_test_loader = tqdm(
            iterable=test_loader,
            desc="Dataset Progress",
            unit="batch",
            leave=False)
    else:
        iter_test_loader = test_loader

    perturbations = torch.empty(test_loader.dataset.data.shape)
    for test_i, (data, target) in enumerate(iter_test_loader):

        data, target = data.to(device), target.to(device)

        # Adversary
        if adversarial_args and adversarial_args["attack"]:
            adversarial_args["attack_args"]["net"] = model
            adversarial_args["attack_args"]["x"] = data
            adversarial_args["attack_args"]["y_true"] = target
            perturbs = adversarial_args['attack'](
                **adversarial_args["attack_args"])

            perturbations[test_i * batch_size: (
                test_i+1)*batch_size] = perturbs.detach().permute(0, 2, 3, 1).cpu()

    return perturbations


def get_layer_outputs(model, test_loader):

    model.eval()

    device = model.parameters().__next__().device
    batch_size = test_loader.batch_size

    activations: dict[nn.Module, torch.Tensor] = {}
    with torch.no_grad():
        for test_i, (data, target) in enumerate(test_loader):

            data, target = data.to(device), target.to(device)

            output = model(data)

            for layer, layer_output in zip(model.layers_of_interest, model.layer_outputs.values()):

                if layer not in activations:
                    layer_output = model.layer_outputs[layer]
                    activations[layer] = torch.empty(
                        len(test_loader.dataset), *layer_output.shape[1:])

                activations[layer][test_i * batch_size: (
                    test_i+1)*batch_size] = (layer_output).detach().cpu()

    return activations


def get_95_index(arr):
    n = np.sum(arr)*0.95
    index = np.searchsorted(np.cumsum(arr), n)
    return index


@ hydra.main(config_path="../configs", config_name="cifar10.yaml")
def main(cfg: DictConfig) -> None:

    init_seeds(cfg)

    use_cuda = cfg.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = init_model(cfg).to(device)

    layer_type = nn.Conv2d
    # layer_type = nn.BatchNorm2d

    wrapped_model = SpecificLayerTypeOutputExtractor_wrapper(
        model, layer_type)

    norm_data = OrderedDict({
        "layer_name": [],
        "L1_nominal_A": np.zeros((0)),
        "L1_nominal_N": np.zeros((0)),
        "L1_ratio_A": np.zeros((0)),
        "L1_ratio_N": np.zeros((0)),
        "L2_nominal_A": np.zeros((0)),
        "L2_nominal_N": np.zeros((0)),
        "L2_ratio_A": np.zeros((0)),
        "L2_ratio_N": np.zeros((0)),
        "Linf_nominal_A": np.zeros((0)),
        "Linf_nominal_N": np.zeros((0)),
        "Linf_ratio_A": np.zeros((0)),
        "Linf_ratio_N": np.zeros((0)),
        "L1_over_L2_A": np.zeros((0)),
        "L1_over_L2_N": np.zeros((0)),
        "L2_over_Linf_A": np.zeros((0)),
        "L2_over_Linf_N": np.zeros((0)),
        "L1_over_Linf_A": np.zeros((0)),
        "L1_over_Linf_N": np.zeros((0)),
        "L1_over_L2_clean_A": np.zeros((0)),
        "L1_over_L2_clean_N": np.zeros((0)),
        "Linf_over_L2_clean_A": np.zeros((0)),
        "Linf_over_L2_clean_N": np.zeros((0)),
        "L0_over_L2_clean_A": np.zeros((0)),
        "L0_over_L2_clean_N": np.zeros((0)),
    })

    if (cfg.model == "resnet" and cfg.cutoff_before == "conv1") or \
            (cfg.model == "vgg" and cfg.cutoff_before == "features.0"):
        original_train_type = cfg.train_type
        cfg.train_type = "all_nat"
        path_N = model_ckpt_namer(cfg)

        cfg.train_type = "all_adv"
        path_A = model_ckpt_namer(cfg)
        cfg.train_type = original_train_type
    else:
        original_code = cfg.retrain_code
        cfg.retrain_code = cfg.retrain_code.replace("x", "N")
        path_N = model_ckpt_namer(cfg)
        cfg.retrain_code = original_code

        cfg.retrain_code = cfg.retrain_code.replace("x", "A")
        path_A = model_ckpt_namer(cfg)
        cfg.retrain_code = original_code

    for train_type in ["A", "N"]:
        test_loader = init_loaders(cfg)[1]

        parameter_dict = torch.load(
            locals()[f"path_{train_type}"], map_location=torch.device(device))

        model.load_state_dict(parameter_dict)

        clean_layer_outputs = get_layer_outputs(
            wrapped_model, test_loader)

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
        perturbations = generate_perturbations(
            model, test_loader, adversarial_args=adversarial_args, progress_bar=True)

        perturbed_data = test_loader.dataset.data.astype(
            np.float32) + perturbations.numpy()*255
        test_loader.dataset.data = perturbed_data.astype(np.uint8)

        adversarial_layer_outputs = get_layer_outputs(
            wrapped_model, test_loader)

        diff_layer_outputs = {
            layer: adversarial_layer_outputs[layer] - clean_layer_outputs[layer] for layer in clean_layer_outputs
        }

        del adversarial_layer_outputs

        for layer_index, layer_name in enumerate(diff_layer_outputs.keys()):

            L0_nominal = torch.sum(
                (diff_layer_outputs[layer_name].abs() > 1e-3).int(), dim=(1, 2, 3))
            L0_clean = torch.sum(
                (clean_layer_outputs[layer_name].abs() > 1e-3).int(), dim=(1, 2, 3))

            L0_ratio = L0_nominal/L0_clean

            L1_nominal = torch.norm(
                diff_layer_outputs[layer_name], p=1, dim=(1, 2, 3))
            L1_clean = torch.norm(
                clean_layer_outputs[layer_name], p=1, dim=(1, 2, 3))
            L1_ratio = L1_nominal/L1_clean

            L2_nominal = torch.norm(
                diff_layer_outputs[layer_name], p=2, dim=(1, 2, 3))
            L2_clean = torch.norm(
                clean_layer_outputs[layer_name], p=2, dim=(1, 2, 3))
            L2_ratio = L2_nominal/L2_clean

            Linf_nominal = diff_layer_outputs[layer_name].abs().amax(
                dim=(1, 2, 3))
            Linf_clean = clean_layer_outputs[layer_name].abs().amax(
                dim=(1, 2, 3))
            Linf_ratio = Linf_nominal / Linf_clean

            if train_type == "A":
                norm_data["layer_name"].extend(
                    [layer_index] * len(L1_nominal))

            norm_data[f"L1_nominal_{train_type}"] = np.append(
                norm_data[f"L1_nominal_{train_type}"], L1_nominal
            )
            norm_data[f"L1_ratio_{train_type}"] = np.append(
                norm_data[f"L1_ratio_{train_type}"], L1_ratio
            )
            norm_data[f"L2_nominal_{train_type}"] = np.append(
                norm_data[f"L2_nominal_{train_type}"], L2_nominal
            )
            norm_data[f"L2_ratio_{train_type}"] = np.append(
                norm_data[f"L2_ratio_{train_type}"], L2_ratio
            )
            norm_data[f"Linf_nominal_{train_type}"] = np.append(
                norm_data[f"Linf_nominal_{train_type}"], Linf_nominal
            )
            norm_data[f"Linf_ratio_{train_type}"] = np.append(
                norm_data[f"Linf_ratio_{train_type}"], Linf_ratio
            )
            norm_data[f"L1_over_L2_{train_type}"] = np.append(
                norm_data[f"L1_over_L2_{train_type}"], L1_nominal/L2_nominal
            )
            norm_data[f"L2_over_Linf_{train_type}"] = np.append(
                norm_data[f"L2_over_Linf_{train_type}"], L2_nominal /
                Linf_nominal
            )
            norm_data[f"L1_over_Linf_{train_type}"] = np.append(
                norm_data[f"L1_over_Linf_{train_type}"], L1_nominal /
                Linf_nominal
            )
            norm_data[f"L0_over_L2_clean_{train_type}"] = np.append(
                norm_data[f"L0_over_L2_clean_{train_type}"], L0_nominal /
                L2_clean
            )
            norm_data[f"L1_over_L2_clean_{train_type}"] = np.append(
                norm_data[f"L1_over_L2_clean_{train_type}"], L1_nominal /
                L2_clean
            )
            norm_data[f"Linf_over_L2_clean_{train_type}"] = np.append(
                norm_data[f"Linf_over_L2_clean_{train_type}"], Linf_nominal /
                L2_clean
            )

        del diff_layer_outputs, clean_layer_outputs

    norm_dataframe = pd.DataFrame(data=norm_data)

    labels = [name for name, module in model.named_modules()
              if isinstance(module, layer_type)]

    def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
        new_cmap = colors.LinearSegmentedColormap.from_list(
            'trunc({n},{a:.2f},{b:.2f})'.format(
                n=cmap.name, a=minval, b=maxval),
            cmap(np.linspace(minval, maxval, n)))
        return new_cmap

    orange_cmap = plt.get_cmap('Oranges')
    blue_cmap = plt.get_cmap('Blues')

    density_estimate = "normalized_counts"

    # works only if density_estimate="counts"
    nb_bins = 100

    y_lim_type = "own"

    l = list(norm_data.keys())[1:]

    for norm_A, norm_N in zip(l[::2], l[1::2]):
        print(norm_A)

        if "ratio" in norm_A:
            p = re.search(r"L(.*?)_", norm_A).group(1)
            if p == "inf":
                p = r"\infty"
            title = r"Histograms of $\frac{||f_l(x)-f_l(x+e)||_{" + \
                p+r"}}{||f_l(x)||_{"+p+r"}}$"
        elif "over" in norm_A:

            p_first, p_second = re.findall(r"L(\w+?)_", norm_A)

            if p_second == "inf":
                p_second = r"\infty"

            if "clean" in norm_A:
                title = r"Histograms of $\frac{||f_l(x)-f_l(x+e)||_{" + \
                    p_first+r"}}{||f_l(x)||_{" + \
                    p_second+r"}}$"
            else:
                title = r"Histograms of $\frac{||f_l(x)-f_l(x+e)||_{" + \
                    p_first+r"}}{||f_l(x)-f_l(x+e)||_{" + \
                    p_second+r"}}$"

        else:
            p = re.search(r"L(.*?)_", norm_A).group(1)
            if p == "inf":
                p = r"\infty"
            title = r"Histograms of $||f_l(x)-f_l(x+e)||_{"+p+"}$"

        fig, axes = joypy.joyplot(norm_dataframe, by="layer_name",
                                  column=[norm_A, norm_N],
                                  labels=labels,
                                  grid="y",
                                  linewidth=1,
                                  legend=False,
                                  figsize=(13, 8),
                                  title=title,
                                  alpha=0.7,
                                  range_style='own',
                                  colormap=[truncate_colormap(orange_cmap, 0.9, 0.35),
                                            truncate_colormap(blue_cmap, 0.9, 0.35)],
                                  bins=nb_bins,
                                  kind=density_estimate,
                                  ylim=y_lim_type,
                                  ylabelsize=25
                                  )

        maxx = max(norm_dataframe[norm_A].max(),
                   norm_dataframe[norm_N].max())
        minn = min(norm_dataframe[norm_A].min(),
                   norm_dataframe[norm_N].min())

        maxx = 0
        for a in axes[:-1]:
            for line in a.lines:
                maxx = max(maxx, line.get_xdata()[
                           get_95_index(line.get_ydata())])
        for a in axes:
            a.set_xlim(
                [0.0, 1.1*maxx])

        plt.xticks(np.linspace(0.0, 1.1*maxx, 6), fontsize=25)

        if "ratio" in norm_A:
            axes[-1].set_xlabel(r"$\ell_{"+p+"}$ ratio", fontsize=25)
        elif "over" in norm_A:

            if "clean" in norm_A:
                axes[-1].set_xlabel(r"$\ell_{" +
                                    p_first+r"}/\ell_{"+p_second+"}$ clean", fontsize=25)

            else:
                axes[-1].set_xlabel(r"$\ell_{" +
                                    p_first+r"}/\ell_{"+p_second+"}$", fontsize=25)
        else:
            axes[-1].set_xlabel(r"$\ell_{"+p+"}$ nominal", fontsize=25)

        if (cfg.model == "resnet" and cfg.cutoff_before == "conv1") or \
                (cfg.model == "vgg" and cfg.cutoff_before == "features.0"):
            legend_label_adv = "All adversarial"
            legend_label_nat = "All natural"
        else:
            legend_label_adv = cfg.retrain_code.replace("x", "A")
            legend_label_nat = cfg.retrain_code.replace("x", "N")

        legend_elements = [Line2D([0], [0], marker='o', color='w', label=legend_label_adv,
                                  markerfacecolor=orange_cmap(0.65), markersize=15),
                           Line2D([0], [0], marker='o', color='w', label=legend_label_nat,
                                  markerfacecolor=blue_cmap(0.65), markersize=15),
                           ]

        fig.legend(handles=legend_elements,
                   loc="upper left", fontsize=25)

        dir_path = os.path.join(
            cfg.directory, "figs", "analysis", "perturbation", os.path.basename(model_ckpt_namer(cfg)).replace(".pt", ""))
        os.makedirs(dir_path, exist_ok=True)

        if layer_type == nn.Conv2d:
            plt.savefig(os.path.join(
                dir_path, f"{norm_A.replace('_A','')}.svg"))
        elif layer_type == nn.BatchNorm2d:
            plt.savefig(os.path.join(
                dir_path, f"{norm_A.replace('_A','_bn')}.svg"))

        plt.close()

        if "ratio" in norm_A:
            p = re.search(r"L(.*?)_", norm_A).group(1)
            if p == "inf":
                p = r"\infty"
            title = r"$\frac{\mathbb{E}[||f_l(x)-f_l(x+e)||_{" + \
                p+r"}]}{\mathbb{E}[||f_l(x)||_{"+p+r"}]}$"
        elif "over" in norm_A:

            p_first, p_second = re.findall(r"L(\w+?)_", norm_A)

            if p_second == "inf":
                p_second = r"\infty"

            if "clean" in norm_A:
                title = r"$\frac{\mathbb{E}[||f_l(x)-f_l(x+e)||_{" + \
                    p_first+r"}]}{\mathbb{E}[||f_l(x)||_{" + \
                    p_second+r"}]}$"
            else:
                title = r"$\frac{\mathbb{E}[||f_l(x)-f_l(x+e)||_{" + \
                    p_first+r"}]}{\mathbb{E}[||f_l(x)-f_l(x+e)||_{" + \
                    p_second+r"}]}$"

        else:
            p = re.search(r"L(.*?)_", norm_A).group(1)
            if p == "inf":
                p = r"\infty"
            title = r"$\mathbb{E}[||f_l(x)-f_l(x+e)||_{"+p+r"}]$"

        means_A = np.array([norm_dataframe[norm_dataframe['layer_name'] == i]
                            [norm_A].mean() for i in range(len(labels))])
        means_N = np.array([norm_dataframe[norm_dataframe['layer_name'] == i]
                            [norm_N].mean() for i in range(len(labels))])

        plt.figure(figsize=(11, 7))
        plt.plot(labels, 20*np.log10(means_A), color=orange_cmap(
            0.65), marker='o', label=legend_label_adv)
        plt.plot(labels, 20*np.log10(means_N), color=blue_cmap(
            0.65), marker='s', label=legend_label_nat)

        plt.grid()
        plt.yticks(fontsize=23)
        plt.legend(fontsize=23)
        plt.xticks(range(len(labels)), labels, rotation=40, ha="right",
                   va="top", rotation_mode="anchor", fontsize=23)
        plt.title(title,  y=1.08, fontsize=23)
        plt.tight_layout()

        if layer_type == nn.Conv2d:
            plt.savefig(os.path.join(
                dir_path, f"{norm_A.replace('_A','_means')}.svg"))
        elif layer_type == nn.BatchNorm2d:
            plt.savefig(os.path.join(
                dir_path, f"{norm_A.replace('_A','_means_bn')}.svg"))

        plt.close()

        plt.figure(figsize=(12, 7))
        plt.plot(labels, means_N/means_A, color="tab:red",
                 marker='o')

        plt.xticks(range(len(labels)), labels, rotation=40, ha="right",
                   va="top", rotation_mode="anchor")

        if "ratio" in norm_A:
            p = re.search(r"L(.*?)_", norm_A).group(1)
            if p == "inf":
                p = r"\infty"
            title = r"$\frac{\mathbb{E}[||f_l(x)-f_l(x+e)||_{" + \
                p+r"}]}{\mathbb{E}[||f_l(x)||_{"+p+r"}]}$"
        elif "over" in norm_A:

            p_first, p_second = re.findall(r"L(\w+?)_", norm_A)

            if p_second == "inf":
                p_second = r"\infty"

            if "clean" in norm_A:
                title = r"$\frac{\mathbb{E}[||f_l(x)-f_l(x+e)||_{" + \
                    p_first+r"}]}{\mathbb{E}[||f_l(x)||_{" + \
                    p_second+r"}]}$"
            else:
                title = r"$\frac{\mathbb{E}[||f_l(x)-f_l(x+e)||_{" + \
                    p_first+r"}]}{\mathbb{E}[||f_l(x)-f_l(x+e)||_{" + \
                    p_second+r"}]}$"

        else:
            p = re.search(r"L(.*?)_", norm_A).group(1)
            if p == "inf":
                p = r"\infty"
            title = r"$\mathbb{E}[||f_l(x)-f_l(x+e)||_{"+p+r"}]$"

        title = f"Ratio of NT {title} over AT {title}"
        plt.title(title,  y=1.08)
        plt.tight_layout()

        if layer_type == nn.Conv2d:
            plt.savefig(os.path.join(
                dir_path, f"{norm_A.replace('_A','_means_ratios')}.svg"))
        elif layer_type == nn.BatchNorm2d:
            plt.savefig(os.path.join(
                dir_path, f"{norm_A.replace('_A','_means_ratios_bn')}.svg"))

        plt.grid()

        plt.close()


if __name__ == "__main__":
    main()
