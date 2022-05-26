import re
import argparse
import xlsxwriter
import os
import hydra
from omegaconf import DictConfig

resnet_layers_single = ["conv1",
                        "bn1",
                        "layer1.0.conv1",
                        "layer1.0.bn1",
                        "layer1.0.conv2",
                        "layer1.0.bn2",
                        "layer1.1.conv1",
                        "layer1.1.bn1",
                        "layer1.1.conv2",
                        "layer1.1.bn2",
                        "layer2.0.conv1",
                        "layer2.0.bn1",
                        "layer2.0.conv2",
                        "layer2.0.bn2",
                        "layer2.0.shortcut.0",
                        "layer2.0.shortcut.1",
                        "layer2.1.conv1",
                        "layer2.1.bn1",
                        "layer2.1.conv2",
                        "layer2.1.bn2",
                        "layer3.0.conv1",
                        "layer3.0.bn1",
                        "layer3.0.conv2",
                        "layer3.0.bn2",
                        "layer3.0.shortcut.0",
                        "layer3.0.shortcut.1",
                        "layer3.1.conv1",
                        "layer3.1.bn1",
                        "layer3.1.conv2",
                        "layer3.1.bn2",
                        "layer4.0.conv1",
                        "layer4.0.bn1",
                        "layer4.0.conv2",
                        "layer4.0.bn2",
                        "layer4.0.shortcut.0",
                        "layer4.0.shortcut.1",
                        "layer4.1.conv1",
                        "layer4.1.bn1",
                        "layer4.1.conv2",
                        "layer4.1.bn2",
                        "linear"]

resnet_layers_cutoff = ["conv1",
                        "bn1",
                        "layer1.0.conv1",
                        "layer1.0.bn1",
                        "layer1.0.conv2",
                        "layer1.0.bn2",
                        "layer1.1.conv1",
                        "layer1.1.bn1",
                        "layer1.1.conv2",
                        "layer1.1.bn2",
                        "layer2.0.conv1",
                        "layer2.0.bn1",
                        "layer2.0.conv2",
                        "layer2.0.bn2",
                        "layer2.0.shortcut.0",
                        "layer2.0.shortcut.1",
                        "layer2.1.conv1",
                        "layer2.1.bn1",
                        "layer2.1.conv2",
                        "layer2.1.bn2",
                        "layer3.0.conv1",
                        "layer3.0.bn1",
                        "layer3.0.conv2",
                        "layer3.0.bn2",
                        "layer3.0.shortcut.0",
                        "layer3.0.shortcut.1",
                        "layer3.1.conv1",
                        "layer3.1.bn1",
                        "layer3.1.conv2",
                        "layer3.1.bn2",
                        "layer4.0.conv1",
                        "layer4.0.bn1",
                        "layer4.0.conv2",
                        "layer4.0.bn2",
                        "layer4.0.shortcut.0",
                        "layer4.0.shortcut.1",
                        "layer4.1.conv1",
                        "layer4.1.bn1",
                        "layer4.1.conv2",
                        "layer4.1.bn2",
                        "linear",
                        "all_layers"]

vgg_layers_single = ["features.0",
                     "features.1",
                     "features.3",
                     "features.4",
                     "features.7",
                     "features.8",
                     "features.10",
                     "features.11",
                     "features.14",
                     "features.15",
                     "features.17",
                     "features.18",
                     "features.20",
                     "features.21",
                     "features.24",
                     "features.25",
                     "features.27",
                     "features.28",
                     "features.30",
                     "features.31",
                     "features.34",
                     "features.35",
                     "features.37",
                     "features.38",
                     "features.40",
                     "features.41",
                     "classifier"]

vgg_layers_cutoff = ["features.0",
                     "features.1",
                     "features.2",
                     "features.4",
                     "features.5",
                     "features.8",
                     "features.9",
                     "features.11",
                     "features.12",
                     "features.15",
                     "features.16",
                     "features.18",
                     "features.19",
                     "features.21",
                     "features.22",
                     "features.25",
                     "features.26",
                     "features.28",
                     "features.29",
                     "features.31",
                     "features.32",
                     "features.35",
                     "features.36",
                     "features.38",
                     "features.39",
                     "features.41",
                     "classifier",
                     "all_layers"]


@hydra.main(config_path="configs", config_name="cifar10.yaml")
def main(cfg: DictConfig) -> None:

    spreadsheet_dir = cfg.directory
    log_dir = os.path.join(cfg.directory, "logs")

    if cfg.model == "resnet":
        n_epochs = 80
        base_logname = os.path.join(
            log_dir, "resnet_sgd_step_lr_1.e-1_wd_2.e-4_mo_9.e-1_ep_80_adv_sgd_step_lr_1.e-1_wd_2.e-4_mo_9.e-1_ep_80_PGD_eps_8_Ns_10_ss_2_Nr_1_rand_")

    elif cfg.model == "vgg":
        n_epochs = 100
        base_logname = os.path.join(
            log_dir, "vgg_adam_step_lr_1.e-3_ep_100_adv_adam_step_lr_1.e-3_ep_100_PGD_eps_8_Ns_10_ss_2_Nr_1_rand_")

    workbook = xlsxwriter.Workbook(
        os.path.join(spreadsheet_dir, cfg.model+".xlsx"))

    worksheet_names = ["Accuracy retrain later layers",
                       "Accuracy retrain earlier layers",
                       "Accuracy retrain single layer",
                       "Loss retrain later layers",
                       "Loss retrain earlier layers",
                       "Loss retrain single layer"]

    bold = workbook.add_format({'bold': True})
    default = workbook.add_format()
    default.set_font_size(10)
    default.set_font_name('Arial')
    default.set_align('center')
    layer_name_format = workbook.add_format()
    layer_name_format.set_align('right')
    layer_name_format.set_font_size(10)
    layer_name_format.set_font_name('Arial')
    default_red = workbook.add_format()
    default_red.set_font_size(10)
    default_red.set_font_name('Arial')
    default_red.set_align('center')
    default_red.set_bg_color("#f4cccc")

    for sheet_name in worksheet_names:
        worksheet = workbook.add_worksheet(sheet_name)
        if cfg.model == "vgg":
            if "single" in sheet_name:
                layer_names = vgg_layers_single
            else:
                layer_names = vgg_layers_cutoff
        elif cfg.model == "resnet":
            if "single" in sheet_name:
                layer_names = resnet_layers_single
            else:
                layer_names = resnet_layers_cutoff

        if "earlier" in sheet_name or "single" in sheet_name:
            retrain_order = "21"
        else:
            retrain_order = "12"
        # row/column notation
        columns = ["cutoff before", "NN", "NA", "AA",
                   "AN", "", "NN", "NA", "AA", "AN"]

        for row_i, layer_name in enumerate([""]+layer_names):

            for column_i, column_name in enumerate(columns):
                if row_i == 0:
                    worksheet.write(0, column_i, column_name, bold)
                elif column_i == 0:
                    worksheet.write(row_i, 0, layer_name, layer_name_format)
                elif column_name == "":
                    continue
                else:
                    if "single" in sheet_name:
                        logname = base_logname + \
                            column_name[0]+retrain_order[0]+column_name[1] + \
                            retrain_order[1]+"_single_" + \
                            layer_name+".log"

                    elif layer_name == "features.0" or layer_name == "conv1":
                        assert column_name[1] in ["A", "N"]
                        if column_name[1] == "N":
                            if cfg.model == "vgg":
                                logname = os.path.join(
                                    log_dir, "vgg_adam_step_lr_1.e-3_ep_100.log")

                            elif cfg.model == "resnet":
                                logname = os.path.join(
                                    log_dir, "resnet_sgd_step_lr_1.e-1_wd_2.e-4_mo_9.e-1_ep_80.log")
                        else:
                            if cfg.model == "vgg":
                                logname = os.path.join(
                                    log_dir, "vgg_adv_adam_step_lr_1.e-3_ep_100_PGD_eps_8_Ns_10_ss_2_Nr_1_rand.log")
                            elif cfg.model == "resnet":
                                logname = os.path.join(
                                    log_dir, "resnet_adv_sgd_step_lr_1.e-1_wd_2.e-4_mo_9.e-1_ep_80_PGD_eps_8_Ns_10_ss_2_Nr_1_rand.log")

                    elif layer_name == "all_layers":
                        assert column_name[0] in ["A", "N"]
                        if column_name[0] == "N":
                            if cfg.model == "vgg":
                                logname = os.path.join(
                                    log_dir, "vgg_adam_step_lr_1.e-3_ep_100.log")
                            elif cfg.model == "resnet":
                                logname = os.path.join(
                                    log_dir, "resnet_sgd_step_lr_1.e-1_wd_2.e-4_mo_9.e-1_ep_80.log")
                        else:
                            if cfg.model == "vgg":
                                logname = os.path.join(
                                    log_dir, "vgg_adv_adam_step_lr_1.e-3_ep_100_PGD_eps_8_Ns_10_ss_2_Nr_1_rand.log")
                            elif cfg.model == "resnet":
                                logname = os.path.join(
                                    log_dir, "resnet_adv_sgd_step_lr_1.e-1_wd_2.e-4_mo_9.e-1_ep_80_PGD_eps_8_Ns_10_ss_2_Nr_1_rand.log")
                    else:
                        logname = base_logname + \
                            column_name[0]+retrain_order[0]+column_name[1] + \
                            retrain_order[1]+"_cutoff_before_" + \
                            layer_name+".log"

                    try:
                        f = open(logname, "r")
                        out = f.read()
                        f.close()

                    except FileNotFoundError:
                        worksheet.write(row_i, column_i, "", default_red)
                        continue

                    if column_i < 5:
                        matches = re.findall(
                            f"ep: {n_epochs-1} .*\n.*Test loss: (\d+\.\d+)\s+acc: (\d+\.\d+)", out)
                    else:
                        matches = re.findall(
                            f"Adversarial test loss: (\d+\.\d+)\s+acc: (\d+\.\d+)", out)

                    if len(matches) > 0:
                        loss, acc = matches[-1]  # get the last run
                        if "Loss" in sheet_name:
                            number = float(loss)
                            s = f"{number:.4f}"
                        else:
                            number = float(acc)*100
                            s = f"{number:.2f}"
                        worksheet.write(row_i, column_i, s, default)
                    else:
                        worksheet.write(row_i, column_i, "", default_red)

    workbook.close()


if __name__ == "__main__":
    main()
