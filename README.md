# Not All Layers Are Equal in Adversarial Training

This repository contains the code for the paper "Not All Layers Are Equal in Adversarial Training". Observations in the paper are achieved by
1) Partial adversarial training: adversarially train some layers, freeze their weights and then naturally train other layers from a fresh initialization. Or the other way around: naturally train some layers, freeze, then adversarially train the rest from scratch.
2) Perturbation-to-signal ratio (PSR) tracking across the layers.

## Requirements

Install the requirements:

```bash
pip install -r requirements.txt
```

Then add current directory to `$PYTHONPATH`:

```bash
export PYTHONPATH="${PYTHONPATH}:/path/containing/project/directory"
```

and change `directory` variable in `src/configs/cifar10.yaml` to `/path/containing/project/directory/layersnotequalinAT`

CIFAR-10 will be downloaded automatically when the code is run.

Following commands assume the name of the folder is `layersnotequalinAT`.


## Basic Commands

### For ResNet

Train all natural and all adversarial models (prerequisite for the rest):

```
python -m layersnotequalinAT.src.train train_type=all_nat
python -m layersnotequalinAT.src.train train_type=all_adv
```

Train all cutoff combinations (earlier layers retrained, later layers retrained):
```
shell/retrain/resnet.sh
```

Train all single layer retrain combinations:
```
shell/retrain/resnet.sh -r="A2A1 N2N1 N2A1 A2N1" -s
```

Train only one cutoff combination:
```
python -m layersnotequalinAT.src.train train_type=retrain retrain_code=A1N2 cutoff_before=layer2.1.bn1
```

Train only one single layer retrain (retrain code always in the form of x2x1):
```
python -m layersnotequalinAT.src.train train_type=retrain retrain_code=A2N1 single_layer_different=layer2.1.bn1
```

Analyze and plot PSR across layers for all adversarial and all natural models:
```
python -m layersnotequalinAT.src.visualization.psr train_type=retrain retrain_code=A1x2 cutoff_before=conv1
```


Resnet list of layers:
- bn1
- layer1.0.conv1
- layer1.0.bn1
- layer1.0.conv2
- layer1.0.bn2
- layer1.1.conv1
- layer1.1.bn1
- layer1.1.conv2
- layer1.1.bn2
- layer2.0.conv1
- layer2.0.bn1
- layer2.0.conv2
- layer2.0.bn2
- layer2.0.shortcut.0
- layer2.0.shortcut.1
- layer2.1.conv1
- layer2.1.bn1
- layer2.1.conv2
- layer2.1.bn2
- layer3.0.conv1
- layer3.0.bn1
- layer3.0.conv2
- layer3.0.bn2
- layer3.0.shortcut.0
- layer3.0.shortcut.1
- layer3.1.conv1
- layer3.1.bn1
- layer3.1.conv2
- layer3.1.bn2
- layer4.0.conv1
- layer4.0.bn1
- layer4.0.conv2
- layer4.0.bn2
- layer4.0.shortcut.0
- layer4.0.shortcut.1
- layer4.1.conv1
- layer4.1.bn1
- layer4.1.conv2
- layer4.1.bn2
- linear

### For VGG

Train all natural and all adversarial models (prerequisite for the rest):

```
python -m layersnotequalinAT.src.train train_type=all_nat model=vgg train.lr=1e-3 train.optimizer=adam train.wd=0 train.n_epochs=100 "train.scheduler_steps=[50,75]"

python -m layersnotequalinAT.src.train train_type=all_adv model=vgg adv_train.lr=1e-3 adv_train.optimizer=adam adv_train.wd=0 adv_train.n_epochs=100 "adv_train.scheduler_steps=[50,75]"
```

Train all cutoff combinations (earlier layers retrained, later layers retrained):
```
shell/retrain/vgg.sh
```

Train all single layer retrain combinations:
```
shell/retrain/vgg.sh -r="A2A1 N2N1 N2A1 A2N1" -s
```

Train only one cutoff combination:
```
python -m layersnotequalinAT.src.train train_type=retrain model=vgg retrain_code=A1N2 cutoff_before=features.18 adv_train.lr=1e-3 adv_train.optimizer=adam adv_train.wd=0 adv_train.n_epochs=100 "adv_train.scheduler_steps=[50,75]" train.lr=1e-3 train.optimizer=adam train.wd=0 train.n_epochs=100 "train.scheduler_steps=[50,75]"
```

Train only one single layer retrain (retrain code always in the form of x2x1):
```
python -m layersnotequalinAT.src.train train_type=retrain retrain_code=A2N1 single_layer_different=features.18 adv_train.lr=1e-3 adv_train.optimizer=adam adv_train.wd=0 adv_train.n_epochs=100 "adv_train.scheduler_steps=[50,75]" train.lr=1e-3 train.optimizer=adam train.wd=0 train.n_epochs=100 "train.scheduler_steps=[50,75]"
```

Analyze and plot PSR across layers for all adversarial and all natural models:
```
python -m layersnotequalinAT.src.visualization.psr train_type=retrain retrain_code=A1x2 cutoff_before=features.0 adv_train.lr=1e-3 adv_train.optimizer=adam adv_train.wd=0 adv_train.n_epochs=100 "adv_train.scheduler_steps=[50,75]" model=vgg  train.lr=1e-3 train.optimizer=adam train.wd=0 train.n_epochs=100 "train.scheduler_steps=[50,75]"
```

VGG list of layers:
- conv (0)
- bn (1)
- relu (2)
- conv (3)
- bn (4)
- relu (5)
- maxpool (6)
- conv (7)
- bn (8)
- relu (9)
- conv (10)
- bn (11)
- relu (12)
- maxpool (13)
- conv (14)
- bn (15)
- relu (16)
- conv (17)
- bn (18)
- relu (19)
- conv (20)
- bn (21)
- relu (22)
- maxpool (23)
- conv (24)
- bn (25)
- relu (26)
- conv (27)
- bn (28)
- relu (29)
- conv (30)
- bn (31)
- relu (32)
- maxpool (33)
- conv (34)
- bn (35)
- relu (36)
- conv (37)
- bn (38)
- relu (39)
- conv (40)
- bn (41)
- relu (42)
- maxpool (43)
- avgpool (44)
- classifier

## Folder Structure 

Adversarial framework folder contains the codes for adversarial attacks, analysis, and adversarial training functions. Src folder contains all the necessary codes for autoencoder, training, testing, models, and utility functions. Repository structure is as follows:

```
Repository
│   README.md                   This file
│   requirements.txt            Required libraries
│   LICENSE                     Apache License
│	
└───src     
    │   extract_tabular_info.py              extracts accuracy&losses from checkpoints and write to xlsx file
    │   retrain.py                           freeze part of model and retrain
    │   test.py                              nat&adv test a checkpoint
    │   train_single_type.py                 train all layers with the same type of training (no freezing)
    │   train_test_functions.py              implementation of train&test functions
    │   train.py                             train wrapper
    │
    │───models
    │       freeze_reinitialize.py 	         model wrapper that freezes&reinitializes
    │       layer_output_extractor.py 	     wrapper with hooks to get intermediate layer outputs
    │       resnet.py                        ResNet
    │       vgg.py                           VGG
    │
    │───shell
    │   │
    │   │───analyze
    │   │       resnet.sh                    runs PSR code for all resnet checkpoints
    │   │       vgg.sh                       runs PSR code for all vgg checkpoints
    │   │
    │   └───retrain
    │           resnet.sh                    given all nat and all adv checkpoints retrains all resnet cutoff combinations
    │           vgg.sh                       given all nat and all adv checkpoints retrains all vgg cutoff combinations
    │       
    │───utils
    │       data_loaders.py                  creates data loaders for CIFAR10
    │       initializers.py                  initializes loaders, models, schedulers, loggers etc.
    │       namers.py                        creates checkpoint names
    │       saver.py                         model saver
    │
    └───visualization
            plot_settings.py                 pretty plot settings
            psr.py                           generate PSR graphs
            
```

## License

Apache License 2.0
