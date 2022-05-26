#!/bin/bash

single_layer_different=false
declare retrain_codes cutoffs gpus
declare OPTIND OPTARG opt
while getopts :r:g:c:s opt; do
    case $opt in
    r)  declare i=1
        retrain_codes[0]=$OPTARG
        while [[ ${OPTIND} -le $# && ${!OPTIND:0:1} != '-' ]]; do
            retrain_codes[i]=${!OPTIND}
            let i++ OPTIND++
        done
        ;;
    g)  declare i=1
        gpus[0]=$OPTARG
        while [[ ${OPTIND} -le $# && ${!OPTIND:0:1} != '-' ]]; do
            gpus[i]=${!OPTIND}
            let i++ OPTIND++
        done
        ;;
    c)  declare i=1
        cutoffs[0]=$OPTARG
        while [[ ${OPTIND} -le $# && ${!OPTIND:0:1} != '-' ]]; do
            cutoffs[i]=${!OPTIND}
            let i++ OPTIND++
        done
        ;;
    s) single_layer_different=true
        ;;
    \?) echo "Invalid option: -$OPTARG" >&2
        ;;
    :) echo "Option -$OPTARG requires an argument." >&2
        exit 1
        ;;
    esac
done


if [ ${#retrain_codes[@]} -eq 0 ]; 
then
    retrain_codes="A1N2 N1N2 N1A2 A1A2 A2A1 N2N1 N2A1 A2N1"
fi

if [ ${#gpus[@]} -eq 0 ]; 
then
    gpus="0 1 2 3"
fi


if [ ${#cutoffs[@]} -eq 0 ]; 
then
    cutoffs='bn1
            layer1.0.conv1
            layer1.0.bn1
            layer1.0.conv2
            layer1.0.bn2
            layer1.1.conv1
            layer1.1.bn1
            layer1.1.conv2
            layer1.1.bn2
            layer2.0.conv1
            layer2.0.bn1
            layer2.0.conv2
            layer2.0.bn2
            layer2.0.shortcut.0
            layer2.0.shortcut.1
            layer2.1.conv1
            layer2.1.bn1
            layer2.1.conv2
            layer2.1.bn2
            layer3.0.conv1
            layer3.0.bn1
            layer3.0.conv2
            layer3.0.bn2
            layer3.0.shortcut.0
            layer3.0.shortcut.1
            layer3.1.conv1
            layer3.1.bn1
            layer3.1.conv2
            layer3.1.bn2
            layer4.0.conv1
            layer4.0.bn1
            layer4.0.conv2
            layer4.0.bn2
            layer4.0.shortcut.0
            layer4.0.shortcut.1
            layer4.1.conv1
            layer4.1.bn1
            layer4.1.conv2
            layer4.1.bn2
            linear'

fi

if $single_layer_different; then
    cutoff_or_single="single_layer_different"
else
    cutoff_or_single="cutoff_before"
fi

for retrain_code in ${retrain_codes[@]}; 
do
    simple_hypersearch "python -m layersnotequalinAT.src.train train_type=retrain retrain_code=$retrain_code $cutoff_or_single={cutoff}" -p cutoff ${cutoffs[@]} | simple_gpu_scheduler --gpus ${gpus[@]}
done
