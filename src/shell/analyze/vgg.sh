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
    if $single_layer_different; 
    then
        cutoffs='features.0
                features.1
                features.3
                features.4
                features.7
                features.8
                features.10
                features.11
                features.14
                features.15
                features.17
                features.18
                features.20
                features.21
                features.24
                features.25
                features.27
                features.28
                features.30
                features.31
                features.34
                features.35
                features.37
                features.38
                features.40
                features.41
                classifier'
    else
        cutoffs='features.1 
                features.2 
                features.4 
                features.5 
                features.8 
                features.9 
                features.11
                features.12
                features.15
                features.16
                features.18
                features.19
                features.21
                features.22
                features.25
                features.26
                features.28
                features.29
                features.31
                features.32
                features.35
                features.36
                features.38
                features.39
                features.41
                classifier'
    fi
fi

if $single_layer_different; then
    cutoff_or_single="single_layer_different"
else
    cutoff_or_single="cutoff_before"
fi

for retrain_code in ${retrain_codes[@]}; 
do
    for file in ${files[@]}; 
    do
        simple_hypersearch "python -m layersnotequalinAT.src.visualization.perturbation train_type=retrain retrain_code=$retrain_code $cutoff_or_single={cutoff} adv_train.lr=1e-3 adv_train.optimizer=adam adv_train.wd=0 adv_train.n_epochs=100 \"adv_train.scheduler_steps=[50,75]\" model=vgg  train.lr=1e-3 train.optimizer=adam train.wd=0 train.n_epochs=100 \"train.scheduler_steps=[50,75]\"" -p cutoff ${cutoffs[@]} | simple_gpu_scheduler --gpus ${gpus[@]}
    done 
done

