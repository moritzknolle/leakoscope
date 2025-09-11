#!/bin/bash

PATIENT_LEVEL_ONLY=False # if set to True only patient/record-level MIA analysis is performed (no aggregate attack success). This is faster

python mem_inf_stats.py --logdir="./logs/chexpert/wrn_28_2" --dataset="chexpert" --patient_level_only=$PATIENT_LEVEL_ONLY --plot_images=True
python mem_inf_stats.py --logdir="./logs/mimic/wrn_28_2" --dataset="mimic" --patient_level_only=$PATIENT_LEVEL_ONLY --plot_images=True
python mem_inf_stats.py --logdir="./logs/embed/wrn_28_2" --dataset="embed" --patient_level_only=$PATIENT_LEVEL_ONLY --plot_images=True --img_size=256,192
python mem_inf_stats.py --logdir="./logs/fairvision/wrn_28_2" --dataset="fairvision" --patient_level_only=$PATIENT_LEVEL_ONLY --plot_images=True
python mem_inf_stats.py --logdir="./logs/fitzpatrick/wrn_28_2" --dataset="fitzpatrick" --patient_level_only=$PATIENT_LEVEL_ONLY --plot_images=True
python mem_inf_stats.py --logdir="./logs/mimic-iv/resnet_300_6" --dataset="mimic-iv-ed" --patient_level_only=$PATIENT_LEVEL_ONLY
python mem_inf_stats.py --logdir="./logs/ptb-xl/resnet1d_128" --dataset="ptb-xl" --patient_level_only=$PATIENT_LEVEL_ONLY

python plots.py --log_scale=True --mia_method="rmia"
python plots.py --log_scale=False --mia_method="rmia"
python plots.py --log_scale=True --mia_method="lira"
python plots.py --log_scale=False --mia_method="lira"

python dp_plots.py --dataset_name='ptb-xl' --patient_level=False --ylim_lower=0.75 --ylim_upper=0.925
python dp_plots.py --dataset_name='ptb-xl' --patient_level=True --ylim_lower=0.75 --ylim_upper=0.925
python dp_plots.py --dataset_name='embed' --patient_level=False --ylim_lower=0.9 --ylim_upper=0.95
python dp_plots.py --dataset_name='embed' --patient_level=True --ylim_lower=0.9 --ylim_upper=0.95


python model_scaling_plots.py --dataset_name='chexpert'
python model_scaling_plots.py --dataset_name='fitzpatrick'

# appendix plots
python scatter_plot.py
python standard_error_ablation.py