#!/bin/bash

# ============================================================================
# Full Reproduction Script for "Disparate privacy risks from medical AI"
# ============================================================================
# 
# This script trains target models across all datasets for membership inference 
# attack evaluation. Each command trains 200 target models using leave-many-out
# data partitioning to enable comprehensive privacy auditing.
#
# RUNTIME WARNING: ~800 hours on a single A100 GPU
# PARALLELIZATION: Each command can be run independently on separate GPUs
#                  to dramatically reduce total runtime
#
# Usage:
#   bash scripts/train_all.sh              # Run full script sequentially
# ============================================================================

conda activate leakoscope

# baseline experiments
bash scripts/run_until_error.sh python chexpert.py --full_dataset=False --n_runs=200 --logdir="./logs/chexpert/wrn_28_2"
bash scripts/run_until_error.sh python mimic.py --full_dataset=False --n_runs=200 --logdir="./logs/mimic/wrn_28_2"
bash scripts/run_until_error.sh python embed.py --full_dataset=False --n_runs=200 --logdir="./logs/embed/wrn_28_2"
bash scripts/run_until_error.sh python fairvision.py --full_dataset=False --n_runs=200 --logdir="./logs/fairvision/wrn_28_2"
bash scripts/run_until_error.sh python fitzpatrick.py --full_dataset=False --n_runs=200 --logdir="./logs/fitzpatrick/wrn_28_2"
bash scripts/run_until_error.sh python ptb-xl.py --full_dataset=False --n_runs=200 --logdir="./logs/ptb-xl/resnet1d_128"
bash scripts/run_until_error.sh python mimic-iv.py --full_dataset=False --n_runs=200 --logdir="./logs/mimic-iv/resnet_300_6"


# model scaling experiments
bash scripts/run_until_error.sh python chexpert.py --full_dataset=False --n_runs=200 --model='wrn_40_4' --logdir="./logs/chexpert/wrn_28_5"
bash scripts/run_until_error.sh python chexpert.py --full_dataset=False --n_runs=200 --model='vit_b_16' --logdir="./logs/chexpert/vit_b_16"
bash scripts/run_until_error.sh python chexpert.py --full_dataset=False --n_runs=200 --model='vit_b_16' --img_size=128,128 --logdir="./logs/chexpert/vit_b_16_128x128"
bash scripts/run_until_error.sh python chexpert.py --full_dataset=False --n_runs=200 --model='vit_l_16' --logdir="./logs/chexpert/vit_l_16"

bash scripts/run_until_error.sh python fitzpatrick.py --full_dataset=False --n_runs=200 --model='wrn_40_4' --logdir="./logs/fitzpatrick/wrn_28_5"
bash scripts/run_until_error.sh python fitzpatrick.py --full_dataset=False --n_runs=200 --model='vit_b_16' --logdir="./logs/fitzpatrick/vit_b_16"
bash scripts/run_until_error.sh python fitzpatrick.py --full_dataset=False --n_runs=200 --model='vit_b_16' --img_size=128,128 --logdir="./logs/fitzpatrick/vit_b_16_128x128"
bash scripts/run_until_error.sh python fitzpatrick.py --full_dataset=False --n_runs=200 --model='vit_l_16' --logdir="./logs/fitzpatrick/vit_l_16"

# differential privacy experiments
bash scripts/run_until_error.sh python embed_dp.py --full_dataset=False --n_runs=200 --epsilon=1 --logdir="./logs/embed/dp/eps1"
bash scripts/run_until_error.sh python embed_dp.py --full_dataset=False --n_runs=200 --epsilon=10 --logdir="./logs/embed/dp/eps10"
bash scripts/run_until_error.sh python embed_dp.py --full_dataset=False --n_runs=200 --epsilon=100 --logdir="./logs/embed/dp/eps100"
bash scripts/run_until_error.sh python embed_dp.py --full_dataset=False --n_runs=200 --epsilon=1000 --logdir="./logs/embed/dp/eps1000"
bash scripts/run_until_error.sh python embed_dp.py --full_dataset=False --n_runs=200 --logdir="./logs/embed/dp/epsinf"

bash scripts/run_until_error.sh python ptb-xl_dp.py --full_dataset=False --n_runs=200 --epsilon=1 --logdir="./logs/ptb-xl/dp/eps1"
bash scripts/run_until_error.sh python ptb-xl_dp.py --full_dataset=False --n_runs=200 --epsilon=10 --logdir="./logs/ptb-xl/dp/eps10"
bash scripts/run_until_error.sh python ptb-xl_dp.py --full_dataset=False --n_runs=200 --epsilon=100 --logdir="./logs/ptb-xl/dp/eps100"
bash scripts/run_until_error.sh python ptb-xl_dp.py --full_dataset=False --n_runs=200 --epsilon=1000 --logdir="./logs/ptb-xl/dp/eps1000"
bash scripts/run_until_error.sh python ptb-xl_dp.py --full_dataset=False --n_runs=200 --logdir="./logs/ptb-xl/dp/epsinf"

# open-source model attacks
conda activate torch
python xrv_scores.py
conda activate med-leak
python xrv_attack.py
