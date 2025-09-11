
# leakoscope🩺: patient-level privacy audits of medical AI

Code to reproduce results from our paper **"Disparate privacy risks from medical AI"**.


## Installation

This codebase requires a functional [jax](https://docs.jax.dev) installation with GPU support as prerequisite. Up-to-date installation instructions can be found [here](https://docs.jax.dev/en/latest/installation.html#conda-installation).

```bash
    conda create -n leakoscope python=3.9.18
    conda activate leakoscope
    # install jax now by following the instructions at https://docs.jax.dev/en/latest/installation.html
    git clone https://github.com/moritzknolle/leakoscope.git
    cd leakoscope
    pip install -r requirements.txt
    mkdir figs
    mkdir logs
```
    
## Datasets
All investigated datasets are publicly available for research purposes, see table below for access links.

| **Name** | **# Patients** | **# Images** |
|------------|--------------|------------|
| [CheXpert](https://stanfordaimi.azurewebsites.net/datasets/192ada7c-4d43-466e-b8bb-b81992bb80cf) | 65,420 | 224,316 |
| [CheXpert (demographic)](https://stanfordaimi.azurewebsites.net/datasets/192ada7c-4d43-466e-b8bb-b81992bb80cf) | 65,420 | 224,316 |
| [MIMIC-CXR](https://physionet.org/content/mimic-cxr-jpg/2.0.0/) | 65,379 | 377,110 |
| [MIMIC-CXR (demographic)](https://physionet.org/content/mimiciv/1.0/) | 65,379 | 377,110 |
| [Fitzpatrick-17k](https://github.com/mattgroh/fitzpatrick17k) | n.a. | 16,523 |
| [Harvard FairVision](https://ophai.hms.harvard.edu/datasets/harvard-fairvision30k) | 30,000 | 30,000 |
| [EMBED](https://registry.opendata.aws/emory-breast-imaging-dataset-embed/) | 23,057 | 451,642 |
| [PTB-XL](https://physionet.org/content/ptb-xl/1.0.3/) | 18,869 | 21,799 |
| [MIMIC-IV-ED](https://physionet.org/content/mimic-iv-ed/2.2/) | 201,213 | 418,007 |

Dataset-specific pre-processing code can be found in ```src/data_utils/notebooks/```.

## Code Structure

This repository is organized as follows:

```
├── src/                          # Core source code
│   ├── data_utils/              # Dataset loading and preprocessing utilities
│   │   ├── notebooks/           # Dataset-specific preprocessing scripts
│   │   └── datasets.py          # Base dataset classes
│   ├── train_utils/             # Training utilities and model definitions
│   │   ├── models/              # Model architectures (WRN, ViT, CNNs)
│   │   ├── training.py          # Core training loops
│   │   └── utils.py             # Training utilities
│   └── privacy_utils/           # Privacy attack implementations
│       ├── lira.py              # LiRA (Likelihood Ratio Attack) implementation
│       ├── rmia.py              # RMIA (Robust Membership Inference Attack) implementation
│       ├── common.py            # Record-level MIA success analysis and other utilities
│       └── plot_utils.py        # Visualization utilities for attack results
├── scripts/                      # Automation scripts
│   ├── train_all.sh             # Full reproduction script (WARNING: ~800h runtime)
│   └── plot_all.sh              # Generate all plots from paper
├── {dataset}.py                  # Main entry scripts for each dataset
├── mem_inf_stats.py             # Privacy attack analysis and plotting
```

### Key Components

- **Entry Scripts**: Each dataset has a corresponding Python script (e.g., `fitzpatrick.py`, `chexpert.py`) that handles model training for that specific dataset
- **Privacy Attacks**: The `privacy_utils/` module implements membership inference attacks (LiRA, RMIA) used to audit trained models
- **Model Zoo**: Support for various architectures including Wide ResNets (WRN), Vision Transformers (ViT), and residual networks
- **Training Pipeline**: Automated training with support for data augmentation, learning rate scheduling, and model checkpointing

### Common Command Line Arguments

All dataset entry scripts support the following key arguments:

| Argument  | Description |  Example |
|----------------|-------------|---------|
| `--n_runs` | Number of models to train | `--n_runs=200` |
| `--full_dataset` | Train on full dataset (True) or random subset for privacy auditing (False) | `--full_dataset=False` |
| `--logdir` | Directory to save training logs and models  | `--logdir="./logs/mylogdir"` |
| `--model` | Model architecture to use | `--model=vit_b_16` |
| `--epochs` | Number of training epochs| `--epochs=200` |
| `--batch_size` | Training batch size  | `--batch_size=512` |
| `--lr` | learning rate | `--lr=1e-2` |
| `--seed` | Random seed for reproducibility | `--seed=123` |

**Available Models**: `wrn_28_2`, `wrn_40_4`, `vit_b_16`, `vit_b_32`, `vit_l_16`, `resnet_1d_128` (dataset-dependent), `resnet_300_6` (dataset-dependent)

### Convenience Script: `scripts/run_until_error.sh`

The `scripts/run_until_error.sh` script is provided for user convenience to automate the training of many models. Instead of manually re-running each dataset-specific entry script for each model you want to train, this script will continuously execute the provided command until an error occurs.

**How it works:**
- Takes any command as arguments and runs it repeatedly in a loop
- Counts successful executions and displays progress
- When the specified number of models (via `--n_runs`) have finished training, the dataset entry script will raise an error and training will terminate automatically
- Syncs wandb runs and cleans up local log directories upon completion

**Usage:**
```bash
bash scripts/run_until_error.sh <your-training-command>
```

**Example:**
```bash
bash scripts/run_until_error.sh python fitzpatrick.py --n_runs=50 --full_dataset=False --logdir="./logs/fitzpatrick"
```

This is particularly useful for privacy auditing workflows where you need to train many target models with the same configuration.

## Example: Reproducing patient-level audit results for Fitzpatrick 17k
Patient/record-level privacy audits require measuring attack performance individually for each record across many target models.
Training many target models can be computationally expensive, thus we recommend to start out with *Fitzpatrick 17k*. For this dataset training a highly performant WRN-28-2 model only takes 3 mins (on a single A100 GPU). The commands below will train 50 target models and then compute and visualise aggregate, record- and patient-level attack success.   

```bash
conda activate leakoscope
bash scripts/run_until_error.sh python fitzpatrick.py --n_runs=50 --full_dataset=False --logdir="./logs/fitzpatrick" # train models on random subsets
python mem_inf_stats.py --logdir="./logs/fitzpatrick" --dataset='fitzpatrick" # perform attacks and plot results
```


## Reproducing attacks on open-source models
Requires an additional [PyTorch](https://pytorch.org/) and [TorchXrayVision](https://mlmed.org/torchxrayvision/) installations (see [here](https://pytorch.org/get-started/locally/) for up-to-date installation instructions). To replicate results for attacking the CheXpert and MIMIC-CXR models from TorchXrayVision, run the following commands below.

```bash
conda activate torch # activate your environment with Pytorch and TorchXrayVision installed
python xrv_scores.py # model inference
conda activate med-leak
python xrv_attack.py # conduct attacks
```

## Reproducing all results

To reproduce all quantitative results presented in the paper simply run the commands below.

```bash
bash scripts/train_all.sh
bash scripts/plot_all.sh
```

**WARNING**: this will take a LONG time (~800h on single A100 GPU)! Fortunately, you can accelerate the process with more GPUs. Simply (re)-run each line in ```scripts/train_all.sh``` as many times as you like for each GPU-machine you have available (underlying logic in ```src/train_utils/training.py``` will handle concurrency).
