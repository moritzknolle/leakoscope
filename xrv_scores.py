# this script requires a functional torch and torchxraysvision installation
import random
from pathlib import Path
from typing import List

import numpy as np
import skimage
import sklearn
import torch
import torchvision
import torchxrayvision as xrv
from absl import app, flags
from scipy.special import logit
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from src.data_utils.constants import CXP_CHALLENGE_LABELS_IDX, CXR_LABELS
from src.data_utils.dataset_factory import get_dataset

FLAGS = flags.FLAGS
flags.DEFINE_integer("img_size", 224, "Size of the image.")
flags.DEFINE_string(
    "save_root",
    "/home/moritz/data_fast/npy",
    "Path to root folder where the memmap files are stored.",
)
flags.DEFINE_string(
    "log_root",
    "./logs/torchxrayvision/",
    "Path to root folder where the memmap files are stored.",
)
flags.DEFINE_integer(
    "N_per", 25_000, "Number of N_in/N_out samples to test the attack on."
)
flags.DEFINE_boolean("debug_images", False, "Debug mode.")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_chexpert(seed: int, transforms: List, batch_size: int = 1):
    dataset = xrv.datasets.CheX_Dataset(
        imgpath="/home/moritz/data/chexpert/CheXpert-v1.0-small",
        csvpath="/home/moritz/data/chexpert/CheXpert-v1.0-small/train.csv",
        transform=transforms,
        data_aug=None,
        unique_patients=False,
        views=["PA", "AP"],
    )
    # re-label the dataset to match the TorchXrayVision labels
    xrv.datasets.relabel_dataset(xrv.datasets.default_pathologies, dataset)
    # we re-create the splits from TorchXrayVision to make sure that all records in train_dataset actually were used to train the model
    gss = sklearn.model_selection.GroupShuffleSplit(
        train_size=0.8, test_size=0.2, random_state=seed
    )
    train_inds, _ = next(gss.split(X=range(len(dataset)), groups=dataset.csv.patientid))
    train_dataset = xrv.datasets.SubsetDataset(dataset, train_inds)
    gss = sklearn.model_selection.GroupShuffleSplit(
        train_size=0.8, test_size=0.2, random_state=seed
    )
    train_inds, _ = next(
        gss.split(X=range(len(train_dataset)), groups=train_dataset.csv.patientid)
    )
    train_dataset = xrv.datasets.SubsetDataset(train_dataset, train_inds)
    return torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=16,
        pin_memory=True,
        prefetch_factor=4,
    )


def get_mimic(seed: int, transforms: List, batch_size: int = 1):
    dataset = xrv.datasets.MIMIC_Dataset(
        imgpath="/home/moritz/data/mimic-cxr/mimic-cxr-jpg/files",
        csvpath="/home/moritz/data/mimic-cxr/mimic-cxr-jpg/mimic-cxr-2.0.0-chexpert.csv.gz",
        metacsvpath="/home/moritz/data/mimic-cxr/mimic-cxr-jpg/mimic-cxr-2.0.0-metadata.csv",
        transform=transforms,
        data_aug=None,
        unique_patients=False,
        views=["PA", "AP"],
    )
    # re-label the dataset to match the TorchXrayVision labels
    xrv.datasets.relabel_dataset(xrv.datasets.default_pathologies, dataset)
    # we re-create the splits from TorchXrayVision to make sure that all records in train_dataset actually were used to train the model
    gss = sklearn.model_selection.GroupShuffleSplit(
        train_size=0.8, test_size=0.2, random_state=seed
    )
    train_inds, _ = next(gss.split(X=range(len(dataset)), groups=dataset.csv.patientid))
    train_dataset = xrv.datasets.SubsetDataset(dataset, train_inds)
    gss = sklearn.model_selection.GroupShuffleSplit(
        train_size=0.8, test_size=0.2, random_state=seed
    )
    train_inds, _ = next(
        gss.split(X=range(len(train_dataset)), groups=train_dataset.csv.patientid)
    )
    train_dataset = xrv.datasets.SubsetDataset(train_dataset, train_inds)
    return torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=16,
        pin_memory=True,
        prefetch_factor=8,  # agressive prefetching as images need to be resized
    )


def main(argv):
    transforms = torchvision.transforms.Compose(
        [xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(FLAGS.img_size)]
    )
    xrv_seed = 0  # seed from the config file on the TorchXrayVision github repository (to replicate exact train-test splits)
    np.random.seed(xrv_seed)
    random.seed(xrv_seed)
    torch.manual_seed(xrv_seed)
    # load datasets
    chex_loader = get_chexpert(seed=xrv_seed, transforms=transforms)
    print(f"Loaded CheXpert with {len(chex_loader)} images.")
    mimic_loader = get_mimic(seed=xrv_seed, transforms=transforms)
    print(f"Loaded MIMIC-CXR with {len(mimic_loader)} images.")

    # load target model (CheXpert)
    chex_model = xrv.models.DenseNet(weights="densenet121-res224-chex").to(device)
    # load target model (MIMIC-CXR)
    mimic_model = xrv.models.DenseNet(weights="densenet121-res224-mimic_ch").to(device)
    # load a pre-trained reference model (Pad Chest)
    reference_model = xrv.models.DenseNet(weights="densenet121-res224-pc").to(device)
    chex_model.eval()
    mimic_model.eval()
    reference_model.eval()

    reference_model_logits = []
    chex_model_logits = []
    mimic_model_logits = []
    labels = []
    # get chexpert predictions from the target and reference models
    for i, batch in enumerate(tqdm(chex_loader, total=FLAGS.N_per, desc="Inference on CheXpert data")):
        x = batch["img"].to(device)
        y = batch["lab"].detach().cpu().numpy().squeeze()
        reference_output = reference_model(x).detach().cpu().numpy().squeeze()
        reference_model_logits.append(
            logit(reference_output)
        )  # we save the logit-transformed probabilities (logit transform is inverse of the sigmoid activation)
        chex_model_output = chex_model(x).detach().cpu().numpy().squeeze()
        chex_model_logits.append(logit(chex_model_output))
        mimic_model_output = mimic_model(x).detach().cpu().numpy().squeeze()
        mimic_model_logits.append(logit(mimic_model_output))
        labels.append(y)
        assert (
            reference_output.shape
            == chex_model_output.shape
            == mimic_model_output.shape
            == y.shape
        ), f"Shapes do not match: {reference_output.shape}, {chex_model_output.shape}, {mimic_model_output.shape}, {y.shape}"
        if i + 1 >= FLAGS.N_per:
            break
    del chex_loader
    # get mimic-cxr predictions from the target and reference models
    for i, batch in enumerate(tqdm(mimic_loader, total=FLAGS.N_per, desc="Inference on MIMIC-CXR data")):
        x = batch["img"].to(device)
        y = batch["lab"].detach().cpu().numpy().squeeze()
        reference_output = reference_model(x).detach().cpu().numpy().squeeze()
        reference_model_logits.append(logit(reference_output))
        chex_model_output = chex_model(x).detach().cpu().numpy().squeeze()
        chex_model_logits.append(logit(chex_model_output))
        mimic_model_output = mimic_model(x).detach().cpu().numpy().squeeze()
        mimic_model_logits.append(logit(mimic_model_output))
        labels.append(y)
        assert (
            reference_output.shape
            == chex_model_output.shape
            == mimic_model_output.shape
            == y.shape
        ), f"Shapes do not match: {reference_output.shape}, {chex_model_output.shpe}, {mimic_model_output}, {y.shape}"
        if i + 1 >= FLAGS.N_per:
            break

    reference_model_logits = np.stack(reference_model_logits)
    chex_model_logits = np.stack(chex_model_logits)
    mimic_model_logits = np.stack(mimic_model_logits)
    # replace NaNs with zeros
    labels = np.nan_to_num(np.stack(labels))
    chex_member_mask = np.concatenate(
        [np.ones(FLAGS.N_per, dtype=np.bool_), np.zeros(FLAGS.N_per, dtype=np.bool_)]
    )
    mimic_member_mask = np.concatenate(
        [
            np.zeros(FLAGS.N_per, dtype=np.bool_),
            np.ones(FLAGS.N_per, dtype=np.bool_),
        ]
    )
    print(chex_member_mask, mimic_member_mask)
    print(
        f"Shapes: member mask={chex_member_mask.shape}, labels={labels.shape}, reference logits={reference_model_logits.shape}, chexpert model logits={chex_model_logits.shape}, mimic model logits={mimic_model_logits.shape}"
    )
    assert (
        len(chex_member_mask)
        == len(mimic_member_mask)
        == len(labels)
        == len(reference_model_logits)
        == len(chex_model_logits)
        == len(mimic_model_logits)
    ), f"Shapes do not match: {len(chex_member_mask)}, {len(mimic_member_mask)}, {len(labels)}, {len(reference_model_logits)}, {len(chex_model_logits)}, {len(mimic_model_logits)}"
    assert (
        chex_member_mask.sum() == mimic_member_mask.sum() == FLAGS.N_per
    ), f"Member mask does not match: {chex_member_mask.sum()} and {mimic_member_mask.sum()} != {FLAGS.N_per}"
    print(
        "reference logits stats:",
        reference_model_logits.min(),
        reference_model_logits.max(),
        reference_model_logits.mean(),
        reference_model_logits.std(),
    )
    print(
        "Chexpert model logits stats:",
        chex_model_logits.min(),
        chex_model_logits.max(),
        chex_model_logits.mean(),
        chex_model_logits.std(),
    )
    print(
        "MIMIC-CXR model logits stats:",
        mimic_model_logits.min(),
        mimic_model_logits.max(),
        mimic_model_logits.mean(),
        mimic_model_logits.std(),
    )
    # sanity check model performance
    mimic_model_aucs = [
        roc_auc_score(labels[:, i], mimic_model_logits[:, i])
        for i in range(labels.shape[1])
    ]
    chex_model_aucs = [
        roc_auc_score(labels[:, i], chex_model_logits[:, i])
        for i in range(labels.shape[1])
    ]
    reference_aucs = [
        roc_auc_score(labels[:, i], reference_model_logits[:, i])
        for i in range(labels.shape[1])
    ]
    print(
        f"....... Chexpert model AUC ({np.nanmean(chex_model_aucs)}): \n",
        dict(zip(xrv.datasets.default_pathologies, chex_model_aucs)),
    )
    print(
        f"....... MIMIC-CXR model AUC ({np.nanmean(mimic_model_aucs)}): \n",
        dict(zip(xrv.datasets.default_pathologies, mimic_model_aucs)),
    )
    print(
        f"....... reference model AUC ({np.nanmean(reference_aucs)}): \n",
        dict(zip(xrv.datasets.default_pathologies, reference_aucs)),
    )
    # log results to disk
    reference_log_dir = Path(FLAGS.log_root) / "reference"
    chexpert_model_log_dir = Path(FLAGS.log_root) / "chexpert"
    mimic_model_log_dir = Path(FLAGS.log_root) / "mimic"
    reference_log_dir.mkdir(parents=True, exist_ok=True)
    chexpert_model_log_dir.mkdir(parents=True, exist_ok=True)
    mimic_model_log_dir.mkdir(parents=True, exist_ok=True)
    # reference model
    np.save(reference_log_dir / "train_logits.npy", reference_model_logits)
    np.save(
        reference_log_dir / "subset_mask.npy", np.zeros_like(chex_member_mask)
    )  # reference model was trained offline on PadChest
    np.save(reference_log_dir / "train_labels.npy", labels)
    # chexpert model
    np.save(chexpert_model_log_dir / "train_logits.npy", chex_model_logits)
    np.save(chexpert_model_log_dir / "subset_mask.npy", chex_member_mask)
    np.save(chexpert_model_log_dir / "train_labels.npy", labels)
    # mimic model
    np.save(mimic_model_log_dir / "train_logits.npy", mimic_model_logits)
    np.save(mimic_model_log_dir / "subset_mask.npy", mimic_member_mask)
    np.save(mimic_model_log_dir / "train_labels.npy", labels)
    print("Done.")


if __name__ == "__main__":
    app.run(main)
