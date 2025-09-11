from pathlib import Path

import numpy as np
import pandas as pd # type: ignore
from typing import Tuple

from src.data_utils.datasets import (CXRDataset, EMBEDataset,
                                           FairVisionDataset,
                                           FitzPatLabelSetting,
                                           FitzpatrickDataset,
                                           PTBXLDataset,
                                           MIMICIVEDDataset,
                                           BaseDataset)


def get_dataset(
    dataset_name: str,
    img_size: Tuple[int, int],
    csv_root: Path,
    data_root: Path,
    save_root: Path,
    get_numpy: bool = False,
    load_from_disk: bool = False,
    overwrite_existing: bool = False,
) -> Tuple[BaseDataset, BaseDataset]: 
    """
    Convenience function to retrieve a dataset by name. Raises a ValueError if the dataset is unknown.
        Args:
            dataset_name: str, name of the dataset
            img_size: int, size of the images
            csv_root: Path, path to the csv files
            data_root: Path, path to the data files
            save_root: Path, path to the save root
            get_numpy: bool, whether to return numpy arrays
            load_from_disk: bool, whether to load from disk
            overwrite_existing: bool, whether to overwrite existing files
        Returns:
            train_dataset: either tuple of numpy arrays or a BaseDataset object
    """
    assert len(img_size) == 2, "img_size must be a tuple of (height, width)"
    if dataset_name == "chexpert":
        cxp_df_path = (
            csv_root / "chexpert_train.csv"
        )
        cxp_df = pd.read_csv(cxp_df_path)
        cxp_test_df = pd.read_csv(csv_root / "chexpert_test.csv")
        train_dataset = CXRDataset(
            df=cxp_df,
            img_path=data_root,
            name="chexpert",
            img_size=img_size,
            split="train",
            save_root=save_root,
        )
        test_dataset = CXRDataset(
            df=cxp_test_df,
            img_path=data_root,
            name="chexpert",
            img_size=img_size,
            split="test",
            save_root=save_root,
        )
    elif dataset_name == "mimic":
        csv_path = (
            csv_root / "mimic_train.csv"
        )
        mimic_df = pd.read_csv(csv_path)
        train_dataset = CXRDataset(
            df=mimic_df,
            img_path=data_root / "files",
            name="mimic",
            img_size=img_size,
            split="train",
            save_root=save_root,
        )
        # use CheXpert test set since labels are verified by experts (consensus voting of three board-certified radiologists)
        cxp_test_df = pd.read_csv(csv_root / "chexpert_test.csv")
        test_dataset = CXRDataset(
            df=cxp_test_df,
            img_path=data_root,
            name="chexpert",
            img_size=img_size,
            split="test",
            save_root=save_root,
        )
    elif dataset_name == "fitzpatrick":
        fitz_df = pd.read_csv(f"{csv_root}/fitzpatrick17k.csv")
        fitz_df = fitz_df[fitz_df["download_success"] == True]
        fitz_train_df = fitz_df[fitz_df["split"] == "train"]
        fitz_test_df = fitz_df[fitz_df["split"] == "test"]
        train_dataset = FitzpatrickDataset(
            df=fitz_train_df,
            img_path=data_root,
            name="fitzpatrick",
            img_size=img_size,
            split="train",
            save_root=save_root,
            label_setting=FitzPatLabelSetting.COARSE,
        )
        test_dataset = FitzpatrickDataset(
            df=fitz_test_df,
            img_path=data_root,
            name="fitzpatrick",
            img_size=img_size,
            split="test",
            save_root=save_root,
            label_setting=FitzPatLabelSetting.COARSE,
        )
    elif dataset_name == "embed":
        embed_df = (
            pd.read_csv("./data/csv/embed.csv", low_memory=False)
        )
        train_df, test_df = (
            embed_df[embed_df.split == "train"],
            embed_df[embed_df.split == "test"],
        )
        train_dataset = EMBEDataset(
            df=train_df,
            img_path=data_root,
            name="embed",
            img_size=img_size,
            split="train",
            save_root=save_root,
        )
        test_dataset = EMBEDataset(
            df=test_df,
            img_path=data_root,
            name="embed",
            img_size=img_size,
            split="test",
            save_root=save_root,
        )
    elif dataset_name == "fairvision":
        fairvision_df = pd.read_csv(csv_root / "fairvision.csv")
        train_df, test_df = (
            fairvision_df[fairvision_df.use != "test"],
            fairvision_df[fairvision_df.use == "test"],
        )
        train_dataset = FairVisionDataset(
            df=train_df,
            img_path=data_root,
            name="fairvision",
            img_size=img_size,
            split="train",
            save_root=save_root,
        )
        test_dataset = FairVisionDataset(
            df=test_df,
            img_path=data_root,
            name="fairvision",
            img_size=img_size,
            split="test",
            save_root=save_root,
        )
    elif dataset_name == "ptb-xl":
        train_df = pd.read_csv(csv_root / "ptb-xl_train.csv")
        test_df = pd.read_csv(csv_root / "ptb-xl_test.csv")
        train_dataset = PTBXLDataset(
            df=train_df,
            img_path=data_root,
            name="ptb-xl",
            split="train",
            save_root=save_root,
            load_base_file=not load_from_disk,
        )
        test_dataset = PTBXLDataset(
            df=test_df,
            img_path=data_root,
            name="ptb-xl",
            split="test",
            save_root=save_root,
            load_base_file=not load_from_disk,
        )
    elif dataset_name == "mimic-iv-ed":
        train_df = pd.read_csv(csv_root / "mimic-iv-ed_train.csv")
        test_df = pd.read_csv(csv_root / "mimic-iv-ed_test.csv")
        train_dataset = MIMICIVEDDataset(
            dataframe=train_df
        )
        test_dataset = MIMICIVEDDataset(
            dataframe=test_df
        )
        # the MIMIC-IV-ED Dataset does not support iterative loading of data, so we return the full data directly
        if get_numpy:
            train_dataset.inputs = train_dataset.__get_all_inputs__()
            train_dataset.targets = train_dataset.__get_all_targets__()
            test_dataset.inputs = test_dataset.__get_all_inputs__()
            test_dataset.targets = test_dataset.__get_all_targets__()
            return train_dataset, test_dataset
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")
    if get_numpy:
        # extract np.arrays
        print("... caching data into memory")
        train_dataset.cache_numpy(
            load_from_disk=load_from_disk, overwrite_existing=overwrite_existing
        )
        test_dataset.cache_numpy(
            load_from_disk=load_from_disk, overwrite_existing=overwrite_existing
        )
        # add channel dimension if necessary
        if len(train_dataset.inputs.shape) == 3 and train_dataset.inputs.shape[1]==train_dataset.inputs.shape[2]:
            train_dataset.inputs = np.expand_dims(train_dataset.inputs, axis=-1)
            test_dataset.inputs = np.expand_dims(test_dataset.inputs, axis=-1)
        # convert to channels last if necessary
        if not train_dataset.inputs.shape[-1] in [1, 3, 12]:
            train_dataset.inputs = np.transpose(train_dataset.inputs, (0, 2, 3, 1))
            test_dataset.inputs = np.transpose(test_dataset.inputs, (0, 2, 3, 1))
        # shape checks
        assert train_dataset.inputs.shape[0] == train_dataset.targets.shape[0]
        assert test_dataset.inputs.shape[0] == test_dataset.targets.shape[0]
        assert train_dataset.inputs.shape[1:] == test_dataset.inputs.shape[1:]
    return train_dataset, test_dataset
