import hashlib
import json
import multiprocessing as mp
from enum import Enum
from pathlib import Path
from typing import Tuple

import keras # type: ignore
import numpy as np
import pandas as pd # type: ignore
from PIL import Image # type: ignore
from tqdm import tqdm # type: ignore

from .constants import (
    CXR_LABELS,
    FITZPATRICK_LABELS_COARSE,
    FITZPATRICK_LABELS_FINE,
    CXRLabelStrategy,
    ECG_LABELS,
)


class BaseDataset:
    """
    Basic Interface common to all datasets. This class is not meant to be used directly but to be subclassed by specific datasets.
    To use this class, you need to implement the __getinput___(), __gettarget___(),  and __len__() methods in the subclass.

    Args:

        df (pd.DataFrame): The dataframe containing the dataset
        img_path (Path): The path to the images
        name (str): The name of the dataset
        num_classes (int): The number of classes in the dataset
        in_channels (int): The number of channels in the images
        img_size (int): The size of the images
        split (str): The split of the dataset (train, val, test)
        save_root (Path): The path to save the dataset
        n_threads (int): The number of threads to use for loading the dataset


    """

    def __init__(
        self,
        df: pd.DataFrame,
        img_path: Path,
        name: str,
        num_classes: int,
        in_channels: int,
        img_size: Tuple[int, int],
        split: str,
        save_root: Path,
        n_threads: int = 32,
    ):
        print(f"... Creating {name} dataset")
        super().__init__()
        self.dataframe = df
        self.img_path = img_path
        self.name = name
        self.num_classes = num_classes
        if not isinstance(save_root, Path):
            save_root = Path(save_root)
        self.save_root = save_root / name
        self.split = split
        self.in_channels = in_channels
        self.img_size = img_size
        self.filename = f"{name}_{split}_{self.img_size[0]}x{self.img_size[1]}"
        self.n_threads = n_threads
        self.save_root.mkdir(parents=True, exist_ok=True)
        if not isinstance(self.img_path, Path):
            self.img_path = Path(self.img_path)
        if not self.img_path.exists():
            print(f"Warning: Image path {self.img_path} does not exist")

    def __getinput__(self, index: int):
        """Returns input corresponding to index i in the dataset"""
        pass

    def __gettarget__(self, index: int):
        """Returns target corresponding to index i in the dataset"""
        pass

    def __getitem__(self, index: int) -> tuple:
        inp = self.__getinput__(index)
        target = self.__gettarget__(index)
        return inp, target

    def _load_from_scratch(self):
        """
        Load the dataset into memory by iterating through it.
        """
        success = False
        convenience_target_fn_exists = callable(
            getattr(self, "__get_all_targets__", None)
        )
        with mp.Pool(self.n_threads) as pool:
            inputs = list(
                tqdm(
                    pool.imap(
                        self.__getinput__, range(self.__len__()), chunksize=32
                    ),
                    desc="Loading inputs",
                    total=self.__len__(),
                )
            )
            if not convenience_target_fn_exists:
                targets = list(
                    tqdm(
                        pool.imap(
                            self.__gettarget__, range(self.__len__()), chunksize=32
                        ),
                        desc="Loading targets",
                        total=self.__len__(),
                    )
                )
                self.targets = np.array(targets, dtype=np.float32)
            else:
                self.targets = self.__get_all_targets__()
            self.inputs = np.array(inputs, dtype=np.float32)
        success = True
        
        return success

    def __len__(self):
        """Returns the length of the dataset"""
        pass

    def _write_metadata(self):
        if self.inputs is None or self.targets is None:
            raise ValueError(
                "Cannot save dataset to disk, inputs and targets are None. Load the dataset first."
            )
        base_file_name = self.save_root / f"{self.filename}"
        df_hash = hashlib.sha1(
            pd.util.hash_pandas_object(self.dataframe).values
        ).hexdigest()
        # Write metadata JSON file
        metadata = {
            "input_shape": self.inputs.shape,
            "target_shape": self.targets.shape,
            "df_hash": df_hash,
        }
        metadata_file = f"{base_file_name}.json"
        with open(metadata_file, "w") as file:
            json.dump(metadata, file)

    def _write_arrays_to_disk(self) -> None:
        """Write a loaded dataset to disk (saves two .npz files and a metadata file)."""
        self._write_metadata()
        base_file_name = self.save_root / f"{self.filename}"
        # write npy files for input and target arrays
        np.save(f"{base_file_name}_inputs.npy", self.inputs)
        np.save(f"{base_file_name}_targets.npy", self.targets)

    def _load_from_file(self) -> bool:
        """Loads dataset from disk (.npz files and the metadata file)."""
        success = False
        base_file_name = self.save_root / f"{self.filename}"
        metadata_file = Path(f"{base_file_name}.json")
        input_file = Path(f"{base_file_name}_inputs.npy")
        target_file = Path(f"{base_file_name}_targets.npy")
        print(f"... looking for dataset at: {self.save_root}")
        print(f"... checking metadata file: {metadata_file}")
        if input_file.exists() and target_file.exists() and metadata_file.is_file():
            # read metadata file
            with open(metadata_file, "r") as file:
                metadata = json.load(file)
                input_shape = tuple(metadata["input_shape"])
                target_shape = tuple(metadata["target_shape"])
            # Read .npy files
            self.inputs = np.load(input_file)
            self.targets = np.load(target_file)
            assert (
                input_shape == self.inputs.shape
            ), f"Input shape mismatch: {input_shape} vs. {self.inputs.shape}"
            assert (
                target_shape == self.targets.shape
            ), f"Target shape mismatch: {target_shape} vs. {self.targets.shape}"
            # check if dataframe has changed since the npz files were created
            current_df_hash = hashlib.sha1(
                pd.util.hash_pandas_object(self.dataframe).values
            ).hexdigest()
            if metadata["df_hash"] != current_df_hash:
                raise ValueError(
                    f"Careful, hash mismatch! Underlying dataframe has changed since npz files were created: {metadata['df_hash']} vs. {current_df_hash}"
                )
            success = True
            print("... succesfully loaded dataset from disk")
        return success

    def cache_numpy(
        self, load_from_disk: bool = True, overwrite_existing: bool = False
    ) -> None:
        """Return the input and target arrays."""
        if load_from_disk:
            success = self._load_from_file()
            if not success:
                print("... loading dataset from disk failed, loading from scratch")
                success = self._load_from_scratch()
        else:
            success = self._load_from_scratch()
        if overwrite_existing and success:
            self._write_arrays_to_disk()
            print("... sucessfully wrote dataset to disk")
        assert success, "loading dataset failed"


class CXRDataset(BaseDataset):
    def __init__(
        self,
        df: pd.DataFrame,
        img_path: Path,
        name: str,
        img_size: Tuple[int, int],
        split: str,
        save_root: Path,
        label_strategy: CXRLabelStrategy = CXRLabelStrategy.UNCERTAIN_ONES,
        frontal_only=False,
    ):

        self.label_strategy = label_strategy
        if frontal_only:
            if "Frontal/Lateral" in df:
                df = df[df["Frontal/Lateral"] == "Frontal"]
            elif "ViewPosition" in df:
                df = df[df.ViewPosition.isin(["AP", "PA"])]
            else:
                raise ValueError(
                    f"Dataframe does not have a 'Frontal/Lateral' or 'ViewPosition' column. Availabel are: {df.columns}"
                )
        super().__init__(
            df=df,
            img_path=img_path,
            name=name,
            num_classes=14,
            in_channels=1,
            img_size=img_size,
            split=split,
            save_root=save_root,
        )
        self.dataframe = self.dataframe.fillna(0)
        if "Path" in self.dataframe.columns:
            self.dataframe.rename(columns={"Path": "path"}, inplace=True)
        assert "path" in self.dataframe.columns, "Dataframe must have a 'path' column"

    def __getinput__(self, index: int):
        """Returns input corresponding to index i in the dataset"""

        path_str = self.dataframe["path"].iloc[index]
        img_path = self.img_path / path_str
        assert img_path.exists(), f"Image path {img_path} does not exist"
        # load in image
        img = Image.open(img_path).resize(self.img_size)
        img = np.array(img, dtype=np.float32)
        img = img / 127.5 - 1
        img = np.expand_dims(img, axis=-1)
        assert img.shape == (
            self.img_size[0],
            self.img_size[1],
            1,
        ), f"Image shape {img.shape} is not ({self.img_size[0]}, {self.img_size[1]}, 1)"
        return img

    def __gettarget__(self, index: int):
        """Returns target corresponding to index i in the dataset"""
        target = self.dataframe.iloc[index][CXR_LABELS]
        target = np.array(target, dtype=np.float16)
        if self.label_strategy == CXRLabelStrategy.UNCERTAIN_ONES:
            target[target == -1.0] = 1.0
        else:
            target[target == -1.0] = 0.0
        return target

    def __get_all_targets__(self):
        """Returns all targets in the dataset"""
        targets = self.dataframe[CXR_LABELS]
        targets = np.array(targets, dtype=np.float16)
        if self.label_strategy == CXRLabelStrategy.UNCERTAIN_ONES:
            targets[targets == -1.0] = 1.0
        else:
            targets[targets == -1.0] = 0.0
        return targets

    def __len__(self) -> int:
        """Returns the length of the dataset"""
        return len(self.dataframe)


class FitzPatLabelSetting(Enum):
    COARSE = 0
    FINE = 1


class FitzpatrickDataset(BaseDataset):
    def __init__(
        self,
        df: pd.DataFrame,
        img_path: Path,
        name: str,
        img_size: Tuple[int, int],
        split: str,
        save_root: Path,
        label_setting: FitzPatLabelSetting = FitzPatLabelSetting.FINE,
    ):

        self.label_setting = label_setting
        num_classes = 3 if label_setting == FitzPatLabelSetting.COARSE else 9

        super().__init__(
            df=df,
            img_path=img_path,
            name=name,
            num_classes=num_classes,
            in_channels=3,
            img_size=img_size,
            split=split,
            save_root=save_root,
        )

    def __getinput__(self, index: int):
        img_path = self.img_path / f"{self.dataframe.iloc[index]['md5hash']}.jpg"
        img = Image.open(img_path)
        img = img.resize(self.img_size)
        img = np.array(img, dtype=np.float32)
        img = img / 127.5 - 1
        return img

    def __gettarget__(self, index: int):
        if self.label_setting == FitzPatLabelSetting.COARSE:
            target = np.array(
                FITZPATRICK_LABELS_COARSE.index(
                    self.dataframe.iloc[index]["three_partition_label"]
                )
            )
        elif self.label_setting == FitzPatLabelSetting.FINE:
            target = np.array(
                FITZPATRICK_LABELS_FINE.index(
                    self.dataframe.iloc[index]["nine_partition_label"]
                )
            )
        n_classes = 3 if self.label_setting == FitzPatLabelSetting.COARSE else 9
        target = keras.utils.to_categorical(target, num_classes=n_classes)
        return target

    def __len__(self) -> int:
        return len(self.dataframe)


class EMBEDataset(BaseDataset):
    def __init__(
        self,
        df: pd.DataFrame,
        img_path: Path,
        name: str,
        img_size: Tuple[int, int],
        split: str,
        save_root: Path,
    ):
        super().__init__(
            df=df,
            img_path=img_path,
            name=name,
            num_classes=4,
            in_channels=1,
            img_size=img_size,
            split=split,
            save_root=save_root,
        )

    def __getinput__(self, index: int):
        file_name = (
            self.dataframe.iloc[index]["anon_dicom_path"].split("/")[-1][:-4] + ".png"
        )
        img_path = self.img_path / str(self.dataframe.iloc[index]["empi_anon"]) / file_name
        img = Image.open(img_path)
        img = img.resize(self.img_size[::-1])
        img = np.array(img, dtype=np.float32)
        img = img / 127.5 - 1
        return img[..., None]

    def __get_all_targets__(self):
        """Returns all targets in the dataset"""
        targets = self.dataframe["BIRADS_DENSITY"]
        targets = np.array(targets, dtype=np.float16)
        targets = keras.utils.to_categorical(targets, num_classes=self.num_classes)
        return targets

    def __gettarget__(self, index: int):
        target = self.dataframe.iloc[index]["BIRADS_DENSITY"]
        target = np.array(target, dtype=np.float16)
        target = keras.utils.to_categorical(target, num_classes=self.num_classes)
        return target

    def __len__(self) -> int:
        return len(self.dataframe)


class FairVisionDataset(BaseDataset):

    def __init__(
        self,
        df: pd.DataFrame,
        img_path: Path,
        name: str,
        img_size: Tuple[int, int],
        split: str,
        save_root: Path,
    ):

        super().__init__(
            df=df,
            img_path=img_path,
            name=name,
            num_classes=7,
            in_channels=1,
            img_size=img_size,
            split=split,
            save_root=save_root,
        )

    def __getinput__(self, index: int):
        file_name = self.dataframe.iloc[index]["slo_file_path"]
        img_path = self.img_path / file_name
        img = Image.open(img_path)
        img = img.resize(self.img_size)
        img = np.array(img, dtype=np.float32)
        img = img / 127.5 - 1
        img = np.expand_dims(img[..., 0], axis=-1)
        return img

    def __gettarget__(self, index: int):
        target = self.dataframe.iloc[index]["disease_label"]
        target = np.array(target, dtype=np.float16)
        target = keras.utils.to_categorical(target, num_classes=self.num_classes)
        return target

    def __len__(self) -> int:
        return len(self.dataframe)


class PTBXLDataset(BaseDataset):

    def __init__(
        self,
        df: pd.DataFrame,
        img_path: Path,  # path to .npy file containing the ECG signals
        name: str,
        split: str,
        save_root: Path,
        load_base_file: bool = True,
    ):

        super().__init__(
            df=df,
            img_path=img_path,
            name=name,
            num_classes=5,
            in_channels=12,
            img_size=(0,0),  # Not used for ECG, but required by the interface
            split=split,
            save_root=save_root,
        )
        if load_base_file:
            self.ecg_inputs = np.load(img_path / f"X_100_{split}.npy")
            print(f"... loaded base .npy file for PTB-XL dataset. shape={self.ecg_inputs.shape}, min= {self.ecg_inputs.min()}, max={self.ecg_inputs.max()}, mean={self.ecg_inputs.mean()}, std={self.ecg_inputs.std()}")


    def __getinput__(self, index: int):
        """Returns input corresponding to index i in the dataset"""
        return self.ecg_inputs[index]

    def __gettarget__(self, index: int):
        target = self.dataframe.iloc[index][ECG_LABELS]
        target = np.array(target, dtype=np.float16)
        return target

    def __get_all_targets__(self):
        """Returns all targets in the dataset"""
        targets = self.dataframe[ECG_LABELS]
        targets = np.array(targets, dtype=np.float16)
        return targets

    def __len__(self) -> int:
        """Returns the length of the dataset"""
        return len(self.dataframe)


class MIMICIVEDDataset:

    def __init__(
        self,
        dataframe: pd.DataFrame,
    ):
        self.dataframe = dataframe
        self.dataframe["gender_binarized"] = 0
        self.dataframe.loc[self.dataframe.gender=="F", "gender_binarized"] = 1
        self.input_variables = [
            "age",
            "gender_binarized",
            "n_ed_30d",
            "n_ed_90d",
            "n_ed_365d",
            "n_hosp_30d",
            "n_hosp_90d",
            "n_hosp_365d",
            "n_icu_30d",
            "n_icu_90d",
            "n_icu_365d",
            "triage_temperature",
            "triage_heartrate",
            "triage_resprate",
            "triage_o2sat",
            "triage_sbp",
            "triage_dbp",
            "triage_pain",
            "triage_acuity",
            "chiefcom_chest_pain",
            "chiefcom_abdominal_pain",
            "chiefcom_headache",
            "chiefcom_shortness_of_breath",
            "chiefcom_back_pain",
            "chiefcom_cough",
            "chiefcom_nausea_vomiting",
            "chiefcom_fever_chills",
            "chiefcom_syncope",
            "chiefcom_dizziness",
            "cci_MI",
            "cci_CHF",
            "cci_PVD",
            "cci_Stroke",
            "cci_Dementia",
            "cci_Pulmonary",
            "cci_Rheumatic",
            "cci_PUD",
            "cci_Liver1",
            "cci_DM1",
            "cci_DM2",
            "cci_Paralysis",
            "cci_Renal",
            "cci_Cancer1",
            "cci_Liver2",
            "cci_Cancer2",
            "cci_HIV",
            "eci_Arrhythmia",
            "eci_Valvular",
            "eci_PHTN",
            "eci_HTN1",
            "eci_HTN2",
            "eci_NeuroOther",
            "eci_Hypothyroid",
            "eci_Lymphoma",
            "eci_Coagulopathy",
            "eci_Obesity",
            "eci_WeightLoss",
            "eci_FluidsLytes",
            "eci_BloodLoss",
            "eci_Anemia",
            "eci_Alcohol",
            "eci_Drugs",
            "eci_Psychoses",
            "eci_Depression",
        ]

    def __len__(self):
        """Returns the length of the dataset"""
        return len(self.dataframe)

    def __get_all_inputs__(self):
        """Returns all inputs in the dataset"""
        inputs = self.dataframe[self.input_variables].copy()
        inputs = np.array(inputs, dtype=np.float32)
        return inputs

    def __get_all_targets__(self):
        """Returns all targets in the dataset"""
        targets = self.dataframe["outcome_hospitalization"]
        targets = np.array(targets, dtype=np.float16)
        return targets[..., None]

    def __getitem__(self, index: int) -> tuple:
        raise NotImplementedError(
            "MIMIC-IV dataset does not support __getitem__ method. Use __get_all_inputs__ and __get_all_targets__ instead."
        )
