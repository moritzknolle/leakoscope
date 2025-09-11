import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np

import wandb


class RetrainLogger:
    """
    A helper class for logging results and metrics of leave-many-out re-training runs.

    Args:
        wandb_run_id (int): The ID of the Weights & Biases run.
        logdir (Path): The directory where the logs will be saved.
        train_idcs (np.ndarray): The indices of the training data.

    Attributes:
        start_time (str): The start time of the retraining process.
        mac_address (str): The MAC address of the machine.
        config (wandb.config): The configuration settings.
        log_dir (Path): The directory where the logs will be saved.
        train_idcs (np.ndarray): The indices of the training data.

    Methods:
        save_metadata: Saves the metadata to a JSON file.
        log: Logs raw predictions (logits) of the training and test data.
        get_mac_address: Retrieves the MAC address of the machine.
    """

    def __init__(
        self,
        logdir: Path,
        subset_idcs: np.ndarray,
        subset_mask: np.ndarray,
    ):
        self.start_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        self.mac_address = self.get_mac_address()
        self.subset_idcs = subset_idcs
        self.subset_mask = subset_mask
        self.base_dir = logdir
        if not self.base_dir.exists():
            self.base_dir.mkdir(parents=True, exist_ok=True)

    def save_metadata(self, config:dict, train_metrics: dict, test_metrics: dict):
        """
        Saves the metadata to a JSON file.

        Args:
            metrics (dict): The metrics to be saved.

        Returns:
            bool: True if the metadata is successfully saved, False otherwise.
        """
        if not isinstance(config, dict):
            raise ValueError(f"Expected config as dictionary, found: {type(config)}")
        self.config = config
        train_metrics = {k: float(v) for k, v in train_metrics.items()}
        test_metrics = {k: float(v) for k, v in test_metrics.items()}
        try:
            info_dict = info_template
            info_dict["wandb_run_id"] = self.wandb_run_id
            info_dict["start_time"] = self.start_time
            info_dict["end_time"] = self.end_time
            info_dict["mac_address"] = self.mac_address
            info_dict["wandb_config"] = self.config
            info_dict["train_metrics"] = train_metrics
            info_dict["test_metrics"] = test_metrics
            with open(self.log_dir / "info.json", "w") as f:
                json.dump(info_dict, f, indent=4)
        except Exception as e:
            print(e)
            return False
        return True

    def maybe_create_log_dir(self):
        if wandb.run is None:
            raise ValueError(
                "You must call wandb.init() before starting a RetrainExperiment()"
            )
        self.wandb_run_id = wandb.run.id
        self.log_dir = self.base_dir / str(self.wandb_run_id)
        if not self.log_dir.exists():
            print(f"... creating log_dir at {self.log_dir}")
            self.log_dir.mkdir(parents=True)

    def log(
        self,
        config:dict,
        train_logits: np.ndarray,
        train_labels: np.ndarray,
        test_logits: np.ndarray,
        test_labels: np.ndarray,
        train_metrics: dict,
        test_metrics: dict,
    ):
        """
        Logs the raw predictions on the training and test data.

        Args:
            train_logits (np.ndarray): The logits of the training data.
            train_labels (np.ndarray): The labels of the training data.
            test_logits (np.ndarray): The logits of the test data.
            test_labels (np.ndarray): The labels of the test data.
            train_metrics (dict): The metrics of the training data.
            test_metrics (dict): The metrics of the test data.

        Returns:
            bool: True if the data is successfully logged, False otherwise.
        """
        self.maybe_create_log_dir()
        self.end_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        assert len(train_logits) == len(
            train_labels
        ), f"Careful! Shape mismatch, {train_logits.shape} {train_labels.shape}"
        try:
            np.save(self.log_dir / "train_logits.npy", train_logits)
            np.save(self.log_dir / "train_labels.npy", train_labels)
            np.save(self.log_dir / "test_logits.npy", test_logits)
            np.save(self.log_dir / "test_labels.npy", test_labels)
            np.save(self.log_dir / "subset_idcs.npy", self.subset_idcs)
            np.save(self.log_dir / "subset_mask.npy", self.subset_mask)
        except Exception as e:
            print(e)
            return False
        metadata_sucess = self.save_metadata(
            config=config,
            train_metrics=train_metrics,
            test_metrics=test_metrics
        )
        assert metadata_sucess, "Failed to save metadata"
        return True

    def get_mac_address(self):
        """
        Retrieves the MAC address of the machine.

        Returns:
            str
        """
        try:
            mac_id = hex(uuid.getnode())
        except Exception as e:
            print(e)
            mac_id = "0000000000"
        return mac_id


info_template = {
    "wandb_run_id": 0,
    "start_time": "",
    "end_time": "",
    "mac_address": "",
    "wandb_config": {},
    "train_metrics": {},
    "test_metrics": {},
}
