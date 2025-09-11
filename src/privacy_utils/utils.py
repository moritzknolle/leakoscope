from enum import Enum
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt #type: ignore
import numpy as np
import pandas as pd #type: ignore
import scipy #type: ignore
from sklearn.metrics import roc_curve #type: ignore
from tqdm import tqdm #type: ignore

from ..data_utils.datasets import BaseDataset


def convert_patientmask_to_recordmask(
    patient_masks: np.ndarray,
    patient_ids: pd.Series,
    dataset: BaseDataset,
    patient_id_col: str,
):
    """
    Convert patient-level subset masks to record-level masks.
    Args:
        patient_masks: np.ndarray, shape=(N_runs, N_patients), boolean masks for patients
        patient_ids: pd.Series, patient ids corresponding to the patient masks
        dataset: BaseDataset, dataset containing the records
        patient_id_col: str, column name in the dataset dataframe that contains patient ids
    Returns:
        record_masks: np.ndarray, shape=(N_runs, N_records), boolean masks for records
    """
    assert patient_masks.shape[1] == len(
        patient_ids
    ), f"Patient masks shape {patient_masks.shape} does not match patient ids shape {len(patient_ids)}"
    # convert patient-level masks to record-level masks
    n_runs, n_patients = patient_masks.shape
    record_patient_ids = dataset.dataframe[patient_id_col]
    record_masks = np.zeros((n_runs, len(dataset)), dtype=bool)
    for i in tqdm(range(n_runs)):
        # selected patients
        selected_patients = patient_ids[patient_masks[i]]
        # selected records
        selected_records_mask = record_patient_ids.isin(selected_patients)
        selected_records_idcs = np.nonzero(selected_records_mask)[0]
        # set record masks
        record_masks[i, selected_records_idcs] = True
    return record_masks


class CurveAverage(Enum):
    vertical = 1
    horizontal = 2
    treshold = 3


def get_curve_summary_stats(
    x_data: List[np.ndarray],
    y_data: List[np.ndarray],
    thresholds: List[np.ndarray],
    mode: CurveAverage,
    n_samples: int = 10_000,
    log_samples: bool = False,
):
    """
    Compute the mean and standard deviation of a collection of ROC curves.
    Args:
        x_data: List[np.ndarray], list of x values for each ROC curve
        y_data: List[np.ndarray], list of y values for each ROC curve
        thresholds: List[np.ndarray], list of thresholds for each ROC curve
        mode: CurveAverage, averaging mode. Either CurveAverage.vertical, CurveAverage.horizontal or CurveAverage.treshold
        n_samples: int, number of samples for interpolation
        log_samples: bool, whether to sample in log space
    Returns:
        x: np.ndarray, x values for the average curve
        mean_y: np.ndarray, mean y values for the average curve
        std_y: np.ndarray, standard deviation of y values for the average curve
    """
    if mode == CurveAverage.horizontal or mode == CurveAverage.treshold:
        raise NotImplementedError("Horizontal averaging is not implemented yet.")
    elif mode == CurveAverage.vertical:
        x = (
            np.linspace(0, 1, n_samples)
            if not log_samples
            else np.logspace(-5, 0, n_samples)
        )
        x = np.flip(x)
        interp_y = []
        for x_dat, y_dat, thresh in zip(x_data, y_data, thresholds):
            interp_function = scipy.interpolate.interp1d(x_dat, y_dat)
            interp_y.append(interp_function(x))
        interp_y = np.stack(interp_y, axis=0)
        mean_y = np.mean(
            interp_y,
            axis=0,
        )
        std_y = np.std(
            interp_y,
            axis=0,
        )
        return x, mean_y, std_y


def confidence_roc_plot(
    preds: np.ndarray,
    targets: np.ndarray,
    color: str,
    averaging_mode: CurveAverage = CurveAverage.vertical,
    log_scale: bool = False,
    n_samples=10_000,
    name: str = "",
    draw_random: bool = True,
    plot_all_curves: bool = False,
    lim_tpr: float = 1e-4,
    lim_fpr: float = 1e-5,
    draw_axes_labels: bool = True,
    draw_legend: bool = False,
    ax: plt.Axes = None,
    save_result: bool = False,
    prefix_str: str = "",
    out_path: str = "./",
    draw_intersect: bool = False,
):
    """
    Plot the average ROC curve with confidence intervals.
    Args:
        preds: np.ndarray, predicted probabilities of shape=(M, N), where M is the number of repetitions and N is the number of samples
        targets: np.ndarray, target labels of shape=(M, N), where M is the number of repetitions and N is the number of samples
        color: str, color of the ROC curve
        averaging_mode: CurveAverage, averaging mode. Either CurveAverage.vertical or CurveAverage.horizontal
        log_scale: bool, whether to plot the ROC curve in log scale
        n_samples: int, number of samples for interpolation
        name: str, name of the curve
        draw_random: bool, whether to draw the random classifier curve
        plot_all_curves: bool, whether to plot all individual ROC curves
        lim_tpr: float, lower limit of the true positive rate
        lim_fpr: float, lower limit of the false positive rate
        draw_axes_labels: bool, whether to draw axes labels
        draw_legend: bool, whether to draw the legend
        ax: plt.Axes, axis to plot the ROC curve
        save_result: bool, whether to save the results
        prefix_str: str, prefix for the saved results
        out_path: str, path to save the results
        draw_intersect: bool, whether to draw the intersection of the ROC curve with the y-axis (typically TPR@~1/NFPR)

    Returns:
        None
    """
    assert (
        preds.shape == targets.shape
    ), f"preds.shape={preds.shape}, targets.shape={targets.shape}"
    assert len(preds.shape) == 2, f"preds.shape={preds.shape}"
    n = preds.shape[1]
    if ax is None:
        ax = plt.gca()
    fpr_list, tpr_list, thresholds_list = [], [], []
    for i in range(preds.shape[0]):
        fpr, tpr, thresholds = roc_curve(targets[i], preds[i])
        fpr_list.append(fpr)
        tpr_list.append(tpr)
        thresholds_list.append(thresholds)
        ax.plot(fpr, tpr, color=color, alpha=0.05) if plot_all_curves else 0

    fpr, mean_tpr, std_tpr = get_curve_summary_stats(
        x_data=fpr_list,
        y_data=tpr_list,
        thresholds=thresholds_list,
        mode=averaging_mode,
        n_samples=n_samples,
        log_samples=log_scale,
    )
    # plot the average roc curve with confidence intervals \pm 1 std
    ax.plot(fpr, mean_tpr, color=color, linewidth=1.5, label=f"{name}", zorder=1)
    ax.fill_between(
        fpr, mean_tpr - std_tpr, mean_tpr + std_tpr, color=color, alpha=0.15
    )
    if draw_intersect:
        tpr_index = np.abs(fpr - (1 / n)).argmin()
        tpr_indicator = ax.scatter(
            lim_fpr,
            mean_tpr[tpr_index],
            color=color,
            marker="x",
            s=15,
            alpha=0.75,
            zorder=3,
        )
        tpr_indicator.set_clip_on(False)  # draw tpr indicator on top of the plot
    if log_scale:
        ax.set_xscale("log")
        ax.set_yscale("log")
        plt.xlim([lim_fpr, 1])
        plt.ylim([lim_tpr, 1])
    else:
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
    if draw_random:
        ax.plot(
            [0, 1],
            [0, 1],
            color="black",
            linestyle="--",
            linewidth=1.5,
            alpha=0.5,
            label="Random",
        )
    if draw_legend:
        ax.legend(
            loc="lower right",
            framealpha=0.9,
        )
    if draw_axes_labels:
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
    ax.spines[["right", "top"]].set_visible(False)
    if save_result:
        log_str = "log" if log_scale else "lin"
        out_path = Path(out_path)
        out_path.mkdir(parents=True, exist_ok=True)
        np.save(f"{out_path}/{prefix_str}_fpr_{log_str}.npy", fpr)
        np.save(f"{out_path}/{prefix_str}_tpr_mean_{log_str}.npy", mean_tpr)
        np.save(f"{out_path}/{prefix_str}_tpr_std_{log_str}.npy", std_tpr)
