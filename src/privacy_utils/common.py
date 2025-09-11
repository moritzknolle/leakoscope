from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy


def load_score(l_dir: Path, logit_transform_func: Callable, verbose: bool = False):
    """
    Loads and transforms MIA scores for a given log directory containing the results (predictions) of a model trained on a random training data subset.

    Args:
        l_dir: The path to the log directory.
    Returns:
        score: The logit transformed loss scores.
        subset_mask: A boolean mask indicating the subset of the data used for training.
        train_labels: The training labels.
    """
    try:
        logits = np.load(l_dir / "train_logits.npy")
        train_labels = np.load(l_dir / "train_labels.npy")
        subset_mask = np.load(l_dir / "subset_mask.npy")
        if verbose:
            print(
                f"Loaded logits of shape {logits.shape}, labels of shape {train_labels.shape}, and subset mask of shape {subset_mask.shape} from {l_dir}"
            )
        if not np.count_nonzero(np.isnan(logits)) == 0:
            raise Exception(f"Found NaNs logits in {l_dir}")
    except Exception as e:
        print(f"Could not load logits from {l_dir}, error message:\n {e}")
        return None
    if not logits.dtype == np.float32:
        print(
            f"WARNING: Logits are not of type float32, but {logits.dtype}. This might lead to numerical instabilities"
        )
        logits = logits.astype(np.float32)
    score = logit_transform_func(logits=logits, labels=train_labels)
    return score, subset_mask, train_labels


def load_score_disjoint(l_dir: Path, logit_transform_func: Callable):
    """
    Loads LiRA scores for a given log directory containing the results (predictions) of a model trained on data disjoint from the training data.

    Args:
        l_dir: The path to the log directory.
        logit_transform_func: The logit transform function to apply to the scores.
    Returns:
        score: The logit transformed loss scores.
        subset_mask: A boolean mask indicating the subset of the data used for training.
        train_labels: The training labels.
    """
    try:
        logits = np.load(l_dir / "eval_logits.npy")
        labels = np.load(l_dir / "eval_labels.npy")
    except Exception as e:
        print(f"Could not load logits from {l_dir}, error: {e}")
        return None
    score = logit_transform_func(logits=logits, labels=labels)
    return score, np.zeros(logits.shape[0], dtype=bool), labels


def aggregate_by_patient(aucs: np.ndarray, patient_ids: pd.Series):
    """
    Aggregates AUCs by patient ID to obtain a single MIA AUC per patient.
    Args:
        aucs: numpy array of shape (n_samples,) containing the record-level MIA AUC scores.
        patient_ids: numpy array of shape (n_samples,) containing the patient ID corresponding to each image.

    Returns:
        patient_auc: dictionary containing the aggregated patient-level MIA AUCs accessible through patient ID.
    """
    assert aucs.shape[0] == len(
        patient_ids
    ), f"Shapes do not match: {aucs.shape} vs {len(patient_ids)}"
    assert isinstance(
        patient_ids, pd.Series
    ), f"Expected patient_ids to be a pandas Series, but got {type(patient_ids)}"
    n_patients = patient_ids.nunique()
    auc_df = pd.DataFrame.from_dict({"aucs": aucs, "patient_id": patient_ids.values})
    aggregated = (
        auc_df.groupby("patient_id")["aucs"]
        .agg(mean="mean", std="std", max="max", count="count")
        .reset_index()
    )
    assert (
        len(aggregated) == n_patients
    ), f"Number of patients do not match dictionary size: {len(aggregated)} vs {n_patients}"
    return aggregated


def roc_analysis_from_gaussian_samplestats(
    mean_in: np.ndarray,
    mean_out: np.ndarray,
    std_in: np.ndarray,
    std_out: np.ndarray,
    N_in: int,
    N_out: int,
    resolution: int = 10_000,
    log_scale: bool = False,
    compute_roc_curves: bool = False,
    eps: float = 1e-10,
    verbose:bool=True,
):
    # average over augmentations if present
    if len(mean_in.shape) == 2:
        mean_in = np.mean(mean_in, axis=1)
        mean_out = np.mean(mean_out, axis=1)
        std_in = np.mean(std_in, axis=1)
        std_out = np.mean(std_out, axis=1)
    std_in += eps  # add small epsilon to avoid division by zero
    a = (mean_in - mean_out) / std_in
    b = std_out / std_in
    a = a[None, :]
    b = b[None, :]
    if log_scale:
        fprs = np.logspace(-5, 0, resolution)[:, None]
    else:
        fprs = np.linspace(0, 1.0, resolution)[:, None]
    if len(a.shape) == 3:
        fprs = fprs[..., None]
    if compute_roc_curves:
        tprs = scipy.stats.norm.cdf(
            a + b * scipy.stats.norm.ppf(fprs)
        )  # analytical solution for the binormal ROC curve
        fprs = np.repeat(
            fprs, tprs.shape[1], axis=1
        ).squeeze()  # repeat fprs to match tprs shape (resolution, n_samples)
    else:
        tprs, fprs = None, None
    aucs = scipy.stats.norm.cdf(
        (mean_in - mean_out) / np.sqrt(std_out**2 + std_in**2)
    )  # analytical solution for the the AUC
    aucs = np.clip(aucs, 0.5, 1-eps) # subtract small epsilon to improve numerical stability
    q_0 = aucs / (2 - aucs)
    q_1 = 2 * aucs**2 / (1 + aucs)
    # standard error of the AUC in the binormal setting (Hanley and McNeil, 1982)
    se_aucs = np.sqrt(
        (
            aucs * (1 - aucs)
            + (N_in - 1) * (q_0 - aucs**2)
            + (N_out - 1) * (q_1 - aucs**2)
        )
        / (N_in * N_out)
    )
    if verbose:
        print(
            f"... Standard Error Summary: SE(AUC) min: {np.min(se_aucs):.3g}, max={np.max(se_aucs):.3g}"
        )
        print(
            f"... AUC Distribution Summary: mu={np.mean(aucs):.3g}, std={np.std(aucs):.3g}, min={np.min(aucs):.3g}, max={np.max(aucs):.3g}, 90%ile={np.percentile(aucs, 90):.3g}, 95%ile={np.percentile(aucs, 95):.3g}, 99%ile={np.percentile(aucs, 99):.3g}"
        )
    if np.count_nonzero(np.isnan(aucs)) != 0:
        problem_records = np.where(np.isnan(aucs))[0]
        print(f"... overview of records (and their values) causing numerical instabilities for AUC computation:")
        print(f"mean_in: {mean_in[problem_records]}")
        print(f"mean_out: {mean_out[problem_records]}")
        print(f"std_in: {std_in[problem_records]}")
        print(f"std_out: {std_out[problem_records]}")
    if np.count_nonzero(np.isnan(se_aucs)) != 0:
        problem_records = np.where(np.isnan(se_aucs))[0]
        print(
            f"... overview of records (and their values) causing numerical instabilities for SE(AUC) computation:"
        )
        print(f"aucs: {aucs[problem_records]}")
    assert (
        np.count_nonzero(np.isnan(aucs)) == 0
    ), f"Found {np.count_nonzero(np.isnan(aucs))} NaN in AUCs: {aucs}, mean={np.nanmean(aucs)}, std={np.nanstd(aucs)}, min={np.nanmin(aucs)}, max={np.nanmax(aucs)}"
    return fprs, tprs, aucs, se_aucs


def record_MIA_ROC_analysis(
    scores: np.ndarray,
    masks: np.ndarray,
    resolution: int = 10_000,
    log_scale: bool = False,
    compute_roc_curves: bool = False,
):
    """
    Performs MIA ROC analyisis at the record-level for each record in the dataset.
    Uses the analytical solution for the ROC curve in the binormal case since the scores are normally distributed.
    Requires scores and masks from many target models trained on random subsets.
    Args:
        scores: A numpy array of shape (n_samples, n_runs) containing the test scores (logit transformed model confidences).
        masks: A numpy array of shape (n_samples, n_runs) containing the subset mask.
        resolution: The resolution for the ROC curve.
        log_scale: Whether the ROC curve will be plotted on a log-scale. If true, this will use a logspace for the FPR.
        compute_roc_curves: Whether to compute the ROC curves.
    Returns:
        fprs: A numpy array of shape (resolution, n_samples) containing the false positive rates.
        tprs: A numpy array of shape (resolution, n_samples) containing the true positive rates.
        aucs: A numpy array of shape (n_samples,) containing the area under the ROC curve.
        se_aucs: A numpy array of shape (n_samples,) containing the standard error of the AUC scores.

    """
    in_scores, out_scores = preprocess_scores(scores, masks)

    # compute ROC curves analytically
    mean_in = np.mean(in_scores, axis=0)
    mean_out = np.mean(out_scores, axis=0)
    std_in = np.std(in_scores, axis=0)
    std_out = np.std(out_scores, axis=0)
    print(
        f"... using N_in={in_scores.shape[0]} and N_out={out_scores.shape[0]} samples to estimate record-level sampling distributions"
    )
    fprs, tprs, aucs, se_aucs = roc_analysis_from_gaussian_samplestats(
        mean_in=mean_in,
        mean_out=mean_out,
        std_in=std_in,
        std_out=std_out,
        N_in=in_scores.shape[0],
        N_out=out_scores.shape[0],
        resolution=resolution,
        log_scale=log_scale,
        compute_roc_curves=compute_roc_curves,
    )
    return fprs, tprs, aucs, se_aucs