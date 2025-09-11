"""
Likelihood Ratio Attack (LiRA) implementation for membership inference attacks.

This module implements the LiRA attack algorithm for auditing membership privacy
in machine learning models. The attack uses likelihood ratios computed from multiple
shadow models to determine if a specific data point was used in training.

References:
- Carlini et al. "Membership Inference Attacks From First Principles" (2022)
"""
import multiprocessing
from functools import partial
from pathlib import Path
from typing import Callable, List, Optional

import keras
import numpy as np
import scipy
from tqdm import tqdm

from ..train_utils.utils import LabelType


def compute_scores(
    log_dirs: List[Path],
    load_score_func: Callable,
    multi_processing: bool = True,
    threads: int = 16,
):
    """
    Load and compute LiRA attack scores from multiple reference models.
    
    This function processes results from leave-many-out retraining experiments to 
    prepare data for the Likelihood Ratio Inference Attack (LiRA). The logit 
    transform converts model confidences such that they are normally distributed.
    
    The logit transform is: logit(p) = log(p / (1 - p))
    where p is the model's predicted probability for the correct class.

    Args:
        log_dirs: List of directories containing reference model training results
        load_score_func: Function that extracts (scores, masks, labels) from each log directory
        multi_processing: Enable parallel loading across multiple processes (default True)
        threads: Number of worker threads for parallel processing (default 16)
        
    Returns:
        tuple: (logit_scores, masks, labels) where:
            - logit_scores: Logit-transformed confidence scores, shape (n_runs, n_samples)
            - masks: Boolean membership masks, shape (n_runs, n_samples)  
            - labels: Training data labels from the first valid run
            
    Note:
        Invalid directories (missing files) are automatically filtered out.
        All runs must have consistent label shapes for proper stacking.
    """
    # check all directories are valid
    log_dirs = [l_dir for l_dir in log_dirs if l_dir.is_dir()]
    if multi_processing:
        with multiprocessing.Pool(processes=threads) as pool:
            results = list(
                tqdm(
                    pool.imap(load_score_func, log_dirs),
                    desc="Loading scores",
                    total=len(log_dirs),
                )
            )
    else:
        results = [
            load_score_func(l_dir)
            for l_dir in tqdm(log_dirs, desc="Loading scores", total=len(log_dirs))
        ]
    logit_scores, masks, label_list = [], [], []
    for r in results:
        if r is not None:
            assert np.count_nonzero(np.isnan(r[0])) == 0, f"Found NaN in logits: {r[0]}"
            l_scores, mask, labels = r
            logit_scores.append(l_scores)
            masks.append(mask)
            label_list.append(labels)
            # check for label mismatch
            if len(label_list) > 2:
                assert np.allclose(
                    label_list[-1], label_list[-2]
                ), "Found label mismatch. Labels are assumed to stay constant between runs. Check for Errors ..."
    logit_score_arr = np.stack(logit_scores, axis=0)
    masks_arr = np.stack(masks, axis=0)
    return logit_score_arr, masks_arr, labels


def loss_logit_transform_multiclass(
    logits: np.ndarray,
    labels: np.ndarray,
    eps: float = 1e-45,
    is_mimic_or_chexpert: bool = False,
):
    """
    Numerically stable implementation of logit transform applied to model confidence (softmax probability output for the correct class):
    logit(p) = log(p / (1 - p))
             = log(p) - log(1 - p)
    See LiRA paper for details.
    Args:
        logits: A numpy array of shape (samples, n_augs, N_classes) representing the raw model outputs
        labels: A numpy array of shape (samples, N_classes), one-hot or multi-hot encoded.
        is_mimic_or_chexpert: A boolean indicating whether the dataset is MIMIC-CXR or CheXpert.
    Returns:
        A numpy array of shape (samples,) containing the logit transformed confidence scores.
    """
    # only for MIMIC-CXR/CheXpert, we overwrite labels with all zeros such that No Finding is indicated correctly
    # this is because there are some weird records in the dataset where all labels are zero
    if is_mimic_or_chexpert:
        zero_rows = np.all(labels == 0, axis=-1)
        zero_label_idcs = np.where(zero_rows)[0]
        labels[zero_label_idcs, 0] = 1.0
    # repeat labels such that they match the shape of the logits (over augmentations of the same record)
    if len(logits.shape) > 2:
        labels = labels[:, None, :]
        labels = np.repeat(labels, logits.shape[1], axis=1)
    assert (
        logits.shape == labels.shape
    ), f"Shapes do not match: {logits.shape} vs {labels.shape}"
    # softmax transformation
    preds = keras.activations.softmax(logits, axis=-1)
    y_true = np.sum(
        preds * labels, axis=-1
    )  # set wrong class probabilities to 0 and sum over the rest (only in the multilabel setting will there be multiple correct classes)
    y_wrong = np.sum(
        preds * np.logical_not(labels), axis=-1
    )  # set the true class probabilities to 0 and sum over the rest
    assert (
        y_wrong.shape == y_true.shape
    ), f"Shapes do not match: {y_wrong.shape} vs {y_true.shape}"
    # numerically stable logit transform
    logit_score = np.log(y_true + eps) - np.log(y_wrong + eps)
    return logit_score


def loss_logit_transform_multilabel(
    logits: np.ndarray,
    labels: np.ndarray,
    eps: float = 1e-45,
    is_mimic_or_chexpert=False,
):
    """
    Numerically stable implementation of logit transform applied to model confidence (sigmoid probability output for each of the correct classes):
    logit(p) = log(p / (1 - p))
             = log(p) - log(1 - p)
    See appendix for details.
    Args:
        logits: A numpy array of shape (samples, n_augs, N_classes)
        labels: A numpy array of shape (samples, N_classes), multi-hot encoded.
        is_mimic_or_chexpert: A boolean indicating whether the dataset is MIMIC-CXR or CheXpert.
    Returns:
        A numpy array of shape (samples, N_classes) containing the logit transformed confidence scores.
    """
    # average over augmentations if present
    if len(logits.shape) > 2:
        logits = np.mean(logits, axis=1)
    assert (
        logits.shape == labels.shape
    ), f"Shapes do not match: {logits.shape} vs {labels.shape}"
    assert (
        logits.shape == labels.shape
    ), f"Shapes do not match: {logits.shape} vs {labels.shape}"
    preds = keras.activations.sigmoid(logits)
    # only for MIMIC-CXR/CheXpert, we overwrite labels with all zeros such that No Finding is indicated correctly.
    if is_mimic_or_chexpert:
        zero_rows = np.all(labels == 0, axis=1)
        zero_label_idcs = np.where(zero_rows)[0]
        labels[zero_label_idcs, 0] = 1.0
    y_true = preds * labels  # set wrong class probabilities to zero
    y_wrong = 1 - y_true
    # logit transform
    logit_score = np.log(y_true + eps) - np.log(y_wrong + eps)
    logit_score *= labels  # set scores of non-present classes to zero
    return logit_score


def loss_logit_transform_binary(
    logits: np.ndarray,
    labels: np.ndarray,
    eps: float = 1e-45,
    is_mimic_or_chexpert: bool = False,
):
    """
    Numerically stable implementation of logit transform applied to model confidence (sigmoid probability output):
    logit(p) = log(p / (1 - p))
             = log(p) - log(1 - p)
    See LiRA paper for details.
    Args:
        logits: A numpy array of shape (samples, n_augs, N_classes) representing the raw model outputs
        labels: A numpy array of shape (samples, N_classes), one-hot or multi-hot encoded.
        is_mimic_or_chexpert: A boolean indicating whether the dataset is MIMIC-CXR or CheXpert.
    Returns:
        A numpy array of shape (samples,) containing the logit transformed confidence scores.
    """
    # repeat labels such that they match the shape of the logits (over augmentations of the same record)
    if len(logits.shape) > 2:
        labels = labels[:, None, :]
        labels = np.repeat(labels, logits.shape[1], axis=1)
    assert (
        logits.shape == labels.shape
    ), f"Shapes do not match: {logits.shape} vs {labels.shape}"
    # softmax transformation
    preds = keras.activations.sigmoid(logits)
    y_true = np.sum(
        preds * labels, axis=-1
    )  # set wrong class probabilities to 0 and sum over the rest (only in the multilabel setting will there be multiple correct classes)
    y_wrong = np.sum(
        preds * np.logical_not(labels), axis=-1
    )  # set the true class probabilities to 0 and sum over the rest
    assert (
        y_wrong.shape == y_true.shape
    ), f"Shapes do not match: {y_wrong.shape} vs {y_true.shape}"
    # numerically stable logit transform
    logit_score = np.log(y_true + eps) - np.log(y_wrong + eps)
    return logit_score[..., None]


def perform_offline_lira(
    train_scores: np.ndarray,
    train_masks: np.ndarray,
    test_scores: np.ndarray,
    test_masks: np.ndarray,
    fix_variance: bool = False,
    disjoint: bool = False,
):
    """
    Convenience method that performs the offline version of the LiRA attack given the training and test scores and masks.
    Args:
        train_scores: A numpy array of shape (n_samples, n_runs, n_classes) containing the training scores.
        train_masks: A numpy array of shape (n_samples, n_runs) containing the training masks.
        test_scores: A numpy array of shape (n_samples, n_classes) containing the test scores.
        test_masks: A numpy array of shape (n_samples,) containing the test masks.
        fix_variance: Whether to fix the variance of the in/out distributions.
        disjoint: Whether the train/test scores are disjoint.
    Returns:
        A tuple containing the LiRA predictions and targets.
    """
    if not disjoint:
        assert (
            train_scores.shape[1] == test_scores.shape[1]
        ), f"Train/Test Shapes do not match: {train_scores.shape} vs {test_scores.shape}"
    dat_out = []
    for j in range(train_scores.shape[1]):
        data_point_mask = train_masks[:, j]
        dat_out.append(train_scores[~data_point_mask, j])
    out_size = min([d.shape[0] for d in dat_out])
    dat_out = np.stack([x[:out_size] for x in dat_out], axis=1)

    mean_out = np.median(dat_out, 0)
    if fix_variance:
        std_out = np.std(dat_out)
    else:
        std_out = np.std(dat_out, 0)
    mia_preds = []
    mia_targets = []
    for ans, sc in zip(test_masks, test_scores):
        score = scipy.stats.norm.logpdf(sc, mean_out, std_out + 1e-30)
        mia_preds.append(score)
        mia_targets.append(ans)
    return np.array(mia_preds), np.array(mia_targets)


def perform_lira(
    train_scores: np.ndarray,
    train_masks: np.ndarray,
    test_scores: np.ndarray,
    test_masks: np.ndarray,
    fix_variance: bool = False,
    offline: bool = False,
    label_type: LabelType = LabelType.MULTICLASS,
    multi_processing: bool = False,
    threads: int = 16,
):
    """
    Convenience method that performs the LiRA attack given the training and test scores and masks.
    Args:
        train_scores: A numpy array of shape (n_samples, n_runs, n_classes) containing the training scores.
        train_masks: A numpy array of shape (n_samples, n_runs) containing the training masks.
        test_scores: A numpy array of shape (n_samples, n_classes) containing the test scores.
        test_masks: A numpy array of shape (n_samples,) containing the test masks.
        fix_variance: Whether to fix the variance of the in/out distributions.
        offline: Whether to perform the offline version of the LiRA attack.
        label_type: The label type.
        multi_processing: Whether to use multiprocessing for the attack.
        threads: The number of threads to use for multiprocessing.
    Returns:
        A tuple containing the LiRA predictions and targets.

    """

    if not offline:
        # offline attack can accomodate train/test scores and masks from different datasets but the number of classes should be the same
        # if the number of classes are not the same this can be fixed by simply adding additional columns
        assert (
            train_scores.shape[1] == test_scores.shape[1]
        ), f"Train/Test Shapes do not match: {train_scores.shape} vs {test_scores.shape}"
    if label_type != LabelType.MULTICLASS:
        raise ValueError("Only multiclass setting is supported")
    assert (
        train_scores.shape[0] == train_masks.shape[0]
    ), f"Shapes do not match: {train_scores.shape} vs {train_masks.shape}"
    assert (
        test_scores.shape[0] == test_masks.shape[0]
    ), f"Shapes do not match: {test_scores.shape} vs {test_masks.shape}"

    if offline:
        if label_type == LabelType.MULTILABEL:
            raise NotImplementedError(
                "Offline attack not implemented for multilabel setting"
            )
        return perform_offline_lira(
            train_scores=train_scores,
            train_masks=train_masks,
            test_scores=test_scores,
            test_masks=test_masks,
            fix_variance=fix_variance,
        )

    dat_in, dat_out = [], []
    for j in range(train_scores.shape[1]):
        data_point_mask = train_masks[:, j]
        dat_in.append(train_scores[data_point_mask, j])
        dat_out.append(train_scores[~data_point_mask, j])

    in_size = min([d.shape[0] for d in dat_in])
    out_size = min([d.shape[0] for d in dat_out])

    # truncate to the minimum size to make array
    dat_in = np.stack([x[:in_size] for x in dat_in], axis=1)
    dat_out = np.stack([x[:out_size] for x in dat_out], axis=1)
    assert (
        dat_in.shape[1] == dat_out.shape[1]
    ), f"Shapes do not match: {dat_in.shape} vs {dat_out.shape}"

    mean_in = np.median(dat_in, 0)
    mean_out = np.median(dat_out, 0)
    if fix_variance:
        std_in = np.std(dat_in)
        std_out = np.std(dat_out)
    else:
        std_in = np.std(dat_in, 0)
        std_out = np.std(dat_out, 0)

    pred_func = partial(
        get_preds_univariate, other_args=[mean_in, mean_out, std_in, std_out]
    )
    print(f"... Computing predictions for N={len(test_scores)} Target Models")
    if multi_processing:
        with multiprocessing.Pool(processes=threads) as pool:
            mia_preds = list(
                tqdm(
                    pool.imap(pred_func, test_scores),
                    desc="Computing LiRA attack",
                    total=len(test_scores),
                )
            )
    else:
        mia_preds = [
            pred_func(scrs=sc)
            for sc in tqdm(
                test_scores, desc="Computing LiRA attack", total=len(test_scores)
            )
        ]
    print("---------------------------------------------------------------------------")
    mia_preds = np.array(mia_preds)
    mia_targets = np.array(test_masks)
    mia_preds = -mia_preds

    return mia_preds, mia_targets


def get_preds_univariate(scrs: np.ndarray, other_args: List):
    """
    Computes the log likelihood ratio test for the univariate setting (multiclass).
    Args:
        scrs: A numpy array of shape (n_samples,) containing the logit transformed model confidences.
        other_args: A list containing the mean and standard deviation of the in/out distributions.
    Returns:
        preds: A numpy array of shape (n_samples,) containing the log likelihood ratio test scores.
    """

    in_mean, out_mean, in_std, out_std = other_args
    pr_in = -scipy.stats.norm.logpdf(scrs, in_mean, in_std + 1e-12)
    pr_out = -scipy.stats.norm.logpdf(scrs, out_mean, out_std + 1e-12)
    preds = pr_in - pr_out
    # average prediction over augmentations
    if len(preds.shape) > 1:
        preds = np.mean(preds, axis=1)
    return preds


def get_preds_multivariate(scrs: np.ndarray, other_args: List):
    """
    Computes the log likelihood ratio test for the multivariate (multilabel) setting.
    Args:
        scrs: A numpy array of shape (n_samples, n_classes) containing the test scores (logit transformed model confidences).
        other_args: A list containing the mean and covariance of the in/out distributions and the labels.
    Returns:
        preds: A numpy array of shape (n_samples,) containing the log likelihood ratio test scores.
    """

    in_mean, out_mean, in_cov, out_cov, t_labels = other_args
    sample_preds = []
    # since dimensionality varies we have to iterate over training records to compute LRT for each sample separately
    for i in range(len(in_mean)):
        present_classes_idcs = np.nonzero(t_labels[i])[0]
        pr_in = -scipy.stats.multivariate_normal.logpdf(
            scrs[i, present_classes_idcs], in_mean[i], in_cov[i]
        )
        pr_out = -scipy.stats.multivariate_normal.logpdf(
            scrs[i, present_classes_idcs], out_mean[i], out_cov[i]
        )
        sample_preds.append(pr_in - pr_out)
    preds = np.array(sample_preds)
    return preds


def preprocess_scores(scores: np.ndarray, masks: np.ndarray):
    dat_in, dat_out = [], []
    for j in range(scores.shape[1]):
        data_point_mask = masks[:, j]
        dat_in.append(scores[data_point_mask, j])
        dat_out.append(scores[~data_point_mask, j])

    in_size = min([d.shape[0] for d in dat_in])
    out_size = min([d.shape[0] for d in dat_out])

    # truncate to the minimum size to make array
    in_scores = np.stack([x[:in_size] for x in dat_in], axis=1)
    out_scores = np.stack([x[:out_size] for x in dat_out], axis=1)
    return in_scores, out_scores
