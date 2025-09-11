"""
Robust Membership Inference Attack (RMIA) implementation for membership inference attacks.

This module implements the RMIA attack algorithm, which uses the relative mass
of data points in the model's output distribution to infer membership status.
RMIA is particularly effective against models with complex decision boundaries.

References:
- Zarifzadeh et al. "Low-Cost High-Power Membership Inference Attacks" (2023)
"""
import multiprocessing
from functools import partial
from pathlib import Path
from typing import Optional

import keras # type ignore
import numpy as np
import scipy # type ignore
from sklearn.metrics import roc_auc_score # type ignore
from tqdm import tqdm # type ignore

from ..train_utils.utils import LabelType


def rmia_transform(
    logits: np.ndarray,
    labels: np.ndarray,
    temperature: float = 1.0,
    is_mimic_or_chexpert=False,
    is_binary: bool = False,
):
    """
    Transform model logits to confidence scores for RMIA attack.
    
    Applies softmax (or sigmoid for binary) activation and extracts probabilities 
    for the true class labels. Includes special handling for medical datasets where
    "No Finding" is represented as all-zero labels.

    Args:
        logits: Rw model predictions array of shape (n_samples, n_augs, n_classes)
        labels: One-hot encoded labels of shape (n_samples, n_augs, n_classes) 
        temperature: Temperature scaling parameter for softmax (default 1.0)
        is_mimic_or_chexpert: Enable special "No Finding" label handling for medical datasets
        is_binary: Use sigmoid instead of softmax for binary classification
    
    Returns:
        np.ndarray: Confidence scores for true class labels, shape (n_samples,) or (n_samples, n_augs)
        
    References:
        Zarifzadeh et al. "Low-Cost High-Power Membership Inference Attacks" (2023)
        https://proceedings.mlr.press/v235/zarifzadeh24a.html
    """
    # only for MIMIC-CXR/CheXpert, we overwrite labels with all zeros such that No Finding is indicated correctly.
    if is_mimic_or_chexpert:
        zero_rows = np.all(labels == 0, axis=-1)
        zero_label_idcs = np.where(zero_rows)[0]
        labels[zero_label_idcs, 0] = 1.0
    # if augmentation axis is present, we need to repeat the labels
    if len(logits.shape) > 2:
        labels = labels[:, None, :]
        labels = np.repeat(labels, logits.shape[1], axis=1)
    assert (
        logits.shape == labels.shape
    ), f"Shapes do not match: {logits.shape} vs {labels.shape}"
    if temperature != 1.0:
        logits /= temperature
    if is_binary:
        preds = keras.activations.sigmoid(logits)
        scores = preds
    else:
        preds = keras.activations.softmax(logits, axis=-1)
        # extract the probabilities for the true class
        scores = np.sum(preds * labels, axis=-1)
    assert (
        scores.shape[0] == logits.shape[0]
    ), f"Expected scores to have shape {logits.shape[0]}, but got {scores.shape[0]}"
    # check augmentation axis is correctly preserved
    if len(logits.shape) > 2:
        assert (
            scores.shape[1] == logits.shape[1]
        ), f"Expected scores to have shape {logits.shape[1]}, but got {scores.shape[1]}"
    return scores


def rmia_transform_multilabel(logits: np.ndarray, labels: np.ndarray):
    """
    Apply the sigmoid activation function on the logits and return the (average) probability/(ies) for the true class(es).

    Args:
        logits: a numpy array of shape (n_samples, n_classes) containing the logits (raw predictions) of the model.
        labels: a numpy array of shape (n_samples, n_classes) containing the multi-hot encoded labels.

    Returns:
        scores: The (average) sigmoid probabilities for the true class(es).
    """
    raise NotImplementedError("Multilabel RMIA is not implemented yet.")


def perform_rmia(
    train_scores: np.ndarray,
    train_masks: np.ndarray,
    test_scores: np.ndarray,
    test_masks: np.ndarray,
    a: float = 0.5,
    n_pop: int = 500,
    gamma: float = 1.0,
    pop_indices: Optional[np.ndarray] = None,
    pcut: float = 0.01,
    offline: bool = False,
    find_optimal_a: bool = False,
    labels: Optional[np.ndarray] = None,
    label_type: LabelType = LabelType.MULTICLASS,
    multi_processing: bool = False,
    threads: int = 16,
):
    """
    Execute Robust Membership Inference Attack (RMIA).
    
    RMIA compares the confidence (predicted probability) on target data points against 
    reference (population) data to infer membership.

    Args:
        train_scores: reference model confidence scores, shape (n_models, n_samples)
        train_masks: Membership masks indicating which samples were used for training each model, 
                     shape (n_models, n_samples)  
        test_scores: Target model scores to evaluate, shape (n_target_models, n_samples)
        test_masks: Ground truth membership for target models, shape (n_target_models, n_samples)
        a: Interpolation parameter for offline RMIA (0.0 to 1.0, default 0.5)
        n_pop: Number of population reference points per target sample (default 500)
        gamma: Decision threshold for membership inference (default 1.0)
        pop_indices: Pre-computed population sample indices (if None, randomly sampled)
        pcut: Proportion to trim from distributions for robustness (default 0.01)
        offline: Use offline RMIA variant that doesn't require IN models (default False)
        find_optimal_a: Auto-tune parameter 'a' for optimal performance (default False)
        labels: Sample labels for multilabel settings (optional)
        label_type: Classification type (BINARY, MULTICLASS, MULTILABEL)
        multi_processing: Enable parallel processing for speed (default False)  
        threads: Number of parallel threads when multiprocessing enabled (default 16)
    
    Returns:
        tuple: (mia_predictions, mia_targets) where:
            - mia_predictions: Attack confidence scores, shape (n_target_models, n_samples)
            - mia_targets: Ground truth membership labels, shape (n_target_models, n_samples)
            
    Note:
        Requires multiple reference models for accurate density estimation. Higher n_pop 
        values improve attack accuracy but increase computational cost.
    """
    n_target_records = test_scores.shape[1]
    print(train_scores.shape, train_masks.shape, test_scores.shape, test_masks.shape)
    if pop_indices is None:
        sample_ratio = 5 * n_pop / n_target_records
        print(
            f"... didn't find population indices for RMIA, sampling {sample_ratio} of the data to attack {n_target_records} target records"
        )
        pop_indices = np.random.binomial(1, sample_ratio, size=(n_target_records))
    else:
        assert (
            pop_indices.max() == 1
        ), "Expected pop_indices to be one-hot encoded indices"
    print(
        f"... sampling {n_pop} records from {np.sum(pop_indices)} population data records for RMIA"
    )
    # for multilabel setting scores will be multivariate
    if label_type == LabelType.MULTILABEL:
        assert (
            train_scores.shape[:-1] == train_masks.shape
        ), f"Shapes do not match: {train_scores.shape} vs {train_masks.shape}"
        assert (
            test_scores.shape[:-1] == test_masks.shape
        ), f"Shapes do not match: {test_scores.shape} vs {test_masks.shape}"
        assert labels is not None, "Labels must be provided for multilabel setting"
    else:
        assert (
            train_scores.shape[0] == train_masks.shape[0]
        ), f"Shapes do not match: {train_scores.shape} vs {train_masks.shape}"
        assert (
            test_scores.shape[0] == test_masks.shape[0]
        ), f"Shapes do not match: {test_scores.shape} vs {test_masks.shape}"
    # average over augmentations if present
    if len(train_scores.shape) > 2:
        train_scores = np.mean(train_scores, axis=-1)
        test_scores = np.mean(test_scores, axis=-1)

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

    # population data selection
    pop_indices = np.nonzero(pop_indices)[0]
    indices = [
        np.random.choice(pop_indices[pop_indices != i], size=n_pop, replace=False)
        for i in range(n_target_records)
    ]
    indices = np.stack(indices)
    for i in range(n_target_records):
        assert (
            np.sum(i == indices[i]) == 0
        ), "Whoops, accidentally picked target record as its own population data"

    pr_z = scipy.stats.trim_mean(dat_out, proportiontocut=pcut, axis=0)
    if find_optimal_a:
        optimal_a = find_optimal_a_value(
            test_scores=test_scores[0],
            test_masks=test_masks[0],
            dat_out=dat_out,
            pr_z=pr_z,
            indices=indices,
            gamma=gamma,
        )
        print(
            f"... Overwriting previous a value ({a}) with newly found optimal value {optimal_a}"
        )
        a = optimal_a
    if offline:
        pr_x = (
            1
            / 2
            * (
                (1 + a) * scipy.stats.trim_mean(dat_out, proportiontocut=pcut, axis=0)
                + (1 - a)
            )
        )
    else:
        pr_x = (
            scipy.stats.trim_mean(dat_in, proportiontocut=pcut, axis=0)
            + scipy.stats.trim_mean(dat_out, proportiontocut=pcut, axis=0)
        ) / 2
    pred_func = partial(
        get_preds_rmia, pr_x=pr_x, pr_z=pr_z, indices=indices, gamma=gamma
    )
    print(f"... Computing predictions for N={len(test_scores)} Target Models")
    if multi_processing:
        with multiprocessing.Pool(processes=threads) as pool:
            mia_preds = list(
                tqdm(
                    pool.imap(pred_func, test_scores),
                    desc="Computing RMIA attack",
                    total=len(test_scores),
                )
            )
    else:
        mia_preds = [
            pred_func(sc)
            for sc in tqdm(
                test_scores, desc="Computing RMIA attack", total=len(test_scores)
            )
        ]
    print("---------------------------------------------------------------------------")
    mia_preds = np.array(mia_preds)
    mia_targets = np.array(test_masks)
    return mia_preds, mia_targets


def find_optimal_a_value(
    test_scores: np.ndarray,
    test_masks: np.ndarray,
    dat_out: np.ndarray,
    pr_z: np.ndarray,
    indices: np.ndarray,
    candidate_values: np.ndarray = np.linspace(0.0, 1.0, 50),
    gamma: float = 1.0,
):
    """
        Convenience function to find the optimal a value for RMIA.
    Args:
        test_scores: numpy array of shape (n_samples,) containing the test scores.
        test_masks: numpy array of shape (n_samples,) containing the test masks.
        dat_out: numpy array of shape (n_samples, n_augs) containing the out data.
        pr_z: numpy array of shape (n_augs,) containing the p(z) values.
        indices: numpy array of shape (n_samples, n_pop) containing the indices of the population data.
        candidate_values: numpy array of candidate values for the a parameter.
        gamma: The gamma parameter for RMIA.
    Returns:
        best_val: The optimal a value that achieves the highest AUC.
    """
    best_auc, best_val = 0.5, 0.5
    for a_val in candidate_values:
        pr_x = (
            1
            / 2
            * (
                (1 + a_val)
                * scipy.stats.trim_mean(dat_out, proportiontocut=0.01, axis=0)
                + (1 - a_val)
            )
        )
        mia_preds = get_preds_rmia(test_scores, pr_x, pr_z, indices, gamma)
        auc = roc_auc_score(test_masks, mia_preds)
        if auc > best_auc:
            best_auc, best_val = auc, a_val
    print(f"... found best a value: {best_val} with AUC: {best_auc}")
    return best_val


def get_preds_rmia(
    test_scores: np.ndarray,
    pr_x: np.ndarray,
    pr_z: np.ndarray,
    indices: np.ndarray,
    gamma: float,
    epsilon: float = 1e-10,
):
    """
    Compute RMIA predictions for the given test scores and population data.
    Args:
        test_scores: numpy array of shape (n_samples,) containing the test scores.
        pr_x: numpy array of shape (n_samples,) containing the p(x) values.
        pr_z: numpy array of shape (n_samples,) containing the p(z) values.
        indices: numpy array of shape (n_samples, n_pop) containing the indices of the population data.
        gamma: The gamma parameter for RMIA.
    Returns:
        mia_preds: numpy array containing the MIA predictions.
    """
    assert (
        len(indices.shape) == 2
    ), f"Expected indices to have shape (n_samples, n_pop), but got {indices.shape}"
    assert (
        test_scores.shape == pr_x.shape == pr_z.shape
    ), f"Shapes do not match: {test_scores.shape} vs {pr_x.shape} vs {pr_z.shape}"
    assert (
        indices.shape[0] == test_scores.shape[0]
    ), f"Expected indices to have shape (n_samples, n_pop), but got {indices.shape}"
    assert (
        indices.max() != 1
    ), f"Expected indices to contain indices, but got one-hot {indices}"
    ratio_x = test_scores / (pr_x + epsilon)
    ratio_z = 1 / ((test_scores / (pr_z+ epsilon)) + epsilon)
    scores = outer_product_at_indices(
        vector_a=ratio_x, vector_b=ratio_z, indices=indices
    )
    mia_preds = np.mean(scores > gamma, axis=1)
    return mia_preds


def outer_product_at_indices(
    vector_a: np.ndarray, vector_b: np.ndarray, indices: np.ndarray
):
    """
    Compute the outer product of two vectors but return only the values indicated by entries. This is much more memory efficient than
            np.take_along_axis(np.outer(vector_a, vector_b), axis=1) as the NxN matrix is never instantiated.
    Args:
        vector_a: numpy array of shape (N,) containing the first vector.
        vector_b: numpy array of shape (N,) containing the second vector.
        indices: numpy array of shape (N, M) containing the indices which we would like to retrieve the value of (from the NxN  matrix).
    """
    assert (
        vector_a.shape == vector_b.shape
    ), f"Shapes do not match: {vector_a.shape} vs {vector_b.shape}"
    assert (
        len(vector_a.shape) == 1
    ), f"Expected vector_a to have shape (N,) but got {vector_a.shape}"
    assert (
        vector_a.shape[0] == indices.shape[0]
    ), f"Expected indices to have shape {vector_a.shape[0]}, but got {indices.shape[0]} instead"

    return vector_a[:, None] * vector_b[indices]
