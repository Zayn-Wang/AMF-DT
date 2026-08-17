# metrics.py
"""
Metric utilities for survival analysis and multi-label classification.

This file contains:
- cox_log_rank: log-rank test between dichotomized risk groups
- CIndex_lifeline: concordance index using lifelines
- MultiLabel_Acc: per-label accuracy for multi-label predictions
"""

import numpy as np
import torch
from lifelines.utils import concordance_index
from lifelines.statistics import logrank_test
from sklearn.metrics import accuracy_score


def cox_log_rank(hazards, labels, survtime_all):
    """
    Perform log-rank test between high-risk and low-risk groups
    based on median of predicted hazards.

    Args:
        hazards (Tensor): model-predicted risk scores.
        labels (Tensor): event indicators (0/1).
        survtime_all (Tensor): survival times.

    Returns:
        float: p-value from log-rank test.
    """
    hazardsdata = hazards.cpu().numpy().reshape(-1)
    median = np.median(hazardsdata)
    hazards_dichotomize = np.zeros([len(hazardsdata)], dtype=int)
    hazards_dichotomize[hazardsdata > median] = 1
    survtime_all = survtime_all.data.cpu().numpy().reshape(-1)
    idx = hazards_dichotomize == 0
    labels = labels.data.cpu().numpy()
    T1 = survtime_all[idx]
    T2 = survtime_all[~idx]
    E1 = labels[idx]
    E2 = labels[~idx]
    results = logrank_test(T1, T2, event_observed_A=E1, event_observed_B=E2)
    pvalue_pred = results.p_value
    return pvalue_pred


def CIndex_lifeline(hazards, labels, survtime_all):
    """
    Compute concordance index (C-index) using lifelines.

    Args:
        hazards (Tensor): model-predicted risk scores.
        labels (Tensor): event indicators (0/1).
        survtime_all (Tensor): survival times.

    Returns:
        float: concordance index.
    """
    labels = labels.cpu().numpy()
    hazards = hazards.cpu().numpy().reshape(-1)
    survtime_all = survtime_all.cpu().numpy()
    return concordance_index(survtime_all, -hazards, labels)


def MultiLabel_Acc(Pred, Y):
    """
    Compute per-label accuracy for multi-label predictions.

    Args:
        Pred (Tensor): predicted binary labels (after thresholding).
        Y (Tensor): ground truth labels.

    Returns:
        np.ndarray: array of accuracies for each label.
    """
    Pred = Pred.cpu().numpy()
    Y = Y.cpu().numpy()
    acc = None
    for i in range(len(Y[1, :])):
        if i == 0:
            acc = accuracy_score(Y[:, i], Pred[:, i])
        else:
            acc = np.concatenate((acc, accuracy_score(Y[:, i], Pred[:, i])), axis=None)
    return acc
