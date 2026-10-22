"""Training and evaluation utilities for PTB-XL multi-label classification."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.utils import resample
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]


def _sigmoid_np(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-logits))


def compute_metrics(
    logits: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute AUC and F1 metrics from raw logits."""
    probs = _sigmoid_np(logits)
    preds = (probs >= threshold).astype(np.int32)
    targets_int = targets.astype(np.int32)

    per_class_auc: dict[str, float] = {}
    auc_values = []
    for idx, class_name in enumerate(SUPERCLASSES):
        if np.unique(targets_int[:, idx]).size < 2:
            auc = float("nan")
        else:
            auc = float(roc_auc_score(targets_int[:, idx], probs[:, idx]))
            auc_values.append(auc)
        per_class_auc[class_name] = auc

    macro_auc = float(np.mean(auc_values)) if auc_values else float("nan")
    return {
        "macro_auc": macro_auc,
        "per_class_auc": per_class_auc,
        "macro_f1": float(f1_score(targets_int, preds, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(targets_int, preds, average="micro", zero_division=0)),
    }

def precision_recall_table(
    logits: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, dict[str, float]]:
    """Per-class precision/recall/F1 table for error analysis."""
    probs = _sigmoid_np(logits)
    preds = (probs >= threshold).astype(np.int32)
    precision, recall, f1, support = precision_recall_fscore_support(
        targets.astype(np.int32),
        preds,
        average=None,
        zero_division=0,
    )
    return {
        class_name: {
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1[idx]),
            "support": int(support[idx]),
        }
        for idx, class_name in enumerate(SUPERCLASSES)
    }

def single_label_confusion(
    logits: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Confusion matrix for records with exactly one true label and one predicted label."""
    probs = _sigmoid_np(logits)
    preds = (probs >= threshold).astype(np.int32)
    single_true = targets.sum(axis=1) == 1
    single_pred = preds.sum(axis=1) == 1
    mask = single_true & single_pred

    y_true = targets[mask].argmax(axis=1)
    y_pred = preds[mask].argmax(axis=1)
    matrix = confusion_matrix(y_true, y_pred, labels=np.arange(len(SUPERCLASSES)))
    return matrix, mask


def bootstrap_auc(
    targets: np.ndarray,
    probs: np.ndarray,
    n_iterations: int = 1000,
    alpha: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """Bootstrap a confidence interval for macro AUC."""
    rng = np.random.default_rng(seed)
    aucs = []
    n_samples = len(targets)

    for _ in range(n_iterations):
        indices = resample(
            np.arange(n_samples),
            replace=True,
            n_samples=n_samples,
            random_state=int(rng.integers(0, 2**32 - 1)),
        )
        sample_targets = targets[indices]
        sample_probs = probs[indices]

        if any(np.unique(sample_targets[:, idx]).size < 2 for idx in range(sample_targets.shape[1])):
            continue
        aucs.append(roc_auc_score(sample_targets, sample_probs, average="macro"))

    lower = np.percentile(aucs, (1 - alpha) / 2 * 100)
    upper = np.percentile(aucs, (1 + alpha) / 2 * 100)
    return {
        "mean": float(np.mean(aucs)),
        "lower": float(lower),
        "upper": float(upper),
        "n_bootstraps": int(len(aucs)),
    }


def high_confidence_errors(
    logits: np.ndarray,
    targets: np.ndarray,
    class_idx: int,
    top_k: int = 20,
    threshold: float = 0.5,
) -> dict[str, np.ndarray]:
    """Return row indices for high-confidence false positives/negatives for one class."""
    probs = _sigmoid_np(logits)
    y_true = targets[:, class_idx].astype(bool)
    y_prob = probs[:, class_idx]
    y_pred = y_prob >= threshold

    false_positive_indices = np.where(y_pred & ~y_true)[0]
    false_negative_indices = np.where(~y_pred & y_true)[0]

    fp_order = false_positive_indices[np.argsort(y_prob[false_positive_indices])[::-1]]
    fn_order = false_negative_indices[np.argsort(y_prob[false_negative_indices])]
    return {
        "false_positives": fp_order[:top_k],
        "false_negatives": fn_order[:top_k],
    }