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

from src.helpers import SUPERCLASSES
from src.logging_utils import log_progress


def get_device(prefer_cuda: bool = True, require_cuda: bool = False) -> torch.device:
    """Return a usable torch device, optionally failing if CUDA cannot initialize."""
    if prefer_cuda and torch.cuda.is_available():
        try:
            torch.empty(1, device="cuda")
            return torch.device("cuda")
        except Exception as exc:
            first_line = str(exc).splitlines()[0]
            message = f"CUDA is visible but failed to initialize: {first_line}"
            if require_cuda:
                raise RuntimeError(message) from exc
            log_progress(f"{message}; falling back to CPU")
    elif require_cuda:
        raise RuntimeError("CUDA is required for this run, but torch.cuda.is_available() is False")
    return torch.device("cpu")


def _non_blocking(device: torch.device | str) -> bool:
    return torch.device(device).type == "cuda"


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


@torch.no_grad()
def predict(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device | str,
    desc: str = "predict",
) -> tuple[np.ndarray, np.ndarray]:
    """Collect raw logits and targets from a DataLoader."""
    model.eval()
    all_logits = []
    all_targets = []

    non_blocking = _non_blocking(device)
    for signals, targets in tqdm(dataloader, desc=desc, leave=False):
        signals = signals.to(device, non_blocking=non_blocking)
        logits = model(signals)
        all_logits.append(logits.detach().cpu().numpy())
        all_targets.append(targets.detach().cpu().numpy())

    return np.vstack(all_logits), np.vstack(all_targets)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device | str,
) -> dict[str, Any]:
    """Run one training epoch."""
    model.train()
    losses = []
    all_logits = []
    all_targets = []

    non_blocking = _non_blocking(device)
    for signals, targets in tqdm(dataloader, desc="train", leave=False):
        signals = signals.to(device, non_blocking=non_blocking)
        targets = targets.to(device, non_blocking=non_blocking)

        optimizer.zero_grad(set_to_none=True)
        logits = model(signals)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        all_logits.append(logits.detach().cpu().numpy())
        all_targets.append(targets.detach().cpu().numpy())

    metrics = compute_metrics(np.vstack(all_logits), np.vstack(all_targets))
    metrics["loss"] = float(np.mean(losses))
    return metrics


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device | str,
    desc: str = "eval",
) -> dict[str, Any]:
    """Evaluate the model on a validation or test DataLoader."""
    model.eval()
    losses = []
    all_logits = []
    all_targets = []

    non_blocking = _non_blocking(device)
    for signals, targets in tqdm(dataloader, desc=desc, leave=False):
        signals = signals.to(device, non_blocking=non_blocking)
        targets = targets.to(device, non_blocking=non_blocking)
        logits = model(signals)
        loss = criterion(logits, targets)

        losses.append(loss.item())
        all_logits.append(logits.detach().cpu().numpy())
        all_targets.append(targets.detach().cpu().numpy())

    metrics = compute_metrics(np.vstack(all_logits), np.vstack(all_targets))
    metrics["loss"] = float(np.mean(losses))
    return metrics


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | None = None,
    device: torch.device | str | None = None,
    epochs: int = 50,
    patience: int = 10,
    checkpoint_path: str | Path = "checkpoints/best_bilstm.pt",
) -> list[dict[str, Any]]:
    """Train with best-val-AUC checkpointing and early stopping."""
    device = device or get_device()
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    model.to(device)

    history: list[dict[str, Any]] = []
    best_auc = -float("inf")
    epochs_without_improvement = 0

    log_progress(
        f"Starting training for up to {epochs} epochs on {device}; "
        f"checkpoint={checkpoint_path}"
    )
    for epoch in range(1, epochs + 1):
        log_progress(f"Epoch {epoch:03d}/{epochs}: training started")
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        log_progress(
            f"Epoch {epoch:03d}/{epochs}: training done "
            f"loss={train_metrics['loss']:.4f} macro_auc={train_metrics['macro_auc']:.4f}"
        )
        val_metrics = evaluate(model, val_loader, criterion, device, desc="val")
        log_progress(
            f"Epoch {epoch:03d}/{epochs}: validation done "
            f"loss={val_metrics['loss']:.4f} macro_auc={val_metrics['macro_auc']:.4f}"
        )

        if scheduler is not None:
            scheduler.step(val_metrics["macro_auc"])

        row = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)

        val_auc = val_metrics["macro_auc"]
        log_progress(
            f"Epoch {epoch:03d} | "
            f"train loss {train_metrics['loss']:.4f} auc {train_metrics['macro_auc']:.4f} | "
            f"val loss {val_metrics['loss']:.4f} auc {val_auc:.4f}"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_macro_auc": val_auc,
                    "history": history,
                },
                checkpoint_path,
            )
            log_progress(f"Saved new best checkpoint with val macro AUC {best_auc:.4f}")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            log_progress(f"Early stopping after {epoch} epochs. Best val macro AUC: {best_auc:.4f}")
            break

    return history


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