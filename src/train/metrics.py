from __future__ import annotations
from typing import Dict, Any, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
    average_precision_score, confusion_matrix, matthews_corrcoef, balanced_accuracy_score,
    brier_score_loss
)

def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    # Ensure 1D arrays
    y_true = np.asarray(y_true).reshape(-1)
    y_prob = np.asarray(y_prob).reshape(-1)

    # Mixup (or label smoothing) can make targets continuous (e.g., 0.3/0.7).
    # For *classification* metrics we must use hard labels.
    if not np.all(np.isin(np.unique(y_true), [0, 1])):
        y_true_hard = (y_true >= 0.5).astype(int)
    else:
        y_true_hard = y_true.astype(int)

    y_pred = (y_prob >= threshold).astype(int)
    acc = accuracy_score(y_true_hard, y_pred)
    f1  = f1_score(y_true_hard, y_pred, zero_division=0)
    prec = precision_score(y_true_hard, y_pred, zero_division=0)
    rec = recall_score(y_true_hard, y_pred, zero_division=0)  # sensitivity
    bal = balanced_accuracy_score(y_true_hard, y_pred)
    mcc = matthews_corrcoef(y_true_hard, y_pred) if len(np.unique(y_true_hard)) > 1 else 0.0
    brier = brier_score_loss(y_true_hard, y_prob)
    # specificity
    cm = confusion_matrix(y_true_hard, y_pred, labels=[0,1])
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp + 1e-9)
    # AUC / AUPR (guard for single-class)
    try:
        auc = roc_auc_score(y_true_hard, y_prob)
    except Exception:
        auc = float("nan")
    try:
        aupr = average_precision_score(y_true_hard, y_prob)
    except Exception:
        aupr = float("nan")
    return {
        "accuracy": float(acc),
        "auroc": float(auc),
        "aupr": float(aupr),
        "f1": float(f1),
        "precision": float(prec),
        "recall_sensitivity": float(rec),
        "specificity": float(spec),
        "balanced_accuracy": float(bal),
        "mcc": float(mcc),
        "brier": float(brier),
        "threshold": float(threshold),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


# from __future__ import annotations
# from typing import Dict, Any, Tuple
# import numpy as np
# from sklearn.metrics import (
#     accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
#     average_precision_score, confusion_matrix, matthews_corrcoef, balanced_accuracy_score,
#     brier_score_loss
# )
#
# def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
#     # Sanitize shapes
#     y_true = np.asarray(y_true).reshape(-1)
#     y_prob = np.asarray(y_prob).reshape(-1)
#     if y_true.shape != y_prob.shape:
#         raise ValueError(f"y_true and y_prob shape mismatch: {y_true.shape} vs {y_prob.shape}")
#     # If Mixup (soft labels) is enabled, y_true may be continuous in [0,1].
#     # scikit-learn classification metrics require discrete labels.
#     uniq = np.unique(y_true)
#     if y_true.dtype.kind in ['f','c'] and not np.all(np.isin(uniq, [0.0, 1.0])):
#         y_true_hard = (y_true >= 0.5).astype(int)
#     else:
#         y_true_hard = y_true.astype(int)
#     y_pred = (y_prob >= threshold).astype(int)
#     acc = accuracy_score(y_true_hard, y_pred)
#     f1  = f1_score(y_true_hard, y_pred, zero_division=0)
#     prec = precision_score(y_true_hard, y_pred, zero_division=0)
#     rec = recall_score(y_true_hard, y_pred, zero_division=0)  # sensitivity
#     bal = balanced_accuracy_score(y_true_hard, y_pred)
#     mcc = matthews_corrcoef(y_true_hard, y_pred) if len(np.unique(y_true_hard)) > 1 else 0.0
#     brier = brier_score_loss(y_true_hard, y_prob)
#     # specificity
#     cm = confusion_matrix(y_true_hard, y_pred, labels=[0,1])
#     tn, fp, fn, tp = cm.ravel()
#     spec = tn / (tn + fp + 1e-9)
#     # AUC / AUPR (guard for single-class)
#     try:
#         auc = roc_auc_score(y_true_hard, y_prob)
#     except Exception:
#         auc = float("nan")
#     try:
#         aupr = average_precision_score(y_true_hard, y_prob)
#     except Exception:
#         aupr = float("nan")
#     return {
#         "accuracy": float(acc),
#         "auroc": float(auc),
#         "aupr": float(aupr),
#         "f1": float(f1),
#         "precision": float(prec),
#         "recall_sensitivity": float(rec),
#         "specificity": float(spec),
#         "balanced_accuracy": float(bal),
#         "mcc": float(mcc),
#         "brier": float(brier),
#         "threshold": float(threshold),
#         "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
#     }
