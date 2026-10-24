"""
Model Evaluation and Cost-Sensitive Threshold Optimization.

Fraud detection requires optimizing asymmetric economic costs:
- False Negatives (FN) incur massive financial loss (stolen funds, chargeback fees).
- False Positives (FP) incur customer friction and operational investigation costs.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
import config


def evaluate_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    cost_fn: float = config.COST_FALSE_NEGATIVE,
    cost_fp: float = config.COST_FALSE_POSITIVE,
) -> Dict[str, Any]:
    """
    Evaluates binary classification predictions at a specified threshold.
    Returns standard ML metrics and financial cost metrics.
    """
    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    total_samples = len(y_true)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except Exception:
        roc_auc = 0.5

    try:
        pr_auc = average_precision_score(y_true, y_prob)
    except Exception:
        pr_auc = 0.0

    total_cost = float((fn * cost_fn) + (fp * cost_fp))
    avg_cost_per_tx = total_cost / total_samples if total_samples > 0 else 0.0

    return {
        "threshold": round(float(threshold), 3),
        "roc_auc": round(float(roc_auc), 4),
        "pr_auc": round(float(pr_auc), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "fpr": round(float(fpr), 4),
        "fnr": round(float(fnr), 4),
        "total_cost": round(total_cost, 2),
        "avg_cost_per_tx": round(avg_cost_per_tx, 4),
        "confusion_matrix": {
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
        },
    }


def optimize_decision_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cost_fn: float = config.COST_FALSE_NEGATIVE,
    cost_fp: float = config.COST_FALSE_POSITIVE,
    thresholds: Optional[np.ndarray] = None,
) -> Tuple[float, Dict[str, Any], List[Dict[str, Any]]]:
    """
    Sweeps thresholds from 0.05 to 0.95 to locate the optimal threshold
    that minimizes total expected financial losses.
    """
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 91)

    curve_records = []
    best_threshold = 0.5
    min_cost = float("inf")
    best_eval = None

    for th in thresholds:
        eval_res = evaluate_predictions(
            y_true, y_prob, threshold=th, cost_fn=cost_fn, cost_fp=cost_fp
        )
        curve_records.append(
            {
                "threshold": round(float(th), 3),
                "cost": eval_res["total_cost"],
                "precision": eval_res["precision"],
                "recall": eval_res["recall"],
                "f1": eval_res["f1"],
                "fp": eval_res["confusion_matrix"]["fp"],
                "fn": eval_res["confusion_matrix"]["fn"],
            }
        )

        if eval_res["total_cost"] < min_cost:
            min_cost = eval_res["total_cost"]
            best_threshold = float(th)
            best_eval = eval_res

    # Default to 0.5 if edge case
    if best_eval is None:
        best_eval = evaluate_predictions(y_true, y_prob, threshold=0.5)
        best_threshold = 0.5

    return best_threshold, best_eval, curve_records


def generate_roc_and_pr_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    max_points: int = 100,
) -> Dict[str, Any]:
    """Generates downsampled curve coordinates for frontend plotting."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)

    # Subsample to max_points for smooth frontend rendering
    def downsample(x, y, n):
        if len(x) <= n:
            return [round(float(v), 4) for v in x], [round(float(v), 4) for v in y]
        idx = np.linspace(0, len(x) - 1, n, dtype=int)
        return [round(float(x[i]), 4) for i in idx], [round(float(y[i]), 4) for i in idx]

    roc_fpr, roc_tpr = downsample(fpr, tpr, max_points)
    pr_recall, pr_precision = downsample(recall, precision, max_points)

    return {
        "roc_curve": {"fpr": roc_fpr, "tpr": roc_tpr},
        "pr_curve": {"recall": pr_recall, "precision": pr_precision},
    }
