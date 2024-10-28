# ood_metrics.py
import numpy as np
from sklearn.metrics import roc_curve

def fpr_at_95_tpr(y_true, y_scores):
    """
    Computes the false positive rate (FPR) at 95% true positive rate (TPR).
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    idx = np.argmin(np.abs(tpr - 0.95))
    return fpr[idx]
