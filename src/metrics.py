"""
AmEx Default Prediction metrics.
"""
import numpy as np


def amex_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    AmEx Default Prediction metric (corrected implementation).
    
    Combines:
      - Normalized Gini coefficient (50% weight)
      - Default rate captured at top 4% (50% weight)
    
    Returns a value where:
      - Perfect prediction → ~1.0
      - Random prediction → ~0.0
      - Inverted prediction → very negative
    """
    y_true = np.asarray(y_true).astype(np.float64)
    y_pred = np.asarray(y_pred).astype(np.float64)
    
    # Sort by predictions (descending)
    indices = np.argsort(y_pred)[::-1]
    sorted_y_true = y_true[indices]
    
    # Normalized Gini 
    weights = np.where(sorted_y_true == 0, 20.0, 1.0)  # AmEx weights
    cum_pos_found = np.cumsum(sorted_y_true * weights)
    total_pos = (sorted_y_true * weights).sum()
    
    cum_neg_found = np.cumsum((1 - sorted_y_true) * weights)
    total_neg = ((1 - sorted_y_true) * weights).sum()
    
    if total_pos == 0 or total_neg == 0:
        return 0.0
    
    lorentz = cum_pos_found / total_pos
    
    # Gini for predictions
    gini_pred = ((cum_neg_found / total_neg) * (lorentz - lorentz / 2)).sum()
    
    # Gini for perfect ranking
    perfect_indices = np.argsort(y_true)[::-1]
    perfect_sorted = y_true[perfect_indices]
    perfect_weights = np.where(perfect_sorted == 0, 20.0, 1.0)
    perfect_cum_pos = np.cumsum(perfect_sorted * perfect_weights)
    perfect_lorentz = perfect_cum_pos / total_pos
    perfect_cum_neg = np.cumsum((1 - perfect_sorted) * perfect_weights)
    gini_perfect = ((perfect_cum_neg / total_neg) * 
                    (perfect_lorentz - perfect_lorentz / 2)).sum()
    
    if gini_perfect == 0:
        return 0.0
    
    normalized_gini = gini_pred / gini_perfect
    
    # Default Rate Captured at top 4% 
    n = len(y_true)
    cutoff = int(0.04 * n)
    weighted_target = sorted_y_true * weights
    
    d = weighted_target[:cutoff].sum() / total_pos
    
    # Final M Score
    M = 0.5 * (normalized_gini + d)
    return float(M)


def amex_metric_lgbm(y_pred, dataset):
    """LightGBM-compatible wrapper."""
    y_true = dataset.get_label()
    score = amex_metric(y_true, y_pred)
    return "amex_m_score", score, True


def amex_metric_sklearn(y_true, y_pred):
    """sklearn-compatible wrapper."""
    return amex_metric(np.array(y_true), np.array(y_pred))