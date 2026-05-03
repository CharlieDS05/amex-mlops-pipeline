"""
AmEx Default Prediction metrics.
Based on the official competition scoring formula.
"""
import numpy as np
import pandas as pd


def amex_metric(y_true, y_pred) -> float:
    """
    Official AmEx Default Prediction metric.
    
    M = 0.5 * (Normalized Gini + Top 4% Default Rate Captured)
    
    Returns:
        - Perfect prediction: ~1.0
        - Random prediction: ~0.0
        - Inverted prediction: very negative
    """
    # Convert to numpy arrays
    y_true = np.asarray(y_true).flatten().astype(np.float64)
    y_pred = np.asarray(y_pred).flatten().astype(np.float64)

    def top_four_percent_captured(y_true, y_pred):
        """Default rate captured at top 4%."""
        df = pd.DataFrame({'target': y_true, 'prediction': y_pred})
        df = df.sort_values('prediction', ascending=False).reset_index(drop=True)
        df['weight'] = df['target'].apply(lambda x: 20.0 if x == 0 else 1.0)
        four_pct_cutoff = int(0.04 * df['weight'].sum())
        df['weight_cumsum'] = df['weight'].cumsum()
        df_cutoff = df.loc[df['weight_cumsum'] <= four_pct_cutoff]
        return (df_cutoff['target'] == 1).sum() / (df['target'] == 1).sum()

    def weighted_gini(y_true, y_pred):
        """Weighted Gini coefficient."""
        df = pd.DataFrame({'target': y_true, 'prediction': y_pred})
        df = df.sort_values('prediction', ascending=False).reset_index(drop=True)
        df['weight'] = df['target'].apply(lambda x: 20.0 if x == 0 else 1.0)
        df['random'] = (df['weight'] / df['weight'].sum()).cumsum()
        total_pos = (df['target'] * df['weight']).sum()
        df['cum_pos_found'] = (df['target'] * df['weight']).cumsum()
        df['lorentz'] = df['cum_pos_found'] / total_pos
        df['gini'] = (df['lorentz'] - df['random']) * df['weight']
        return df['gini'].sum()

    def normalized_weighted_gini(y_true, y_pred):
        """Gini divided by Gini of perfect ranking."""
        return weighted_gini(y_true, y_pred) / weighted_gini(y_true, y_true)

    g = normalized_weighted_gini(y_true, y_pred)
    d = top_four_percent_captured(y_true, y_pred)
    
    return 0.5 * (g + d)


def amex_metric_lgbm(y_pred, dataset):
    """LightGBM-compatible wrapper."""
    y_true = dataset.get_label()
    score = amex_metric(y_true, y_pred)
    return "amex_m_score", score, True


def amex_metric_sklearn(y_true, y_pred):
    """sklearn-compatible wrapper."""
    return amex_metric(y_true, y_pred)