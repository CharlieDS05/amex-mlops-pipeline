import numpy as np


def amex_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Métrica oficial de la competencia AmEx."""
    def gini(y_true, y_pred):
        n = len(y_true)
        sorted_idx = np.argsort(-y_pred)
        y_sorted = y_true[sorted_idx]
        total_pos = y_true.sum()
        lorenz = np.cumsum(y_sorted) / total_pos
        gini_sum = lorenz[:-1].sum() / (n - 1)
        return 2 * gini_sum - 1 + (1 / n)

    gini_pred = gini(y_true, y_pred)
    gini_perfect = gini(y_true, y_true)
    normalized_gini = gini_pred / gini_perfect

    n = len(y_true)
    top_4pct = int(np.ceil(0.04 * n))
    sorted_idx = np.argsort(-y_pred)
    top_idx = sorted_idx[:top_4pct]
    d = y_true[top_idx].sum() / y_true.sum()

    M = 0.5 * (normalized_gini + d)
    return float(M)


def amex_metric_lgbm(y_pred, dataset):
    """Wrapper para LightGBM."""
    y_true = dataset.get_label()
    score = amex_metric(y_true, y_pred)
    return "amex_m_score", score, True