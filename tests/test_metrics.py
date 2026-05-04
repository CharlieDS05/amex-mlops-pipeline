"""
Tests for the AmEx Default Prediction metric.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest
from metrics import amex_metric


class TestAmexMetric:
    """Tests for the amex_metric function."""

    def test_perfect_prediction_returns_one(self):
        """Perfect predictions should return M Score very close to 1.0."""
        y_true = np.array([0, 0, 0, 1, 1])
        y_pred = np.array([0.01, 0.02, 0.03, 0.98, 0.99])
        score = amex_metric(y_true, y_pred)
        assert score > 0.95, f"Expected ~1.0, got {score}"

    def test_random_prediction_close_to_zero(self):
        """Random predictions should return M Score close to 0."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 10000)
        y_pred = np.random.random(10000)
        score = amex_metric(y_true, y_pred)
        assert abs(score) < 0.1, f"Expected ~0.0, got {score}"

    def test_inverted_prediction_negative(self):
        """Inverted predictions should give very negative M Score."""
        y_true = np.array([0, 0, 0, 1, 1])
        y_pred = np.array([0.99, 0.98, 0.97, 0.02, 0.01])
        score = amex_metric(y_true, y_pred)
        assert score < -0.5, f"Expected very negative, got {score}"

    def test_returns_float(self):
        """Function should always return a float type."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0.3, 0.7, 0.2, 0.8])
        score = amex_metric(y_true, y_pred)
        assert isinstance(score, float)

    def test_handles_list_input(self):
        """Function should accept Python lists, not just numpy arrays."""
        y_true = [0, 0, 1, 1]
        y_pred = [0.1, 0.2, 0.8, 0.9]
        score = amex_metric(y_true, y_pred)
        assert isinstance(score, float)