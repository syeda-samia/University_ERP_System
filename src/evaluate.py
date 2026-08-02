"""Metrics, cross-validation summaries, and feature importance for both
classification (Risk, Dropout, Fee Default) and regression (GPA) modules."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate

logger = logging.getLogger(__name__)


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Accuracy, precision, recall, and F1 (binary, zero_division-safe).

    Args:
        y_true: Ground-truth binary labels.
        y_pred: Predicted binary labels.

    Returns:
        Dict with keys "accuracy", "precision", "recall", "f1".
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """R^2, RMSE, and MAE.

    Args:
        y_true: Ground-truth continuous values.
        y_pred: Predicted continuous values.

    Returns:
        Dict with keys "r2", "rmse", "mae".
    """
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def cross_validate_classifier(
    model: BaseEstimator, x: np.ndarray, y: np.ndarray, n_splits: int = 5, random_state: int = 42
) -> Dict[str, Any]:
    """k-fold (stratified) cross-validation for a classifier, scoring F1.

    Args:
        model: An unfitted scikit-learn classifier.
        x: Feature matrix.
        y: Binary labels.
        n_splits: Number of folds.
        random_state: Seed for fold shuffling.

    Returns:
        Dict with "f1_mean", "f1_std", "f1_scores".
    """
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    results = cross_validate(model, x, y, cv=kfold, scoring="f1")
    scores = results["test_score"]
    logger.info("%d-fold CV for %s: F1 mean=%.4f std=%.4f", n_splits, type(model).__name__, scores.mean(), scores.std())
    return {"f1_mean": float(scores.mean()), "f1_std": float(scores.std()), "f1_scores": scores.tolist()}


def cross_validate_regressor(
    model: BaseEstimator, x: np.ndarray, y: np.ndarray, n_splits: int = 5, random_state: int = 42
) -> Dict[str, Any]:
    """k-fold cross-validation for a regressor, scoring R^2.

    Args:
        model: An unfitted scikit-learn regressor.
        x: Feature matrix.
        y: Continuous targets.
        n_splits: Number of folds.
        random_state: Seed for fold shuffling.

    Returns:
        Dict with "r2_mean", "r2_std", "r2_scores".
    """
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    results = cross_validate(model, x, y, cv=kfold, scoring="r2")
    scores = results["test_score"]
    logger.info("%d-fold CV for %s: R^2 mean=%.4f std=%.4f", n_splits, type(model).__name__, scores.mean(), scores.std())
    return {"r2_mean": float(scores.mean()), "r2_std": float(scores.std()), "r2_scores": scores.tolist()}


def get_feature_importance(model: BaseEstimator, feature_names: List[str]) -> pd.DataFrame:
    """Extract feature importance (tree-based) or absolute coefficients (linear).

    Args:
        model: A fitted scikit-learn estimator exposing `feature_importances_` or `coef_`.
        feature_names: Column names matching the model's input features.

    Returns:
        DataFrame with columns ["Feature", "Importance"], sorted descending.
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(np.ravel(model.coef_))
    else:
        logger.warning("Model %s exposes no importance/coef attribute", type(model).__name__)
        return pd.DataFrame(columns=["Feature", "Importance"])

    df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    return df.sort_values("Importance", ascending=False).reset_index(drop=True)
