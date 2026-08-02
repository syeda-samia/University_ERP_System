"""Trains all four ML modules (Student Risk, Dropout, Fee Default, GPA),
builds the enrollment forecast from real historical data, and saves a single
consolidated model bundle.

Usage:
    python -m src.train
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler

from src.dataset import build_enrollment_forecast, load_and_build
from src.encoding import apply_label_encoders, fit_label_encoders
from src.evaluate import (
    compute_classification_metrics,
    compute_regression_metrics,
    cross_validate_classifier,
    cross_validate_regressor,
    get_feature_importance,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", message=".*delayed.*should be used with.*Parallel.*", category=UserWarning)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "models" / "university_dataset.xlsx"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "erp_models.pkl"

RANDOM_STATE = 42

CLASSIFIER_CANDIDATES = {
    "LogisticRegression": LogisticRegression(max_iter=2000),
    "RandomForest": RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
}
REGRESSOR_CANDIDATES = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
}

RF_CLASSIFIER_PARAMS = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 5, 10, 15],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", None],
}
RF_REGRESSOR_PARAMS = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 5, 10, 15],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}


def split_70_15_15(x: pd.DataFrame, y: pd.Series, stratify: bool) -> Tuple[Any, Any, Any, Any, Any, Any]:
    """70/15/15 train/val/test split, stratified for classification targets."""
    strat = y if stratify else None
    x_train, x_temp, y_train, y_temp = train_test_split(
        x, y, test_size=0.30, random_state=RANDOM_STATE, stratify=strat
    )
    strat_temp = y_temp if stratify else None
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=strat_temp
    )
    return x_train, x_val, x_test, y_train, y_val, y_test


def train_classifier_module(
    module_name: str, features: pd.DataFrame, target: pd.Series
) -> Dict[str, Any]:
    """Compare, cross-validate, tune, and evaluate a binary classifier module.

    Args:
        module_name: Used for logging only.
        features: Feature matrix (already numeric/encoded).
        target: Binary labels.

    Returns:
        Dict bundle with model, scaler, metrics, feature_importance, etc.
    """
    x_train, x_val, x_test, y_train, y_val, y_test = split_70_15_15(features, target, stratify=True)

    scaler = StandardScaler().fit(x_train)
    x_train_s, x_val_s, x_test_s = scaler.transform(x_train), scaler.transform(x_val), scaler.transform(x_test)

    cv_results = {}
    for name, model in CLASSIFIER_CANDIDATES.items():
        logger.info("[%s] 5-fold CV for %s", module_name, name)
        cv_results[name] = cross_validate_classifier(model, x_train_s, y_train)

    best_name = max(cv_results, key=lambda n: cv_results[n]["f1_mean"])
    logger.info("[%s] Best candidate by CV F1: %s", module_name, best_name)

    if best_name == "RandomForest":
        search = RandomizedSearchCV(
            RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
            RF_CLASSIFIER_PARAMS, n_iter=15, cv=5, scoring="f1", random_state=RANDOM_STATE, n_jobs=-1,
        )
        search.fit(x_train_s, y_train)
        tuned_model = search.best_estimator_
        logger.info("[%s] Best params: %s (CV F1=%.4f)", module_name, search.best_params_, search.best_score_)
    else:
        tuned_model = CLASSIFIER_CANDIDATES[best_name]
        tuned_model.fit(x_train_s, y_train)

    val_metrics = compute_classification_metrics(y_val, tuned_model.predict(x_val_s))
    logger.info("[%s] Validation metrics: %s", module_name, val_metrics)

    x_trainval_s = np.vstack([x_train_s, x_val_s])
    y_trainval = pd.concat([y_train, y_val])
    tuned_model.fit(x_trainval_s, y_trainval)

    test_metrics = compute_classification_metrics(y_test, tuned_model.predict(x_test_s))
    logger.info("[%s] Held-out test metrics: %s", module_name, test_metrics)

    feature_importance = get_feature_importance(tuned_model, list(features.columns))

    return {
        "model": tuned_model,
        "model_name": best_name,
        "scaler": scaler,
        "feature_columns": list(features.columns),
        "cv_results": cv_results,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "feature_importance": feature_importance.to_dict(orient="records"),
    }


def train_regressor_module(module_name: str, features: pd.DataFrame, target: pd.Series) -> Dict[str, Any]:
    """Compare, cross-validate, tune, and evaluate the GPA regressor module."""
    x_train, x_val, x_test, y_train, y_val, y_test = split_70_15_15(features, target, stratify=False)

    scaler = StandardScaler().fit(x_train)
    x_train_s, x_val_s, x_test_s = scaler.transform(x_train), scaler.transform(x_val), scaler.transform(x_test)

    cv_results = {}
    for name, model in REGRESSOR_CANDIDATES.items():
        logger.info("[%s] 5-fold CV for %s", module_name, name)
        cv_results[name] = cross_validate_regressor(model, x_train_s, y_train)

    best_name = max(cv_results, key=lambda n: cv_results[n]["r2_mean"])
    logger.info("[%s] Best candidate by CV R^2: %s", module_name, best_name)

    if best_name == "RandomForest":
        search = RandomizedSearchCV(
            RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
            RF_REGRESSOR_PARAMS, n_iter=15, cv=5, scoring="r2", random_state=RANDOM_STATE, n_jobs=-1,
        )
        search.fit(x_train_s, y_train)
        tuned_model = search.best_estimator_
        logger.info("[%s] Best params: %s (CV R^2=%.4f)", module_name, search.best_params_, search.best_score_)
    else:
        tuned_model = REGRESSOR_CANDIDATES[best_name]
        tuned_model.fit(x_train_s, y_train)

    val_metrics = compute_regression_metrics(y_val, tuned_model.predict(x_val_s))
    logger.info("[%s] Validation metrics: %s", module_name, val_metrics)

    x_trainval_s = np.vstack([x_train_s, x_val_s])
    y_trainval = pd.concat([y_train, y_val])
    tuned_model.fit(x_trainval_s, y_trainval)

    test_metrics = compute_regression_metrics(y_test, tuned_model.predict(x_test_s))
    residual_std = float(np.std(y_test.values - tuned_model.predict(x_test_s)))
    logger.info("[%s] Held-out test metrics: %s", module_name, test_metrics)

    feature_importance = get_feature_importance(tuned_model, list(features.columns))

    return {
        "model": tuned_model,
        "model_name": best_name,
        "scaler": scaler,
        "feature_columns": list(features.columns),
        "cv_results": cv_results,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "residual_std": residual_std,
        "feature_importance": feature_importance.to_dict(orient="records"),
    }


def main() -> None:
    """Train all modules and save the consolidated bundle."""
    df = load_and_build(DATA_PATH)
    logger.info("Training dataset: %d students", len(df))

    # --- Student Risk ---
    risk_features = df[["CGPA", "AttendancePercentage", "AssignmentsSubmitted", "AssignmentsTotal", "Backlogs"]].copy()
    risk_features["CompletionRate"] = (
        risk_features["AssignmentsSubmitted"] / risk_features["AssignmentsTotal"].replace(0, 1)
    ).clip(upper=1.0)
    risk_features = risk_features[["AttendancePercentage", "CGPA", "CompletionRate", "Backlogs"]]
    student_risk_bundle = train_classifier_module("student_risk", risk_features, df["may_fail_label"])

    # --- Dropout ---
    dropout_features = df[["AttendancePercentage", "CGPA", "FeesPaid", "LMSActivity"]]
    dropout_bundle = train_classifier_module("dropout", dropout_features, df["dropout_label"])

    # --- Fee Default ---
    fee_categorical_cols = ["Gender", "Department", "YearLevel", "Status", "City", "Scholarship"]
    fee_encoders = fit_label_encoders(df, fee_categorical_cols)
    fee_features = apply_label_encoders(df, fee_encoders, fee_categorical_cols)
    fee_features = fee_features[fee_categorical_cols + ["CGPA"]]
    fee_default_bundle = train_classifier_module("fee_default", fee_features, df["fee_default_label"])
    fee_default_bundle["encoders"] = fee_encoders
    fee_default_bundle["categorical_columns"] = fee_categorical_cols

    # --- GPA ---
    gpa_features = df[["AttendancePercentage", "AssignmentAvg", "QuizAvg", "ExamAvg", "CGPA"]]
    gpa_bundle = train_regressor_module("gpa", gpa_features, df["gpa_target"])

    # --- Enrollment Forecast (real historical data, simple trend model) ---
    enrollment_bundle = build_enrollment_forecast(df)

    valid_ranges = {
        "CGPA": (float(df["CGPA"].min()), float(df["CGPA"].max())),
        "AttendancePercentage": (float(df["AttendancePercentage"].min()), float(df["AttendancePercentage"].max())),
        "Backlogs": (float(df["Backlogs"].min()), float(df["Backlogs"].max())),
        "QuizAvg": (float(df["QuizAvg"].min()), float(df["QuizAvg"].max())),
        "AssignmentAvg": (float(df["AssignmentAvg"].min()), float(df["AssignmentAvg"].max())),
        "ExamAvg": (float(df["ExamAvg"].min()), float(df["ExamAvg"].max())),
        "LabAvg": (float(df["LabAvg"].min()), float(df["LabAvg"].max())),
    }

    bundle = {
        "student_risk": student_risk_bundle,
        "dropout": dropout_bundle,
        "fee_default": fee_default_bundle,
        "gpa": gpa_bundle,
        "enrollment_forecast": enrollment_bundle,
        "valid_ranges": valid_ranges,
        "n_training_students": len(df),
    }

    MODEL_DIR.mkdir(exist_ok=True)
    try:
        joblib.dump(bundle, MODEL_PATH)
    except OSError as exc:
        logger.error("Failed to save model bundle to %s: %s", MODEL_PATH, exc)
        raise
    logger.info("Saved consolidated model bundle to %s", MODEL_PATH)

    print("\n=== Module summary (test-set metrics) ===")
    for name in ["student_risk", "dropout", "fee_default"]:
        m = bundle[name]["test_metrics"]
        print(f"{name:>15} ({bundle[name]['model_name']:>18}): acc={m['accuracy']:.3f} prec={m['precision']:.3f} rec={m['recall']:.3f} f1={m['f1']:.3f}")
    gm = bundle["gpa"]["test_metrics"]
    print(f"{'gpa':>15} ({bundle['gpa']['model_name']:>18}): R2={gm['r2']:.3f} RMSE={gm['rmse']:.3f} MAE={gm['mae']:.3f}")
    print(f"\nEnrollment forecast trend R^2: {enrollment_bundle['trend_r2']:.3f}")
    print("Forecast:", enrollment_bundle["forecast"])


if __name__ == "__main__":
    main()
