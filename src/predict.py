"""Runs a validated, cleaned student dataframe through whichever of the 6
prediction modules have the columns available for them."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.dataset import build_enrollment_forecast
from src.encoding import apply_label_encoders

logger = logging.getLogger(__name__)


class ModelLoadError(RuntimeError):
    """Raised when the model bundle cannot be loaded."""


def load_model_bundle(path: Path) -> Dict[str, Any]:
    """Load the trained model bundle produced by `src/train.py`.

    Args:
        path: Path to the joblib-serialized bundle.

    Returns:
        Dict of module name -> module bundle (model/scaler/metrics/etc).

    Raises:
        ModelLoadError: If the file is missing or cannot be deserialized.
    """
    import joblib

    if not path.exists():
        raise ModelLoadError(f"Model file not found: {path}. Run `python -m src.train` first.")
    try:
        bundle = joblib.load(path)
    except Exception as exc:
        raise ModelLoadError(f"Could not load model bundle from {path}: {exc}") from exc

    required = {"student_risk", "dropout", "fee_default", "gpa"}
    missing = required - set(bundle.keys())
    if missing:
        raise ModelLoadError(f"Model bundle at {path} is missing modules: {sorted(missing)}")
    return bundle


def _risk_level(pct: float) -> str:
    if pct < 30:
        return "Low"
    if pct < 70:
        return "Medium"
    return "High"


def predict_student_risk(df: pd.DataFrame, bundle: Dict[str, Any]) -> pd.DataFrame:
    """Predict may-fail risk for each student.

    Args:
        df: Cleaned student dataframe (must have CGPA, AttendancePercentage,
            AssignmentsSubmitted, AssignmentsTotal, Backlogs).
        bundle: Output of `load_model_bundle`.

    Returns:
        Dataframe with StudentID, risk_probability (%), risk_level.
    """
    module = bundle["student_risk"]
    completion_rate = (df["AssignmentsSubmitted"] / df["AssignmentsTotal"].replace(0, 1)).clip(upper=1.0)
    features = pd.DataFrame({
        "AttendancePercentage": df["AttendancePercentage"],
        "CGPA": df["CGPA"],
        "CompletionRate": completion_rate,
        "Backlogs": df["Backlogs"],
    })[module["feature_columns"]]

    scaled = module["scaler"].transform(features)
    proba = module["model"].predict_proba(scaled)[:, 1] * 100

    return pd.DataFrame({
        "StudentID": df["StudentID"].values,
        "risk_probability": proba.round(1),
        "risk_level": [_risk_level(p) for p in proba],
    })


def predict_dropout(df: pd.DataFrame, bundle: Dict[str, Any]) -> pd.DataFrame:
    """Predict dropout risk for each student.

    Args:
        df: Cleaned student dataframe (must have AttendancePercentage, CGPA,
            FeesPaid, LMSActivity).
        bundle: Output of `load_model_bundle`.

    Returns:
        Dataframe with StudentID, dropout_probability (%), risk_level.
    """
    module = bundle["dropout"]
    features = df[module["feature_columns"]]
    scaled = module["scaler"].transform(features)
    proba = module["model"].predict_proba(scaled)[:, 1] * 100

    return pd.DataFrame({
        "StudentID": df["StudentID"].values,
        "dropout_probability": proba.round(1),
        "risk_level": [_risk_level(p) for p in proba],
    })


def predict_fee_default(df: pd.DataFrame, bundle: Dict[str, Any]) -> pd.DataFrame:
    """Predict fee-default risk for each student.

    Args:
        df: Cleaned student dataframe (must have Gender, Department,
            YearLevel, Status, City, Scholarship, CGPA).
        bundle: Output of `load_model_bundle`.

    Returns:
        Dataframe with StudentID, default_probability (%), risk_level.
    """
    module = bundle["fee_default"]
    encoded = apply_label_encoders(df, module["encoders"], module["categorical_columns"])
    features = encoded[module["feature_columns"]]
    scaled = module["scaler"].transform(features)
    proba = module["model"].predict_proba(scaled)[:, 1] * 100

    return pd.DataFrame({
        "StudentID": df["StudentID"].values,
        "default_probability": proba.round(1),
        "risk_level": [_risk_level(p) for p in proba],
    })


def predict_gpa(df: pd.DataFrame, bundle: Dict[str, Any]) -> pd.DataFrame:
    """Predict next-term GPA for each student, with an approximate interval.

    Args:
        df: Cleaned student dataframe (must have AttendancePercentage,
            AssignmentAvg, QuizAvg, ExamAvg, CGPA).
        bundle: Output of `load_model_bundle`.

    Returns:
        Dataframe with StudentID, predicted_gpa, gpa_low, gpa_high.
    """
    module = bundle["gpa"]
    features = df[module["feature_columns"]]
    scaled = module["scaler"].transform(features)
    pred = module["model"].predict(scaled)
    pred = np.clip(pred, 0.0, 4.0)
    residual_std = module.get("residual_std", 0.0)

    return pd.DataFrame({
        "StudentID": df["StudentID"].values,
        "predicted_gpa": pred.round(2),
        "gpa_low": np.clip(pred - 1.96 * residual_std, 0.0, 4.0).round(2),
        "gpa_high": np.clip(pred + 1.96 * residual_std, 0.0, 4.0).round(2),
    })


WEAK_AREA_THRESHOLD = 50.0


def recommend_weak_areas(df: pd.DataFrame) -> pd.DataFrame:
    """Flag which academic area(s) each student is weakest in.

    Rule-based (score < 50 in a given area), not a trained model. Adapted
    from the original per-course recommendation engine to work on the
    aggregate Quiz/Assignment/Exam/Lab averages present in the generic
    template, so it works for any university's data, not just the original
    per-course exam records (see README "Model methodology").

    Args:
        df: Cleaned student dataframe (must have QuizAvg, AssignmentAvg,
            ExamAvg, LabAvg).

    Returns:
        Dataframe with StudentID, status ("Good Standing"/"Needs Attention"),
        weak_areas (comma-joined string of areas scoring below 50).
    """
    area_cols = {"QuizAvg": "Quizzes", "AssignmentAvg": "Assignments", "ExamAvg": "Exams", "LabAvg": "Labs"}
    records = []
    for _, row in df.iterrows():
        weak = [label for col, label in area_cols.items() if row[col] < WEAK_AREA_THRESHOLD]
        records.append({
            "StudentID": row["StudentID"],
            "status": "Needs Attention" if weak else "Good Standing",
            "weak_areas": ", ".join(weak) if weak else "None",
        })
    return pd.DataFrame(records)


def get_enrollment_forecast(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute an enrollment forecast from the uploaded data's own EnrollmentDate
    column (falls back to raising if fewer than 2 distinct years are present).

    Args:
        df: Cleaned student dataframe (must have EnrollmentDate).

    Returns:
        Dict with historical counts, forecast counts, and trend R^2.
    """
    return build_enrollment_forecast(df)


def run_all_modules(df: pd.DataFrame, bundle: Dict[str, Any], available_modules: List[str]) -> Dict[str, Any]:
    """Run every module whose required columns are present.

    Args:
        df: Cleaned, validated student dataframe.
        bundle: Output of `load_model_bundle`.
        available_modules: Module keys to run (from `ValidationReport.available_modules`).

    Returns:
        Dict of module name -> result (a dataframe, or a dict for enrollment_forecast).
        Modules that raise are logged and simply omitted from the result,
        rather than aborting the whole run.
    """
    results: Dict[str, Any] = {}
    runners = {
        "student_risk": lambda: predict_student_risk(df, bundle),
        "dropout": lambda: predict_dropout(df, bundle),
        "fee_default": lambda: predict_fee_default(df, bundle),
        "gpa": lambda: predict_gpa(df, bundle),
        "recommendation": lambda: recommend_weak_areas(df),
        "enrollment_forecast": lambda: get_enrollment_forecast(df),
    }
    for module_name in available_modules:
        runner = runners.get(module_name)
        if runner is None:
            continue
        try:
            results[module_name] = runner()
        except Exception:
            logger.exception("Module '%s' failed during prediction; skipping it", module_name)
    return results
