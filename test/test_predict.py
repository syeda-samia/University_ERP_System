"""Tests for src/predict.py: model loading and end-to-end prediction for all
6 modules, using the real trained bundle and sample template when available."""

from pathlib import Path

import pandas as pd
import pytest

from src.predict import (
    ModelLoadError,
    load_model_bundle,
    predict_dropout,
    predict_fee_default,
    predict_gpa,
    predict_student_risk,
    recommend_weak_areas,
    run_all_modules,
)
from src.validation import clean_and_coerce, read_uploaded_file, validate_upload


def test_load_model_bundle_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ModelLoadError):
        load_model_bundle(tmp_path / "missing.pkl")


@pytest.fixture
def bundle(model_path: Path):
    if not model_path.exists():
        pytest.skip("Model bundle not trained yet; run `python -m src.train` first.")
    return load_model_bundle(model_path)


@pytest.fixture
def sample_df(sample_template_path: Path):
    if not sample_template_path.exists():
        pytest.skip("Sample template not generated yet.")
    with open(sample_template_path, "rb") as f:
        raw = read_uploaded_file(f.read(), sample_template_path.name)
    return clean_and_coerce(raw)


def test_load_model_bundle_success(bundle) -> None:
    assert "student_risk" in bundle
    assert "dropout" in bundle
    assert "fee_default" in bundle
    assert "gpa" in bundle


def test_predict_student_risk_returns_valid_probabilities(bundle, sample_df) -> None:
    result = predict_student_risk(sample_df, bundle)
    assert len(result) == len(sample_df)
    assert (result["risk_probability"] >= 0).all() and (result["risk_probability"] <= 100).all()
    assert set(result["risk_level"].unique()) <= {"Low", "Medium", "High"}


def test_predict_dropout_returns_valid_probabilities(bundle, sample_df) -> None:
    result = predict_dropout(sample_df, bundle)
    assert len(result) == len(sample_df)
    assert (result["dropout_probability"] >= 0).all() and (result["dropout_probability"] <= 100).all()


def test_predict_fee_default_handles_unseen_category(bundle, sample_df) -> None:
    modified = sample_df.copy()
    modified.loc[0, "Department"] = "Department That Does Not Exist"
    result = predict_fee_default(modified, bundle)
    assert len(result) == len(modified)
    assert not result["default_probability"].isna().any()


def test_predict_gpa_interval_contains_point_estimate(bundle, sample_df) -> None:
    result = predict_gpa(sample_df, bundle)
    assert (result["gpa_low"] <= result["predicted_gpa"]).all()
    assert (result["predicted_gpa"] <= result["gpa_high"]).all()
    assert (result["predicted_gpa"] >= 0).all() and (result["predicted_gpa"] <= 4).all()


def test_recommend_weak_areas_flags_low_scores() -> None:
    df = pd.DataFrame({
        "StudentID": ["S1", "S2"],
        "QuizAvg": [30, 90],
        "AssignmentAvg": [40, 85],
        "ExamAvg": [20, 88],
        "LabAvg": [35, 92],
    })
    result = recommend_weak_areas(df)
    assert result.loc[0, "status"] == "Needs Attention"
    assert result.loc[1, "status"] == "Good Standing"
    assert "Quizzes" in result.loc[0, "weak_areas"]


def test_run_all_modules_end_to_end(bundle, sample_df) -> None:
    report = validate_upload(sample_df.assign(EnrollmentDate=sample_df["EnrollmentDate"]))
    results = run_all_modules(sample_df, bundle, report.available_modules)
    assert "student_risk" in results
    assert "gpa" in results
    assert "recommendation" in results
