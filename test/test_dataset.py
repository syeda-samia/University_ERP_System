"""Tests for src/dataset.py: feature engineering, label construction, and
the enrollment forecast trend model."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.dataset import (
    build_enrollment_forecast,
    build_student_features,
    build_training_targets,
    load_raw_sheets,
)


def test_load_raw_sheets_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_raw_sheets(tmp_path / "missing.xlsx")


def test_build_student_features_no_missing_values(raw_sheets: dict) -> None:
    features = build_student_features(raw_sheets)
    assert not features[["CGPA", "AttendancePercentage", "QuizAvg", "ExamAvg"]].isna().any().any()


def test_build_student_features_attendance_in_range(raw_sheets: dict) -> None:
    features = build_student_features(raw_sheets)
    assert (features["AttendancePercentage"] >= 0).all()
    assert (features["AttendancePercentage"] <= 100).all()


def test_build_student_features_simulated_columns_present(raw_sheets: dict) -> None:
    features = build_student_features(raw_sheets)
    assert features["FeesPaid"].isin([0, 1]).all()
    assert (features["LMSActivity"] >= 0).all() and (features["LMSActivity"] <= 100).all()


def test_build_training_targets_labels_are_binary(raw_sheets: dict) -> None:
    features = build_student_features(raw_sheets)
    df = build_training_targets(features)
    assert set(df["dropout_label"].unique()) <= {0, 1}
    assert set(df["may_fail_label"].unique()) <= {0, 1}
    assert set(df["fee_default_label"].unique()) <= {0, 1}


def test_build_training_targets_gpa_in_range(raw_sheets: dict) -> None:
    features = build_student_features(raw_sheets)
    df = build_training_targets(features)
    assert (df["gpa_target"] >= 0).all()
    assert (df["gpa_target"] <= 4).all()


def test_build_training_targets_may_fail_matches_rule(raw_sheets: dict) -> None:
    features = build_student_features(raw_sheets)
    df = build_training_targets(features)
    expected = ((df["CGPA"] < 2.0) | (df["ExamAvg"] < 50)).astype(int)
    assert (df["may_fail_label"] == expected).all()


def test_build_enrollment_forecast_basic() -> None:
    df = pd.DataFrame({
        "EnrollmentDate": pd.to_datetime(
            ["2020-06-01"] * 10 + ["2021-06-01"] * 15 + ["2022-06-01"] * 20
        )
    })
    result = build_enrollment_forecast(df, years_ahead=3)
    assert result["historical"] == {2020: 10, 2021: 15, 2022: 20}
    assert len(result["forecast"]) == 3
    assert result["slope_per_year"] > 0  # clearly increasing trend


def test_build_enrollment_forecast_requires_two_years() -> None:
    df = pd.DataFrame({"EnrollmentDate": pd.to_datetime(["2022-01-01"] * 5)})
    with pytest.raises(ValueError):
        build_enrollment_forecast(df)


def test_build_enrollment_forecast_excludes_partial_final_year() -> None:
    df = pd.DataFrame({
        "EnrollmentDate": pd.to_datetime(
            ["2020-06-01"] * 10 + ["2021-06-01"] * 10 + ["2022-06-01"] * 10 + ["2023-03-01"] * 2
        )
    })
    result = build_enrollment_forecast(df)
    assert result["excluded_partial_year"] == 2023
