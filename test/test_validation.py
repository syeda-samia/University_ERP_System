"""Tests for src/validation.py: file reading and upload validation."""

import pandas as pd
import pytest

from src.validation import (
    UnsupportedFileTypeError,
    clean_and_coerce,
    read_uploaded_file,
    validate_upload,
)


def test_read_uploaded_file_rejects_unsupported_extension() -> None:
    with pytest.raises(UnsupportedFileTypeError):
        read_uploaded_file(b"whatever", "notes.txt")


def test_read_uploaded_file_rejects_empty_csv() -> None:
    with pytest.raises(ValueError):
        read_uploaded_file(b"col1,col2\n", "empty.csv")


def test_read_uploaded_file_parses_csv() -> None:
    df = read_uploaded_file(b"StudentID,CGPA\nSTU1,3.0\nSTU2,2.5\n", "students.csv")
    assert len(df) == 2
    assert list(df.columns) == ["StudentID", "CGPA"]


def test_validate_upload_valid_row_passes(valid_student_row: dict) -> None:
    df = pd.DataFrame(valid_student_row)
    report = validate_upload(df)
    assert report.is_valid
    assert set(report.available_modules) == {
        "student_risk", "dropout", "fee_default", "gpa", "recommendation", "enrollment_forecast",
    }


def test_validate_upload_missing_student_id_errors() -> None:
    df = pd.DataFrame({"CGPA": [3.0, 2.5]})
    report = validate_upload(df)
    assert not report.is_valid
    assert any("StudentID" in e for e in report.errors)


def test_validate_upload_duplicate_student_id_errors(valid_student_row: dict) -> None:
    row = valid_student_row.copy()
    df = pd.concat([pd.DataFrame(row), pd.DataFrame(row)], ignore_index=True)
    report = validate_upload(df)
    assert not report.is_valid
    assert any("duplicate" in e.lower() for e in report.errors)


def test_validate_upload_partial_columns_skips_some_modules(valid_student_row: dict) -> None:
    df = pd.DataFrame(valid_student_row)[["StudentID", "CGPA", "AttendancePercentage", "AssignmentAvg", "QuizAvg", "ExamAvg"]]
    report = validate_upload(df)
    assert report.is_valid  # gpa module alone is enough to be valid
    assert report.available_modules == ["gpa"]
    assert any("student_risk" in w for w in report.warnings)


def test_validate_upload_no_usable_modules_errors() -> None:
    df = pd.DataFrame({"StudentID": ["S1", "S2"], "SomeUnrelatedColumn": [1, 2]})
    report = validate_upload(df)
    assert not report.is_valid


def test_validate_upload_out_of_range_warns(valid_student_row: dict) -> None:
    row = valid_student_row.copy()
    row["CGPA"] = [9.9]  # way outside 0-4
    df = pd.DataFrame(row)
    report = validate_upload(df)
    assert report.is_valid
    assert any("CGPA" in w and "range" in w for w in report.warnings)


def test_validate_upload_unknown_category_warns(valid_student_row: dict) -> None:
    row = valid_student_row.copy()
    row["Gender"] = ["NonBinary"]
    df = pd.DataFrame(row)
    report = validate_upload(df)
    assert report.is_valid
    assert any("Gender" in w for w in report.warnings)


def test_clean_and_coerce_clips_out_of_range(valid_student_row: dict) -> None:
    row = valid_student_row.copy()
    row["CGPA"] = [9.9]
    df = pd.DataFrame(row)
    cleaned = clean_and_coerce(df)
    assert cleaned["CGPA"].iloc[0] == 4.0


def test_clean_and_coerce_fills_missing_numeric(valid_student_row: dict) -> None:
    row = valid_student_row.copy()
    row["CGPA"] = [None]
    df = pd.DataFrame(row)
    cleaned = clean_and_coerce(df)
    assert not pd.isna(cleaned["CGPA"].iloc[0])
