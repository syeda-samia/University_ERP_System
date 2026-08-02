"""Shared pytest fixtures for the University ERP System test suite."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BASE_DIR = Path(__file__).resolve().parent.parent


@pytest.fixture
def raw_sheets() -> dict:
    """Small synthetic Students/Attendance/ExamRecords sheets, shaped like
    the real university_dataset.xlsx, for isolated unit tests."""
    rng = np.random.default_rng(7)
    n_students = 40
    student_ids = [f"STU{i:04d}" for i in range(1, n_students + 1)]

    students = pd.DataFrame({
        "StudentID": student_ids,
        "Gender": rng.choice(["Male", "Female"], n_students),
        "Department": rng.choice(["Computer Science", "Economics", "Biology"], n_students),
        "YearLevel": rng.choice(["Freshman", "Sophomore", "Junior", "Senior"], n_students),
        "EnrollmentDate": pd.to_datetime(rng.choice(pd.date_range("2020-01-01", "2023-12-31"), n_students)),
        "CGPA": rng.uniform(1.5, 4.0, n_students).round(2),
        "Status": rng.choice(["Active", "Inactive", "Graduated"], n_students),
        "City": rng.choice(["Karachi", "Lahore", "Quetta"], n_students),
        "Scholarship": rng.choice(["Yes", "No"], n_students),
    })

    attendance_rows = []
    for sid in student_ids:
        for i in range(10):
            attendance_rows.append({
                "StudentID": sid, "CourseID": f"CRS{i % 5:03d}",
                "Status": rng.choice(["Present", "Absent", "Late"], p=[0.8, 0.15, 0.05]),
            })
    attendance = pd.DataFrame(attendance_rows)

    exam_rows = []
    for sid in student_ids:
        for i, exam_type in enumerate(["Midterm", "Final", "Quiz", "Assignment", "Lab Exam"]):
            pct = rng.uniform(30, 100)
            exam_rows.append({
                "StudentID": sid, "CourseID": f"CRS{i % 5:03d}", "ExamType": exam_type,
                "Percentage": pct, "Grade": "F" if pct < 50 else "B",
            })
    exam_records = pd.DataFrame(exam_rows)

    return {"Students": students, "Attendance": attendance, "ExamRecords": exam_records}


@pytest.fixture
def model_path() -> Path:
    """Path to the trained consolidated model bundle (may not exist until
    `src.train` has been run)."""
    return BASE_DIR / "models" / "erp_models.pkl"


@pytest.fixture
def sample_template_path() -> Path:
    return BASE_DIR / "sample_data" / "sample_students_template.xlsx"


@pytest.fixture
def valid_student_row() -> dict:
    """A single valid, complete student row covering every module's required columns."""
    return {
        "StudentID": ["STU9001"],
        "CGPA": [3.2],
        "AttendancePercentage": [82.0],
        "AssignmentsSubmitted": [8],
        "AssignmentsTotal": [10],
        "Backlogs": [0],
        "QuizAvg": [75.0],
        "AssignmentAvg": [80.0],
        "ExamAvg": [78.0],
        "LabAvg": [85.0],
        "FeesPaid": [1],
        "LMSActivity": [70.0],
        "Gender": ["Male"],
        "YearLevel": ["Junior"],
        "Status": ["Active"],
        "Scholarship": ["Yes"],
        "Department": ["Computer Science"],
        "City": ["Karachi"],
        "EnrollmentDate": ["2022-09-01"],
    }
