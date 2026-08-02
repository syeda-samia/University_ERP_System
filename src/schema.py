"""Canonical student-data schema shared by the Excel/CSV upload template,
data validation, model training, and prediction.

Any university can use this system with their own data by filling in this
one flat, per-student template. Not every column is required for every
module — see MODULE_REQUIRED_COLUMNS. A university missing a given column
(e.g. no LMS, so no LMSActivity) simply won't get that one module's
predictions; the rest still run.
"""

from __future__ import annotations

from typing import Dict, List

# ---------------------------------------------------------------------------
# Column definitions: name -> (dtype, description, valid values or range)
# ---------------------------------------------------------------------------

NUMERIC_COLUMNS = {
    "CGPA": (0.0, 4.0),
    "AttendancePercentage": (0.0, 100.0),
    "AssignmentsSubmitted": (0, 1000),
    "AssignmentsTotal": (1, 1000),
    "Backlogs": (0, 50),
    "QuizAvg": (0.0, 100.0),
    "AssignmentAvg": (0.0, 100.0),
    "ExamAvg": (0.0, 100.0),
    "LabAvg": (0.0, 100.0),
    "FeesPaid": (0, 1),
    "LMSActivity": (0.0, 100.0),
}

CATEGORICAL_COLUMNS = {
    "Gender": ["Male", "Female"],
    "YearLevel": ["Freshman", "Sophomore", "Junior", "Senior"],
    "Status": ["Active", "Inactive", "Graduated"],
    "Scholarship": ["Yes", "No"],
    # Department and City are open-vocabulary (any university's own values);
    # unseen categories at inference are bucketed into "Other" rather than rejected.
}

OPEN_CATEGORICAL_COLUMNS = ["Department", "City"]

DATE_COLUMNS = ["EnrollmentDate"]

IDENTIFIER_COLUMNS = ["StudentID"]

ALL_TEMPLATE_COLUMNS: List[str] = (
    IDENTIFIER_COLUMNS
    + list(NUMERIC_COLUMNS.keys())
    + list(CATEGORICAL_COLUMNS.keys())
    + OPEN_CATEGORICAL_COLUMNS
    + DATE_COLUMNS
)

COLUMN_DESCRIPTIONS: Dict[str, str] = {
    "StudentID": "Unique student identifier (e.g. STU0001). Required.",
    "CGPA": "Cumulative GPA on a 0-4 scale.",
    "AttendancePercentage": "Overall class attendance, 0-100.",
    "AssignmentsSubmitted": "Number of assignments submitted this term.",
    "AssignmentsTotal": "Number of assignments assigned this term.",
    "Backlogs": "Number of currently failed/backlog courses.",
    "QuizAvg": "Average quiz score, 0-100.",
    "AssignmentAvg": "Average assignment score, 0-100.",
    "ExamAvg": "Average midterm/final exam score, 0-100.",
    "LabAvg": "Average lab exam score, 0-100.",
    "FeesPaid": "1 if current tuition fees are paid in full, else 0.",
    "LMSActivity": "LMS engagement score, 0-100 (e.g. logins/activity index).",
    "Gender": "Male or Female.",
    "YearLevel": "Freshman, Sophomore, Junior, or Senior.",
    "Status": "Current enrollment status: Active, Inactive, or Graduated.",
    "Scholarship": "Yes or No.",
    "Department": "Academic department (any value accepted).",
    "City": "Student's home city (any value accepted).",
    "EnrollmentDate": "Date the student first enrolled (YYYY-MM-DD).",
}

# ---------------------------------------------------------------------------
# Per-module column requirements: a module runs only if ALL of its required
# columns are present (and StudentID is always required).
# ---------------------------------------------------------------------------

MODULE_REQUIRED_COLUMNS: Dict[str, List[str]] = {
    "student_risk": ["CGPA", "AttendancePercentage", "AssignmentsSubmitted", "AssignmentsTotal", "Backlogs"],
    "dropout": ["AttendancePercentage", "CGPA", "FeesPaid", "LMSActivity"],
    "fee_default": ["Gender", "Department", "YearLevel", "Status", "City", "Scholarship", "CGPA"],
    "gpa": ["AttendancePercentage", "AssignmentAvg", "QuizAvg", "ExamAvg", "CGPA"],
    "recommendation": ["QuizAvg", "AssignmentAvg", "ExamAvg", "LabAvg"],
    "enrollment_forecast": ["EnrollmentDate"],
}

MODULE_LABELS: Dict[str, str] = {
    "student_risk": "Student Risk Prediction",
    "dropout": "Dropout Prediction",
    "fee_default": "Fee Default Prediction",
    "gpa": "GPA Prediction",
    "recommendation": "Recommendation Engine",
    "enrollment_forecast": "Enrollment Forecasting",
}
