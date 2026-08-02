"""Excel/CSV upload reading and validation against the canonical student schema.

Design principle: fail loudly on things that would silently corrupt
predictions (missing StudentID, no usable columns at all), but degrade
gracefully on everything else (an optional column missing just means one
fewer module runs; an out-of-range value gets clipped and counted, not
rejected) -- so "any university" can use this with whatever subset of data
they actually have.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import pandas as pd

from src.schema import (
    ALL_TEMPLATE_COLUMNS,
    CATEGORICAL_COLUMNS,
    DATE_COLUMNS,
    IDENTIFIER_COLUMNS,
    MODULE_REQUIRED_COLUMNS,
    NUMERIC_COLUMNS,
)

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = (".csv", ".xlsx", ".xls")

# ============================================
# 1. COLUMN MAPPING FOR FLEXIBLE IMPORT
# ============================================

COLUMN_MAPPING = {
    # Student ID variations
    "StudentID": ["StudentID", "Student ID", "ID", "Student_Id", "student_id", "Student Id"],
    "CGPA": ["CGPA", "Cgpa", "GPA", "Gpa", "Current CGPA", "CurrentGPA"],
    "AttendancePercentage": ["AttendancePercentage", "Attendance", "Attendance %", "Attendance%", "Att %"],
    "Department": ["Department", "Dept", "Major", "Program", "Faculty"],
    "Gender": ["Gender", "Sex"],
    "YearLevel": ["YearLevel", "Year", "Year Level", "Semester", "Term", "Academic Year"],
    "Status": ["Status", "Enrollment Status", "Student Status", "Enrollment"],
    "City": ["City", "Location", "Hometown", "City/Town"],
    "Scholarship": ["Scholarship", "Scholarship Status", "Financial Aid", "Scholarship?"],
    "FeesPaid": ["FeesPaid", "Fees Paid", "Fee Status", "Tuition Paid", "Fee Paid"],
    "LMSActivity": ["LMSActivity", "LMS Activity", "Online Activity", "VLE Activity", "LMS"],
    "QuizAvg": ["QuizAvg", "Quiz Average", "Quiz Score", "Avg Quiz"],
    "AssignmentAvg": ["AssignmentAvg", "Assignment Average", "Assignment Score", "Avg Assignment"],
    "ExamAvg": ["ExamAvg", "Exam Average", "Exam Score", "Avg Exam"],
    "LabAvg": ["LabAvg", "Lab Average", "Lab Score", "Avg Lab"],
    "AssignmentsSubmitted": ["AssignmentsSubmitted", "Assignments Submitted", "Submitted"],
    "AssignmentsTotal": ["AssignmentsTotal", "Assignments Total", "Total Assignments"],
    "Backlogs": ["Backlogs", "Backlog Courses", "Failed Courses", "Backlog"],
    "EnrollmentDate": ["EnrollmentDate", "Enrollment Date", "Admission Date", "Enrolled Date"]
}


def detect_columns(df: pd.DataFrame) -> Dict[str, str]:
    """
    Auto-detect columns by matching with mapping
    
    Returns:
        {standard_name: actual_column_name}
    """
    detected = {}
    actual_cols = [str(col).strip() for col in df.columns]
    
    for standard, variants in COLUMN_MAPPING.items():
        for variant in variants:
            # Case-insensitive match
            for actual in actual_cols:
                if actual.lower() == variant.lower():
                    detected[standard] = actual
                    break
            if standard in detected:
                break
    
    return detected


def map_columns_to_schema(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    """
    Rename columns to standard ERP schema
    
    Args:
        df: Original dataframe
        mapping: {standard_name: actual_column_name}
    
    Returns:
        Dataframe with renamed columns
    """
    df = df.copy()
    rename_dict = {v: k for k, v in mapping.items()}
    df.rename(columns=rename_dict, inplace=True)
    return df


def get_sample_template() -> pd.DataFrame:
    """Generate sample template for download"""
    sample_data = {
        "StudentID": ["S001", "S002", "S003", "S004", "S005"],
        "CGPA": [3.5, 2.8, 1.9, 3.2, 2.5],
        "AttendancePercentage": [85, 70, 45, 90, 60],
        "Department": ["Computer Science", "Business", "Engineering", "Mathematics", "Physics"],
        "Gender": ["Female", "Male", "Male", "Female", "Male"],
        "YearLevel": ["Junior", "Sophomore", "Freshman", "Senior", "Sophomore"],
        "Status": ["Active", "Active", "At Risk", "Active", "Probation"],
        "City": ["Karachi", "Lahore", "Islamabad", "Peshawar", "Quetta"],
        "Scholarship": ["Yes", "No", "No", "Yes", "No"],
        "FeesPaid": [1, 0, 0, 1, 1],
        "LMSActivity": [75, 60, 30, 85, 55],
        "QuizAvg": [80, 70, 45, 88, 65],
        "AssignmentAvg": [85, 65, 40, 82, 60],
        "ExamAvg": [75, 60, 35, 78, 58],
        "LabAvg": [80, 70, 50, 85, 65],
        "AssignmentsSubmitted": [8, 6, 3, 9, 7],
        "AssignmentsTotal": [10, 10, 10, 10, 10],
        "Backlogs": [0, 2, 5, 0, 3],
        "EnrollmentDate": ["2023-09-01", "2023-09-01", "2024-01-15", "2022-09-01", "2023-09-01"]
    }
    return pd.DataFrame(sample_data)


# ============================================
# 2. USER-FRIENDLY VALIDATION MESSAGES
# ============================================

class UnsupportedFileTypeError(ValueError):
    """Raised when an uploaded file's extension isn't csv/xlsx/xls."""


@dataclass
class ValidationReport:
    """Result of validating an uploaded student dataframe.

    Attributes:
        errors: Blocking problems -- the file cannot be processed at all.
        warnings: Non-blocking problems -- processing continues, with the
            affected rows/columns/modules called out.
        available_modules: Module keys (from schema.MODULE_REQUIRED_COLUMNS)
            whose required columns are all present.
        missing_columns: Template columns not present in the upload at all.
        detected_columns: Columns that were auto-detected and mapped.
        mapped_columns: Original -> Standard column mapping.
    """

    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    available_modules: List[str] = field(default_factory=list)
    missing_columns: List[str] = field(default_factory=list)
    detected_columns: List[str] = field(default_factory=list)
    mapped_columns: Dict[str, str] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return not self.errors

    def get_user_message(self) -> str:
        """Generate professional, user-friendly validation message"""
        
        # Blocking errors
        if self.errors:
            msg = "❌ **Validation Failed**\n\n"
            msg += "The uploaded file doesn't match the required format.\n\n"
            msg += "**Issues found:**\n"
            for err in self.errors:
                msg += f"• {err}\n"
            
            msg += "\n📥 **Download the sample template** from the sidebar to see the required format."
            return msg
        
        # Success with professional report
        msg = "✅ **File Validated Successfully!**\n\n"
        
        # Show detected/mapped columns
        if self.detected_columns:
            msg += "**Detected columns:**\n"
            for col in self.detected_columns:
                msg += f"• {col}\n"
            msg += "\n"
        
        # Show missing optional columns (if any)
        if self.missing_columns:
            required_cols = ["StudentID", "CGPA", "AttendancePercentage", "Department"]
            missing_required = [c for c in required_cols if c in self.missing_columns]
            missing_optional = [c for c in self.missing_columns if c not in required_cols]
            
            if missing_required:
                msg += "⚠️ **Some required columns are missing:**\n"
                for col in missing_required:
                    msg += f"• `{col}`\n"
                msg += "\n*Please ensure your file contains these columns.*\n\n"
            
            if missing_optional:
                msg += "ℹ️ **Optional columns not found:**\n"
                for col in missing_optional[:5]:  # Show first 5
                    msg += f"• `{col}`\n"
                if len(missing_optional) > 5:
                    msg += f"• ... and {len(missing_optional) - 5} more\n"
                msg += "\n*These modules will be skipped if columns are missing.*\n\n"
        
        # Show available modules
        if self.available_modules:
            module_names = {
                "student_risk": "Student Risk Prediction",
                "dropout": "Dropout Prediction",
                "fee_default": "Fee Default Prediction",
                "gpa": "GPA Prediction",
                "recommendations": "Recommendation Engine",
                "enrollment_forecast": "Enrollment Forecasting"
            }
            msg += "**Available modules:**\n"
            for module in self.available_modules:
                name = module_names.get(module, module)
                msg += f"✅ {name}\n"
            msg += "\n"
        
        # Show warnings (if any)
        if self.warnings:
            msg += "⚠️ **Warnings (non-blocking):**\n"
            for warn in self.warnings[:3]:  # Show first 3 warnings
                msg += f"• {warn}\n"
            if len(self.warnings) > 3:
                msg += f"• ... and {len(self.warnings) - 3} more warnings\n"
            msg += "\n"
        
        msg += "✅ **Ready for prediction!** Click 'Run Predictions' to proceed."
        return msg


# ============================================
# 3. CORE VALIDATION FUNCTION (Updated)
# ============================================

def read_uploaded_file(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Parse an uploaded CSV/XLSX file into a dataframe.

    Args:
        file_bytes: Raw file content.
        filename: Original filename, used only to detect the format.

    Returns:
        The parsed dataframe (columns not yet validated).

    Raises:
        UnsupportedFileTypeError: If the extension isn't csv/xlsx/xls.
        ValueError: If the file can't be parsed (corrupt/empty/wrong format).
    """
    lower_name = filename.lower()
    if not lower_name.endswith(SUPPORTED_EXTENSIONS):
        raise UnsupportedFileTypeError(
            f"Unsupported file type for '{filename}'. Please upload a .csv, .xlsx, or .xls file."
        )

    buffer = io.BytesIO(file_bytes)
    try:
        if lower_name.endswith(".csv"):
            df = pd.read_csv(buffer)
        else:
            df = pd.read_excel(buffer)
    except Exception as exc:
        raise ValueError(f"Could not read '{filename}': the file may be corrupt or in an unexpected format ({exc}).") from exc

    if df.empty:
        raise ValueError(f"'{filename}' contains no rows.")

    df.columns = df.columns.str.strip()
    return df


def validate_upload(df: pd.DataFrame, strict_mode: bool = False) -> ValidationReport:
    """Validate an uploaded student dataframe against the canonical schema.

    Args:
        df: Raw uploaded dataframe (post `read_uploaded_file`).
        strict_mode: If True, require all columns exactly match ERP schema.
                     If False, auto-detect and map columns.

    Returns:
        A ValidationReport with user-friendly messages.
    """
    report = ValidationReport()
    present_columns = set(df.columns)
    
    # ============================================
    # STEP 1: Auto-detect columns (if not strict mode)
    # ============================================
    if not strict_mode:
        mapping = detect_columns(df)
        report.mapped_columns = mapping
        
        # Add detected columns to report
        for std, actual in mapping.items():
            report.detected_columns.append(f"{std} → '{actual}'")
        
        # Map the dataframe for validation
        df = map_columns_to_schema(df, mapping)
        present_columns = set(df.columns)
    
    # ============================================
    # STEP 2: Validate StudentID (MANDATORY)
    # ============================================
    if 'StudentID' not in present_columns:
        report.errors.append("Required column 'StudentID' is missing.")
        report.errors.append("Please ensure your file has a column for student identification.")
        return report
    
    if df['StudentID'].isna().any():
        report.errors.append("Column 'StudentID' has missing values -- every row must have a unique student ID.")
    
    if df['StudentID'].duplicated().any():
        n_dupes = int(df['StudentID'].duplicated().sum())
        report.errors.append(f"Column 'StudentID' has {n_dupes} duplicate value(s) -- student IDs must be unique.")
    
    if not report.is_valid:
        return report

    # ============================================
    # STEP 3: Check other required columns (only if strict_mode)
    # ============================================
    if strict_mode:
        required = ["StudentID", "CGPA", "AttendancePercentage", "Department"]
        for col in required:
            if col not in present_columns:
                report.errors.append(f"Required column '{col}' is missing.")
        
        if report.errors:
            return report
    
    # ============================================
    # STEP 4: Check optional columns
    # ============================================
    report.missing_columns = [c for c in ALL_TEMPLATE_COLUMNS if c not in present_columns]

    # ============================================
    # STEP 5: Numeric column validation (with warnings)
    # ============================================
    for col, (low, high) in NUMERIC_COLUMNS.items():
        if col not in present_columns:
            continue
        coerced = pd.to_numeric(df[col], errors="coerce")
        n_bad = int(coerced.isna().sum() - df[col].isna().sum())
        if n_bad > 0:
            report.warnings.append(f"Column '{col}' has {n_bad} non-numeric value(s); those rows will be treated as missing.")
        n_missing = int(coerced.isna().sum())
        if n_missing > 0:
            report.warnings.append(f"Column '{col}' has {n_missing} missing value(s); they will be filled with the median.")
        out_of_range = coerced.dropna()[(coerced.dropna() < low) | (coerced.dropna() > high)]
        if len(out_of_range) > 0:
            report.warnings.append(
                f"Column '{col}' has {len(out_of_range)} value(s) outside the expected range ({low}-{high}); they will be clipped."
            )

    # ============================================
    # STEP 6: Categorical column validation
    # ============================================
    for col, allowed in CATEGORICAL_COLUMNS.items():
        if col not in present_columns:
            continue
        unknown = set(df[col].dropna().astype(str).unique()) - set(allowed)
        if unknown:
            # Only show first 5 unknown values to avoid clutter
            show_unknown = sorted(unknown)[:5]
            if len(unknown) > 5:
                report.warnings.append(
                    f"Column '{col}' has {len(unknown)} unrecognized values (e.g. {show_unknown}); "
                    "they will be mapped to the most common category."
                )
            else:
                report.warnings.append(
                    f"Column '{col}' has unrecognized value(s) {show_unknown}; "
                    "they will be mapped to the most common category."
                )

    # ============================================
    # STEP 7: Date column validation
    # ============================================
    for col in DATE_COLUMNS:
        if col not in present_columns:
            continue
        parsed = pd.to_datetime(df[col], errors="coerce")
        n_bad = int(parsed.isna().sum() - df[col].isna().sum())
        if n_bad > 0:
            report.warnings.append(f"Column '{col}' has {n_bad} value(s) that aren't valid dates; they will be excluded from forecasting.")

    # ============================================
    # STEP 8: Module availability
    # ============================================
    for module, required_cols in MODULE_REQUIRED_COLUMNS.items():
        if all(c in present_columns for c in required_cols):
            report.available_modules.append(module)
        else:
            missing = [c for c in required_cols if c not in present_columns]
            report.warnings.append(f"Module '{module}' will be skipped -- missing column(s): {missing}.")

    # ============================================
    # STEP 9: Final check - at least one module should work
    # ============================================
    if not report.available_modules:
        report.errors.append(
            "None of the 6 prediction modules have enough columns to run. "
            "Please check the sample template for the required columns."
        )

    return report


def clean_and_coerce(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce, clip, and impute an already-validated dataframe for prediction.

    Assumes `validate_upload` has already been run and returned no errors.

    Args:
        df: Raw uploaded dataframe.

    Returns:
        A cleaned copy: numeric columns coerced/clipped/imputed, unknown
        categorical values left as-is (handled downstream by the fallback
        encoders in `src.encoding`), dates parsed.
    """
    df = df.copy()
    if "StudentID" in df.columns:
        df["StudentID"] = df["StudentID"].astype(str).str.strip()

    for col, (low, high) in NUMERIC_COLUMNS.items():
        if col not in df.columns:
            continue
        coerced = pd.to_numeric(df[col], errors="coerce")
        if coerced.isna().any():
            median = coerced.median()
            coerced = coerced.fillna(median if pd.notna(median) else (low + high) / 2)
        df[col] = coerced.clip(lower=low, upper=high)

    for col in DATE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    return df


# ============================================
# 4. CONVENIENCE FUNCTION FOR STREAMLIT
# ============================================

def validate_uploaded_file(uploaded_file, strict_mode: bool = False) -> Tuple[bool, pd.DataFrame, str]:
    """
    Complete validation pipeline for uploaded files
    
    Returns:
        (is_valid, mapped_dataframe, user_message)
    """
    try:
        # Read file
        df = read_uploaded_file(uploaded_file.getvalue(), uploaded_file.name)
        
        # Validate
        report = validate_upload(df, strict_mode=strict_mode)
        
        # Get user message
        message = report.get_user_message()
        
        if report.is_valid:
            # Clean data
            cleaned_df = clean_and_coerce(df)
            return True, cleaned_df, message
        else:
            return False, None, message
            
    except UnsupportedFileTypeError as e:
        return False, None, f"❌ {str(e)}"
    except ValueError as e:
        return False, None, f"❌ {str(e)}"
    except Exception as e:
        logger.exception("Unexpected error during file validation")
        return False, None, f"❌ Unexpected error: {str(e)}"
