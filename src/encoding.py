"""Simple, robust label encoding for open-vocabulary categorical columns
(Department, City, Gender, YearLevel, Status, Scholarship). Unseen categories
at inference are mapped to the most frequent training category rather than
raising an error, so a new university's data with different department names
degrades gracefully instead of crashing.
"""

from __future__ import annotations

from typing import Dict, List

import pandas as pd


def fit_label_encoders(df: pd.DataFrame, columns: List[str]) -> Dict[str, Dict[str, int]]:
    """Fit a category->int mapping per column, most-frequent category first (code 0).

    Args:
        df: Training dataframe.
        columns: Categorical column names to encode.

    Returns:
        Dict of column -> {category: code}. Code 0 is always the most
        frequent training category, used as the fallback for unseen values.
    """
    encoders = {}
    for col in columns:
        ordered_categories = df[col].astype(str).value_counts().index.tolist()
        encoders[col] = {cat: i for i, cat in enumerate(ordered_categories)}
    return encoders


def apply_label_encoders(
    df: pd.DataFrame, encoders: Dict[str, Dict[str, int]], columns: List[str]
) -> pd.DataFrame:
    """Apply fitted encoders, mapping unseen categories to code 0 (most frequent).

    Args:
        df: Dataframe to transform (not mutated).
        encoders: Output of `fit_label_encoders`.
        columns: Categorical column names to encode.

    Returns:
        Copy of `df` with each categorical column replaced by its integer code.
    """
    df = df.copy()
    for col in columns:
        mapping = encoders[col]
        df[col] = df[col].astype(str).map(mapping).fillna(0).astype(int)
    return df
