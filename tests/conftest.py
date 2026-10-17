"""Shared fixtures for the test suite.

Most tests use a small SYNTHETIC dataset built here. It exists only to
exercise the code and is never presented as real student data. Tests that
need the real dataset or the saved model skip themselves if those files
are missing.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

from src.data_loader import RAW_DATA_DIR, TARGET_COLUMN, load_student_data
from src.predict import CATEGORICAL_VALUES, FEATURE_COLUMNS, MODEL_PATH, NUMERIC_RANGES
from src.preprocessing import build_preprocessor, identify_column_types
from src.train import build_model_pipeline


def make_students(n: int = 80, seed: int = 0) -> pd.DataFrame:
    """Build a small valid dataset with the same columns as the real one.

    Categorical values cycle through every allowed category, so each
    category is guaranteed to appear. G3 follows G2 closely and the first
    five students get a final grade of 0.
    """
    rng = np.random.default_rng(seed)
    data = {}
    for offset, column in enumerate(FEATURE_COLUMNS):
        if column in NUMERIC_RANGES:
            low, high = NUMERIC_RANGES[column]
            if column == "absences":
                high = 20  # keep the synthetic values modest
            data[column] = rng.integers(low, high + 1, size=n)
        else:
            values = CATEGORICAL_VALUES[column]
            data[column] = [values[(i + offset) % len(values)] for i in range(n)]

    df = pd.DataFrame(data)
    noise = rng.integers(-1, 2, size=n)
    df[TARGET_COLUMN] = np.clip(df["G2"] + noise, 0, 20)
    df.loc[df.index[:5], TARGET_COLUMN] = 0
    return df


@pytest.fixture
def students_df() -> pd.DataFrame:
    """A fresh synthetic dataset (32 features + G3)."""
    return make_students()


@pytest.fixture
def valid_record() -> dict:
    """One valid student: the first row of student-mat.csv (without G3)."""
    return {
        "school": "GP", "sex": "F", "age": 18, "address": "U",
        "famsize": "GT3", "Pstatus": "A", "Medu": 4, "Fedu": 4,
        "Mjob": "at_home", "Fjob": "teacher", "reason": "course",
        "guardian": "mother", "traveltime": 2, "studytime": 2, "failures": 0,
        "schoolsup": "yes", "famsup": "no", "paid": "no", "activities": "no",
        "nursery": "yes", "higher": "yes", "internet": "no", "romantic": "no",
        "famrel": 4, "freetime": 3, "goout": 4, "Dalc": 1, "Walc": 1,
        "health": 3, "absences": 6, "G1": 5, "G2": 6,
    }


@pytest.fixture
def small_model(students_df):
    """A small fitted pipeline (preprocessing + tiny Random Forest)."""
    X = students_df.drop(columns=[TARGET_COLUMN])
    y = students_df[TARGET_COLUMN]
    numeric_cols, categorical_cols = identify_column_types(X)
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    pipeline = build_model_pipeline(
        preprocessor, RandomForestRegressor(n_estimators=10, random_state=0)
    )
    pipeline.fit(X, y)
    return pipeline


@pytest.fixture
def real_students() -> pd.DataFrame:
    """The real dataset, or skip the test if it is not in data/raw/."""
    if not (RAW_DATA_DIR / "student-mat.csv").exists():
        pytest.skip("data/raw/student-mat.csv not found")
    return load_student_data("mat")


@pytest.fixture
def real_model_path():
    """Path of the real saved model, or skip the test if it does not exist."""
    if not MODEL_PATH.exists():
        pytest.skip("models/student_performance_model.joblib not found")
    return MODEL_PATH