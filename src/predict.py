"""Load the saved model pipeline and generate validated predictions.

The model is trained and saved by `src/save_model.py`. This module never
retrains: the Streamlit app only imports from here.

Security note: joblib files can execute code when loaded. Only load model
files that you created yourself or obtained from a trusted source.
"""

import json
import warnings
from numbers import Real

import joblib
import numpy as np
import pandas as pd
import sklearn

from src.data_loader import PROJECT_ROOT

MODEL_PATH = PROJECT_ROOT / "models" / "student_performance_model.joblib"
METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"
GRADE_MIN, GRADE_MAX = 0, 20

YES_NO = ["yes", "no"]
JOBS = ["teacher", "health", "services", "at_home", "other"]

# Input columns in the same order as the training data (G3 is the target).
FEATURE_COLUMNS = [
    "school", "sex", "age", "address", "famsize", "Pstatus", "Medu", "Fedu",
    "Mjob", "Fjob", "reason", "guardian", "traveltime", "studytime", "failures",
    "schoolsup", "famsup", "paid", "activities", "nursery", "higher",
    "internet", "romantic", "famrel", "freetime", "goout", "Dalc", "Walc",
    "health", "absences", "G1", "G2",
]

# Whole-number features and their allowed (min, max), from the dataset
# documentation. 'failures' uses 0-3, the range observed in student-mat.csv.
NUMERIC_RANGES = {
    "age": (15, 22), "Medu": (0, 4), "Fedu": (0, 4),
    "traveltime": (1, 4), "studytime": (1, 4), "failures": (0, 3),
    "famrel": (1, 5), "freetime": (1, 5), "goout": (1, 5),
    "Dalc": (1, 5), "Walc": (1, 5), "health": (1, 5),
    "absences": (0, 93), "G1": (0, 20), "G2": (0, 20),
}

CATEGORICAL_VALUES = {
    "school": ["GP", "MS"], "sex": ["F", "M"], "address": ["U", "R"],
    "famsize": ["LE3", "GT3"], "Pstatus": ["T", "A"],
    "Mjob": JOBS, "Fjob": JOBS,
    "reason": ["home", "reputation", "course", "other"],
    "guardian": ["mother", "father", "other"],
    "schoolsup": YES_NO, "famsup": YES_NO, "paid": YES_NO,
    "activities": YES_NO, "nursery": YES_NO, "higher": YES_NO,
    "internet": YES_NO, "romantic": YES_NO,
}


class InputValidationError(ValueError):
    """Raised when input data does not match the expected format."""


class ModelNotFoundError(FileNotFoundError):
    """Raised when the saved model file does not exist."""


def _validate_value(column: str, value):
    """Check one value. Returns (clean_value, error_message_or_None)."""
    if column in NUMERIC_RANGES:
        low, high = NUMERIC_RANGES[column]
        if isinstance(value, bool) or not isinstance(value, Real):
            return None, f"'{column}' must be a number, got {value!r}"
        if not np.isfinite(value):
            return None, f"'{column}' must be a finite number, got {value!r}"
        if float(value) != int(value):
            return None, f"'{column}' must be a whole number, got {value!r}"
        if not low <= value <= high:
            return None, f"'{column}' must be between {low} and {high}, got {value}"
        return int(value), None

    allowed = CATEGORICAL_VALUES[column]
    if value not in allowed:
        return None, f"'{column}' must be one of {allowed}, got {value!r}"
    return value, None


def validate_record(record: dict) -> dict:
    """Validate one student's features and return a cleaned copy.

    Raises:
        InputValidationError: listing every problem found.
    """
    problems = []
    missing = [c for c in FEATURE_COLUMNS if c not in record]
    unexpected = [c for c in record if c not in FEATURE_COLUMNS]
    if missing:
        problems.append(f"Missing columns: {missing}")
    if unexpected:
        problems.append(f"Unexpected columns: {unexpected}")

    cleaned = {}
    for column in FEATURE_COLUMNS:
        if column in record:
            value, error = _validate_value(column, record[column])
            if error:
                problems.append(error)
            else:
                cleaned[column] = value

    if problems:
        raise InputValidationError("; ".join(problems))
    return cleaned


def validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Validate many students at once. Returns a clean DataFrame."""
    if df.empty:
        raise InputValidationError("No rows to predict.")

    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    unexpected = [c for c in df.columns if c not in FEATURE_COLUMNS]
    if missing or unexpected:
        raise InputValidationError(
            f"Missing columns: {missing}; Unexpected columns: {unexpected}"
        )

    rows, errors = [], []
    for position, record in enumerate(df.to_dict(orient="records")):
        try:
            rows.append(validate_record(record))
        except InputValidationError as exc:
            errors.append(f"Row {position}: {exc}")
    if errors:
        raise InputValidationError("\n".join(errors))
    return pd.DataFrame(rows, columns=FEATURE_COLUMNS)


def load_metadata() -> dict | None:
    """Return the saved model metadata, or None if the file is missing."""
    if not METADATA_PATH.exists():
        return None
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))


def load_model(model_path=MODEL_PATH):
    """Load the saved pipeline and check it matches this module's schema."""
    if not model_path.exists():
        raise ModelNotFoundError(
            f"Model file not found at {model_path}. "
            "Create it with: python -m src.save_model"
        )
    pipeline = joblib.load(model_path)

    expected = list(getattr(pipeline, "feature_names_in_", []))
    if sorted(expected) != sorted(FEATURE_COLUMNS):
        raise ValueError(
            "The saved model expects different input columns than predict.py. "
            "Re-create it with: python -m src.save_model"
        )

    metadata = load_metadata()
    if metadata and metadata.get("sklearn_version") != sklearn.__version__:
        warnings.warn(
            f"Model was saved with scikit-learn {metadata.get('sklearn_version')} "
            f"but {sklearn.__version__} is installed. Predictions may be unreliable; "
            "re-create the model with: python -m src.save_model"
        )
    return pipeline


def predict_grades(records, model=None) -> np.ndarray:
    """Predict final grades for one or more students.

    Args:
        records: a dict (one student) or a DataFrame (many students) with
            exactly the columns in FEATURE_COLUMNS.
        model: an already-loaded pipeline; loaded from disk if None.

    Returns:
        Predicted G3 values, clipped to the valid 0-20 range.
    """
    if isinstance(records, dict):
        # validate_record reports every problem at once, including bad values
        records = pd.DataFrame([validate_record(records)])
    clean = validate_dataframe(records)
    if model is None:
        model = load_model()
    return np.clip(model.predict(clean), GRADE_MIN, GRADE_MAX)


def predict_grade(record: dict, model=None) -> float:
    """Predict the final grade for a single student."""
    return float(predict_grades(record, model)[0])


if __name__ == "__main__":
    from src.data_loader import TARGET_COLUMN, load_student_data

    model = load_model()
    students = load_student_data("mat")

    # Smoke test only: this row may have been in the training data, so this
    # says nothing about accuracy. Real accuracy was measured in Phase 7.
    sample = students.drop(columns=[TARGET_COLUMN]).iloc[0].to_dict()
    prediction = predict_grade(sample, model)
    print(f"Sample student: predicted G3 = {prediction:.2f} "
          f"(recorded G3 = {students[TARGET_COLUMN].iloc[0]})")

    # Invalid input should be rejected with a clear message
    bad = dict(sample)
    bad["age"] = 40
    bad["sex"] = "X"
    bad["extra_column"] = 1
    del bad["G2"]
    try:
        predict_grade(bad, model)
    except InputValidationError as exc:
        print("\nInvalid input was rejected, as intended:")
        print(" ", exc)