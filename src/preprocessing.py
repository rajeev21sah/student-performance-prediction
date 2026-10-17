"""Preprocessing utilities for the Student Performance project.

Key rule: every transformation that learns from data (imputer medians,
scaler mean/std, encoder categories) is fitted on the TRAINING set only.
"""

from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data_loader import TARGET_COLUMN, load_student_data

RANDOM_STATE = 42
TEST_SIZE = 0.2
EARLY_GRADE_COLUMNS = ["G1", "G2"]  # earlier grades from the same course
GRADE_MIN, GRADE_MAX = 0, 20


@dataclass
class PreparedData:
    """Everything later phases need, bundled together."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    numeric_cols: list
    categorical_cols: list
    preprocessor: ColumnTransformer  # NOT fitted yet


def validate_data(df: pd.DataFrame) -> None:
    """Raise ValueError if the raw data is not in the expected form."""
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' is missing.")
    if df.empty:
        raise ValueError("The dataset is empty.")
    if df[TARGET_COLUMN].isna().any():
        raise ValueError("The target column contains missing values.")
    out_of_range = ~df[TARGET_COLUMN].between(GRADE_MIN, GRADE_MAX)
    if out_of_range.any():
        raise ValueError(
            f"{int(out_of_range.sum())} target values are outside "
            f"[{GRADE_MIN}, {GRADE_MAX}]."
        )


def split_features_target(df: pd.DataFrame, *, include_early_grades: bool):
    """Separate the input features (X) from the target (y).

    Args:
        df: Raw dataset.
        include_early_grades: If False, G1 and G2 are dropped, which models
            an early-warning scenario where those grades do not exist yet.
    """
    validate_data(df)
    X = df.drop(columns=[TARGET_COLUMN])
    if not include_early_grades:
        X = X.drop(columns=EARLY_GRADE_COLUMNS)
    y = df[TARGET_COLUMN]
    return X, y


def identify_column_types(X: pd.DataFrame):
    """Return (numeric_columns, categorical_columns) as lists."""
    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    categorical_cols = X.select_dtypes(exclude="number").columns.tolist()
    return numeric_cols, categorical_cols


def build_preprocessor(numeric_cols: list, categorical_cols: list) -> ColumnTransformer:
    """Build the (unfitted) preprocessing pipeline.

    Numeric:     median imputation -> standardization
    Categorical: most-frequent imputation -> one-hot encoding
                 (binary columns become a single 0/1 column)
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",  # unseen category -> all zeros
                    drop="if_binary",
                    sparse_output=False,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ]
    )


def prepare_data(subject: str = "mat", *, include_early_grades: bool) -> PreparedData:
    """Load data, split it, and build the unfitted preprocessor."""
    df = load_student_data(subject)
    X, y = split_features_target(df, include_early_grades=include_early_grades)
    numeric_cols, categorical_cols = identify_column_types(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=(y == 0),  # keep the share of zero grades similar in both sets
    )

    return PreparedData(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        preprocessor=build_preprocessor(numeric_cols, categorical_cols),
    )


def check_preprocessing(data: PreparedData) -> None:
    """Fit on TRAIN only, transform both sets, and sanity-check the result."""
    X_train_t = data.preprocessor.fit_transform(data.X_train)
    X_test_t = data.preprocessor.transform(data.X_test)  # transform only!

    n_features = X_train_t.shape[1]
    assert X_train_t.shape[0] == len(data.X_train), "Train row count changed"
    assert X_test_t.shape[0] == len(data.X_test), "Test row count changed"
    assert X_test_t.shape[1] == n_features, "Train/test column mismatch"
    assert not pd.DataFrame(X_train_t).isna().any().any(), "NaN in train output"
    assert not pd.DataFrame(X_test_t).isna().any().any(), "NaN in test output"

    names = data.preprocessor.get_feature_names_out()
    print(f"Features after preprocessing: {n_features}")
    print("First 8 feature names:", list(names[:8]))

    # Scaled numeric columns should be ~mean 0, ~std 1 on TRAIN only
    numeric_train = pd.DataFrame(
        X_train_t[:, : len(data.numeric_cols)], columns=data.numeric_cols
    )
    print("\nTrain numeric means (should be ~0):",
          numeric_train.mean().abs().max().round(6))
    print("Train numeric stds  (should be ~1):",
          numeric_train.std(ddof=0).round(3).unique())


if __name__ == "__main__":
    for include in (True, False):
        label = "WITH G1/G2" if include else "WITHOUT G1/G2"
        print("=" * 60)
        print(label)
        print("=" * 60)
        prepared = prepare_data("mat", include_early_grades=include)
        print("X_train:", prepared.X_train.shape, "| X_test:", prepared.X_test.shape)
        print("Numeric columns:", len(prepared.numeric_cols),
              "| Categorical columns:", len(prepared.categorical_cols))
        print("Zero grades in train:", int((prepared.y_train == 0).sum()),
              "| in test:", int((prepared.y_test == 0).sum()))
        check_preprocessing(prepared)
        print()