"""Tests for src/preprocessing.py."""

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from src.preprocessing import (
    EARLY_GRADE_COLUMNS,
    build_preprocessor,
    identify_column_types,
    prepare_data,
    split_features_target,
    validate_data,
)


def _fitted_preprocessor(students_df, include_early=True):
    """Split synthetic data, fit the preprocessor on train only."""
    X, _ = split_features_target(students_df, include_early_grades=include_early)
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=0)
    numeric_cols, categorical_cols = identify_column_types(X_train)
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    preprocessor.fit(X_train)
    return preprocessor, X, X_train, X_test, numeric_cols


# --- validate_data -----------------------------------------------------------


def test_valid_data_passes_validation(students_df):
    validate_data(students_df)  # must not raise


def test_missing_target_is_rejected(students_df):
    with pytest.raises(ValueError, match="Target column"):
        validate_data(students_df.drop(columns=["G3"]))


def test_empty_dataset_is_rejected(students_df):
    with pytest.raises(ValueError, match="empty"):
        validate_data(students_df.iloc[0:0])


def test_missing_target_values_are_rejected(students_df):
    students_df["G3"] = students_df["G3"].astype(float)
    students_df.loc[0, "G3"] = np.nan
    with pytest.raises(ValueError, match="missing values"):
        validate_data(students_df)


def test_out_of_range_target_is_rejected(students_df):
    students_df.loc[0, "G3"] = 25
    with pytest.raises(ValueError, match="outside"):
        validate_data(students_df)


# --- splitting features and target -------------------------------------------


def test_split_removes_target_and_keeps_early_grades(students_df):
    X, y = split_features_target(students_df, include_early_grades=True)
    assert "G3" not in X.columns
    assert set(EARLY_GRADE_COLUMNS) <= set(X.columns)
    assert X.shape[1] == 32
    assert y.equals(students_df["G3"])


def test_split_can_drop_early_grades(students_df):
    X, _ = split_features_target(students_df, include_early_grades=False)
    assert X.shape[1] == 30
    assert not set(EARLY_GRADE_COLUMNS) & set(X.columns)


def test_split_validates_the_data(students_df):
    with pytest.raises(ValueError):
        split_features_target(students_df.drop(columns=["G3"]), include_early_grades=True)


def test_column_types_are_identified(students_df):
    X, _ = split_features_target(students_df, include_early_grades=True)
    numeric_cols, categorical_cols = identify_column_types(X)
    assert len(numeric_cols) == 15
    assert len(categorical_cols) == 17
    assert "G2" in numeric_cols and "sex" in categorical_cols


# --- the preprocessing pipeline ----------------------------------------------


@pytest.mark.parametrize("include_early, expected_features", [(True, 45), (False, 43)])
def test_preprocessor_output_shape_and_no_missing_values(
    students_df, include_early, expected_features
):
    preprocessor, _, X_train, X_test, _ = _fitted_preprocessor(students_df, include_early)
    train_t = preprocessor.transform(X_train)
    test_t = preprocessor.transform(X_test)

    assert train_t.shape == (len(X_train), expected_features)
    assert test_t.shape == (len(X_test), expected_features)
    assert not np.isnan(train_t).any()
    assert not np.isnan(test_t).any()


def test_numeric_columns_are_standardized_on_training_data(students_df):
    preprocessor, _, X_train, _, numeric_cols = _fitted_preprocessor(students_df)
    block = preprocessor.transform(X_train)[:, : len(numeric_cols)]
    assert np.allclose(block.mean(axis=0), 0, atol=1e-8)
    assert np.allclose(block.std(axis=0), 1)


def test_scaler_is_fitted_on_training_data_only(students_df):
    preprocessor, X, X_train, X_test, numeric_cols = _fitted_preprocessor(students_df)
    scaler = preprocessor.named_transformers_["num"].named_steps["scaler"]

    assert np.allclose(scaler.mean_, X_train[numeric_cols].mean().to_numpy())
    assert not np.allclose(scaler.mean_, X[numeric_cols].mean().to_numpy())

    before = scaler.mean_.copy()
    preprocessor.transform(X_test)  # transforming test data must not change anything
    assert np.array_equal(before, scaler.mean_)


def test_missing_numeric_value_is_imputed(students_df):
    preprocessor, _, _, X_test, _ = _fitted_preprocessor(students_df)
    X_test = X_test.copy()
    X_test["age"] = X_test["age"].astype(float)
    X_test.loc[X_test.index[0], "age"] = np.nan

    assert not np.isnan(preprocessor.transform(X_test)).any()


@pytest.mark.filterwarnings("ignore:Found unknown categories")
def test_unknown_category_does_not_crash(students_df):
    preprocessor, _, _, X_test, _ = _fitted_preprocessor(students_df)
    X_test = X_test.copy()
    X_test["Mjob"] = "astronaut"

    output = preprocessor.transform(X_test)
    assert output.shape[0] == len(X_test)
    assert not np.isnan(output).any()


# --- prepare_data on the real dataset ----------------------------------------


@pytest.mark.parametrize("include_early, n_columns", [(True, 32), (False, 30)])
def test_prepare_data_shapes(real_students, include_early, n_columns):
    data = prepare_data("mat", include_early_grades=include_early)
    assert data.X_train.shape == (316, n_columns)
    assert data.X_test.shape == (79, n_columns)
    assert len(data.y_train) == 316 and len(data.y_test) == 79
    assert "G3" not in data.X_train.columns


def test_train_and_test_sets_do_not_overlap(real_students):
    data = prepare_data("mat", include_early_grades=True)
    assert set(data.X_train.index).isdisjoint(data.X_test.index)
    assert len(data.X_train) + len(data.X_test) == len(real_students)


def test_split_is_reproducible(real_students):
    first = prepare_data("mat", include_early_grades=True)
    second = prepare_data("mat", include_early_grades=True)
    pd.testing.assert_index_equal(first.X_train.index, second.X_train.index)
    pd.testing.assert_index_equal(first.X_test.index, second.X_test.index)


def test_zero_grades_are_spread_across_both_sets(real_students):
    data = prepare_data("mat", include_early_grades=True)
    total_zeros = int((data.y_train == 0).sum() + (data.y_test == 0).sum())
    assert total_zeros == int((real_students["G3"] == 0).sum())

    train_share = (data.y_train == 0).mean()
    test_share = (data.y_test == 0).mean()
    assert abs(train_share - test_share) < 0.03


def test_prepared_preprocessor_is_not_fitted_yet(real_students):
    data = prepare_data("mat", include_early_grades=True)
    assert not hasattr(data.preprocessor, "transformers_")