"""Tests for input validation in src/predict.py."""

import pandas as pd
import pytest

from src.predict import (
    CATEGORICAL_VALUES,
    FEATURE_COLUMNS,
    NUMERIC_RANGES,
    InputValidationError,
    validate_dataframe,
    validate_record,
)


def test_schema_covers_every_feature_exactly_once():
    assert len(FEATURE_COLUMNS) == 32
    assert len(set(FEATURE_COLUMNS)) == 32
    assert set(NUMERIC_RANGES) | set(CATEGORICAL_VALUES) == set(FEATURE_COLUMNS)
    assert not set(NUMERIC_RANGES) & set(CATEGORICAL_VALUES)


def test_valid_record_is_accepted(valid_record):
    cleaned = validate_record(valid_record)
    assert list(cleaned) == FEATURE_COLUMNS
    assert cleaned == valid_record


def test_whole_number_float_is_converted_to_int(valid_record):
    valid_record["age"] = 17.0
    cleaned = validate_record(valid_record)
    assert cleaned["age"] == 17
    assert isinstance(cleaned["age"], int)


@pytest.mark.parametrize(
    "column, value",
    [("age", 15), ("age", 22), ("absences", 0), ("absences", 93),
     ("failures", 3), ("G1", 0), ("G2", 20)],
)
def test_boundary_values_are_accepted(valid_record, column, value):
    valid_record[column] = value
    assert validate_record(valid_record)[column] == value


@pytest.mark.parametrize(
    "column, value",
    [("age", 14), ("age", 23), ("absences", -1), ("absences", 94),
     ("failures", 4), ("failures", -1), ("G1", 21), ("G2", -1),
     ("Medu", 5), ("studytime", 0)],
)
def test_out_of_range_values_are_rejected(valid_record, column, value):
    valid_record[column] = value
    with pytest.raises(InputValidationError, match=column):
        validate_record(valid_record)


@pytest.mark.parametrize(
    "bad_value", ["17", None, True, float("nan"), float("inf"), 17.5]
)
def test_non_whole_or_non_numeric_age_is_rejected(valid_record, bad_value):
    valid_record["age"] = bad_value
    with pytest.raises(InputValidationError, match="age"):
        validate_record(valid_record)


@pytest.mark.parametrize(
    "column, value",
    [("sex", "X"), ("sex", "female"), ("school", "gp"), ("Mjob", "doctor"),
     ("higher", "maybe"), ("famsize", 3)],
)
def test_invalid_category_is_rejected(valid_record, column, value):
    valid_record[column] = value
    with pytest.raises(InputValidationError, match=column):
        validate_record(valid_record)


def test_missing_column_is_reported(valid_record):
    del valid_record["G2"]
    with pytest.raises(InputValidationError, match="Missing columns.*G2"):
        validate_record(valid_record)


def test_unexpected_column_is_reported(valid_record):
    valid_record["extra_column"] = 1
    with pytest.raises(InputValidationError, match="Unexpected columns.*extra_column"):
        validate_record(valid_record)


def test_target_column_is_not_accepted_as_input(valid_record):
    valid_record["G3"] = 10
    with pytest.raises(InputValidationError, match="G3"):
        validate_record(valid_record)


def test_all_problems_are_reported_together(valid_record):
    valid_record["age"] = 40
    valid_record["sex"] = "X"
    valid_record["extra_column"] = 1
    del valid_record["G2"]
    with pytest.raises(InputValidationError) as info:
        validate_record(valid_record)
    message = str(info.value)
    for fragment in ["G2", "extra_column", "age", "sex"]:
        assert fragment in message


def test_dataframe_of_valid_rows_is_accepted(valid_record):
    clean = validate_dataframe(pd.DataFrame([valid_record, valid_record]))
    assert list(clean.columns) == FEATURE_COLUMNS
    assert len(clean) == 2


def test_dataframe_column_order_does_not_matter(valid_record):
    df = pd.DataFrame([valid_record])[list(reversed(FEATURE_COLUMNS))]
    assert list(validate_dataframe(df).columns) == FEATURE_COLUMNS


def test_empty_dataframe_is_rejected():
    with pytest.raises(InputValidationError, match="No rows"):
        validate_dataframe(pd.DataFrame(columns=FEATURE_COLUMNS))


def test_dataframe_with_missing_column_is_rejected(valid_record):
    df = pd.DataFrame([valid_record]).drop(columns=["G1"])
    with pytest.raises(InputValidationError, match="Missing columns.*G1"):
        validate_dataframe(df)


def test_dataframe_error_names_the_bad_row(valid_record):
    bad = dict(valid_record, age=40)
    with pytest.raises(InputValidationError, match="Row 1"):
        validate_dataframe(pd.DataFrame([valid_record, bad]))


def test_every_real_student_passes_validation(real_students):
    features = real_students.drop(columns=["G3"])
    assert len(validate_dataframe(features)) == len(real_students)