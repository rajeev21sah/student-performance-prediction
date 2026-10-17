"""Tests for model loading, saving, and prediction in src/predict.py."""

import json
import warnings

import joblib
import numpy as np
import pandas as pd
import pytest
import sklearn
from sklearn.linear_model import LinearRegression

import src.predict as predict
from src.predict import (
    FEATURE_COLUMNS,
    InputValidationError,
    ModelNotFoundError,
    load_metadata,
    load_model,
    predict_grade,
    predict_grades,
)


class FakeModel:
    """Stands in for a model that returns out-of-range values."""

    def predict(self, X):
        return np.array([25.0, -3.0])


# --- prediction output -------------------------------------------------------


def test_predict_grade_returns_a_float_in_range(small_model, valid_record):
    grade = predict_grade(valid_record, small_model)
    assert isinstance(grade, float)
    assert 0 <= grade <= 20


def test_predict_grades_handles_several_students(small_model, valid_record):
    df = pd.DataFrame([valid_record] * 3)
    predictions = predict_grades(df, small_model)
    assert predictions.shape == (3,)
    assert np.isfinite(predictions).all()


def test_predictions_are_clipped_to_the_valid_range(valid_record):
    df = pd.DataFrame([valid_record, valid_record])
    predictions = predict_grades(df, FakeModel())
    assert predictions.tolist() == [20.0, 0.0]


# --- invalid input -----------------------------------------------------------


def test_invalid_input_is_rejected_before_any_model_is_needed(valid_record):
    valid_record["age"] = 99
    with pytest.raises(InputValidationError, match="age"):
        predict_grade(valid_record)  # model=None would trigger loading, so validation must come first


def test_missing_column_is_rejected_when_predicting(small_model, valid_record):
    del valid_record["G2"]
    with pytest.raises(InputValidationError, match="G2"):
        predict_grade(valid_record, small_model)


# --- missing, wrong, or outdated model files ---------------------------------


def test_missing_model_file_gives_clear_error(tmp_path):
    with pytest.raises(ModelNotFoundError, match="save_model"):
        load_model(tmp_path / "missing.joblib")


def test_missing_model_error_is_a_file_not_found_error():
    assert issubclass(ModelNotFoundError, FileNotFoundError)


def test_model_with_wrong_columns_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(predict, "METADATA_PATH", tmp_path / "no_metadata.json")
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [3.0, 1.0, 2.0]})
    path = tmp_path / "wrong.joblib"
    joblib.dump(LinearRegression().fit(X, [1, 2, 3]), path)

    with pytest.raises(ValueError, match="different input columns"):
        load_model(path)


def test_missing_metadata_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(predict, "METADATA_PATH", tmp_path / "no_metadata.json")
    assert load_metadata() is None


def test_version_mismatch_gives_a_warning(small_model, tmp_path, monkeypatch):
    metadata = tmp_path / "metadata.json"
    metadata.write_text(json.dumps({"sklearn_version": "0.0.1"}))
    monkeypatch.setattr(predict, "METADATA_PATH", metadata)
    path = tmp_path / "model.joblib"
    joblib.dump(small_model, path)

    with pytest.warns(UserWarning, match="scikit-learn"):
        load_model(path)


def test_matching_version_gives_no_version_warning(small_model, tmp_path, monkeypatch):
    metadata = tmp_path / "metadata.json"
    metadata.write_text(json.dumps({"sklearn_version": sklearn.__version__}))
    monkeypatch.setattr(predict, "METADATA_PATH", metadata)
    path = tmp_path / "model.joblib"
    joblib.dump(small_model, path)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        load_model(path)
    assert not any("scikit-learn" in str(w.message) for w in caught)


# --- serialization -----------------------------------------------------------


def test_saved_model_reloads_and_predicts_identically(
    small_model, students_df, tmp_path, monkeypatch
):
    monkeypatch.setattr(predict, "METADATA_PATH", tmp_path / "no_metadata.json")
    path = tmp_path / "model.joblib"
    joblib.dump(small_model, path)

    reloaded = load_model(path)

    X = students_df[FEATURE_COLUMNS]
    assert np.allclose(small_model.predict(X), reloaded.predict(X))


# --- the real saved model ----------------------------------------------------


def test_real_saved_model_expects_the_documented_columns(real_model_path):
    model = load_model()
    assert set(model.feature_names_in_) == set(FEATURE_COLUMNS)


def test_real_saved_model_predicts_the_sample_student(real_model_path, valid_record):
    # 6.23 is the value printed by `python -m src.predict` in Phase 9
    grade = predict_grade(valid_record, load_model())
    assert grade == pytest.approx(6.23, abs=0.05)