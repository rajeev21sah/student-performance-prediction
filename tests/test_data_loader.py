"""Tests for src/data_loader.py."""

import pytest

import src.data_loader as data_loader
from src.data_loader import TARGET_COLUMN, load_student_data


def test_unknown_subject_raises_value_error():
    with pytest.raises(ValueError, match="Unknown subject"):
        load_student_data("xyz")


def test_missing_file_gives_helpful_error(tmp_path, monkeypatch):
    monkeypatch.setattr(data_loader, "RAW_DATA_DIR", tmp_path)  # empty folder
    with pytest.raises(FileNotFoundError, match="archive.ics.uci.edu"):
        load_student_data("mat")


def test_semicolon_separated_file_is_parsed(students_df, tmp_path, monkeypatch):
    students_df.to_csv(tmp_path / "student-mat.csv", sep=";", index=False)
    monkeypatch.setattr(data_loader, "RAW_DATA_DIR", tmp_path)

    loaded = load_student_data("mat")

    assert list(loaded.columns) == list(students_df.columns)
    assert loaded.shape == students_df.shape


def test_real_dataset_has_expected_structure(real_students):
    assert real_students.shape == (395, 33)
    assert TARGET_COLUMN in real_students.columns
    assert real_students[TARGET_COLUMN].between(0, 20).all()
    assert real_students.isna().sum().sum() == 0