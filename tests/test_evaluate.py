"""Tests for the evaluation helpers in src/evaluate.py."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

import src.evaluate as evaluate
from src.evaluate import (
    bootstrap_mae_interval,
    compute_metrics,
    error_by_group,
    select_final_model,
)


def test_compute_metrics_on_known_values():
    metrics = compute_metrics([0, 2], [1, 1])
    assert metrics["mae"] == pytest.approx(1.0)
    assert metrics["mse"] == pytest.approx(1.0)
    assert metrics["rmse"] == pytest.approx(1.0)
    assert metrics["r2"] == pytest.approx(0.0)


def test_error_by_group_on_known_values():
    table = error_by_group(pd.Series([0, 0, 10, 12]), np.array([2.0, 0.0, 9.0, 12.0]))
    table = table.set_index("group")

    assert table.loc["G3 == 0", "n"] == 2
    assert table.loc["G3 == 0", "mae"] == pytest.approx(1.0)
    assert table.loc["G3 == 0", "rmse"] == pytest.approx(np.sqrt(2.0))
    assert table.loc["G3 > 0", "mae"] == pytest.approx(0.5)
    assert table.loc["All test students", "n"] == 4
    assert table.loc["All test students", "mae"] == pytest.approx(0.75)


def test_bootstrap_interval_of_constant_errors_is_that_constant():
    low, high = bootstrap_mae_interval(np.full(10, 1.0))
    assert low == pytest.approx(1.0) and high == pytest.approx(1.0)


def test_bootstrap_interval_is_ordered_and_within_the_data_range():
    errors = np.array([0.0, 1.0, 2.0, 3.0, 10.0])
    low, high = bootstrap_mae_interval(errors)
    assert low <= high
    assert errors.min() <= low and high <= errors.max()


def _write_comparison(path):
    pd.DataFrame(
        {
            "setup": ["with G1/G2"] * 3 + ["without G1/G2"] * 3,
            "model": ["Baseline (median)", "Ridge (alpha=10)", "Random Forest"] * 2,
            "cv_mae": [3.4, 1.3, 1.0, 0.1, 0.5, 2.9],  # lower values in the OTHER setup
        }
    ).to_csv(path, index=False)


def test_final_model_is_the_best_candidate_in_the_requested_setup(tmp_path, monkeypatch):
    path = tmp_path / "model_comparison.csv"
    _write_comparison(path)
    monkeypatch.setattr(evaluate, "COMPARISON_FILE", path)

    name, estimator, _ = select_final_model("with G1/G2")

    assert name == "Random Forest"
    assert isinstance(estimator, RandomForestRegressor)


def test_baselines_are_never_selected(tmp_path, monkeypatch):
    path = tmp_path / "model_comparison.csv"
    _write_comparison(path)
    monkeypatch.setattr(evaluate, "COMPARISON_FILE", path)

    # In 'without G1/G2' the baseline row has the lowest MAE (0.1) but must be ignored
    name, _, _ = select_final_model("without G1/G2")
    assert name == "Ridge (alpha=10)"


def test_selection_without_comparison_file_gives_clear_error(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate, "COMPARISON_FILE", tmp_path / "missing.csv")
    with pytest.raises(FileNotFoundError, match="src.train"):
        select_final_model("with G1/G2")


def test_unknown_setup_is_rejected(tmp_path, monkeypatch):
    path = tmp_path / "model_comparison.csv"
    _write_comparison(path)
    monkeypatch.setattr(evaluate, "COMPARISON_FILE", path)
    with pytest.raises(ValueError, match="No candidate models"):
        select_final_model("nonsense setup")