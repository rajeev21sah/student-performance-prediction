"""Final model selection and one-time test-set evaluation.

Workflow:
  1. Select the final model using CROSS-VALIDATION results only.
  2. Fit it on the full training set.
  3. Evaluate it on the untouched test set.

The test set must not be used to choose between models.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data_loader import PROJECT_ROOT
from src.preprocessing import RANDOM_STATE, PreparedData, prepare_data
from src.train import (
    RESULTS_DIR,
    build_model_pipeline,
    get_candidate_models,
    save_results,
)

FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
COMPARISON_FILE = RESULTS_DIR / "model_comparison.csv"
FINAL_SETUP = "with G1/G2"
N_BOOTSTRAP = 2000


def select_final_model(setup: str = FINAL_SETUP):
    """Pick the candidate with the lowest CV MAE for the given setup.

    Returns:
        (model_name, unfitted_estimator, cv_row) where cv_row holds its CV scores.
    """
    if not COMPARISON_FILE.exists():
        raise FileNotFoundError(
            f"{COMPARISON_FILE} not found. Run `python -m src.train` first."
        )
    results = pd.read_csv(COMPARISON_FILE)
    candidates = results[
        (results["setup"] == setup) & (~results["model"].str.startswith("Baseline"))
    ]
    if candidates.empty:
        raise ValueError(f"No candidate models found for setup '{setup}'.")

    best_row = candidates.sort_values("cv_mae").iloc[0]
    estimator, _ = get_candidate_models()[best_row["model"]]
    return best_row["model"], clone(estimator), best_row


def compute_metrics(y_true, y_pred) -> dict:
    """Standard regression metrics."""
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "r2": r2_score(y_true, y_pred),
    }


def bootstrap_mae_interval(abs_errors: np.ndarray, n_resamples: int = N_BOOTSTRAP):
    """95% bootstrap interval for the MAE (resampling test-set errors)."""
    rng = np.random.default_rng(RANDOM_STATE)
    means = [
        rng.choice(abs_errors, size=len(abs_errors), replace=True).mean()
        for _ in range(n_resamples)
    ]
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def error_by_group(y_true: pd.Series, y_pred: np.ndarray) -> pd.DataFrame:
    """MAE and RMSE for zero-grade students, other students, and everyone."""
    y_true = np.asarray(y_true)
    groups = {
        "G3 == 0": y_true == 0,
        "G3 > 0": y_true > 0,
        "All test students": np.ones(len(y_true), dtype=bool),
    }
    rows = []
    for label, mask in groups.items():
        if mask.sum() == 0:
            continue
        errors = y_true[mask] - y_pred[mask]
        rows.append(
            {
                "group": label,
                "n": int(mask.sum()),
                "mae": float(np.abs(errors).mean()),
                "rmse": float(np.sqrt((errors**2).mean())),
                "mean_actual": float(y_true[mask].mean()),
                "mean_predicted": float(y_pred[mask].mean()),
            }
        )
    return pd.DataFrame(rows)


def plot_actual_vs_predicted(y_true, y_pred, title: str, path) -> None:
    """Scatter of actual vs predicted grades; zero-grade students highlighted."""
    y_true = np.asarray(y_true)
    zero = y_true == 0
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true[~zero], y_pred[~zero], alpha=0.6, label="G3 > 0")
    ax.scatter(y_true[zero], y_pred[zero], alpha=0.8, marker="x", label="G3 == 0")
    ax.plot([0, 20], [0, 20], "k--", linewidth=1, label="Perfect prediction")
    ax.set_xlabel("Actual G3")
    ax.set_ylabel("Predicted G3")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_residuals(y_true, y_pred, title: str, path) -> None:
    """Residuals vs predicted values, plus a histogram of residuals."""
    y_true = np.asarray(y_true)
    residuals = y_true - y_pred
    zero = y_true == 0

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(y_pred[~zero], residuals[~zero], alpha=0.6, label="G3 > 0")
    axes[0].scatter(y_pred[zero], residuals[zero], alpha=0.8, marker="x", label="G3 == 0")
    axes[0].axhline(0, color="k", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Predicted G3")
    axes[0].set_ylabel("Residual (actual - predicted)")
    axes[0].set_title("Residuals vs predicted")
    axes[0].legend()

    axes[1].hist(residuals, bins=20)
    axes[1].set_xlabel("Residual (actual - predicted)")
    axes[1].set_ylabel("Number of students")
    axes[1].set_title("Distribution of residuals")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_final_evaluation(setup: str = FINAL_SETUP) -> None:
    """Select the final model, evaluate it once on the test set, save outputs."""
    include_early = setup == "with G1/G2"
    data: PreparedData = prepare_data("mat", include_early_grades=include_early)

    model_name, estimator, cv_row = select_final_model(setup)
    print(f"Setup: {setup}")
    print(f"Selected by lowest CV MAE: {model_name}")
    print(f"  CV MAE = {cv_row['cv_mae']:.3f} (+/- {cv_row['cv_mae_std']:.3f}), "
          f"CV RMSE = {cv_row['cv_rmse']:.3f}, CV R2 = {cv_row['cv_r2']:.3f}")

    # Final model: fit on the FULL training set, predict the test set
    pipeline = build_model_pipeline(data.preprocessor, estimator)
    pipeline.fit(data.X_train, data.y_train)
    y_pred = pipeline.predict(data.X_test)

    # Reference: the median baseline, evaluated the same way
    baseline = build_model_pipeline(data.preprocessor, DummyRegressor(strategy="median"))
    baseline.fit(data.X_train, data.y_train)
    y_pred_base = baseline.predict(data.X_test)

    rows = []
    for name, preds in [("Baseline (median)", y_pred_base), (model_name, y_pred)]:
        metrics = compute_metrics(data.y_test, preds)
        rows.append({"setup": setup, "model": name, **metrics})
    test_results = pd.DataFrame(rows)

    print(f"\nTEST SET RESULTS ({len(data.y_test)} students, evaluated once)")
    print(test_results.round(3).to_string(index=False))

    abs_errors = np.abs(np.asarray(data.y_test) - y_pred)
    low, high = bootstrap_mae_interval(abs_errors)
    print(f"\nTest MAE 95% bootstrap interval: [{low:.3f}, {high:.3f}]")

    groups = error_by_group(data.y_test, y_pred)
    print("\nERROR BY GROUP")
    print(groups.round(3).to_string(index=False))

    # Largest individual errors (for manual inspection)
    predictions = pd.DataFrame(
        {
            "actual": np.asarray(data.y_test),
            "predicted": y_pred,
            "residual": np.asarray(data.y_test) - y_pred,
        },
        index=data.y_test.index,
    )
    worst = predictions.reindex(
        predictions["residual"].abs().sort_values(ascending=False).index
    ).head(8)
    print("\n8 LARGEST ERRORS")
    print(worst.round(2).to_string())

    # Save figures and tables
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_actual_vs_predicted(
        data.y_test, y_pred, f"{model_name}: actual vs predicted (test set)",
        FIGURES_DIR / "test_actual_vs_predicted.png",
    )
    plot_residuals(
        data.y_test, y_pred, f"{model_name}: residuals (test set)",
        FIGURES_DIR / "test_residuals.png",
    )
    save_results(test_results, "test_results.csv")
    save_results(groups, "test_error_by_group.csv")
    save_results(predictions.reset_index(names="row_index"), "test_predictions.csv")
    print(f"Figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    run_final_evaluation()