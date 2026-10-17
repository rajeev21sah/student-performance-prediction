"""Permutation importance and partial dependence for the final model.

These tools describe what the MODEL relies on. They do not show what causes
academic performance.
"""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import PartialDependenceDisplay, permutation_importance

from src.evaluate import FIGURES_DIR, FINAL_SETUP, select_final_model
from src.preprocessing import RANDOM_STATE, PreparedData, prepare_data
from src.train import build_model_pipeline, save_results

N_REPEATS = 30  # shuffles per column; more repeats give a steadier estimate
TOP_N = 15  # how many features to show in the bar chart
PDP_FEATURES = ["G2", "absences", "failures"]


def fit_final_pipeline(setup: str = FINAL_SETUP):
    """Re-fit the selected final model on the training set.

    Returns:
        (model_name, fitted_pipeline, prepared_data)
    """
    include_early = setup == "with G1/G2"
    data = prepare_data("mat", include_early_grades=include_early)
    model_name, estimator, _ = select_final_model(setup)

    pipeline = build_model_pipeline(data.preprocessor, estimator)
    pipeline.fit(data.X_train, data.y_train)
    return model_name, pipeline, data


def compute_permutation_importance(pipeline, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Permutation importance of each ORIGINAL input column.

    Shuffling is done on the raw columns before the pipeline, so a column like
    'Mjob' is shuffled as one unit instead of as separate one-hot columns.
    The score is the increase in MAE caused by shuffling (in grade points).
    """
    result = permutation_importance(
        pipeline,
        X,
        y,
        scoring="neg_mean_absolute_error",
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    table = pd.DataFrame(
        {
            "feature": X.columns,
            "mae_increase_mean": result.importances_mean,
            "mae_increase_std": result.importances_std,
        }
    )
    return table.sort_values("mae_increase_mean", ascending=False).reset_index(drop=True)


def plot_permutation_importance(table: pd.DataFrame, model_name: str, path) -> None:
    """Horizontal bar chart of the top features, with spread across shuffles."""
    top = table.head(TOP_N).iloc[::-1]  # reversed so the largest bar is on top
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top["feature"], top["mae_increase_mean"], xerr=top["mae_increase_std"])
    ax.set_xlabel("Increase in test MAE when the feature is shuffled (grade points)")
    ax.set_title(f"Permutation importance: {model_name} (test set)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_partial_dependence(pipeline, X_train: pd.DataFrame, model_name: str, path) -> None:
    """Average predicted grade as selected features vary."""
    # scikit-learn does not allow partial dependence on integer columns,
    # so pass a copy in which integer columns are converted to floats.
    X_float = X_train.copy()
    int_cols = X_float.select_dtypes(include="integer").columns
    X_float[int_cols] = X_float[int_cols].astype(float)

    fig, axes = plt.subplots(1, len(PDP_FEATURES), figsize=(15, 4))
    PartialDependenceDisplay.from_estimator(
        pipeline, X_float, PDP_FEATURES, kind="average", ax=axes
    )
    for ax in axes:
        ax.set_ylabel("Average predicted G3")
    fig.suptitle(f"Partial dependence: {model_name} (training data)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_explainability(setup: str = FINAL_SETUP) -> None:
    """Compute and save importance and partial dependence outputs."""
    model_name, pipeline, data = fit_final_pipeline(setup)
    print(f"Explaining: {model_name} ({setup})")

    table = compute_permutation_importance(pipeline, data.X_test, data.y_test)
    print(f"\nPERMUTATION IMPORTANCE (test set, {N_REPEATS} shuffles per column)")
    print(table.head(TOP_N).round(3).to_string(index=False))

    positive = int((table["mae_increase_mean"] > 0).sum())
    print(f"\nColumns with a positive mean importance: {positive} of {len(table)}")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_permutation_importance(
        table, model_name, FIGURES_DIR / "permutation_importance.png"
    )
    plot_partial_dependence(
        pipeline, data.X_train, model_name, FIGURES_DIR / "partial_dependence.png"
    )
    save_results(table, "permutation_importance.csv")
    print(f"Figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    run_explainability()