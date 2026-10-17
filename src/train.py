"""Model training and cross-validated evaluation utilities.

All scores in this module come from cross-validation on the TRAINING set.
The test set is never used here; it is reserved for the final evaluation.
"""

import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from src.data_loader import PROJECT_ROOT
from src.preprocessing import RANDOM_STATE, PreparedData, prepare_data

RESULTS_DIR = PROJECT_ROOT / "reports" / "model_results"
N_SPLITS = 5
SCORING = {
    "mae": "neg_mean_absolute_error",  # sklearn maximizes scores, so errors are negated
    "rmse": "neg_root_mean_squared_error",
    "r2": "r2",
}
SETUPS = {"with G1/G2": True, "without G1/G2": False}


def make_cv_splits(X: pd.DataFrame, y: pd.Series, n_splits: int = N_SPLITS) -> list:
    """Create reproducible CV folds that keep the share of zero grades similar.

    The same folds are reused for every model, so comparisons are fair.
    """
    splitter = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE
    )
    return list(splitter.split(X, y == 0))


def build_model_pipeline(preprocessor, model) -> Pipeline:
    """Combine an (unfitted) preprocessor and a model into one pipeline.

    The preprocessor is cloned so each pipeline gets its own copy, and inside
    cross-validation it is re-fitted on each fold's training part only.
    """
    return Pipeline(
        steps=[("preprocessor", clone(preprocessor)), ("model", model)]
    )


def get_baseline_models() -> dict:
    """Return {name: (estimator, parameter summary)} for the baselines."""
    return {
        "Baseline (mean)": (DummyRegressor(strategy="mean"), "strategy=mean"),
        "Baseline (median)": (DummyRegressor(strategy="median"), "strategy=median"),
    }


def get_candidate_models() -> dict:
    """Return {name: (estimator, parameter summary)} for the real models."""
    return {
        "Linear Regression": (LinearRegression(), "default"),
        "Ridge (alpha=1)": (Ridge(alpha=1.0), "alpha=1"),
        "Ridge (alpha=10)": (Ridge(alpha=10.0), "alpha=10"),
        "Ridge (alpha=100)": (Ridge(alpha=100.0), "alpha=100"),
        "Decision Tree (depth 3)": (
            DecisionTreeRegressor(
                max_depth=3, min_samples_leaf=5, random_state=RANDOM_STATE
            ),
            "max_depth=3, min_samples_leaf=5",
        ),
        "Decision Tree (depth 5)": (
            DecisionTreeRegressor(
                max_depth=5, min_samples_leaf=5, random_state=RANDOM_STATE
            ),
            "max_depth=5, min_samples_leaf=5",
        ),
        "Random Forest": (
            RandomForestRegressor(
                n_estimators=300,
                min_samples_leaf=3,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            "n_estimators=300, min_samples_leaf=3",
        ),
        "HistGradientBoosting": (
            HistGradientBoostingRegressor(
                learning_rate=0.05,
                max_iter=200,
                max_depth=3,
                min_samples_leaf=10,
                random_state=RANDOM_STATE,
            ),
            "learning_rate=0.05, max_iter=200, max_depth=3, min_samples_leaf=10",
        ),
    }


def evaluate_with_cv(
    name: str, model, data: PreparedData, setup: str, params: str = ""
) -> dict:
    """Cross-validate one model on the training set and summarize the scores."""
    pipeline = build_model_pipeline(data.preprocessor, model)
    splits = make_cv_splits(data.X_train, data.y_train)

    scores = cross_validate(
        pipeline,
        data.X_train,
        data.y_train,
        cv=splits,
        scoring=SCORING,
    )

    return {
        "setup": setup,
        "model": name,
        "params": params,
        "cv_mae": -scores["test_mae"].mean(),
        "cv_mae_std": scores["test_mae"].std(),
        "cv_rmse": -scores["test_rmse"].mean(),
        "cv_rmse_std": scores["test_rmse"].std(),
        "cv_r2": scores["test_r2"].mean(),
        "cv_r2_std": scores["test_r2"].std(),
    }


def run_model_comparison() -> pd.DataFrame:
    """Evaluate baselines and candidate models for both feature setups."""
    all_models = {**get_baseline_models(), **get_candidate_models()}

    rows = []
    for setup_name, include_early in SETUPS.items():
        data = prepare_data("mat", include_early_grades=include_early)
        for model_name, (model, params) in all_models.items():
            print(f"  evaluating [{setup_name}] {model_name} ...")
            rows.append(evaluate_with_cv(model_name, model, data, setup_name, params))

    results = pd.DataFrame(rows)

    # How much better than the best baseline (within the same setup)?
    is_baseline = results["model"].str.startswith("Baseline")
    best_baseline_mae = results[is_baseline].groupby("setup")["cv_mae"].min()
    best_baseline_rmse = results[is_baseline].groupby("setup")["cv_rmse"].min()
    results["mae_gain_vs_baseline"] = (
        results["setup"].map(best_baseline_mae) - results["cv_mae"]
    )
    results["rmse_gain_vs_baseline"] = (
        results["setup"].map(best_baseline_rmse) - results["cv_rmse"]
    )
    return results


def save_results(results: pd.DataFrame, filename: str) -> None:
    """Save a results table to reports/model_results/."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / filename
    results.to_csv(path, index=False)
    print(f"\nSaved results to: {path}")


def print_comparison(results: pd.DataFrame) -> None:
    """Print one table per setup, best (lowest MAE) first."""
    columns = [
        "model", "cv_mae", "cv_mae_std", "cv_rmse", "cv_rmse_std",
        "cv_r2", "mae_gain_vs_baseline",
    ]
    for setup_name in SETUPS:
        subset = results[results["setup"] == setup_name].sort_values("cv_mae")
        print("\n" + "=" * 70)
        print(f"{setup_name.upper()} (5-fold CV on the training set, sorted by MAE)")
        print("=" * 70)
        print(subset[columns].round(3).to_string(index=False))


if __name__ == "__main__":
    print("Running 5-fold cross-validation for all models...")
    comparison = run_model_comparison()
    print_comparison(comparison)
    save_results(comparison, "model_comparison.csv")