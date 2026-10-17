"""Train the selected final model on the training set and save it to disk.

Run once after Phase 7:  python -m src.save_model

The saved file contains the COMPLETE pipeline (preprocessing + model), fitted
on the training set only, so it is exactly the model evaluated in Phase 7.
"""

import json

import joblib
import numpy as np
import sklearn
from sklearn.metrics import mean_absolute_error

from src.evaluate import FINAL_SETUP, select_final_model
from src.predict import FEATURE_COLUMNS, METADATA_PATH, MODEL_PATH, load_model
from src.preprocessing import RANDOM_STATE, prepare_data
from src.train import build_model_pipeline


def save_final_model(setup: str = FINAL_SETUP) -> None:
    """Fit the selected model on the training set, save it, and verify it."""
    include_early = setup == "with G1/G2"
    data = prepare_data("mat", include_early_grades=include_early)

    if list(data.X_train.columns) != FEATURE_COLUMNS:
        raise ValueError(
            "Training columns do not match predict.FEATURE_COLUMNS. "
            "predict.py is built for the 'with G1/G2' setup."
        )

    model_name, estimator, _ = select_final_model(setup)
    print(f"Selected model: {model_name} ({setup})")

    pipeline = build_model_pipeline(data.preprocessor, estimator)
    pipeline.fit(data.X_train, data.y_train)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH, compress=3)
    size_kb = MODEL_PATH.stat().st_size / 1024
    print(f"Saved model to: {MODEL_PATH} ({size_kb:.0f} KB)")

    metadata = {
        "model_name": model_name,
        "setup": setup,
        "sklearn_version": sklearn.__version__,
        "random_state": RANDOM_STATE,
        "n_train_rows": int(len(data.X_train)),
        "feature_columns": FEATURE_COLUMNS,
        "model_file": MODEL_PATH.name,
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved metadata to: {METADATA_PATH}")

    # Verification: the reloaded model must behave exactly like the original
    reloaded = load_model()
    original_predictions = pipeline.predict(data.X_test)
    reloaded_predictions = reloaded.predict(data.X_test)
    identical = bool(np.allclose(original_predictions, reloaded_predictions))
    test_mae = mean_absolute_error(data.y_test, reloaded_predictions)

    print(f"\nPredictions identical after reload: {identical}")
    print(f"Reloaded model test MAE: {test_mae:.3f} "
          "(should match the Random Forest row in test_results.csv)")
    if not identical:
        raise RuntimeError("Reloaded model does not match the original.")


if __name__ == "__main__":
    save_final_model()