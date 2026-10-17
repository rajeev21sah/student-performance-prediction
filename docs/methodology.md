# Methodology

This document records what was done in the project. Numbers come from runs of the scripts in `src/` and are stored in `reports/model_results/`.

## 1. Data

- Source: UCI Student Performance dataset (Cortez, 2008), Mathematics file `student-mat.csv`.
- Size: 395 rows, 33 columns: 32 input features and the target `G3` (final grade, 0-20). 17 columns are text and 15 numeric predictors are numbers; many numeric predictors are ordered codes (for example 1-5), not true measurements.
- Quality checks (notebook `01_exploratory_data_analysis.ipynb`): no missing values, no duplicate rows, all values inside the documented ranges, consistent category labels. `failures` takes the values 0-3 in this file.
- 38 students (9.6%) have `G3 = 0`. All of them have 0 recorded absences, and the zero-grade group hides a relationship between absences and grades among the other students (Spearman correlation of `absences` with `G3`: 0.018 for all students, -0.244 for students with `G3 > 0`).
- `student-por.csv` was not used: the documentation states that 382 students appear in both files, so a merged dataset could place the same student in both the training and test sets.

## 2. Two feature setups

- **With G1/G2:** 32 input columns. `G1` and `G2` are earlier grades in the same course; their Spearman correlations with `G3` are 0.878 and 0.957.
- **Without G1/G2:** 30 input columns, modelling an early-warning situation.

Both setups were compared in cross-validation. The final model uses the setup **with G1/G2**, so it is a late-term predictor.

## 3. Preprocessing

Built with scikit-learn (`src/preprocessing.py`):

- Numeric columns: median imputation, then standardization.
- Categorical columns: most-frequent imputation, then one-hot encoding (`drop="if_binary"`, `handle_unknown="ignore"`).
- Result: 45 features with `G1`/`G2`, 43 without.
- Imputation is a safeguard (the data has no missing values). All 38 zero-grade students and the `absences` outliers were **kept**.
- Every transformation is fitted on training data only. Inside cross-validation, it is refitted on each fold's training part.

## 4. Splitting

- 80/20 train/test split with `random_state=42`, stratified on whether `G3 == 0`.
- Training set: 316 students (30 with `G3 = 0`). Test set: 79 students (8 with `G3 = 0`).
- The test set was not used to choose between models or settings.

## 5. Evaluation protocol

- 5-fold cross-validation on the training set, shuffled, `random_state=42`, stratified on `G3 == 0`. The same folds are used for every model.
- Metrics: MAE, RMSE, R² (cross-validation); MAE, MSE, RMSE, R² (test).
- Baselines: `DummyRegressor` predicting the training mean or median.

## 6. Candidate models

| Model                | Settings                                                                   |
| -------------------- | -------------------------------------------------------------------------- |
| Linear Regression    | defaults                                                                   |
| Ridge                | alpha = 1, 10, 100                                                         |
| Decision Tree        | max_depth = 3 or 5, min_samples_leaf = 5                                   |
| Random Forest        | 300 trees, min_samples_leaf = 3                                            |
| HistGradientBoosting | learning_rate = 0.05, max_iter = 200, max_depth = 3, min_samples_leaf = 10 |

Tree-based models use `random_state=42`. Settings are a small fixed set of sensible values, not the result of an extensive search.

## 7. Cross-validation results

**With G1/G2**

| Model                   | MAE   | MAE std | RMSE  | R²     |
| ----------------------- | ----- | ------- | ----- | ------ |
| Random Forest           | 1.040 | 0.151   | 1.665 | 0.859  |
| Decision Tree (depth 5) | 1.087 | 0.252   | 1.869 | 0.813  |
| HistGradientBoosting    | 1.095 | 0.160   | 1.686 | 0.856  |
| Decision Tree (depth 3) | 1.173 | 0.198   | 1.848 | 0.827  |
| Ridge (alpha=10)        | 1.333 | 0.114   | 2.021 | 0.798  |
| Ridge (alpha=1)         | 1.399 | 0.128   | 2.080 | 0.786  |
| Linear Regression       | 1.415 | 0.131   | 2.094 | 0.783  |
| Ridge (alpha=100)       | 1.425 | 0.081   | 2.178 | 0.769  |
| Baseline (median)       | 3.431 | 0.118   | 4.642 | -0.023 |
| Baseline (mean)         | 3.461 | 0.134   | 4.621 | -0.013 |

**Without G1/G2**

| Model                   | MAE   | MAE std | RMSE  | R²     |
| ----------------------- | ----- | ------- | ----- | ------ |
| Random Forest           | 3.080 | 0.237   | 4.038 | 0.217  |
| HistGradientBoosting    | 3.284 | 0.284   | 4.230 | 0.139  |
| Ridge (alpha=100)       | 3.323 | 0.192   | 4.368 | 0.091  |
| Baseline (median)       | 3.431 | 0.118   | 4.642 | -0.023 |
| Decision Tree (depth 3) | 3.437 | 0.154   | 4.480 | 0.043  |
| Decision Tree (depth 5) | 3.451 | 0.189   | 4.648 | -0.031 |
| Baseline (mean)         | 3.461 | 0.134   | 4.621 | -0.013 |
| Ridge (alpha=10)        | 3.476 | 0.239   | 4.528 | 0.018  |
| Ridge (alpha=1)         | 3.598 | 0.273   | 4.682 | -0.051 |
| Linear Regression       | 3.623 | 0.281   | 4.716 | -0.067 |

## 8. Model selection

The model with the lowest cross-validated MAE, excluding baselines, in the setup with G1/G2 was chosen: Random Forest. The top three models differ by less than the fold-to-fold standard deviation, so the choice is a tie-break by rule, not evidence of a clear winner.

## 9. Test-set evaluation

The selected model was fitted on all 316 training students and evaluated once on the 79 test students.

| Model             | MAE   | MSE    | RMSE  | R²     |
| ----------------- | ----- | ------ | ----- | ------ |
| Baseline (median) | 3.253 | 20.165 | 4.490 | -0.033 |
| Random Forest     | 0.895 | 2.094  | 1.447 | 0.893  |

| Group  | n   | MAE   | RMSE  | Mean actual | Mean predicted |
| ------ | --- | ----- | ----- | ----------- | -------------- |
| G3 = 0 | 8   | 2.265 | 3.141 | 0.0         | 2.265          |
| G3 > 0 | 71  | 0.741 | 1.104 | 11.352      | 11.072         |

For the 8 zero-grade test students, the model predicted below 1.1 for five of them and 3.5, 4.9, and 6.4 for the other three. One student with an actual grade of 11 was predicted at 5.25.

95% bootstrap interval for the Random Forest test MAE (2,000 resamples of the 79 test errors, seed 42): 0.665 to 1.173. The cross-validated MAE of 1.040 lies inside it. The interval does not include variation from retraining the model or from selecting it among several candidates.

## 10. Explainability

- **Permutation importance** (`src/explainability.py`): columns of the raw test data are shuffled 30 times each, scored by MAE increase. Top results: `G2` 3.211 (± 0.346), `absences` 0.525 (± 0.114), `age` 0.051 (± 0.032); all other columns are 0.018 or below, about the size of their standard deviations.
- **Partial dependence** on the training data for `G2`, `absences`, and `failures`. The `G2` curve rises steeply from about 4 to about 18. The `absences` curve steps from about 9.3 at 0-1 absences to about 11 from 2 absences upward and is flat after that. The `failures` curve is nearly flat.
- `G1` scores near zero because it is highly correlated with `G2`, not because it is uninformative.
- These tools show what the model uses. They do not show what causes academic performance.

## 11. Saving, loading, and validation

- The full fitted pipeline (preprocessing and model) is saved with Joblib to `models/student_performance_model.joblib`, together with `models/model_metadata.json` (model name, setup, scikit-learn version, training rows, feature list).
- The saved model is fitted on the training set only, so it is the model that was evaluated.
- `src/predict.py` validates every input: exact column set, whole numbers within the documented ranges, and allowed category values. It reports all problems at once, clips predictions to 0-20, and warns if the scikit-learn version differs from the one that saved the model.
- Models are never retrained when the app starts.
- Only load model files from sources you trust: Joblib files can execute code when loaded.

## 12. Testing

The pytest suite (89 tests) covers data loading, input validation (including boundary values), preprocessing and leakage rules, prediction output, serialization, missing or wrong model files, version warnings, and model-selection logic. Unit tests use a small synthetic dataset to exercise the code (never presented as real student data); a few tests use the real dataset and saved model and skip themselves if they are missing. The suite was checked to fail when a validation range was deliberately widened.

## 13. Reproducibility

- Python 3.11.5, library versions pinned in `requirements.txt`.
- Random seed 42 for the split, cross-validation folds, tree-based models, bootstrap, and permutation shuffles.
- Run the scripts in the order given in the README.

## 14. What was not done

- No extensive hyperparameter search, nested cross-validation, or repeated cross-validation.
- No prediction intervals.
- No measurement of performance across subgroups (school, sex, address type, and so on).
- The setup without `G1`/`G2` was not evaluated on the test set.
- The Portuguese file was not analysed.
- No automated tests for the Streamlit pages, `train.py`, `save_model.py`, or `explainability.py`.
