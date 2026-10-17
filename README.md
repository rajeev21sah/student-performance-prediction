# Student Performance Prediction

An educational machine learning project that predicts the final grade (`G3`, 0-20) of secondary-school students in a Mathematics course, using regression. It includes data exploration, a leakage-safe preprocessing pipeline, model comparison with cross-validation, a one-time test-set evaluation, explainability, a validated prediction module, a Streamlit app, and automated tests.

> **Not for decisions about real students.** Predictions are statistical estimates, not official grades. See [Limitations](#limitations).

## Problem statement

Given a student's background information and the grades from the first two periods (`G1`, `G2`), estimate the final-period grade `G3`. The task is regression on a 0-20 scale. Because the model needs `G1` and `G2`, it is a **late-term** predictor, not an early-warning tool.

## Dataset

- **Name:** Student Performance, UCI Machine Learning Repository
- **File used:** `student-mat.csv` (Mathematics): 395 students, 33 columns (32 input features + target `G3`)
- **Source:** https://archive.ics.uci.edu/dataset/320/student+performance
- **Citation:** Cortez, P. (2008). Student Performance [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5TG7T
- **License:** CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/legalcode.txt)
- `student-por.csv` (Portuguese) is included but **not used** for modelling. Per the dataset documentation, 382 students appear in both files, so merging them could put the same student in both the training and test sets.

More detail is in `data/raw/SOURCE.md`.

## Technologies

Python 3.11, pandas, NumPy, Matplotlib, Seaborn, scikit-learn, Joblib, Streamlit, pytest, Jupyter Notebook, Git. Exact versions are in `requirements.txt`.

## Project structure

```text
student-performance-prediction/
├── app.py                      Streamlit application
├── requirements.txt
├── pytest.ini
├── README.md
├── .gitignore
├── data/raw/                   dataset files and SOURCE.md
├── notebooks/
│   └── 01_exploratory_data_analysis.ipynb
├── src/
│   ├── data_loader.py          load the dataset and print a structural overview
│   ├── preprocessing.py        validation, train/test split, preprocessing pipeline
│   ├── train.py                baselines, candidate models, cross-validation
│   ├── evaluate.py             model selection and one-time test evaluation
│   ├── explainability.py       permutation importance, partial dependence
│   ├── save_model.py           fit and save the final pipeline
│   └── predict.py              input validation, model loading, prediction
├── models/                     saved model (created locally, not committed)
├── reports/
│   ├── figures/                plots from the notebook and the scripts
│   └── model_results/          CSV result tables
├── tests/                      pytest test suite
└── docs/
    ├── methodology.md
    └── limitations.md
```

## Installation

These commands are for Windows PowerShell, run from the project folder.

```powershell
python --version                      # tested with Python 3.11.5
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks the activation script, run `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` once and try again.

## Data preparation

The dataset files are in `data/raw/`. To download them again, use the link above and place `student-mat.csv`, `student-por.csv`, and `student.txt` in `data/raw/`.

There is no separate data-cleaning step: the checks found no missing values, duplicates, or out-of-range values, and all preprocessing (imputation, scaling, encoding) runs inside the model pipeline, fitted on training data only. Two optional commands inspect this stage:

```powershell
python -m src.data_loader       # dataset structure and summary statistics
python -m src.preprocessing     # runs the split and preprocessing checks
```

The exploratory analysis is in `notebooks/01_exploratory_data_analysis.ipynb`.

## Model training

Run these in order from the project folder (each later step reads files written by an earlier one):

```powershell
python -m src.train             # cross-validate baselines and 8 candidate models
python -m src.evaluate          # select the final model, evaluate once on the test set
python -m src.explainability    # permutation importance and partial dependence
python -m src.save_model        # fit and save models/student_performance_model.joblib
```

## Model evaluation

- Models are compared with 5-fold cross-validation on the training set (316 students). The same folds are used for every model.
- The final model is chosen by lowest cross-validated MAE and then evaluated **once** on the held-out test set (79 students).
- Reported metrics: MAE, MSE, RMSE, and R². MAE and RMSE are in grade points.

Outputs are written to `reports/model_results/` and `reports/figures/`.

## Running the app

```powershell
python -m src.save_model        # once, if models\student_performance_model.joblib does not exist
streamlit run app.py
```

Then open http://localhost:8501. To make the app reachable only from your own computer, use `streamlit run app.py --server.address localhost`. The app has pages for Home, Student Prediction, Data Explorer, Model Evaluation, Feature Importance, and About. Do not enter real students' data if the app is shared or deployed publicly.

## Running the tests

```powershell
python -m pytest -q
```

The suite has 89 tests covering data loading, input validation, preprocessing (including the rule that the scaler is fitted on training data only), prediction output, model saving and loading, missing or incompatible model files, and the model-selection logic. The Streamlit pages, the cross-validation code in `train.py`, `save_model.py`, and `explainability.py` are not covered by automated tests; they were checked by running them.

## Results

All numbers below come from runs of the scripts above (random seed 42). Full tables are in `reports/model_results/`.

### Cross-validation (setup with G1/G2, 316 training students)

| Model                                | CV MAE          | CV RMSE | CV R²  |
| ------------------------------------ | --------------- | ------- | ------ |
| Random Forest                        | 1.040 (± 0.151) | 1.665   | 0.859  |
| Decision Tree (depth 5)              | 1.087 (± 0.252) | 1.869   | 0.813  |
| HistGradientBoosting                 | 1.095 (± 0.160) | 1.686   | 0.856  |
| Ridge (alpha=10), best linear model  | 1.333 (± 0.114) | 2.021   | 0.798  |
| Baseline (always predict the median) | 3.431 (± 0.118) | 4.642   | -0.023 |

The top three models are within the fold-to-fold variation of one another. Random Forest was selected by the lowest-MAE rule, not because it was shown to be clearly better than the other two.

### Final model on the test set (79 students, evaluated once)

| Model             | MAE   | RMSE  | R²     |
| ----------------- | ----- | ----- | ------ |
| Baseline (median) | 3.253 | 4.490 | -0.033 |
| Random Forest     | 0.895 | 1.447 | 0.893  |

| Group           | Students | MAE   | RMSE  |
| --------------- | -------- | ----- | ----- |
| Final grade = 0 | 8        | 2.265 | 3.141 |
| Final grade > 0 | 71       | 0.741 | 1.104 |

The test set is small (only 8 students with a final grade of 0), so these numbers are noisy.

### Without the earlier grades

When `G1` and `G2` are removed, the best cross-validated model (Random Forest) reaches an MAE of 3.080 (± 0.237) and R² of 0.217, compared with 3.431 for the baseline. Most other models in that setup were no better than the baseline. This setup was not evaluated on the test set, because the final model uses the setup with `G1`/`G2`.

### What the model relies on

Permutation importance on the test set: `G2` is by far the most important input (shuffling it raises MAE by about 3.2 grade points), followed by `absences` (about 0.5). Other inputs are close to noise. These figures describe what the model uses, not what causes grades.

## Limitations

- Small data: 395 students from two schools in one country.
- Needs `G1` and `G2`, so it is not an early-warning tool.
- Some students receive a final grade of 0 after ordinary earlier grades; the model catches some of them and misses others.
- Every student with a final grade of 0 has 0 recorded absences, and the model has learned this pattern. It may reflect how records were kept, not attendance behaviour.
- The test set is small, so the reported error is uncertain.
- Fairness across student groups was not measured.

See `docs/limitations.md` for details.

## Future improvements

- Use repeated or nested cross-validation and a wider hyperparameter search.
- Handle zero grades with a separate approach (for example, a classifier for "final grade is 0" followed by a regressor).
- Add prediction intervals instead of a single average error figure.
- Measure performance across subgroups (for example by school, sex, or address type).
- Analyse the Portuguese file, taking care to avoid the 382 overlapping students.
- Add automated tests for the app and the training code, and run the tests in continuous integration.

## License and citation

The code is released under the MIT License (see `LICENSE`). The dataset is licensed separately under CC BY 4.0; please cite it as shown above.

A 95% bootstrap interval for the test MAE (resampling the 79 test students 2,000 times) is 0.665 to 1.173 grade points. It reflects only the small test set, not variation from retraining or from choosing among models.
