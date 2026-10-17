# Limitations

## Data

- **Small and specific.** 395 students from two secondary schools in Portugal, in one subject (Mathematics). The model may not apply to other schools, countries, subjects, or time periods.
- **Self-reported and coded features.** Many inputs (study time, alcohol use, family relationships, health) are self-reported on coarse scales.
- **Small groups.** Some groups are very small, for example 17 students with 2 past failures and 16 with 3. Averages for them are unreliable.
- **Dataset documentation mismatch.** The documentation allows more past failures than the maximum of 3 seen in this file.

## Target and zero grades

- 38 of 395 students (9.6%) have a final grade of 0. The data cannot say why (for example whether they left the course or missed the final assessment).
- In the test set, the model recognized five of eight zero-grade students (predictions below 1.1) and missed three (predictions of 3.5, 4.9, and 6.4). One student whose final grade was 11 looked almost identical to the missed zero-grade students on the available features and was predicted at 5.25. Using these features, such cases cannot be reliably separated.
- The average error figure hides these large misses. Report errors for zero-grade and other students separately.

## Recorded absences

- Every student with a final grade of 0 has 0 recorded absences, and the model has learned this. The partial dependence curve suggests it mainly separates 0-1 absences from the rest.
- This may reflect how records were kept (for example, absences not logged for students who left), not attendance behaviour. For students with high grades the effect on predictions was small in a spot check, but it is larger for students with low grades.
- Among students with a non-zero final grade, more absences go with lower grades (Spearman -0.244), the opposite direction of what the zero-grade pattern suggests. The app shows a warning when absences are 0 or 1.

## Late-term use only

The model needs the first two period grades and relies mostly on the second (`G2`). It estimates how a student is likely to finish given how they are already doing, so it is not an early-warning tool. Models without the earlier grades performed much worse: the best reached an R² of 0.217 in cross-validation, and most were no better than guessing the median.

## Evaluation uncertainty

- The test set has 79 students (8 with a final grade of 0), so test metrics are noisy.
- The test set happened to be somewhat easier than the average cross-validation fold (the baseline also scored better on it), so the test MAE of 0.895 should not be read as better than the cross-validated 1.040.
- The three best models are within the fold-to-fold variation of each other. Random Forest was chosen by a fixed rule.
- Hyperparameters came from a small fixed set, not an extensive search, so cross-validated scores for the chosen model are slightly optimistic.

- A bootstrap interval for the test MAE is wide (0.665 to 1.173), and it understates the real uncertainty because it ignores retraining variability and model selection.

## Explainability

- Permutation importance and partial dependence describe the model, not causes.
- `G1` and `G2` are strongly correlated, so each one's importance understates how much the model gains from grade information.
- Features with importance near zero cannot be ranked on a test set this small.
- Partial dependence averages over all students and can hide differences between groups.

## Fairness and ethics

- Inputs such as sex, age, address type, parents' education and jobs, and alcohol use can reflect social background rather than ability. Using them could lead to unfair treatment.
- Performance across student groups was not measured, so the model may be less accurate for some groups.
- The dataset contains personal information about minors. Do not enter real students' data into a shared or public deployment of the app.
- Predictions must not be used alone to grade, rank, label, or make decisions about a student.

## Software and deployment

- The model file is tied to scikit-learn 1.9.1; the app warns if the version differs.
- Joblib files can execute code when loaded. Only load models you created or trust.
- Streamlit pages, `train.py`, `save_model.py`, and `explainability.py` have no automated tests.
- The app is a local educational demo. It has no authentication, logging, or monitoring.
