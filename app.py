"""Streamlit application: Student Performance Prediction.

Run from the project root with:  streamlit run app.py

Stage 1 contains the Home and About pages. Other pages are added later.
All results shown here are read from files produced by earlier phases.
"""
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.data_loader import PROJECT_ROOT, TARGET_COLUMN, load_student_data
from src.predict import (
    CATEGORICAL_VALUES,
    MODEL_PATH,
    NUMERIC_RANGES,
    InputValidationError,
    ModelNotFoundError,
    load_metadata,
    load_model,
    predict_grade,
)

TEST_RESULTS_PATH = PROJECT_ROOT / "reports" / "model_results" / "test_results.csv"
RESULTS_DIR = PROJECT_ROOT / "reports" / "model_results"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide",
)


@st.cache_data
def load_dataset() -> pd.DataFrame:
    """Load the raw dataset once and reuse it across reruns."""
    return load_student_data("mat")


@st.cache_data
def load_test_results() -> pd.DataFrame | None:
    """Load the saved test-set results, or None if the file is missing."""
    if not TEST_RESULTS_PATH.exists():
        return None
    return pd.read_csv(TEST_RESULTS_PATH)


def show_disclaimer() -> None:
    """Standard reminder that predictions are estimates."""
    st.warning(
        "Predictions are statistical estimates, **not official grades** and "
        "not guaranteed outcomes. They must not be used on their own to make "
        "decisions about any student."
    )


def page_home() -> None:
    st.title("🎓 Student Performance Prediction")
    st.caption("A machine learning project: predicting a student's final grade (G3, 0-20)")

    st.header("Introduction")
    st.write(
        "This application predicts the final grade of a secondary-school "
        "student in a Mathematics course from background information and the "
        "grades of the first two periods. It was built as an end-to-end "
        "machine learning project: data inspection, preprocessing, model "
        "comparison, evaluation, and explanation."
    )

    st.header("Objectives")
    st.markdown(
        "- Train and compare several regression models on real student data.\n"
        "- Evaluate them honestly: cross-validation for selection, an untouched test set for the final check.\n"
        "- Explain which inputs the model relies on, without claiming causes.\n"
        "- Make the trained model usable through a simple, validated interface."
    )

    st.header("Dataset overview")
    try:
        df = load_dataset()
    except FileNotFoundError as exc:
        st.error(str(exc))
    else:
        zero_share = (df[TARGET_COLUMN] == 0).mean() * 100
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Students", len(df))
        col2.metric("Input features", df.shape[1] - 1)
        col3.metric("Mean final grade (G3)", f"{df[TARGET_COLUMN].mean():.1f}")
        col4.metric("Students with G3 = 0", f"{zero_share:.1f}%")
        st.caption(
            "Source: UCI Machine Learning Repository, Student Performance "
            "dataset (Mathematics course file)."
        )
        with st.expander("Preview the first rows"):
            st.dataframe(df.head(10))

    st.header("Model information")
    metadata = load_metadata()
    if metadata is None or not MODEL_PATH.exists():
        st.warning(
            "No saved model was found. Create it by running "
            "`python -m src.save_model` in the project folder."
        )
    else:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Model", metadata["model_name"])
        col2.metric("Feature setup", metadata["setup"])
        col3.metric("Training students", metadata["n_train_rows"])
        col4.metric("scikit-learn", metadata["sklearn_version"])

    results = load_test_results()
    if results is None:
        st.info(
            "Test-set results were not found. Create them with "
            "`python -m src.evaluate`."
        )
    else:
        st.subheader("Test-set results")
        st.dataframe(
            results.drop(columns=["setup"], errors="ignore").round(3),
            hide_index=True,
        )
        st.caption(
            "Measured once on students the model never saw during training or "
            "model selection. MAE and RMSE are in grade points (0-20 scale). "
            "The test set is small, so these numbers are noisy."
        )

    show_disclaimer()


def page_about() -> None:
    st.title("About this project")

    st.header("Methodology")
    st.markdown(
        "1. **Data inspection and exploration:** checked missing values, "
        "duplicates, value ranges, distributions, and relationships.\n"
        "2. **Preprocessing:** numeric columns imputed and standardized; "
        "categorical columns one-hot encoded. All fitted on training data only.\n"
        "3. **Splitting:** 80/20 train/test split, stratified on whether the "
        "final grade is zero, with a fixed random seed.\n"
        "4. **Baseline:** a model that always predicts the median grade.\n"
        "5. **Model comparison:** linear, Ridge, decision tree, random forest, "
        "and gradient boosting models compared with 5-fold cross-validation "
        "on the training set.\n"
        "6. **Final evaluation:** the model with the lowest cross-validated "
        "error was evaluated once on the test set.\n"
        "7. **Explanation:** permutation importance and partial dependence."
    )

    st.header("Technology stack")
    st.markdown(
        "Python, pandas, NumPy, Matplotlib, scikit-learn, Joblib, Streamlit, "
        "pytest, Git and GitHub, Jupyter Notebook."
    )

    st.header("Dataset source")
    st.markdown(
        "Cortez, P. (2008). *Student Performance* [Dataset]. UCI Machine "
        "Learning Repository. https://doi.org/10.24432/C5TG7T  \n"
        "Licence: Creative Commons Attribution 4.0 (CC BY 4.0). Records come "
        "from two secondary schools in Portugal."
    )

    st.header("Ethical considerations")
    st.markdown(
        "- **Not for high-stakes decisions.** Do not use predictions alone to "
        "grade, rank, label, or make decisions about a student.\n"
        "- **Sensitive inputs.** Some features (sex, age, home address type, "
        "parents' education and jobs, alcohol use, relationship status) can "
        "reflect social background rather than ability, and could lead to "
        "unfair treatment if relied on.\n"
        "- **Privacy.** The dataset contains personal information about "
        "minors. Do not enter real students' data into a shared or public "
        "deployment of this app.\n"
        "- **Association, not cause.** The explanations show what the model "
        "uses, not what causes academic performance."
    )

    st.header("Limitations")
    st.markdown(
        "- **Small dataset:** 395 students from two schools in one country, "
        "so the model may not transfer to other schools or countries.\n"
        "- **Needs earlier grades:** the model uses the first two period "
        "grades, so it works late in the term and is not an early-warning tool.\n"
        "- **Sudden zero grades:** some students end with a grade of 0 and "
        "look like ordinary low performers beforehand. The model catches "
        "some of them and misses others.\n"
        "- **Recorded absences:** every student with a zero final grade has "
        "0 recorded absences, and the model has learned this pattern. It may "
        "reflect how records were kept, not attendance behaviour.\n"
        "- **Small test set:** the test results are based on 79 students "
        "and carry substantial uncertainty.\n"
        "- **One subject:** only the Mathematics file was used."
    )


# ---------------------------------------------------------------------------
# Student Prediction page
# ---------------------------------------------------------------------------

# Readable labels for the dataset's coded values (codes are what the model uses)
EDUCATION = {
    0: "0 – none",
    1: "1 – primary (4th grade)",
    2: "2 – 5th to 9th grade",
    3: "3 – secondary education",
    4: "4 – higher education",
}
TRAVEL_TIME = {
    1: "1 – under 15 min",
    2: "2 – 15 to 30 min",
    3: "3 – 30 min to 1 hour",
    4: "4 – over 1 hour",
}
STUDY_TIME = {
    1: "1 – under 2 hours",
    2: "2 – 2 to 5 hours",
    3: "3 – 5 to 10 hours",
    4: "4 – over 10 hours",
}
JOB_LABELS = {
    "teacher": "Teacher",
    "health": "Health care related",
    "services": "Civil services (e.g. administrative, police)",
    "at_home": "At home",
    "other": "Other",
}
CATEGORY_LABELS = {
    "school": {"GP": "Gabriel Pereira", "MS": "Mousinho da Silveira"},
    "sex": {"F": "Female", "M": "Male"},
    "address": {"U": "Urban", "R": "Rural"},
    "famsize": {"LE3": "3 or fewer members", "GT3": "More than 3 members"},
    "Pstatus": {"T": "Living together", "A": "Living apart"},
    "Mjob": JOB_LABELS,
    "Fjob": JOB_LABELS,
    "reason": {
        "home": "Close to home",
        "reputation": "School reputation",
        "course": "Course preference",
        "other": "Other",
    },
    "guardian": {"mother": "Mother", "father": "Father", "other": "Other"},
}


def _scale_labels(low_text: str, high_text: str) -> dict:
    """Labels for a 1-5 scale, naming only the two ends."""
    return {1: f"1 – {low_text}", 2: "2", 3: "3", 4: "4", 5: f"5 – {high_text}"}


@st.cache_resource
def _load_model_cached(modified_time: float):
    """Load the model once. The file's modified time is the cache key, so a
    re-saved model file is picked up automatically."""
    return load_model()


def get_model():
    """Return the saved pipeline, or raise ModelNotFoundError."""
    if not MODEL_PATH.exists():
        raise ModelNotFoundError(
            "No saved model was found. Create it by running "
            "`python -m src.save_model` in the project folder."
        )
    return _load_model_cached(MODEL_PATH.stat().st_mtime)


def get_test_mae() -> float | None:
    """Mean absolute error of the final model on the test set, if available."""
    results = load_test_results()
    if results is None:
        return None
    final = results[~results["model"].str.startswith("Baseline")]
    return float(final["mae"].iloc[0]) if not final.empty else None


def _category_input(column: str, label: str, default: str, help_text=None):
    """Dropdown for a text-valued feature with readable labels."""
    options = CATEGORICAL_VALUES[column]
    labels = CATEGORY_LABELS.get(column, {})
    return st.selectbox(
        label,
        options,
        index=options.index(default),
        format_func=lambda value: labels.get(value, value.capitalize()),
        key=column,
        help=help_text,
    )


def _coded_input(column: str, label: str, labels: dict, default: int, help_text=None):
    """Dropdown for a number-coded feature with readable labels."""
    options = list(labels)
    return st.selectbox(
        label,
        options,
        index=options.index(default),
        format_func=lambda value: labels[value],
        key=column,
        help=help_text,
    )


def _number_input(column: str, label: str, default: int, help_text=None):
    """Whole-number input limited to the range the model accepts."""
    low, high = NUMERIC_RANGES[column]
    return st.number_input(
        label,
        min_value=low,
        max_value=high,
        value=default,
        step=1,
        key=column,
        help=help_text,
    )


def build_input_form() -> tuple[bool, dict]:
    """Draw the input form. Returns (was_submitted, record)."""
    scale_quality = _scale_labels("very bad", "excellent")
    scale_level = _scale_labels("very low", "very high")
    scale_health = _scale_labels("very bad", "very good")

    with st.form("student_form"):
        st.subheader("Grades so far")
        col1, col2 = st.columns(2)
        with col1:
            g1 = _number_input("G1", "First period grade (G1, 0-20)", 11)
        with col2:
            g2 = _number_input("G2", "Second period grade (G2, 0-20)", 11)

        st.subheader("Student and family")
        col1, col2, col3 = st.columns(3)
        with col1:
            school = _category_input("school", "School", "GP")
            sex = _category_input("sex", "Sex", "F")
            age = _number_input("age", "Age", 17)
            address = _category_input("address", "Home address type", "U")
        with col2:
            famsize = _category_input("famsize", "Family size", "GT3")
            pstatus = _category_input("Pstatus", "Parents' cohabitation", "T")
            guardian = _category_input("guardian", "Guardian", "mother")
            medu = _coded_input("Medu", "Mother's education", EDUCATION, 3)
        with col3:
            fedu = _coded_input("Fedu", "Father's education", EDUCATION, 2)
            mjob = _category_input("Mjob", "Mother's job", "other")
            fjob = _category_input("Fjob", "Father's job", "other")

        st.subheader("School and study")
        col1, col2, col3 = st.columns(3)
        with col1:
            reason = _category_input("reason", "Reason for choosing the school", "course")
            traveltime = _coded_input("traveltime", "Travel time to school", TRAVEL_TIME, 1)
            studytime = _coded_input("studytime", "Weekly study time", STUDY_TIME, 2)
            failures = _number_input(
                "failures", "Past class failures", 0, help_text="0 to 3 recorded in the data"
            )
        with col2:
            schoolsup = _category_input("schoolsup", "Extra educational support", "no")
            famsup = _category_input("famsup", "Family educational support", "yes")
            paid = _category_input("paid", "Extra paid classes in this subject", "no")
            activities = _category_input("activities", "Extra-curricular activities", "yes")
        with col3:
            nursery = _category_input("nursery", "Attended nursery school", "yes")
            higher = _category_input("higher", "Wants higher education", "yes")
            absences = _number_input(
                "absences",
                "Number of school absences",
                4,
                help_text="Enter the number actually recorded. See the note under the result.",
            )

        st.subheader("Home and lifestyle")
        col1, col2, col3 = st.columns(3)
        with col1:
            internet = _category_input("internet", "Internet access at home", "yes")
            romantic = _category_input("romantic", "In a romantic relationship", "no")
            famrel = _coded_input("famrel", "Quality of family relationships", scale_quality, 4)
        with col2:
            freetime = _coded_input("freetime", "Free time after school", scale_level, 3)
            goout = _coded_input("goout", "Going out with friends", scale_level, 3)
            health = _coded_input("health", "Current health status", scale_health, 4)
        with col3:
            dalc = _coded_input("Dalc", "Workday alcohol consumption", scale_level, 1)
            walc = _coded_input("Walc", "Weekend alcohol consumption", scale_level, 2)

        submitted = st.form_submit_button("Predict final grade", type="primary")

    record = {
        "school": school, "sex": sex, "age": age, "address": address,
        "famsize": famsize, "Pstatus": pstatus, "Medu": medu, "Fedu": fedu,
        "Mjob": mjob, "Fjob": fjob, "reason": reason, "guardian": guardian,
        "traveltime": traveltime, "studytime": studytime, "failures": failures,
        "schoolsup": schoolsup, "famsup": famsup, "paid": paid,
        "activities": activities, "nursery": nursery, "higher": higher,
        "internet": internet, "romantic": romantic, "famrel": famrel,
        "freetime": freetime, "goout": goout, "Dalc": dalc, "Walc": walc,
        "health": health, "absences": absences, "G1": g1, "G2": g2,
    }
    return submitted, record


def show_prediction_result(prediction: float, record: dict) -> None:
    """Show the predicted grade, its context, and the relevant warnings."""
    st.success("Prediction complete")

    col1, col2 = st.columns(2)
    col1.metric("Estimated final grade (G3)", f"{prediction:.1f} / 20")
    mae = get_test_mae()
    if mae is not None:
        col2.metric("Average error on the test set", f"about {mae:.1f} grade points")

    # Where does this estimate sit among the students in the dataset?
    grades = load_dataset()[TARGET_COLUMN]
    q25, q75 = grades.quantile([0.25, 0.75])
    if prediction < q25:
        position = f"in the lower quarter of students in the dataset (below {q25:.0f})"
    elif prediction > q75:
        position = f"in the upper quarter of students in the dataset (above {q75:.0f})"
    else:
        position = f"in the middle half of students in the dataset ({q25:.0f} to {q75:.0f})"
    st.write(f"This estimate falls {position}.")

    if record["absences"] <= 1:
        st.warning(
            "**Recorded absences of 0 or 1:** in the training data, every student "
            "with a final grade of 0 had 0 recorded absences, and the model has "
            "learned this pattern. It may reflect how records were kept, not "
            "attendance behaviour. The estimate is less reliable here, and the "
            "model may give a lower grade for fewer absences."
        )

    with st.expander("How to read this prediction"):
        st.markdown(
            "- This is an **estimate**, not an official grade or a guaranteed outcome.\n"
            "- The model relies mostly on the **second-period grade (G2)**, so the "
            "estimate largely reflects how the student was already doing.\n"
            "- The typical error is an **average**. In the test set, some students "
            "were off by about 5 grade points or more, mostly students whose final "
            "grade was 0 after ordinary earlier grades. The model cannot foresee "
            "every sudden change.\n"
            "- The model describes patterns in 395 students from two schools in "
            "Portugal. It may not apply to other schools or countries.\n"
            "- Features such as sex, age, address type, parents' education and "
            "jobs can reflect social background. The model uses them as inputs, "
            "but that does not mean they cause better or worse grades.\n"
            "- Do not use this estimate alone to make decisions about a student."
        )

    with st.expander("Inputs sent to the model"):
        st.dataframe(pd.DataFrame([record]).T.rename(columns={0: "value"}))


def page_prediction() -> None:
    st.title("Student Prediction")
    st.write(
        "Enter a student's information and the grades from the first two "
        "periods. The model estimates the final grade (G3) on the 0-20 scale. "
        "The starting values are only placeholders: replace all of them with "
        "the student's actual information."
    )
    st.info(
        "Do not enter real students' data if this app is shared or deployed "
        "publicly. The model needs the first two period grades, so it is "
        "meant for use late in the term."
    )

    try:
        model = get_model()
    except ModelNotFoundError as exc:
        st.error(str(exc))
        return

    submitted, record = build_input_form()
    if submitted:
        try:
            prediction = predict_grade(record, model)
        except InputValidationError as exc:
            st.error(f"The input is not valid: {exc}")
        else:
            show_prediction_result(prediction, record)

    show_disclaimer()

# ---------------------------------------------------------------------------
# Shared helpers for the result pages
# ---------------------------------------------------------------------------


@st.cache_data
def read_result_csv(filename: str) -> pd.DataFrame | None:
    """Read a results table from reports/model_results/, or None if missing."""
    path = RESULTS_DIR / filename
    return pd.read_csv(path) if path.exists() else None


def show_figure(filename: str, caption: str, command: str) -> None:
    """Show a saved figure, or explain how to create it."""
    path = FIGURES_DIR / filename
    if path.exists():
        st.image(str(path), caption=caption)
    else:
        st.info(f"Figure `{filename}` not found. Create it by running `{command}`.")


def show_missing_results(filename: str, command: str) -> None:
    st.error(f"`{filename}` was not found. Create it by running `{command}`.")


# ---------------------------------------------------------------------------
# Data Explorer page
# ---------------------------------------------------------------------------


def _is_many_valued_number(series: pd.Series, threshold: int) -> bool:
    return pd.api.types.is_numeric_dtype(series) and series.nunique() > threshold


def _plot_distribution(df: pd.DataFrame, column: str) -> None:
    series = df[column]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    if _is_many_valued_number(series, 25):
        ax.hist(series, bins=20)
    else:
        counts = series.value_counts().sort_index()
        ax.bar(counts.index.astype(str), counts.values)
    ax.set_xlabel(column)
    ax.set_ylabel("Number of students")
    ax.set_title(f"Distribution of {column}")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _plot_relationship(df: pd.DataFrame, column: str) -> None:
    series = df[column]
    fig, ax = plt.subplots(figsize=(7, 4))
    if _is_many_valued_number(series, 10):
        ax.scatter(series, df[TARGET_COLUMN], alpha=0.4)
    else:
        groups = sorted(series.unique())
        ax.boxplot(
            [df.loc[series == g, TARGET_COLUMN] for g in groups],
            tick_labels=[str(g) for g in groups],
        )
    ax.set_xlabel(column)
    ax.set_ylabel("Final grade (G3)")
    ax.set_title(f"G3 by {column}")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    if not _is_many_valued_number(series, 10):
        table = df.groupby(column)[TARGET_COLUMN].agg(["count", "mean"]).round(2)
        st.dataframe(table.rename(columns={"count": "students", "mean": "mean G3"}))
        st.caption(
            "Check the group sizes: averages for small groups are unreliable."
        )


def page_data_explorer() -> None:
    st.title("Data Explorer")
    st.write(
        "Explore the dataset behind the model. These charts show **associations "
        "in this sample**, not causes."
    )
    try:
        df = load_dataset()
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    tab_summary, tab_distributions, tab_relationships = st.tabs(
        ["Summary", "Distributions", "Relationships with G3"]
    )

    with tab_summary:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Students", len(df))
        col2.metric("Columns", df.shape[1])
        col3.metric("Missing values", int(df.isna().sum().sum()))
        col4.metric("Duplicate rows", int(df.duplicated().sum()))

        st.subheader("Column types")
        types = pd.DataFrame(
            {"type": df.dtypes.astype(str), "distinct values": df.nunique()}
        )
        st.dataframe(types)

        st.subheader("Summary statistics (numeric columns)")
        st.dataframe(df.describe().T.round(2))
        st.caption(
            "Many numeric columns are ordered codes (for example 1 to 5), not "
            "true measurements. See the dataset documentation for their meaning."
        )

    with tab_distributions:
        column = st.selectbox(
            "Feature", list(df.columns), index=list(df.columns).index(TARGET_COLUMN)
        )
        _plot_distribution(df, column)
        if column == TARGET_COLUMN:
            zeros = int((df[TARGET_COLUMN] == 0).sum())
            st.write(
                f"{zeros} students ({zeros / len(df):.1%}) have a final grade of 0, "
                "which forms a separate group at the low end."
            )

    with tab_relationships:
        exclude_zero = st.checkbox(
            "Exclude students with G3 = 0",
            value=False,
            help="Zero grades are a separate group. Excluding them shows how "
            "relationships look among the other students.",
        )
        data = df[df[TARGET_COLUMN] > 0] if exclude_zero else df
        st.caption(f"Using {len(data)} students.")

        features = [c for c in df.columns if c != TARGET_COLUMN]
        column = st.selectbox("Feature", features, index=features.index("failures"))
        _plot_relationship(data, column)

        with st.expander("Spearman correlation of numeric features with G3"):
            numeric = data.select_dtypes(include="number")
            correlations = (
                numeric.corr(method="spearman")[TARGET_COLUMN]
                .drop(TARGET_COLUMN)
                .sort_values()
            )
            st.bar_chart(correlations)
            st.caption(
                "Spearman measures how consistently two variables rise and fall "
                "together. It does not show cause, and it can miss relationships "
                "that are not steadily increasing or decreasing. Hover over a bar "
                "for its exact value."
            )


# ---------------------------------------------------------------------------
# Model Evaluation page
# ---------------------------------------------------------------------------


def page_model_evaluation() -> None:
    st.title("Model Evaluation")
    st.write(
        "Models were compared with 5-fold cross-validation on the training "
        "set. The model with the lowest cross-validated error was then "
        "evaluated **once** on a test set that played no part in choosing it."
    )

    comparison = read_result_csv("model_comparison.csv")
    if comparison is None:
        show_missing_results("model_comparison.csv", "python -m src.train")
        return

    st.header("Model comparison (cross-validation)")
    setup = st.radio("Feature setup", list(comparison["setup"].unique()), horizontal=True)
    subset = comparison[comparison["setup"] == setup].sort_values("cv_mae")

    columns = ["model", "cv_mae", "cv_mae_std", "cv_rmse", "cv_r2", "mae_gain_vs_baseline"]
    st.dataframe(subset[columns].round(3), hide_index=True)

    plot_df = subset.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(plot_df["model"], plot_df["cv_mae"], xerr=plot_df["cv_mae_std"])
    ax.set_xlabel("Cross-validated MAE in grade points (lower is better)")
    ax.set_title(f"Model comparison: {setup}")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption(
        "Bars show the average over 5 folds and whiskers show one standard "
        "deviation across folds. A difference smaller than the whiskers is not "
        "convincing evidence that one model is better. 'mae_gain_vs_baseline' "
        "is how much lower the error is than the best always-predict-the-same-"
        "grade baseline in the same setup; zero or negative means no better "
        "than guessing."
    )

    st.header("Final model on the test set")
    test_results = read_result_csv("test_results.csv")
    if test_results is None:
        show_missing_results("test_results.csv", "python -m src.evaluate")
    else:
        st.dataframe(test_results.drop(columns=["setup"], errors="ignore").round(3), hide_index=True)
        st.caption(
            "MAE and RMSE are in grade points on the 0-20 scale. R² is the "
            "share of variation in final grades explained. The test set is "
            "small, so these numbers are noisy."
        )

    st.subheader("Errors for students with a final grade of 0 vs the rest")
    by_group = read_result_csv("test_error_by_group.csv")
    if by_group is None:
        show_missing_results("test_error_by_group.csv", "python -m src.evaluate")
    else:
        st.dataframe(by_group.round(3), hide_index=True)

    predictions = read_result_csv("test_predictions.csv")
    if predictions is not None:
        zero_students = predictions[predictions["actual"] == 0]
        with st.expander("Predictions for test students whose final grade was 0"):
            st.dataframe(zero_students.round(2), hide_index=True)
            st.caption(
                "Residual = actual minus predicted. The model recognizes some of "
                "these students and misses others, so an average error figure "
                "hides large misses for a few individuals."
            )

    st.header("Prediction plots (test set)")
    col1, col2 = st.columns(2)
    with col1:
        show_figure(
            "test_actual_vs_predicted.png",
            "Actual vs predicted grades. Points on the dashed line are perfect predictions.",
            "python -m src.evaluate",
        )
    with col2:
        show_figure(
            "test_residuals.png",
            "Residuals (actual minus predicted).",
            "python -m src.evaluate",
        )

    show_disclaimer()


# ---------------------------------------------------------------------------
# Feature Importance page
# ---------------------------------------------------------------------------


def page_feature_importance() -> None:
    st.title("Feature Importance")
    st.write(
        "Which inputs does the model rely on? This page uses **permutation "
        "importance**: one input column is shuffled at a time, which breaks its "
        "link to the grade, and we measure how much the model's error on the "
        "test set increases."
    )

    table = read_result_csv("permutation_importance.csv")
    if table is None:
        show_missing_results("permutation_importance.csv", "python -m src.explainability")
        return

    top_n = (
        st.slider("Number of features to show", 5, len(table), min(15, len(table)))
        if len(table) > 5
        else len(table)
    )
    top = table.head(top_n).iloc[::-1]

    fig, ax = plt.subplots(figsize=(8, 0.35 * top_n + 1.5))
    ax.barh(top["feature"], top["mae_increase_mean"], xerr=top["mae_increase_std"])
    ax.set_xlabel("Increase in test MAE when the feature is shuffled (grade points)")
    ax.set_title("Permutation importance")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption(
        "Bars show the average over repeated shuffles and whiskers show one "
        "standard deviation. A bar that is not clearly longer than its whisker "
        "cannot be told apart from noise."
    )

    with st.expander("Full table"):
        full = table.copy()
        full["clearly above noise (rough guide)"] = np_where_yes(
            full["mae_increase_mean"] > 2 * full["mae_increase_std"]
        )
        st.dataframe(full.round(3), hide_index=True)
        st.caption(
            "The last column is only a rough guide: it marks features whose "
            "average increase is more than twice its standard deviation."
        )

    st.header("Partial dependence")
    st.write(
        "These curves show the model's **average** predicted grade as one input "
        "changes, with all other students' values left as they are."
    )
    show_figure(
        "partial_dependence.png",
        "Average predicted final grade as G2, absences, and failures vary.",
        "python -m src.explainability",
    )

    st.header("How to read these results")
    with st.expander("What importance does and does not mean", expanded=True):
        st.markdown(
            "- Importance describes what the **model** relies on. It does **not** "
            "show what causes academic performance.\n"
            "- A feature can score high because it stands in for something else, "
            "or because of how the data was recorded.\n"
            "- A low score does not mean a feature is irrelevant in the real "
            "world. It only means this model gains little from it.\n"
            "- With a small test set, small scores are mostly noise, so avoid "
            "ranking features that are close to zero."
        )
    with st.expander("Correlated features"):
        st.markdown(
            "The first-period grade (G1) and second-period grade (G2) are strongly "
            "related. When one is shuffled, the model can still use the other, so "
            "each one's importance understates how much the model gains from "
            "grade information as a whole."
        )
    with st.expander("Recorded absences"):
        st.markdown(
            "Absences rank high, but the partial dependence curve suggests the "
            "model mainly separates students with 0 or 1 recorded absences from "
            "the rest, not the amount of absence. In the training data, every "
            "student with a final grade of 0 had 0 recorded absences, which may "
            "reflect how records were kept rather than attendance behaviour."
        )
    with st.expander("Limits of partial dependence"):
        st.markdown(
            "Averaging over all students can hide differences between groups, and "
            "it assumes one feature can change while the others stay fixed, which "
            "is unrealistic for related features."
        )

    show_disclaimer()


def np_where_yes(mask: pd.Series) -> pd.Series:
    """Turn a True/False column into 'yes'/'no' text."""
    return mask.map({True: "yes", False: "no"})


PAGES = {
    "Home": page_home,
    "Student Prediction": page_prediction,
    "Data Explorer": page_data_explorer,
    "Model Evaluation": page_model_evaluation,
    "Feature Importance": page_feature_importance,
    "About": page_about,
}


def main() -> None:
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", list(PAGES))
    st.sidebar.caption("Student Performance Prediction · educational project")
    PAGES[selection]()


main()