"""Data loading utilities for the Student Performance project.

The UCI Student Performance dataset stores two subjects in separate CSV
files, using a semicolon (;) as the separator instead of a comma.
"""

from pathlib import Path

import pandas as pd

# Project root = the folder that contains `src/`
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

DATA_FILES = {
    "mat": "student-mat.csv",  # Mathematics
    "por": "student-por.csv",  # Portuguese language
}

TARGET_COLUMN = "G3"  # Final grade, on a 0-20 scale


def load_student_data(subject: str = "mat") -> pd.DataFrame:
    """Load the raw student dataset for one subject.

    Args:
        subject: "mat" for Mathematics or "por" for Portuguese.

    Returns:
        The raw data as a pandas DataFrame.

    Raises:
        ValueError: If the subject is not "mat" or "por".
        FileNotFoundError: If the CSV file is not in data/raw/.
    """
    if subject not in DATA_FILES:
        raise ValueError(
            f"Unknown subject '{subject}'. Choose from: {list(DATA_FILES)}"
        )

    file_path = RAW_DATA_DIR / DATA_FILES[subject]
    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {file_path}. Download it from "
            "https://archive.ics.uci.edu/dataset/320/student+performance "
            "and place the CSV files in data/raw/."
        )

    return pd.read_csv(file_path, sep=";")


def describe_dataset(df: pd.DataFrame) -> None:
    """Print a structural overview of the dataset."""
    print("=" * 60)
    print("SHAPE (rows, columns):", df.shape)

    print("\nCOLUMN NAMES:")
    print(list(df.columns))

    print("\nDATA TYPES:")
    print(df.dtypes.to_string())

    print("\nFIRST 5 ROWS:")
    print(df.head().to_string())

    print("\nSUMMARY STATISTICS (numerical columns):")
    print(df.describe().T.to_string())

    print("\nMISSING VALUES PER COLUMN (only columns with any):")
    missing = df.isna().sum()
    print(missing[missing > 0].to_string() if missing.any() else "None")

    if TARGET_COLUMN in df.columns:
        print(f"\nTARGET '{TARGET_COLUMN}' SUMMARY:")
        print(df[TARGET_COLUMN].describe().to_string())
    else:
        print(f"\nWARNING: target column '{TARGET_COLUMN}' not found.")
    print("=" * 60)


if __name__ == "__main__":
    data = load_student_data("mat")
    describe_dataset(data)