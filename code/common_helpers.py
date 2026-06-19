import csv
import pandas as pd

def read_dataset(file_path: str) -> pd.DataFrame:
    """Read a CSV dataset into a pandas DataFrame."""
    return pd.read_csv(file_path)

def write_dataset(file_path: str, df: pd.DataFrame) -> None:
    """Write a pandas DataFrame to a CSV file, matching sample_claims.csv format exactly:
    - All fields quoted
    - Boolean values as lowercase strings ('true'/'false')
    """
    # Normalize boolean columns to lowercase string before writing
    bool_cols = ['evidence_standard_met', 'valid_image']
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda v: 'true' if str(v).lower() in ('true', '1', 'yes') else 'false'
            )

    df.to_csv(file_path, index=False, quoting=csv.QUOTE_ALL)
