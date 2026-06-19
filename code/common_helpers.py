import pandas as pd

def read_dataset(file_path: str) -> pd.DataFrame:
    """Read a CSV dataset into a pandas DataFrame."""
    return pd.read_csv(file_path)

def write_dataset(file_path: str, df: pd.DataFrame) -> None:
    """Write a pandas DataFrame to a CSV file."""
    df.to_csv(file_path, index=False)
