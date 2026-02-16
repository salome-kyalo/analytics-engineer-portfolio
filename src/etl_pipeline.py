"""
Telco Churn ETL Pipeline
Author: Salome Kyalo

This script:
1. Loads raw telco data
2. Cleans and validates fields
3. Engineers analytical features
4. Exports processed dataset
"""

import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    """Load dataset from CSV."""
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Basic data cleaning and validation."""
    df = df.dropna()

    # Convert churn label to numeric
    df["Churn Value"] = df["Churn Label"].map({"Yes": 1, "No": 0})

    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Create analytical features."""

    # Revenue per month
    df["Revenue Per Month"] = (
        df["Total Revenue"] / df["Tenure in Months"].replace(0, 1)
    )

    # High value customer flag
    df["High Value Customer"] = (
        df["Revenue Per Month"] > df["Revenue Per Month"].median()
    )

    # Engagement score
    service_cols = [
        "Online Security",
        "Online Backup",
        "Device Protection Plan",
        "Premium Tech Support",
        "Streaming TV",
        "Streaming Movies",
        "Streaming Music",
    ]

    df["Engagement Score"] = (
        df[service_cols]
        .apply(lambda row: (row == "Yes").sum(), axis=1)
    )

    return df


def save_data(df: pd.DataFrame, path: str):
    """Export processed dataset."""
    df.to_csv(path, index=False)


def main():
    input_path = "data/telco_raw.csv"
    output_path = "data/telco_processed.csv"

    df = load_data(input_path)
    df = clean_data(df)
    df = feature_engineering(df)
    save_data(df, output_path)

    print("ETL pipeline completed successfully.")


if __name__ == "__main__":
    main()
