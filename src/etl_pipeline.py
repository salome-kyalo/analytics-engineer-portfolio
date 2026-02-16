"""
Telco Customer Churn — ETL Pipeline
Author: Salome Kyalo

This script:
1. Loads segmented telco churn datasets
2. Merges them into a unified customer-level dataset
3. Cleans and validates business-logic nulls
4. Engineers analytical features
5. Exports a processed dataset ready for modeling
"""

import os
import pandas as pd


# ------------------------------------------------------------
# Load Data
# ------------------------------------------------------------

def load_data(base_path: str) -> dict:
    """
    Load all customer-level CSV files.
    Returns dictionary of DataFrames.
    """

    datasets = {
        "demographics": pd.read_csv(os.path.join(base_path, "Telco_customer_churn_demographics.csv")),
        "location": pd.read_csv(os.path.join(base_path, "Telco_customer_churn_location.csv")),
        "services": pd.read_csv(os.path.join(base_path, "Telco_customer_churn_services.csv")),
        "status": pd.read_csv(os.path.join(base_path, "Telco_customer_churn_status.csv")),
    }

    return datasets


# ------------------------------------------------------------
# Merge Tables
# ------------------------------------------------------------

def merge_tables(datasets: dict) -> pd.DataFrame:
    """
    Merge customer-level tables using LEFT JOIN on Customer ID.
    """

    df = datasets["demographics"] \
        .merge(datasets["location"], on="Customer ID", how="left") \
        .merge(datasets["services"], on="Customer ID", how="left") \
        .merge(datasets["status"], on="Customer ID", how="left")

    return df


# ------------------------------------------------------------
# Clean & Validate
# ------------------------------------------------------------

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate dataset integrity and handle business-logic nulls.
    """

    # Ensure customer ID uniqueness
    assert df["Customer ID"].is_unique, "Customer IDs are not unique"

    # Business-logic null handling
    df["Churn Reason"] = df["Churn Reason"].fillna("Not Churned")
    df["Churn Category"] = df["Churn Category"].fillna("Not Churned")
    df["Offer"] = df["Offer"].fillna("No Offer")
    df["Internet Type"] = df["Internet Type"].fillna("No Internet")

    # Encode churn label numerically
    df["Churn Value"] = df["Churn Label"].map({"Yes": 1, "No": 0})

    return df


# ------------------------------------------------------------
# Feature Engineering
# ------------------------------------------------------------

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create analytical features for churn analysis.
    """

    # Revenue efficiency
    df["Revenue Per Month"] = (
        df["Total Revenue"] / df["Tenure in Months"].replace(0, 1)
    )

    # High-value segmentation
    df["High Value Customer"] = (
        df["Revenue Per Month"] > df["Revenue Per Month"].median()
    )

    # Engagement score (service adoption count)
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


# ------------------------------------------------------------
# Save Output
# ------------------------------------------------------------

def save_data(df: pd.DataFrame, output_path: str):
    """
    Export cleaned and engineered dataset.
    """
    df.to_csv(output_path, index=False)


# ------------------------------------------------------------
# Main Pipeline Execution
# ------------------------------------------------------------

def main():

    base_path = "data"
    output_path = "data/customer_churn_processed.csv"

    print("Loading datasets...")
    datasets = load_data(base_path)

    print("Merging tables...")
    df = merge_tables(datasets)

    print("Cleaning and validating...")
    df = clean_data(df)

    print("Engineering features...")
    df = feature_engineering(df)

    print("Saving processed dataset...")
    save_data(df, output_path)

    print("ETL pipeline completed successfully.")


if __name__ == "__main__":
    main()
