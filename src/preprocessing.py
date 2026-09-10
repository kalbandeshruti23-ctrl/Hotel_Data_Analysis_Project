import pandas as pd
import numpy as np


def load_data(file_path):
    """Load hotel booking dataset."""
    df = pd.read_csv(file_path)
    return df


def preprocess_data(df):
    """Clean and prepare hotel booking data."""

    df = df.copy()

    # --------------------------------------------------
    # 1. Remove unnecessary personal information
    # --------------------------------------------------
    personal_columns = [
        "name",
        "email",
        "phone-number",
        "credit_card"
    ]

    df.drop(
        columns=personal_columns,
        errors="ignore",
        inplace=True
    )

    # --------------------------------------------------
    # 2. Remove duplicate records
    # --------------------------------------------------
    df.drop_duplicates(inplace=True)

    # --------------------------------------------------
    # 3. Handle missing values
    # --------------------------------------------------

    if "children" in df.columns:
        df["children"] = df["children"].fillna(0)

    if "country" in df.columns:
        df["country"] = df["country"].fillna(
            df["country"].mode()[0]
        )

    if "agent" in df.columns:
        df["agent"] = df["agent"].fillna(0)

    if "company" in df.columns:
        df["company"] = df["company"].fillna(0)

    # --------------------------------------------------
    # 4. Create arrival date
    # --------------------------------------------------

    df["arrival_date"] = pd.to_datetime(
        df["arrival_date_year"].astype(str)
        + "-"
        + df["arrival_date_month"]
        + "-"
        + df["arrival_date_day_of_month"].astype(str),
        errors="coerce"
    )

    # --------------------------------------------------
    # 5. Feature Engineering
    # --------------------------------------------------

    df["total_stay_nights"] = (
        df["stays_in_weekend_nights"]
        + df["stays_in_week_nights"]
    )

    df["total_guests"] = (
        df["adults"]
        + df["children"]
        + df["babies"]
    )

    df["total_previous_bookings"] = (
        df["previous_bookings_not_canceled"]
        + df["previous_cancellations"]
    )

    df["estimated_revenue"] = (
        df["adr"]
        * df["total_stay_nights"]
    )

    df["arrival_month_num"] = (
        df["arrival_date"].dt.month
    )

    df["arrival_day"] = (
        df["arrival_date"].dt.day
    )

    df["arrival_weekday"] = (
        df["arrival_date"].dt.dayofweek
    )

    # --------------------------------------------------
    # 6. Remove invalid stays
    # --------------------------------------------------

    df = df[
        df["total_stay_nights"] >= 0
    ]

    # Remove records with zero guests
    df = df[
        df["total_guests"] > 0
    ]

    return df


if __name__ == "__main__":

    file_path = "../data/hotel_booking_10000.csv"

    df = load_data(file_path)

    print("Original dataset shape:", df.shape)

    df = preprocess_data(df)

    print("Processed dataset shape:", df.shape)

    print("\nMissing values:")
    print(
        df.isnull().sum()
        .sort_values(ascending=False)
        .head(10)
    )

    print("\nFirst 5 records:")
    print(df.head())