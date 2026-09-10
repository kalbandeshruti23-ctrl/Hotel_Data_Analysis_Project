import os
import pandas as pd

from src.preprocessing import (
    load_data,
    preprocess_data
)

from src.eda import generate_eda

from src.segmentation import (
    prepare_segmentation_data,
    find_optimal_k,
    apply_kmeans,
    apply_dbscan
)

from src.forecasting import (
    prepare_demand_data,
    linear_regression_forecast,
    arima_forecast
)


# ==========================================
# PROJECT CONFIGURATION
# ==========================================

DATA_PATH = "data/hotel_booking_10000.csv"

os.makedirs(
    "results",
    exist_ok=True
)

os.makedirs(
    "models",
    exist_ok=True
)

os.makedirs(
    "visualizations",
    exist_ok=True
)


# ==========================================
# 1. LOAD DATA
# ==========================================

print("=" * 60)
print("HOTEL DATA ANALYSIS PROJECT")
print("=" * 60)

print("\nLoading dataset...")

df = load_data(DATA_PATH)

print(
    "Original dataset shape:",
    df.shape
)


# ==========================================
# 2. PREPROCESSING
# ==========================================

print("\nPreprocessing dataset...")

df = preprocess_data(df)

print(
    "Processed dataset shape:",
    df.shape
)


# Save processed dataset

df.to_csv(
    "results/processed_hotel_data.csv",
    index=False
)

print(
    "Processed dataset saved."
)


# ==========================================
# 3. EDA
# ==========================================

print("\nGenerating EDA...")

demand_data = generate_eda(df)

print(
    "EDA completed."
)


# ==========================================
# 4. CUSTOMER SEGMENTATION
# ==========================================

print("\n" + "=" * 60)
print("CUSTOMER SEGMENTATION")
print("=" * 60)

segmentation_df, scaled_data, scaler = (
    prepare_segmentation_data(df)
)


# Find optimal K

best_k = find_optimal_k(
    scaled_data
)


# K-Means

kmeans_result, kmeans_model, kmeans_score = (
    apply_kmeans(
        df,
        scaled_data,
        best_k
    )
)


# DBSCAN

dbscan_result, dbscan_model = (
    apply_dbscan(
        df,
        scaled_data
    )
)


# Save clustering results

kmeans_result.to_csv(
    "results/kmeans_customer_segments.csv",
    index=False
)

dbscan_result.to_csv(
    "results/dbscan_customer_segments.csv",
    index=False
)


# ==========================================
# 5. DEMAND FORECASTING
# ==========================================

print("\n" + "=" * 60)
print("DEMAND FORECASTING")
print("=" * 60)

demand_data = prepare_demand_data(df)


# Linear Regression

(
    linear_model,
    linear_predictions,
    linear_mae,
    linear_mse,
    linear_rmse
) = linear_regression_forecast(
    demand_data
)


# ARIMA

(
    arima_model,
    arima_predictions,
    arima_mae,
    arima_mse,
    arima_rmse
) = arima_forecast(
    demand_data
)


# ==========================================
# 6. MODEL COMPARISON
# ==========================================

comparison = pd.DataFrame({

    "Model": [
        "Linear Regression",
        "ARIMA"
    ],

    "MAE": [
        linear_mae,
        arima_mae
    ],

    "MSE": [
        linear_mse,
        arima_mse
    ],

    "RMSE": [
        linear_rmse,
        arima_rmse
    ]

})


print("\nMODEL COMPARISON")
print(
    comparison.round(4)
)


comparison.to_csv(
    "results/model_comparison.csv",
    index=False
)


# ==========================================
# 7. FINAL SUMMARY
# ==========================================

print("\n" + "=" * 60)
print("PROJECT EXECUTION COMPLETED")
print("=" * 60)

print(
    "\nBest K:",
    best_k
)

print(
    "K-Means Silhouette Score:",
    round(kmeans_score, 4)
)

print("\nForecasting Results:")

print(
    comparison.round(4)
)

print(
    "\nAll results saved inside the results/ folder."
)

print(
    "All graphs saved inside the visualizations/ folder."
)