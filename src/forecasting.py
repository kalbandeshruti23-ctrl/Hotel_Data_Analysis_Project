import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)

from statsmodels.tsa.arima.model import ARIMA


def prepare_demand_data(df):

    demand = (
        df.groupby("arrival_date")
        .size()
        .reset_index(
            name="booking_demand"
        )
    )

    demand = demand.sort_values(
        "arrival_date"
    )

    demand["day_number"] = range(
        len(demand)
    )

    demand["month"] = (
        demand["arrival_date"].dt.month
    )

    demand["day_of_week"] = (
        demand["arrival_date"].dt.dayofweek
    )

    demand["year"] = (
        demand["arrival_date"].dt.year
    )

    return demand


def linear_regression_forecast(demand):

    features = [
        "day_number",
        "month",
        "day_of_week",
        "year"
    ]

    X = demand[features]
    y = demand["booking_demand"]

    split_index = int(
        len(demand) * 0.8
    )

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    model = LinearRegression()

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    mse = mean_squared_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(mse)

    print("\nLinear Regression Results")

    print("MAE:", round(mae, 4))
    print("MSE:", round(mse, 4))
    print("RMSE:", round(rmse, 4))

    plt.figure(figsize=(14, 6))

    plt.plot(
        y_test.index,
        y_test.values,
        label="Actual"
    )

    plt.plot(
        y_test.index,
        predictions,
        label="Predicted"
    )

    plt.title(
        "Linear Regression - Actual vs Predicted Demand"
    )

    plt.xlabel("Time")
    plt.ylabel("Booking Demand")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "visualizations/"
        "linear_regression_forecast.png"
    )

    plt.close()

    return model, predictions, mae, mse, rmse


def arima_forecast(demand):

    series = demand[
        "booking_demand"
    ].astype(float)

    split_index = int(
        len(series) * 0.8
    )

    train = series.iloc[
        :split_index
    ]

    test = series.iloc[
        split_index:
    ]

    print(
        "\nTraining ARIMA model..."
    )

    model = ARIMA(
        train,
        order=(5, 1, 0)
    )

    fitted_model = model.fit()

    predictions = fitted_model.forecast(
        steps=len(test)
    )

    mae = mean_absolute_error(
        test,
        predictions
    )

    mse = mean_squared_error(
        test,
        predictions
    )

    rmse = np.sqrt(mse)

    print("\nARIMA Results")

    print("MAE:", round(mae, 4))
    print("MSE:", round(mse, 4))
    print("RMSE:", round(rmse, 4))

    plt.figure(figsize=(14, 6))

    plt.plot(
        test.index,
        test.values,
        label="Actual"
    )

    plt.plot(
        test.index,
        predictions,
        label="ARIMA Forecast"
    )

    plt.title(
        "ARIMA - Actual vs Forecasted Demand"
    )

    plt.xlabel("Time")
    plt.ylabel("Booking Demand")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "visualizations/"
        "arima_forecast.png"
    )

    plt.close()

    return (
        fitted_model,
        predictions,
        mae,
        mse,
        rmse
    )