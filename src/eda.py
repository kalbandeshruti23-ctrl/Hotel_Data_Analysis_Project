import os
import matplotlib.pyplot as plt
import seaborn as sns


def create_visualization_folder():

    os.makedirs(
        "visualizations",
        exist_ok=True
    )


def hotel_distribution(df):

    plt.figure(figsize=(8, 5))

    sns.countplot(
        data=df,
        x="hotel"
    )

    plt.title(
        "Distribution of Hotel Bookings"
    )

    plt.xlabel("Hotel Type")
    plt.ylabel("Number of Bookings")

    plt.tight_layout()

    plt.savefig(
        "visualizations/hotel_distribution.png"
    )

    plt.close()


def cancellation_distribution(df):

    plt.figure(figsize=(8, 5))

    sns.countplot(
        data=df,
        x="is_canceled"
    )

    plt.title(
        "Booking Cancellation Distribution"
    )

    plt.xlabel(
        "Cancellation Status "
        "(0 = Not Cancelled, 1 = Cancelled)"
    )

    plt.ylabel("Number of Bookings")

    plt.tight_layout()

    plt.savefig(
        "visualizations/cancellation_distribution.png"
    )

    plt.close()


def customer_type_distribution(df):

    plt.figure(figsize=(10, 5))

    sns.countplot(
        data=df,
        x="customer_type",
        order=df["customer_type"].value_counts().index
    )

    plt.title(
        "Customer Type Distribution"
    )

    plt.xlabel("Customer Type")
    plt.ylabel("Number of Bookings")

    plt.xticks(rotation=30)

    plt.tight_layout()

    plt.savefig(
        "visualizations/customer_type_distribution.png"
    )

    plt.close()


def market_segment_distribution(df):

    plt.figure(figsize=(10, 5))

    sns.countplot(
        data=df,
        x="market_segment",
        order=df["market_segment"].value_counts().index
    )

    plt.title(
        "Market Segment Distribution"
    )

    plt.xlabel("Market Segment")
    plt.ylabel("Number of Bookings")

    plt.xticks(rotation=30)

    plt.tight_layout()

    plt.savefig(
        "visualizations/market_segment_distribution.png"
    )

    plt.close()


def adr_distribution(df):

    plt.figure(figsize=(10, 5))

    sns.histplot(
        df["adr"],
        bins=40,
        kde=True
    )

    plt.title(
        "Distribution of Average Daily Rate (ADR)"
    )

    plt.xlabel("ADR")
    plt.ylabel("Frequency")

    plt.tight_layout()

    plt.savefig(
        "visualizations/adr_distribution.png"
    )

    plt.close()


def demand_over_time(df):

    daily_demand = (
        df.groupby("arrival_date")
        .size()
        .reset_index(
            name="booking_demand"
        )
    )

    plt.figure(figsize=(14, 6))

    plt.plot(
        daily_demand["arrival_date"],
        daily_demand["booking_demand"]
    )

    plt.title(
        "Hotel Booking Demand Over Time"
    )

    plt.xlabel("Arrival Date")
    plt.ylabel("Number of Bookings")

    plt.xticks(rotation=45)

    plt.tight_layout()

    plt.savefig(
        "visualizations/demand_over_time.png"
    )

    plt.close()

    return daily_demand


def generate_eda(df):

    create_visualization_folder()

    hotel_distribution(df)

    cancellation_distribution(df)

    customer_type_distribution(df)

    market_segment_distribution(df)

    adr_distribution(df)

    daily_demand = demand_over_time(df)

    print(
        "EDA visualizations generated successfully."
    )

    return daily_demand