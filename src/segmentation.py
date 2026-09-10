import os
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score


def prepare_segmentation_data(df):

    features = [
        "lead_time",
        "stays_in_weekend_nights",
        "stays_in_week_nights",
        "adults",
        "children",
        "babies",
        "adr",
        "is_repeated_guest",
        "previous_cancellations",
        "previous_bookings_not_canceled",
        "booking_changes",
        "days_in_waiting_list",
        "required_car_parking_spaces",
        "total_of_special_requests"
    ]

    segmentation_df = df[features].copy()

    segmentation_df = segmentation_df.fillna(0)

    scaler = StandardScaler()

    scaled_data = scaler.fit_transform(
        segmentation_df
    )

    return segmentation_df, scaled_data, scaler


def find_optimal_k(scaled_data):

    inertias = []
    silhouette_scores = []

    k_values = range(2, 11)

    for k in k_values:

        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        labels = model.fit_predict(
            scaled_data
        )

        inertias.append(
            model.inertia_
        )

        silhouette_scores.append(
            silhouette_score(
                scaled_data,
                labels
            )
        )

    # Elbow graph
    plt.figure(figsize=(8, 5))

    plt.plot(
        list(k_values),
        inertias,
        marker="o"
    )

    plt.title(
        "Elbow Method for Optimal K"
    )

    plt.xlabel("Number of Clusters")
    plt.ylabel("Inertia")

    plt.tight_layout()

    plt.savefig(
        "visualizations/elbow_method.png"
    )

    plt.close()

    # Silhouette graph
    plt.figure(figsize=(8, 5))

    plt.plot(
        list(k_values),
        silhouette_scores,
        marker="o"
    )

    plt.title(
        "Silhouette Score for K-Means"
    )

    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Score")

    plt.tight_layout()

    plt.savefig(
        "visualizations/silhouette_scores.png"
    )

    plt.close()

    best_k = list(k_values)[
        silhouette_scores.index(
            max(silhouette_scores)
        )
    ]

    print(
        "Best K based on Silhouette Score:",
        best_k
    )

    return best_k


def apply_kmeans(
    df,
    scaled_data,
    best_k
):

    model = KMeans(
        n_clusters=best_k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(
        scaled_data
    )

    result = df.copy()

    result["KMeans_Cluster"] = labels

    score = silhouette_score(
        scaled_data,
        labels
    )

    print(
        "K-Means Silhouette Score:",
        round(score, 4)
    )

    return result, model, score


def apply_dbscan(
    df,
    scaled_data
):

    model = DBSCAN(
        eps=1.5,
        min_samples=10
    )

    labels = model.fit_predict(
        scaled_data
    )

    result = df.copy()

    result["DBSCAN_Cluster"] = labels

    number_of_clusters = len(
        set(labels)
        - {-1}
    )

    number_of_noise = list(
        labels
    ).count(-1)

    print(
        "DBSCAN clusters:",
        number_of_clusters
    )

    print(
        "DBSCAN noise points:",
        number_of_noise
    )

    return result, model