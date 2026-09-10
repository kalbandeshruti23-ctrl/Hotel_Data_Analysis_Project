import streamlit as st
import pandas as pd
import numpy as np
import os
import base64


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Hotel Data Analytics Dashboard",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# FILE PATHS
# ============================================================

DATA_FILE = "data/hotel_booking_10000.csv"

PROCESSED_FILE = "results/processed_hotel_data.csv"

KMEANS_FILE = "results/kmeans_customer_segments.csv"

DBSCAN_FILE = "results/dbscan_customer_segments.csv"

MODEL_FILE = "results/model_comparison.csv"

VIS_DIR = "visualizations"

BACKGROUND_IMAGE = "assets/hotel_background.jpg"


# ============================================================
# BACKGROUND IMAGE
# ============================================================

def set_background(image_path):

    if not os.path.exists(image_path):
        return

    with open(image_path, "rb") as image_file:

        encoded_image = base64.b64encode(
            image_file.read()
        ).decode()

    st.markdown(
        f"""
        <style>

        /* ==================================================
           MAIN STREAMLIT BACKGROUND
           ================================================== */

        .stApp {{

            background-image:

                linear-gradient(
                    rgba(248, 250, 252, 0.94),
                    rgba(248, 250, 252, 0.94)
                ),

                url("data:image/jpeg;base64,{encoded_image}");

            background-size: cover;

            background-position: center;

            background-attachment: fixed;

        }}


        /* ==================================================
           TOP STREAMLIT HEADER
           ================================================== */

        header[data-testid="stHeader"] {{

            background-color:
                rgba(255,255,255,0.05);

        }}


        </style>
        """,
        unsafe_allow_html=True
    )


set_background(BACKGROUND_IMAGE)


# ============================================================
# CUSTOM CSS
# NO CUSTOM DIV ELEMENTS USED
# ============================================================

st.markdown(
    """
    <style>

    /* ======================================================
       GENERAL
       ====================================================== */

    .main .block-container {

        max-width: 1450px;

        padding-top: 2rem;

        padding-bottom: 3rem;

    }


    /* ======================================================
       SIDEBAR
       ====================================================== */

    section[data-testid="stSidebar"] {

        background:
            linear-gradient(
                180deg,
                #061A33 0%,
                #082B4F 50%,
                #064E5B 100%
            );

    }


    section[data-testid="stSidebar"] * {

        color: white !important;

    }


    section[data-testid="stSidebar"] hr {

        border-color:
            rgba(255,255,255,0.20);

    }


    /* ======================================================
       MAIN TITLE
       ====================================================== */

    h1 {

        color: #0F2F56 !important;

        font-size: 38px !important;

        font-weight: 850 !important;

        margin-bottom: 5px !important;

    }


    h2 {

        color: #123B67 !important;

        font-weight: 800 !important;

    }


    h3 {

        color: #164E63 !important;

        font-weight: 750 !important;

    }


    /* ======================================================
       CAPTION
       ====================================================== */

    .stCaption {

        color: #475569 !important;

        font-size: 16px;

    }


    /* ======================================================
       KPI CARDS
       ====================================================== */

    div[data-testid="stMetric"] {

        background:
            rgba(255,255,255,0.96);

        border-radius: 18px;

        padding: 20px;

        border:
            1px solid #D9E2EC;

        box-shadow:
            0 8px 25px
            rgba(15,23,42,0.10);

        min-height: 120px;

        transition:
            transform 0.2s ease,
            box-shadow 0.2s ease;

    }


    div[data-testid="stMetric"]:hover {

        transform:
            translateY(-4px);

        box-shadow:
            0 14px 32px
            rgba(8,145,178,0.18);

        border-color:
            #67E8F9;

    }


    div[data-testid="stMetricLabel"] {

        color:
            #475569 !important;

        font-weight:
            700 !important;

    }


    div[data-testid="stMetricValue"] {

        color:
            #0F172A !important;

        font-size:
            28px !important;

        font-weight:
            850 !important;

    }


    /* ======================================================
       DATAFRAME
       ====================================================== */

    div[data-testid="stDataFrame"] {

        border-radius: 14px;

        overflow: hidden;

        box-shadow:
            0 5px 18px
            rgba(15,23,42,0.07);

    }


    /* ======================================================
       BUTTON
       ====================================================== */

    .stButton > button {

        background:
            linear-gradient(
                135deg,
                #2563EB,
                #0891B2
            );

        color: white;

        border: none;

        border-radius: 10px;

        font-weight: 700;

    }


    .stButton > button:hover {

        background:
            linear-gradient(
                135deg,
                #1D4ED8,
                #0E7490
            );

        color: white;

    }


    /* ======================================================
       SELECTBOX
       ====================================================== */

    div[data-baseweb="select"] > div {

        border-radius: 9px;

    }


    /* ======================================================
       TABS
       ====================================================== */

    button[data-baseweb="tab"] {

        font-weight: 700;

    }


    /* ======================================================
       IMAGE ROUNDED CORNERS
       ====================================================== */

    img {

        border-radius: 14px;

    }


    /* ======================================================
       SUCCESS MESSAGE
       ====================================================== */

    div[data-testid="stAlert"] {

        border-radius: 12px;

    }


    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    if os.path.exists(PROCESSED_FILE):

        data = pd.read_csv(
            PROCESSED_FILE
        )

    elif os.path.exists(DATA_FILE):

        data = pd.read_csv(
            DATA_FILE
        )

    else:

        return None

    if "arrival_date" in data.columns:

        data["arrival_date"] = pd.to_datetime(
            data["arrival_date"],
            errors="coerce"
        )

    return data


@st.cache_data
def load_kmeans():

    if os.path.exists(KMEANS_FILE):

        return pd.read_csv(
            KMEANS_FILE
        )

    return None


@st.cache_data
def load_dbscan():

    if os.path.exists(DBSCAN_FILE):

        return pd.read_csv(
            DBSCAN_FILE
        )

    return None


@st.cache_data
def load_model_results():

    if os.path.exists(MODEL_FILE):

        return pd.read_csv(
            MODEL_FILE
        )

    return None


# ============================================================
# LOAD FILES
# ============================================================

df = load_data()

kmeans_df = load_kmeans()

dbscan_df = load_dbscan()

model_results = load_model_results()


# ============================================================
# DATA CHECK
# ============================================================

if df is None:

    st.error(
        "❌ Hotel dataset was not found."
    )

    st.info(
        "Run `python main.py` first."
    )

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "# 🏨 Hotel Analytics"
    )

    st.caption(
        "Insights for Better Decisions"
    )

    st.markdown("---")

    st.markdown(
        "### 🧭 Navigation"
    )

    page = st.radio(
        "Select Page",
        [
            "🏠 Dashboard",
            "📊 EDA Analysis",
            "👥 Customer Segmentation",
            "📈 Demand Forecasting"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")

    st.markdown(
        "### 🔎 Filters"
    )


    # --------------------------------------------------------
    # HOTEL FILTER
    # --------------------------------------------------------

    if "hotel" in df.columns:

        hotel_options = (
            ["All"]
            +
            sorted(
                df["hotel"]
                .dropna()
                .unique()
                .tolist()
            )
        )

        selected_hotel = st.selectbox(
            "🏨 Hotel Type",
            hotel_options
        )

    else:

        selected_hotel = "All"


    # --------------------------------------------------------
    # CUSTOMER FILTER
    # --------------------------------------------------------

    if "customer_type" in df.columns:

        customer_options = (
            ["All"]
            +
            sorted(
                df["customer_type"]
                .dropna()
                .unique()
                .tolist()
            )
        )

        selected_customer = st.selectbox(
            "👥 Customer Type",
            customer_options
        )

    else:

        selected_customer = "All"


    st.markdown("---")

    st.markdown(
        "### 🛠 Technologies"
    )

    st.write("🐍 Python")

    st.write("🐼 Pandas")

    st.write("🔢 NumPy")

    st.write("📊 Matplotlib")

    st.write("🎨 Seaborn")

    st.write("🤖 Scikit-learn")

    st.write("📈 ARIMA")

    st.write("🌐 Streamlit")


# ============================================================
# APPLY FILTERS
# ============================================================

filtered_df = df.copy()


if (
    selected_hotel != "All"
    and "hotel" in filtered_df.columns
):

    filtered_df = filtered_df[
        filtered_df["hotel"]
        == selected_hotel
    ]


if (
    selected_customer != "All"
    and "customer_type" in filtered_df.columns
):

    filtered_df = filtered_df[
        filtered_df["customer_type"]
        == selected_customer
    ]


# ============================================================
# MAIN TITLE
# ============================================================

st.title(
    "🏨 Hotel Data Analytics Dashboard"
)

st.caption(
    "Customer Segmentation  •  Demand Forecasting  •  Exploratory Data Analysis"
)


st.markdown("---")


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.header(
        "🎯 Project Overview"
    )

    st.info(
        """
        This dashboard analyzes hotel booking data to
        understand customer behavior, identify customer
        segments using clustering algorithms, and forecast
        future hotel demand using machine learning models.
        """
    )


    # ========================================================
    # KPIs
    # ========================================================

    total_bookings = len(
        filtered_df
    )


    if "total_guests" in filtered_df.columns:

        total_guests = filtered_df[
            "total_guests"
        ].sum()

    else:

        total_guests = 0


    if "adr" in filtered_df.columns:

        average_adr = filtered_df[
            "adr"
        ].mean()

    else:

        average_adr = 0


    if "is_canceled" in filtered_df.columns:

        cancellation_rate = (
            filtered_df[
                "is_canceled"
            ].mean()
            * 100
        )

    else:

        cancellation_rate = 0


    # ========================================================
    # KPI DISPLAY
    # ========================================================

    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "🛏️ Total Bookings",
            f"{total_bookings:,}"
        )


    with col2:

        st.metric(
            "👥 Total Guests",
            f"{total_guests:,.0f}"
        )


    with col3:

        st.metric(
            "💰 Average ADR",
            f"₹ {average_adr:,.2f}"
        )


    with col4:

        st.metric(
            "❌ Cancellation Rate",
            f"{cancellation_rate:.2f}%"
        )


    st.markdown("")


    # ========================================================
    # BOOKING OVERVIEW
    # ========================================================

    st.header(
        "📊 Booking Overview"
    )


    # --------------------------------------------------------
    # ROW 1
    # --------------------------------------------------------

    col1, col2 = st.columns(2)


    with col1:

        path = os.path.join(
            VIS_DIR,
            "hotel_distribution.png"
        )

        if os.path.exists(path):

            st.image(
                path,
                caption="🏨 Hotel Distribution",
                use_container_width=True
            )


    with col2:

        path = os.path.join(
            VIS_DIR,
            "cancellation_distribution.png"
        )

        if os.path.exists(path):

            st.image(
                path,
                caption="❌ Cancellation Distribution",
                use_container_width=True
            )


    # --------------------------------------------------------
    # ROW 2
    # --------------------------------------------------------

    col1, col2 = st.columns(2)


    with col1:

        path = os.path.join(
            VIS_DIR,
            "customer_type_distribution.png"
        )

        if os.path.exists(path):

            st.image(
                path,
                caption="👥 Customer Type Distribution",
                use_container_width=True
            )


    with col2:

        path = os.path.join(
            VIS_DIR,
            "market_segment_distribution.png"
        )

        if os.path.exists(path):

            st.image(
                path,
                caption="🎯 Market Segment Distribution",
                use_container_width=True
            )


    # --------------------------------------------------------
    # ROW 3
    # --------------------------------------------------------

    col1, col2 = st.columns(2)


    with col1:

        path = os.path.join(
            VIS_DIR,
            "adr_distribution.png"
        )

        if os.path.exists(path):

            st.image(
                path,
                caption="💰 ADR Distribution",
                use_container_width=True
            )


    with col2:

        path = os.path.join(
            VIS_DIR,
            "demand_over_time.png"
        )

        if os.path.exists(path):

            st.image(
                path,
                caption="📈 Demand Over Time",
                use_container_width=True
            )


# ============================================================
# EDA PAGE
# ============================================================

elif page == "📊 EDA Analysis":

    st.header(
        "📊 Exploratory Data Analysis"
    )

    st.info(
        """
        Exploratory Data Analysis helps identify booking
        patterns, cancellation behavior, customer types,
        market segments, pricing patterns, and demand trends.
        """
    )


    # ========================================================
    # DATE FILTER
    # ========================================================

    eda_df = filtered_df.copy()


    if "arrival_date" in eda_df.columns:

        valid_dates = (
            eda_df["arrival_date"]
            .dropna()
        )


        if len(valid_dates) > 0:

            min_date = (
                valid_dates
                .min()
                .date()
            )

            max_date = (
                valid_dates
                .max()
                .date()
            )


            selected_dates = st.date_input(
                "📅 Select Arrival Date Range",
                value=(
                    min_date,
                    max_date
                ),
                min_value=min_date,
                max_value=max_date
            )


            if (
                isinstance(
                    selected_dates,
                    tuple
                )
                and len(selected_dates) == 2
            ):

                start_date, end_date = (
                    selected_dates
                )


                eda_df = eda_df[
                    (
                        eda_df[
                            "arrival_date"
                        ].dt.date
                        >= start_date
                    )
                    &
                    (
                        eda_df[
                            "arrival_date"
                        ].dt.date
                        <= end_date
                    )
                ]


    st.metric(
        "📋 Selected Records",
        f"{len(eda_df):,}"
    )


    st.markdown("---")


    # ========================================================
    # EDA VISUALIZATIONS
    # ========================================================

    col1, col2 = st.columns(2)


    with col1:

        path = os.path.join(
            VIS_DIR,
            "hotel_distribution.png"
        )

        if os.path.exists(path):

            st.image(
                path,
                caption="🏨 Hotel Distribution",
                use_container_width=True
            )


    with col2:

        path = os.path.join(
            VIS_DIR,
            "cancellation_distribution.png"
        )

        if os.path.exists(path):

            st.image(
                path,
                caption="❌ Cancellation Distribution",
                use_container_width=True
            )


    col1, col2 = st.columns(2)


    with col1:

        path = os.path.join(
            VIS_DIR,
            "customer_type_distribution.png"
        )

        if os.path.exists(path):

            st.image(
                path,
                caption="👥 Customer Type Distribution",
                use_container_width=True
            )


    with col2:

        path = os.path.join(
            VIS_DIR,
            "market_segment_distribution.png"
        )

        if os.path.exists(path):

            st.image(
                path,
                caption="🎯 Market Segment Distribution",
                use_container_width=True
            )


    col1, col2 = st.columns(2)


    with col1:

        path = os.path.join(
            VIS_DIR,
            "adr_distribution.png"
        )

        if os.path.exists(path):

            st.image(
                path,
                caption="💰 ADR Distribution",
                use_container_width=True
            )


    with col2:

        path = os.path.join(
            VIS_DIR,
            "demand_over_time.png"
        )

        if os.path.exists(path):

            st.image(
                path,
                caption="📈 Demand Over Time",
                use_container_width=True
            )


# ============================================================
# CUSTOMER SEGMENTATION
# ============================================================

elif page == "👥 Customer Segmentation":

    st.header(
        "👥 Customer Segmentation"
    )

    st.info(
        """
        K-Means clustering groups customers based on booking
        behavior. DBSCAN is used to discover dense groups and
        identify potential outliers.
        """
    )


    # ========================================================
    # K-MEANS
    # ========================================================

    if kmeans_df is not None:

        st.subheader(
            "🎯 K-Means Customer Segmentation"
        )


        if "KMeans_Cluster" in kmeans_df.columns:

            clusters = sorted(
                kmeans_df[
                    "KMeans_Cluster"
                ]
                .dropna()
                .unique()
                .tolist()
            )


            selected_cluster = st.selectbox(
                "Select Customer Segment",
                clusters
            )


            cluster_data = kmeans_df[
                kmeans_df[
                    "KMeans_Cluster"
                ]
                == selected_cluster
            ]


            # ------------------------------------------------
            # CLUSTER KPIs
            # ------------------------------------------------

            col1, col2, col3, col4 = st.columns(4)


            with col1:

                st.metric(
                    "👥 Customers",
                    f"{len(cluster_data):,}"
                )


            with col2:

                if "adr" in cluster_data.columns:

                    st.metric(
                        "💰 Average ADR",
                        f"₹ {cluster_data['adr'].mean():,.2f}"
                    )


            with col3:

                if "lead_time" in cluster_data.columns:

                    st.metric(
                        "📅 Average Lead Time",
                        f"{cluster_data['lead_time'].mean():.1f}"
                    )


            with col4:

                if "total_stay_nights" in cluster_data.columns:

                    st.metric(
                        "🛏️ Average Stay",
                        f"{cluster_data['total_stay_nights'].mean():.1f}"
                    )


            st.subheader(
                "📋 Customer Segment Profile"
            )


            profile_columns = [

                "lead_time",

                "adr",

                "total_stay_nights",

                "total_guests",

                "is_repeated_guest",

                "total_previous_bookings",

                "total_of_special_requests",

                "is_canceled"

            ]


            available_columns = [

                column

                for column in profile_columns

                if column in cluster_data.columns

            ]


            if available_columns:

                profile = (
                    cluster_data[
                        available_columns
                    ]
                    .describe()
                    .T
                    .round(2)
                )


                st.dataframe(
                    profile,
                    use_container_width=True
                )


            # ------------------------------------------------
            # K-MEANS GRAPHS
            # ------------------------------------------------

            st.subheader(
                "📊 K-Means Model Analysis"
            )


            col1, col2 = st.columns(2)


            with col1:

                path = os.path.join(
                    VIS_DIR,
                    "elbow_method.png"
                )

                if os.path.exists(path):

                    st.image(
                        path,
                        caption="Elbow Method",
                        use_container_width=True
                    )


            with col2:

                path = os.path.join(
                    VIS_DIR,
                    "silhouette_scores.png"
                )

                if os.path.exists(path):

                    st.image(
                        path,
                        caption="Silhouette Score Analysis",
                        use_container_width=True
                    )


    else:

        st.warning(
            "K-Means results not found. "
            "Run `python main.py` first."
        )


    # ========================================================
    # DBSCAN
    # ========================================================

    st.markdown("---")

    st.subheader(
        "🔎 DBSCAN Analysis"
    )


    if dbscan_df is not None:

        if "DBSCAN_Cluster" in dbscan_df.columns:

            total_records = len(
                dbscan_df
            )


            noise_points = (
                dbscan_df[
                    "DBSCAN_Cluster"
                ]
                == -1
            ).sum()


            cluster_count = (
                dbscan_df[
                    "DBSCAN_Cluster"
                ]
                .nunique()
            )


            col1, col2, col3 = st.columns(3)


            with col1:

                st.metric(
                    "🎯 DBSCAN Clusters",
                    cluster_count
                )


            with col2:

                st.metric(
                    "⚠️ Noise / Outliers",
                    f"{noise_points:,}"
                )


            with col3:

                st.metric(
                    "📋 Total Records",
                    f"{total_records:,}"
                )


            st.dataframe(
                dbscan_df.head(100),
                use_container_width=True
            )


    else:

        st.warning(
            "DBSCAN results not found."
        )


# ============================================================
# DEMAND FORECASTING
# ============================================================

elif page == "📈 Demand Forecasting":

    st.header(
        "📈 Demand Forecasting"
    )


    st.info(
        """
        Demand forecasting estimates future hotel booking
        demand. Linear Regression and ARIMA models are
        evaluated using MAE, MSE, and RMSE.
        """
    )


    # ========================================================
    # MODEL COMPARISON
    # ========================================================

    if model_results is not None:

        st.subheader(
            "🤖 Model Performance Comparison"
        )


        st.dataframe(
            model_results,
            use_container_width=True
        )


        # ----------------------------------------------------
        # BEST MODEL
        # ----------------------------------------------------

        if "RMSE" in model_results.columns:

            best_index = (
                model_results[
                    "RMSE"
                ]
                .idxmin()
            )


            best_model = (
                model_results
                .loc[
                    best_index,
                    "Model"
                ]
            )


            best_rmse = (
                model_results
                .loc[
                    best_index,
                    "RMSE"
                ]
            )


            st.success(
                f"🏆 Best Forecasting Model: "
                f"{best_model} "
                f"| RMSE: {best_rmse:.4f}"
            )


    else:

        st.warning(
            "Model comparison file not found."
        )


    # ========================================================
    # FORECAST GRAPHS
    # ========================================================

    st.subheader(
        "📊 Forecast Visualization"
    )


    col1, col2 = st.columns(2)


    with col1:

        path = os.path.join(
            VIS_DIR,
            "linear_regression_forecast.png"
        )


        if os.path.exists(path):

            st.image(
                path,
                caption="Linear Regression Demand Forecast",
                use_container_width=True
            )


    with col2:

        path = os.path.join(
            VIS_DIR,
            "arima_forecast.png"
        )


        if os.path.exists(path):

            st.image(
                path,
                caption="ARIMA Demand Forecast",
                use_container_width=True
            )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "🏨 Hotel Data Analysis for Customer Segmentation and Demand Forecasting"
)

st.caption(
    "Python • Pandas • NumPy • Scikit-learn • "
    "K-Means • DBSCAN • Linear Regression • ARIMA • Streamlit"
)

st.caption(
    "More Bookings • Better Insights • Smarter Decisions"
)