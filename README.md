# 🏨 Hotel Data Analysis for Customer Segmentation and Demand Forecasting

## 📌 Project Overview

This project focuses on analyzing hotel booking data to understand customer booking behavior, segment customers into meaningful groups, and forecast hotel room demand.

The project combines **Exploratory Data Analysis (EDA)**, **K-Means Clustering**, **DBSCAN**, **Linear Regression**, and **ARIMA** to identify customer patterns and predict future demand.

A **Streamlit-based interactive dashboard** is also developed to visualize the analysis results in an easy-to-understand format.

---

## 🎯 Objectives

The main objectives of this project are:

1. To clean and preprocess hotel booking data.
2. To perform Exploratory Data Analysis (EDA) and identify important booking patterns.
3. To segment hotel customers using **K-Means clustering**.
4. To compare customer segmentation using **DBSCAN**.
5. To forecast hotel room demand using **Linear Regression**.
6. To compare forecasting performance with **ARIMA**.
7. To identify important factors affecting hotel bookings and demand.
8. To develop an interactive **Streamlit dashboard** for data visualization and analysis.

---

## 📊 Dataset

The project uses hotel booking data containing information about customer reservations, booking behavior, stay duration, room types, arrival information, and other booking-related attributes.

For this project, a prepared dataset containing **10,000 hotel booking records** is used.

### Important Features

Some of the important attributes include:

* Hotel type
* Lead time
* Arrival date
* Arrival month
* Arrival week
* Arrival day
* Weekend nights
* Weekday nights
* Adults
* Children
* Babies
* Meal type
* Country
* Market segment
* Distribution channel
* Room type
* Booking status
* Average Daily Rate (ADR)
* Required car parking spaces
* Special requests
* Total stay duration
* Total guests
* Estimated revenue

---

## 🛠️ Tools and Technologies

| Technology   | Purpose                                 |
| ------------ | --------------------------------------- |
| Python       | Programming and model development       |
| Pandas       | Data manipulation and preprocessing     |
| NumPy        | Numerical computations                  |
| Matplotlib   | Data visualization                      |
| Seaborn      | Statistical visualization               |
| Scikit-learn | Machine learning and evaluation         |
| Statsmodels  | ARIMA time-series forecasting           |
| SciPy        | Scientific and statistical computations |
| Joblib       | Saving machine-learning models          |
| Streamlit    | Interactive web dashboard               |
| Git & GitHub | Version control and project hosting     |
| VS Code      | Development environment                 |

---

## 🔬 Methodology

The project follows the following workflow:

```text
Hotel Booking Dataset
        ↓
Data Loading
        ↓
Data Cleaning
        ↓
Data Preprocessing
        ↓
Exploratory Data Analysis
        ↓
Feature Engineering
        ↓
Customer Segmentation
   ┌───────────────┐
   ↓               ↓
K-Means          DBSCAN
   ↓               ↓
Customer Segments
        ↓
Demand Forecasting
   ┌───────────────┐
   ↓               ↓
Linear Regression  ARIMA
   ↓               ↓
Model Comparison
        ↓
Streamlit Dashboard
```

---

## 🧹 Data Preprocessing

The preprocessing stage prepares the raw hotel booking data for analysis and machine learning.

The major preprocessing steps include:

* Handling missing values
* Removing unnecessary columns
* Detecting and handling outliers
* Converting categorical variables
* Encoding categorical features
* Scaling numerical features
* Creating new useful features

### Feature Engineering

Additional features are created to improve analysis:

* `total_stay_nights`
* `total_guests`
* `estimated_revenue`
* `arrival_month_num`
* `arrival_day`
* `arrival_weekday`

Sensitive/unnecessary fields such as customer names, email addresses, phone numbers, and credit-card information are excluded from the analysis.

---

## 📈 Exploratory Data Analysis

EDA is performed to understand the characteristics of hotel bookings.

The analysis includes:

* Booking distribution
* Customer distribution
* Room type analysis
* Lead-time analysis
* Stay-duration analysis
* Monthly booking trends
* Customer preferences
* Market segment analysis
* Cancellation patterns
* Revenue-related analysis

Visualizations are generated using **Matplotlib** and **Seaborn**.

---

## 👥 Customer Segmentation

### K-Means Clustering

K-Means clustering is used to divide customers into groups based on their booking behavior.

The number of clusters is evaluated using the **Silhouette Score** for different values of K.

The best-performing K is selected based on clustering quality.

Customer segments can help hotels understand groups such as:

* Frequent/valuable customers
* Short-stay customers
* Long-stay customers
* High-revenue customers
* Customers with specific booking patterns

### DBSCAN

**DBSCAN (Density-Based Spatial Clustering of Applications with Noise)** is also implemented as an additional clustering technique.

It helps identify:

* Dense customer groups
* Irregular customer patterns
* Potential outliers/noise

The results of DBSCAN are compared with K-Means segmentation.

---

## 🔮 Demand Forecasting

### Linear Regression

Linear Regression is used as the primary forecasting approach to estimate hotel room demand based on historical booking information and engineered features.

The dataset is divided into training and testing portions for model evaluation.

### ARIMA

**ARIMA (AutoRegressive Integrated Moving Average)** is implemented as an additional time-series forecasting technique.

It is used to analyze historical demand patterns and compare its forecasting behavior with Linear Regression.

---

## 📊 Model Comparison

The project compares the forecasting models using appropriate evaluation metrics.

The comparison includes:

* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)
* R² Score, where applicable

The model comparison results are saved for further analysis.

---

## 🖥️ Streamlit Dashboard

An interactive Streamlit dashboard is developed to present the project results.

The dashboard provides sections for:

### 🏠 Dashboard

Provides an overview of the hotel analytics project.

### 📊 Exploratory Data Analysis

Displays important charts and booking trends.

### 👥 Customer Segmentation

Displays customer clusters generated using K-Means and DBSCAN.

### 📈 Demand Forecasting

Displays demand forecasting results from Linear Regression and ARIMA.

### 📋 Model Comparison

Provides a comparison of forecasting models and their evaluation metrics.

---

## 📁 Project Structure

```text
Hotel_Data_Analysis_Project/
│
├── data/
│   └── hotel_booking_10000.csv
│
├── models/
│   └── saved machine learning models
│
├── results/
│   ├── processed_hotel_data.csv
│   ├── kmeans_customer_segments.csv
│   ├── dbscan_customer_segments.csv
│   └── model_comparison.csv
│
├── visualizations/
│   └── generated charts and plots
│
├── src/
│   ├── preprocessing.py
│   ├── eda.py
│   ├── segmentation.py
│   └── forecasting.py
│
├── main.py
├── app.py
├── README.md
└── .gitignore
```

---

## ▶️ How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/kalbandeshruti23-ctrl/Hotel_Data_Analysis_Project.git
```

### 2. Open the Project

```bash
cd Hotel_Data_Analysis_Project
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Virtual Environment

For Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 5. Install Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels scipy joblib streamlit
```

### 6. Run the Main Analysis

```bash
python main.py
```

### 7. Run the Streamlit Dashboard

```bash
streamlit run app.py
```

The dashboard will open in your web browser.

---

## 📌 Key Learnings

Through this project, the following concepts are demonstrated:

* Data cleaning and preprocessing
* Feature engineering
* Exploratory Data Analysis
* Customer segmentation
* Unsupervised machine learning
* K-Means clustering
* DBSCAN clustering
* Time-series analysis
* Linear Regression
* ARIMA forecasting
* Model evaluation
* Data visualization
* Streamlit dashboard development
* Git and GitHub project management

---

## 📊 Expected Business Benefits

The analysis can help hotels:

* Understand customer booking behavior
* Identify valuable customer segments
* Analyze room preferences
* Understand seasonal booking patterns
* Estimate future room demand
* Improve resource planning
* Support marketing strategies
* Improve customer-targeted services
* Make data-driven business decisions

---

## 🚀 Future Enhancements

Possible future improvements include:

* Integration with real-time hotel booking data
* Advanced forecasting models such as Prophet, XGBoost, or LSTM
* Automated demand alerts
* Dynamic pricing recommendations
* Customer lifetime value prediction
* Personalized hotel recommendations
* Cloud deployment
* Real-time dashboard updates

---

## 👩‍💻 Author

**Shruti Kalbande**

M.Tech – Computer Science / Big Data Analytics

### Project

**Hotel Data Analysis for Customer Segmentation and Demand Forecasting**

---

## 📜 License

This project is developed for **academic and educational purposes**.
