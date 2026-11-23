# 🏪 RetailIQ
### Retail Business Intelligence & Machine Learning Platform

**Understand performance. Predict outcomes. Make smarter retail decisions.**

RetailIQ is a machine learning-powered retail intelligence platform developed using the Rossmann Store Sales dataset. It combines data exploration, predictive modelling, and interactive visualizations to help understand retail performance and uncover meaningful business insights.

The platform integrates five machine learning models into a unified Streamlit application, providing capabilities ranging from sales forecasting and customer footfall prediction to store segmentation and closure prediction.

---

## 📌 Project Overview

Retail businesses generate large amounts of operational data, but transforming that data into meaningful decisions can be challenging.

RetailIQ addresses this by combining historical retail data with machine learning techniques to explore business performance, identify patterns, and generate predictions.

### Key Objectives

- Analyze historical retail sales and customer trends.
- Forecast daily store sales.
- Predict customer footfall.
- Classify sales performance.
- Identify groups of stores with similar characteristics.
- Predict store operating status.

---

## ✨ Features

| Feature | Description |
|---|---|
| Dataset Explorer | Explore historical sales, customers, promotions, and store characteristics. |
| Sales Forecasting | Predict expected store sales using regression. |
| Customer Footfall | Estimate customer visits using historical patterns. |
| Sales Performance | Classify performance into Low, Medium, and High categories. |
| Store Segmentation | Group stores into meaningful clusters using K-Means. |
| Store Closure | Predict whether a store is expected to be Open or Closed. |

### Interface

- Dark-themed analytics dashboard.
- Interactive visualizations.
- Custom sidebar navigation.
- Manual store identification using the `ST-0001` format.
- Calendar-based date selection.
- Integrated machine learning predictions.

---

## 🤖 Machine Learning Models

RetailIQ implements five machine learning components.

| # | Component | Algorithm | Validation Metric |
|---|---|---|---|
| 1 | Sales Forecasting | HistGradientBoostingRegressor | R²: 61.66% |
| 2 | Customer Footfall | RandomForestRegressor | R²: 73.42% |
| 3 | Sales Performance | HistGradientBoostingClassifier | Accuracy: 72.28% |
| 4 | Store Segmentation | K-Means | Silhouette: 0.3144 |
| 5 | Store Closure | DecisionTreeClassifier | Accuracy: 99.80% |

### Model Details

**1. Sales Forecasting**

Uses HistGradientBoostingRegressor to estimate daily sales based on store characteristics, promotions, calendar information, and other relevant features.

**2. Customer Footfall Forecasting**

Uses RandomForestRegressor to estimate the number of customers visiting a store. The model operates independently of sales values.

**3. Sales Performance Classification**

Uses HistGradientBoostingClassifier to categorize sales performance into three classes:

- Low
- Medium
- High

**4. Store Segmentation**

Uses K-Means clustering to group stores based on historical operational and performance characteristics.

Three clusters were selected based on the highest Silhouette Score among the tested configurations.

**5. Store Closure Prediction**

Uses DecisionTreeClassifier to predict whether a store is expected to be open or closed based on historical operating patterns and calendar features.

### Evaluation Notes

- Regression models are evaluated using R², MAE, RMSE, and MAPE.
- Classification models are evaluated using accuracy, precision, recall, F1-score, and confusion matrices.
- Store segmentation is evaluated using the Silhouette Score.

The reported metrics are historical validation results, not guarantees of future performance.

The Store Closure model's high validation score should be interpreted in the context of predictable operating schedules and historical patterns. It does not guarantee detection of unexpected closures.

---

## 📊 Dataset

RetailIQ uses the **Rossmann Store Sales dataset**, originally provided through the Kaggle Rossmann Store Sales competition.

The dataset contains historical information about retail stores, including:

- Daily sales.
- Customer counts.
- Store identifiers.
- Promotional activities.
- State and school holidays.
- Store types and assortment.
- Competition information.
- Store operating status.

### Dataset Files

| File | Description |
|---|---|
| `train.csv` | Historical training records. |
| `store.csv` | Store-level metadata. |
| `test.csv` | Test records for the original competition. |

The training dataset contains approximately 1 million records across 1,115 stores.

**Original dataset:** [Rossmann Store Sales – Kaggle](https://www.kaggle.com/competitions/rossmann-store-sales)

---

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Frontend | Streamlit |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Data Visualization | Plotly, Matplotlib, Seaborn |
| Model Persistence | Joblib |
| Development Environment | Jupyter Notebook, VS Code |

---

## 📁 Project Structure

```text
RetailIQ/
│
├── dataset/
│   ├── store.csv
│   ├── train.csv
│   └── test.csv
│
├── models/
│   ├── model_1_sales_forecasting/
│   ├── model_2_customer_footfall/
│   ├── model_3_sales_classification/
│   ├── model_4_store_segmentation/
│   └── model_5_store_closure/
│
├── notebooks/
│   ├── Data Cleaning & Modeling.ipynb
│   └── Data Preparation & EDA.ipynb
│
├── frontend/
│   ├── app.py
│   │
│   ├── pages/
│   │   ├── 1_Dataset_Explorer.py
│   │   ├── 2_Sales_Forecasting.py
│   │   ├── 3_Customer_Footfall.py
│   │   ├── 4_Sales_Performance.py
│   │   ├── 5_Store_Segmentation.py
│   │   └── 6_Store_Closure.py
│   │
│   ├── components/
│   │   ├── sidebar.py
│   │   ├── charts.py
│   │   └── metrics.py
│   │
│   └── utils/
│       ├── model_loader.py
│       └── data_loader.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Installation & Setup

### Prerequisites

- Python 3.10 or later.
- Git.
- pip.

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/RetailIQ.git
cd RetailIQ
```

Replace `YOUR_USERNAME` with your GitHub username.

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

**Windows:**

```powershell
venv\Scripts\activate
```

**Linux / macOS:**

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch RetailIQ

From the project root directory, execute:

```bash
streamlit run frontend/app.py
```

The application will open in your default browser.

If it does not open automatically, visit:

```text
http://localhost:8501
```

---

## 📈 Model Development Workflow

The machine learning development process follows these stages:

1. **Data Preparation:** Load, inspect, and clean the Rossmann datasets.
2. **Exploratory Data Analysis:** Identify patterns, relationships, and trends.
3. **Feature Engineering:** Create calendar, promotional, and store-related features.
4. **Model Development:** Train five models for different retail intelligence tasks.
5. **Model Evaluation:** Evaluate performance using appropriate validation metrics.
6. **Model Persistence:** Save trained pipelines using Joblib.
7. **Application Integration:** Load the saved models into the Streamlit frontend.

The Jupyter notebooks contain the data preparation, exploration, and model development workflows.

---

## ⚠️ Limitations

- Models are trained on historical Rossmann data from 2013–2015.
- Predictions for dates outside the historical training period may have reduced reliability.
- Future promotional schedules and unexpected external events may not be represented.
- Store segmentation is exploratory and its clusters are not ground-truth business categories.
- Historical operating patterns do not guarantee accurate prediction of unexpected store closures.

RetailIQ is intended as an analytical and predictive demonstration platform, not a replacement for actual retail operational systems.

---

## 🚀 Future Improvements

Potential future enhancements include:

- Advanced forecasting algorithms.
- Additional model optimization and hyperparameter tuning.
- More detailed store-level performance analytics.
- Integration of external economic and seasonal indicators.
- Improved model explainability.
- Deployment to a cloud platform.
- Real-time data integration.

---

## 👨‍💻 Development

RetailIQ is an academic machine learning and business intelligence project developed to explore the practical application of predictive analytics in the retail domain.

**Project:** RetailIQ  
**Domain:** Retail Analytics & Machine Learning  
**Dataset:** Rossmann Store Sales  
**Frontend:** Streamlit  
**Models:** Five Machine Learning Components

---

*RetailIQ — Turning retail data into actionable insights.*