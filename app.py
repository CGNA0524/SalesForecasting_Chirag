import streamlit as st
import pandas as pd
import os

# -----------------------------------------
# Page Configuration
# -----------------------------------------

st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📈",
    layout="wide"
)

# -----------------------------------------
# Sidebar
# -----------------------------------------

st.sidebar.title("📊 Dashboard Menu")

st.sidebar.markdown("""
### Project Overview

- Sales Forecasting
- Model Comparison
- Category Forecast
- Anomaly Detection
- Product Clustering
- Customer Segmentation

---
Prepared by **Chirag Nagra**
""")

# -----------------------------------------
# Title
# -----------------------------------------

st.title("📈 Sales Forecasting & Demand Intelligence Dashboard")

st.markdown("""
This dashboard summarizes the complete analysis of the Superstore Sales Forecasting Project using Machine Learning and Time Series Forecasting.
""")

st.divider()

# -----------------------------------------
# KPI Cards
# -----------------------------------------

c1,c2,c3,c4=st.columns(4)

c1.metric("Best Model","SARIMA")
c2.metric("Forecast Horizon","3 Months")
c3.metric("Isolation Forest","11 Anomalies")
c4.metric("Z-Score","2 Outliers")

st.divider()

# -----------------------------------------
# Model Performance
# -----------------------------------------

st.header("🏆 Model Performance")

comparison=pd.DataFrame({

"Model":["SARIMA","Prophet","XGBoost"],

"MAE":[19244.49,20250.79,36206.50],

"RMSE":[19950.07,22318.41,41996.14],

"MAPE":[20.53,21.86,36.71]

})

st.dataframe(comparison,use_container_width=True)

best=comparison.loc[
comparison["RMSE"].idxmin(),
"Model"
]

st.success(f"✅ Best Forecasting Model : {best}")

st.divider()

# -----------------------------------------
# Forecast Report
# -----------------------------------------

st.header("📅 Category & Region Forecast")

forecast_path="report/category_region_forecast.csv"

if os.path.exists(forecast_path):

    forecast=pd.read_csv(forecast_path,index_col=0)

    st.dataframe(forecast,use_container_width=True)

else:

    st.warning("Forecast report not found.")

st.divider()

# -----------------------------------------
# Model Comparison CSV
# -----------------------------------------

st.header("📋 Model Comparison Report")

model_path="report/model_comparison.csv"

if os.path.exists(model_path):

    model=pd.read_csv(model_path)

    if "Unnamed: 0" in model.columns:
        model=model.drop(columns=["Unnamed: 0"])

    st.dataframe(model,use_container_width=True)

st.divider()

# -----------------------------------------
# Anomaly Detection
# -----------------------------------------

st.header("🚨 Anomaly Detection")

a1,a2=st.columns(2)

a1.info("Isolation Forest detected **11 anomalous weeks**.")

a2.warning("Z-Score detected **2 extreme outliers**.")

st.divider()

# -----------------------------------------
# Product Clustering
# -----------------------------------------

st.header("📦 Product Demand Segmentation")

st.markdown("""

Three product clusters were identified using **K-Means Clustering**.

### Cluster 1
High Demand Products

### Cluster 2
Medium Demand Products

### Cluster 3
Low Demand Products

""")

st.divider()

# -----------------------------------------
# Customer Segmentation
# -----------------------------------------

st.header("👥 Customer Segmentation")

st.markdown("""

Customers were segmented using **RFM Analysis**.

- ⭐ Premium Customers
- 👤 Regular Customers
- 🔄 Low Value Customers

""")

st.divider()

# -----------------------------------------
# Charts
# -----------------------------------------

st.header("📊 Generated Charts")

chart_folder="charts"

if os.path.exists(chart_folder):

    images=[f for f in os.listdir(chart_folder) if f.endswith(".png")]

    if len(images)==0:

        st.warning("No chart images found.")

    else:

        cols=st.columns(2)

        for i,img in enumerate(images):

            with cols[i%2]:

                st.image(
                    os.path.join(chart_folder,img),
                    caption=img,
                    use_container_width=True
                )

else:

    st.warning("Charts folder not found.")

st.divider()

# -----------------------------------------
# Business Insights
# -----------------------------------------

st.header("💡 Business Insights")

st.markdown("""

- SARIMA achieved the highest forecasting accuracy.

- Technology category is expected to generate the highest future sales.

- Isolation Forest successfully detected unusual sales behaviour.

- Customer segmentation helps identify premium and inactive customers.

- Product clustering supports better inventory planning.

""")

st.divider()

# -----------------------------------------
# Footer
# -----------------------------------------

st.caption("Sales Forecasting & Demand Intelligence Dashboard | Machine Learning Internship Project | Chirag Nagra")