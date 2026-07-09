"""
Sales Forecasting & Demand Intelligence Dashboard
--------------------------------------------------
4-page interactive Streamlit app:
  1. Sales Overview        -> yearly/monthly trends, region+category filters
  2. Forecast Explorer     -> pick Category/Region, pick horizon (1-3 months), SARIMA forecast + MAE/RMSE
  3. Anomaly Report        -> Isolation Forest + Z-Score anomalies on weekly sales
  4. Product Demand Segments -> K-Means clustering of sub-categories (PCA visualised)

Run with:
    streamlit run app.py
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from scipy.stats import zscore
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX

# ------------------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------------------
st.set_page_config(
    page_title="Sales Forecasting & Demand Intelligence",
    page_icon="📈",
    layout="wide",
)

# ------------------------------------------------------------------------
# DATA LOADING
# ------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading sales data...")
def load_data():
    candidates = ["train.csv", "data/train.csv", "./data/train.csv"]
    path_found = next((p for p in candidates if os.path.exists(p)), None)

    if path_found is None:
        st.error(
            "❌ train.csv not found. Place it in the same folder as app.py "
            "(or in a `data/` subfolder next to app.py)."
        )
        st.stop()

    try:
        df = pd.read_csv(path_found, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(path_found, encoding="ISO-8859-1")

    df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True, errors="coerce")
    if "Ship Date" in df.columns:
        df["Ship Date"] = pd.to_datetime(df["Ship Date"], dayfirst=True, errors="coerce")
        df["Shipping Days"] = (df["Ship Date"] - df["Order Date"]).dt.days

    df = df.dropna(subset=["Order Date", "Sales"])
    df["Year"] = df["Order Date"].dt.year
    df["Month"] = df["Order Date"].dt.month
    return df


df = load_data()

# ------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# ------------------------------------------------------------------------
st.sidebar.title("📊 Dashboard Menu")

page = st.sidebar.radio(
    "Navigate to:",
    [
        "🏠 1. Sales Overview",
        "🔮 2. Forecast Explorer",
        "🚨 3. Anomaly Report",
        "📦 4. Product Demand Segments",
        "👥 5. Customer Segmentation (Bonus)",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
**Sales Forecasting & Demand Intelligence System**

Built with SARIMA, Prophet, XGBoost, Isolation Forest,
Z-Score and K-Means.

Prepared by **Chirag Nagra**
"""
)

# ------------------------------------------------------------------------
# SHARED HELPERS
# ------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_monthly_series(_df, category="All", region="All"):
    d = _df.copy()
    if category != "All":
        d = d[d["Category"] == category]
    if region != "All":
        d = d[d["Region"] == region]
    ts = d.groupby(pd.Grouper(key="Order Date", freq="ME"))["Sales"].sum()
    return ts.asfreq("ME").fillna(0)


@st.cache_data(show_spinner="Training SARIMA model...")
def sarima_forecast(ts, horizon):
    """Backtests on the last 3 actual months (for MAE/RMSE) and forecasts
    `horizon` months into the future using the full series."""
    if len(ts) < 15:
        return None

    # --- Backtest: hold out last 3 months to measure accuracy ---
    train, test = ts.iloc[:-3], ts.iloc[-3:]
    bt_model = SARIMAX(
        train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12),
        enforce_stationarity=False, enforce_invertibility=False,
    )
    bt_fitted = bt_model.fit(disp=False)
    test_pred = bt_fitted.forecast(steps=3)

    mae = mean_absolute_error(test, test_pred)
    rmse = np.sqrt(mean_squared_error(test, test_pred))

    # --- Future forecast: fit on full series, forecast `horizon` months ahead ---
    full_model = SARIMAX(
        ts, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12),
        enforce_stationarity=False, enforce_invertibility=False,
    )
    full_fitted = full_model.fit(disp=False)
    future = full_fitted.get_forecast(steps=horizon)
    future_mean = future.predicted_mean
    future_ci = future.conf_int()

    return {
        "mae": mae,
        "rmse": rmse,
        "test": test,
        "test_pred": test_pred,
        "future_mean": future_mean,
        "future_ci": future_ci,
    }


@st.cache_data(show_spinner="Detecting anomalies...")
def detect_anomalies(_df):
    weekly = (
        _df.groupby(pd.Grouper(key="Order Date", freq="W"))["Sales"]
        .sum()
        .reset_index()
    )

    iso = IsolationForest(contamination=0.05, random_state=42)
    weekly["Isolation_Forest"] = iso.fit_predict(weekly[["Sales"]]) == -1

    weekly["Z_Score"] = zscore(weekly["Sales"])
    weekly["Z_Score_Anomaly"] = weekly["Z_Score"].abs() > 3

    return weekly


@st.cache_data(show_spinner="Running clustering...")
def cluster_products(_df):
    monthly = (
        _df.groupby(["Sub-Category", pd.Grouper(key="Order Date", freq="ME")])["Sales"]
        .sum()
        .reset_index()
    )

    growth = monthly.groupby("Sub-Category")["Sales"].apply(
        lambda x: ((x.iloc[-1] - x.iloc[0]) / x.iloc[0]) * 100 if x.iloc[0] != 0 else 0
    )

    features = pd.DataFrame({
        "Total Sales": monthly.groupby("Sub-Category")["Sales"].sum(),
        "Growth Rate (%)": growth,
        "Sales Volatility": monthly.groupby("Sub-Category")["Sales"].std(),
        "Avg Order Value": monthly.groupby("Sub-Category")["Sales"].mean(),
    }).fillna(0)

    scaled = StandardScaler().fit_transform(features)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    features["Cluster"] = kmeans.fit_predict(scaled)

    # Auto-label clusters by mean total sales (high -> low)
    cluster_rank = (
        features.groupby("Cluster")["Total Sales"].mean().sort_values(ascending=False)
    )
    label_map = {
        cluster_rank.index[0]: "High Volume, Stable Demand",
        cluster_rank.index[1]: "Moderate Demand",
        cluster_rank.index[2]: "Low Volume / Niche Demand",
    }
    features["Demand Segment"] = features["Cluster"].map(label_map)

    pca = PCA(n_components=2)
    coords = pca.fit_transform(scaled)
    features["PCA1"] = coords[:, 0]
    features["PCA2"] = coords[:, 1]

    return features.reset_index().rename(columns={"index": "Sub-Category"})


@st.cache_data(show_spinner="Segmenting customers (RFM)...")
def segment_customers(_df):
    snapshot_date = _df["Order Date"].max() + pd.Timedelta(days=1)

    rfm = _df.groupby("Customer ID").agg(
        Recency=("Order Date", lambda x: (snapshot_date - x.max()).days),
        Frequency=("Order ID", "nunique"),
        Monetary=("Sales", "sum"),
    )
    if "Customer Name" in _df.columns:
        rfm["Customer Name"] = _df.groupby("Customer ID")["Customer Name"].first()

    scaled = StandardScaler().fit_transform(rfm[["Recency", "Frequency", "Monetary"]])

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    rfm["Cluster"] = kmeans.fit_predict(scaled)

    # Auto-label by mean monetary value (high -> low)
    rank = rfm.groupby("Cluster")["Monetary"].mean().sort_values(ascending=False)
    label_map = {
        rank.index[0]: "⭐ Premium Customers",
        rank.index[1]: "👤 Regular Customers",
        rank.index[2]: "🔄 Low Value Customers",
    }
    rfm["Segment"] = rfm["Cluster"].map(label_map)

    pca = PCA(n_components=2)
    coords = pca.fit_transform(scaled)
    rfm["PCA1"] = coords[:, 0]
    rfm["PCA2"] = coords[:, 1]

    return rfm.reset_index()


CUSTOMER_STRATEGY = {
    "⭐ Premium Customers": "High monetary value & frequent buyers — reward with loyalty perks and early access to new products.",
    "👤 Regular Customers": "Steady, moderate spenders — nurture with regular engagement and cross-sell offers.",
    "🔄 Low Value Customers": "Infrequent or low spenders — target with reactivation discounts and win-back campaigns.",
}

STOCKING_STRATEGY = {
    "High Volume, Stable Demand": "Prioritize inventory allocation; avoid stockouts; negotiate bulk supplier deals.",
    "Moderate Demand": "Maintain balanced stock levels; monitor monthly trends before reordering.",
    "Low Volume / Niche Demand": "Keep minimal stock; use just-in-time ordering or seasonal promotions to clear inventory.",
}

# ------------------------------------------------------------------------
# PAGE 1: SALES OVERVIEW
# ------------------------------------------------------------------------
if page.startswith("🏠"):
    st.title("🏠 Sales Overview Dashboard")
    st.caption("High-level view of historical sales performance across the business.")

    # --- Interactive filters ---
    f1, f2, f3 = st.columns(3)
    with f1:
        categories = st.multiselect(
            "Category", sorted(df["Category"].unique()), default=list(df["Category"].unique())
        )
    with f2:
        regions = st.multiselect(
            "Region", sorted(df["Region"].unique()), default=list(df["Region"].unique())
        )
    with f3:
        segments = st.multiselect(
            "Segment", sorted(df["Segment"].unique()), default=list(df["Segment"].unique())
        )

    filtered = df[
        df["Category"].isin(categories)
        & df["Region"].isin(regions)
        & df["Segment"].isin(segments)
    ]

    if filtered.empty:
        st.warning("No data matches the selected filters.")
        st.stop()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Sales", f"${filtered['Sales'].sum():,.0f}")
    k2.metric("Total Orders", f"{filtered['Order ID'].nunique():,}")
    k3.metric("Avg Order Value", f"${filtered['Sales'].mean():,.2f}")
    k4.metric("Date Range", f"{filtered['Order Date'].dt.year.min()}–{filtered['Order Date'].dt.year.max()}")

    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("📅 Total Sales by Year")
        yearly = filtered.groupby("Year")["Sales"].sum().reset_index()
        fig_year = px.bar(yearly, x="Year", y="Sales", text_auto=".2s", color="Sales",
                           color_continuous_scale="Blues")
        fig_year.update_layout(showlegend=False)
        st.plotly_chart(fig_year, use_container_width=True)

    with c2:
        st.subheader("🏷️ Sales by Category")
        cat_sales = filtered.groupby("Category")["Sales"].sum().reset_index()
        fig_cat = px.pie(cat_sales, names="Category", values="Sales", hole=0.45)
        st.plotly_chart(fig_cat, use_container_width=True)

    st.subheader("📈 Monthly Sales Trend")
    monthly = filtered.groupby(pd.Grouper(key="Order Date", freq="ME"))["Sales"].sum().reset_index()
    fig_trend = px.line(monthly, x="Order Date", y="Sales", markers=True)
    fig_trend.update_traces(line_width=3)
    st.plotly_chart(fig_trend, use_container_width=True)

    st.subheader("🌍 Sales by Region & Category")
    region_cat = filtered.groupby(["Region", "Category"])["Sales"].sum().reset_index()
    fig_region = px.bar(
        region_cat, x="Region", y="Sales", color="Category", barmode="group", text_auto=".2s"
    )
    st.plotly_chart(fig_region, use_container_width=True)

# ------------------------------------------------------------------------
# PAGE 2: FORECAST EXPLORER
# ------------------------------------------------------------------------
elif page.startswith("🔮"):
    st.title("🔮 Forecast Explorer")
    st.caption("Select a segment and forecast horizon to generate a live SARIMA forecast.")

    c1, c2, c3 = st.columns(3)
    with c1:
        group_type = st.selectbox("Explore by", ["Category", "Region"])
    with c2:
        options = ["All"] + sorted(df[group_type].unique().tolist())
        group_value = st.selectbox(group_type, options)
    with c3:
        horizon = st.select_slider("Forecast Horizon (months ahead)", options=[1, 2, 3], value=3)

    category_filter = group_value if group_type == "Category" else "All"
    region_filter = group_value if group_type == "Region" else "All"

    ts = get_monthly_series(df, category=category_filter, region=region_filter)

    if len(ts) < 15:
        st.warning("Not enough historical data for this selection to build a reliable SARIMA forecast.")
    else:
        result = sarima_forecast(ts, horizon)

        m1, m2 = st.columns(2)
        m1.metric("Model MAE (backtest, last 3 months)", f"${result['mae']:,.2f}")
        m2.metric("Model RMSE (backtest, last 3 months)", f"${result['rmse']:,.2f}")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=ts.index, y=ts.values, mode="lines", name="Actual Sales", line=dict(width=2)
        ))
        fig.add_trace(go.Scatter(
            x=result["test"].index, y=result["test_pred"].values, mode="lines+markers",
            name="Backtest Forecast (last 3 months)", line=dict(dash="dot", color="orange")
        ))
        fig.add_trace(go.Scatter(
            x=result["future_mean"].index, y=result["future_mean"].values, mode="lines+markers",
            name=f"Future Forecast (+{horizon} months)", line=dict(color="green", width=3)
        ))
        fig.add_trace(go.Scatter(
            x=list(result["future_ci"].index) + list(result["future_ci"].index[::-1]),
            y=list(result["future_ci"].iloc[:, 1]) + list(result["future_ci"].iloc[:, 0][::-1]),
            fill="toself", fillcolor="rgba(0,176,80,0.15)", line=dict(color="rgba(255,255,255,0)"),
            name="Confidence Interval", showlegend=True,
        ))
        fig.update_layout(
            title=f"SARIMA Forecast — {group_type}: {group_value}",
            xaxis_title="Date", yaxis_title="Sales", hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader(f"📋 Next {horizon}-Month Forecast Values")
        forecast_table = pd.DataFrame({
            "Month": result["future_mean"].index.strftime("%b %Y"),
            "Forecasted Sales": result["future_mean"].values.round(2),
            "Lower CI": result["future_ci"].iloc[:, 0].values.round(2),
            "Upper CI": result["future_ci"].iloc[:, 1].values.round(2),
        })
        st.dataframe(forecast_table, use_container_width=True, hide_index=True)

# ------------------------------------------------------------------------
# PAGE 3: ANOMALY REPORT
# ------------------------------------------------------------------------
elif page.startswith("🚨"):
    st.title("🚨 Anomaly Report")
    st.caption("Weekly sales anomalies detected using Isolation Forest and Z-Score methods.")

    weekly = detect_anomalies(df)

    method = st.radio("Detection Method", ["Isolation Forest", "Z-Score", "Both"], horizontal=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=weekly["Order Date"], y=weekly["Sales"], mode="lines", name="Weekly Sales",
        line=dict(color="royalblue"),
    ))

    if method in ("Isolation Forest", "Both"):
        iso_pts = weekly[weekly["Isolation_Forest"]]
        fig.add_trace(go.Scatter(
            x=iso_pts["Order Date"], y=iso_pts["Sales"], mode="markers", name="Isolation Forest Anomaly",
            marker=dict(color="red", size=11, symbol="circle"),
        ))

    if method in ("Z-Score", "Both"):
        z_pts = weekly[weekly["Z_Score_Anomaly"]]
        fig.add_trace(go.Scatter(
            x=z_pts["Order Date"], y=z_pts["Sales"], mode="markers", name="Z-Score Anomaly",
            marker=dict(color="darkorange", size=14, symbol="x"),
        ))

    fig.update_layout(title="Weekly Sales with Detected Anomalies", xaxis_title="Week", yaxis_title="Sales",
                       hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    m1, m2 = st.columns(2)
    m1.metric("Isolation Forest Anomalies", int(weekly["Isolation_Forest"].sum()))
    m2.metric("Z-Score Anomalies", int(weekly["Z_Score_Anomaly"].sum()))

    st.subheader("📋 Detected Anomaly Weeks")
    anomaly_table = weekly[weekly["Isolation_Forest"] | weekly["Z_Score_Anomaly"]].copy()
    anomaly_table["Detected By"] = anomaly_table.apply(
        lambda r: ", ".join(
            [m for m, flag in [("Isolation Forest", r["Isolation_Forest"]),
                                ("Z-Score", r["Z_Score_Anomaly"])] if flag]
        ), axis=1,
    )
    anomaly_table = anomaly_table[["Order Date", "Sales", "Z_Score", "Detected By"]].rename(
        columns={"Order Date": "Week", "Z_Score": "Z-Score"}
    )
    anomaly_table["Week"] = anomaly_table["Week"].dt.strftime("%d %b %Y")
    st.dataframe(anomaly_table.round(2), use_container_width=True, hide_index=True)

# ------------------------------------------------------------------------
# PAGE 4: PRODUCT DEMAND SEGMENTS
# ------------------------------------------------------------------------
elif page.startswith("📦"):
    st.title("📦 Product Demand Segments")
    st.caption("Sub-categories clustered by sales volume, growth, volatility and order value (K-Means + PCA).")

    clusters = cluster_products(df)

    c1, c2 = st.columns([2, 1])
    with c1:
        fig = px.scatter(
            clusters, x="PCA1", y="PCA2", color="Demand Segment", text="Sub-Category",
            size="Total Sales", hover_data=["Total Sales", "Growth Rate (%)", "Sales Volatility"],
        )
        fig.update_traces(textposition="top center")
        fig.update_layout(title="Product Clusters (PCA-reduced)")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("💡 Stocking Strategy")
        for segment, strategy in STOCKING_STRATEGY.items():
            st.markdown(f"**{segment}**")
            st.caption(strategy)
            st.markdown("")

    st.subheader("📋 Sub-Category → Demand Segment")
    display_table = clusters[
        ["Sub-Category", "Demand Segment", "Total Sales", "Growth Rate (%)", "Sales Volatility", "Avg Order Value"]
    ].sort_values("Total Sales", ascending=False)
    st.dataframe(display_table.round(2), use_container_width=True, hide_index=True)

# ------------------------------------------------------------------------
# PAGE 5 (BONUS): CUSTOMER SEGMENTATION (RFM)
# ------------------------------------------------------------------------
elif page.startswith("👥"):
    st.title("👥 Customer Segmentation (RFM Analysis)")
    st.caption(
        "Bonus page — customers clustered by Recency, Frequency and Monetary value using K-Means."
    )

    if "Customer ID" not in df.columns:
        st.warning("Customer ID column not found in the dataset — cannot compute RFM segments.")
    else:
        rfm = segment_customers(df)

        counts = rfm["Segment"].value_counts()
        cols = st.columns(len(counts))
        for col, (seg, count) in zip(cols, counts.items()):
            col.metric(seg, f"{count} customers")

        c1, c2 = st.columns([2, 1])
        with c1:
            fig = px.scatter(
                rfm, x="PCA1", y="PCA2", color="Segment",
                size="Monetary", hover_data=["Customer Name", "Recency", "Frequency", "Monetary"]
                if "Customer Name" in rfm.columns else ["Recency", "Frequency", "Monetary"],
            )
            fig.update_layout(title="Customer Segments (PCA-reduced)")
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.subheader("💡 Engagement Strategy")
            for segment, strategy in CUSTOMER_STRATEGY.items():
                st.markdown(f"**{segment}**")
                st.caption(strategy)
                st.markdown("")

        st.subheader("📋 Customer → Segment")
        cols_to_show = [c for c in ["Customer Name", "Customer ID", "Segment", "Recency", "Frequency", "Monetary"] if c in rfm.columns]
        st.dataframe(
            rfm[cols_to_show].sort_values("Monetary", ascending=False).round(2),
            use_container_width=True, hide_index=True,
        )