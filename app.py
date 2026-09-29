import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Reforestation & Tree Planting Monitor", layout="wide")
st.title("🌱 Tree Planting & Restoration Monitoring System")

# 1. Synthetic/Demo Data Loader
@st.cache_data
def load_restoration_data():
    np.random.seed(42)
    n_plots = 180
    species_list = ["Dipterocarpus alatus", "Afzelia xylocarpa", "Pterocarpus macrocarpus", "Hopea odorata", "Dalbergia cochinchinensis"]
    zones = ["Community Forest", "Riparian Buffer", "Degraded Core Zone", "Agroforestry Belt"]
    
    dates = pd.date_range(start="2021-01-01", end="2025-12-31", periods=n_plots)
    
    df = pd.DataFrame({
        "Plot_ID": [f"PLOT-{1000 + i}" for i in range(n_plots)],
        "Planting_Date": dates,
        "Zone": np.random.choice(zones, size=n_plots),
        "Dominant_Species": np.random.choice(species_list, size=n_plots),
        "Trees_Planted": np.random.randint(200, 2500, size=n_plots),
        "Area_ha": np.round(np.random.uniform(0.5, 5.0, size=n_plots), 2),
        "Survival_Rate_pct": np.round(np.random.uniform(55, 96, size=n_plots), 1),
        "Mean_Height_m": np.round(np.random.uniform(1.2, 6.8, size=n_plots), 2),
    })
    
    # Calculate derived monitoring indicators
    df["Live_Trees"] = (df["Trees_Planted"] * (df["Survival_Rate_pct"] / 100)).astype(int)
    # Approximate biomass carbon accumulation (tCO2e) based on live stems and area
    df["Estimated_tCO2e"] = np.round(df["Live_Trees"] * df["Mean_Height_m"] * 0.015, 2)
    return df

df = load_restoration_data()

# 2. Sidebar Filters
st.sidebar.header("Restoration Filters")

min_date = df["Planting_Date"].min().date()
max_date = df["Planting_Date"].max().date()

start_date = st.sidebar.date_input("Start Date", min_date)
end_date = st.sidebar.date_input("End Date", max_date)

selected_zones = st.sidebar.multiselect(
    "Management Zone", 
    options=df["Zone"].unique(), 
    default=df["Zone"].unique()
)

selected_species = st.sidebar.multiselect(
    "Dominant Species", 
    options=df["Dominant_Species"].unique(), 
    default=df["Dominant_Species"].unique()
)

# Apply Filters
filtered_df = df[
    (df["Planting_Date"].dt.date >= start_date) &
    (df["Planting_Date"].dt.date <= end_date) &
    (df["Zone"].isin(selected_zones)) &
    (df["Dominant_Species"].isin(selected_species))
]

# 3. Top Metrics Row
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Trees Planted", f"{filtered_df['Trees_Planted'].sum():,}")
c2.metric("Restored Area (ha)", f"{filtered_df['Area_ha'].sum():,.2f} ha")
c3.metric("Avg. Survival Rate", f"{filtered_df['Survival_Rate_pct'].mean():.1f}%")
c4.metric("Est. Carbon (tCO₂e)", f"{filtered_df['Estimated_tCO2e'].sum():,.1f}")

st.divider()

# 4. Trends and Distribution
col_trend, col_pie = st.columns([2, 1])

with col_trend:
    st.subheader("📈 Cumulative Trees Planted Over Time")
    trend_df = (
        filtered_df.sort_values("Planting_Date")
        .set_index("Planting_Date")
        .resample("ME")["Trees_Planted"]
        .sum()
        .cumsum()
        .reset_index()
    )
    fig1 = px.line(
        trend_df,
        x="Planting_Date",
        y="Trees_Planted",
        markers=True,
        labels={"Planting_Date": "Month", "Trees_Planted": "Cumulative Trees Planted"}
    )
    st.plotly_chart(fig1, use_container_width=True)

with col_pie:
    st.subheader("🌳 Area by Management Zone")
    zone_area = filtered_df.groupby("Zone")["Area_ha"].sum().reset_index()
    fig2 = px.bar(
        zone_area, 
        x="Zone", 
        y="Area_ha", 
        color="Zone",
        labels={"Area_ha": "Area (ha)"}
    )
    st.plotly_chart(fig2, use_container_width=True)

# 5. Species Ranking and Survival Diagnostics
col_rank, col_scatter = st.columns([1, 1])

with col_rank:
    st.subheader("🏆 Survival Rate by Species")
    species_perf = (
        filtered_df.groupby("Dominant_Species")["Survival_Rate_pct"]
        .mean()
        .reset_index()
        .sort_values(by="Survival_Rate_pct", ascending=True)
    )
    fig3 = px.bar(
        species_perf,
        x="Survival_Rate_pct",
        y="Dominant_Species",
        orientation="h",
        color="Survival_Rate_pct",
        color_continuous_scale="greens",
        labels={"Survival_Rate_pct": "Avg Survival Rate (%)", "Dominant_Species": "Species"}
    )
    st.plotly_chart(fig3, use_container_width=True)

with col_scatter:
    st.subheader("🌱 Tree Height vs. Biomass Carbon")
    fig4 = px.scatter(
        filtered_df,
        x="Mean_Height_m",
        y="Estimated_tCO2e",
        color="Zone",
        size="Trees_Planted",
        hover_data=["Plot_ID", "Dominant_Species"]
    )
    st.plotly_chart(fig4, use_container_width=True)

st.divider()

# 6. Machine Learning Growth Prediction
st.subheader("🤖 Biomass Carbon Sequestration Prediction (Regression)")

ml_df = df.dropna().copy()
ml_df["Months_Since_Planting"] = (
    (pd.Timestamp.today() - ml_df["Planting_Date"]).dt.days // 30
)

X = ml_df[["Months_Since_Planting", "Trees_Planted", "Area_ha"]].values
y = ml_df["Estimated_tCO2e"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
pred = model.predict(X_test)

score = r2_score(y_test, pred)
st.write(f"**Model R² Score:** `{score:.4f}`")

eval_df = pd.DataFrame({"Actual_tCO2e": y_test, "Predicted_tCO2e": pred}).head(35)
st.line_chart(eval_df)