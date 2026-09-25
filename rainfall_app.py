import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Rainfall Prediction in India",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    .dashboard-title {
        font-size: 2.3rem;
        font-weight: 700;
        color: #4dabf7;
        margin-bottom: 0;
    }

    .dashboard-subtitle {
        color: #aab4c3;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }

    .kpi-card {
        background: linear-gradient(135deg, #172033, #111827);
        border: 1px solid #26354d;
        border-radius: 12px;
        padding: 18px;
        min-height: 120px;
    }

    .kpi-title {
        color: #9ca9bd;
        font-size: 0.85rem;
    }

    .kpi-value {
        color: white;
        font-size: 1.8rem;
        font-weight: 700;
        margin-top: 8px;
    }

    .kpi-note {
        color: #6ee7b7;
        font-size: 0.75rem;
        margin-top: 5px;
    }

    .section-title {
        font-size: 1.35rem;
        font-weight: 650;
        color: #e8eef7;
        margin-top: 1rem;
        margin-bottom: 0.7rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    historical_file = "rainfall in india 1901-2015.csv"
    district_file = "district_wise_rainfall_normal.csv"

    historical = pd.read_csv(historical_file)
    district = pd.read_csv(district_file)

    return historical, district


try:
    historical, district = load_data()

except Exception as e:
    st.error("Unable to load the CSV files.")
    st.write("Make sure both CSV files are in the same folder as rainfall_app.py")
    st.exception(e)
    st.stop()


# ============================================================
# CLEAN COLUMN NAMES
# ============================================================

historical.columns = historical.columns.str.strip()
district.columns = district.columns.str.strip()

historical["YEAR"] = pd.to_numeric(
    historical["YEAR"], errors="coerce"
)

historical["ANNUAL"] = pd.to_numeric(
    historical["ANNUAL"], errors="coerce"
)

# Monthly columns
months = [
    "JAN", "FEB", "MAR", "APR", "MAY", "JUN",
    "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"
]

months = [m for m in months if m in historical.columns]

for col in months:
    historical[col] = pd.to_numeric(
        historical[col], errors="coerce"
    )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🌧️ Rainfall Dashboard")

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Dashboard Section",
    [
        "Overview",
        "Historical Trends",
        "Seasonal & Monthly Analysis",
        "Regional Analysis",
        "Rainfall Prediction",
        "Data Explorer"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info(
    "Historical rainfall data: 1901–2015\n\n"
    "Prediction horizon: 2016"
)


# ============================================================
# COMMON CALCULATIONS
# ============================================================

annual_data = (
    historical
    .groupby("YEAR", as_index=False)["ANNUAL"]
    .mean()
)

annual_data = annual_data.dropna()

long_term_mean = annual_data["ANNUAL"].mean()

latest_year = int(annual_data["YEAR"].max())

latest_value = float(
    annual_data.loc[
        annual_data["YEAR"] == latest_year,
        "ANNUAL"
    ].iloc[0]
)

subdivision_count = historical["SUBDIVISION"].nunique()

# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="dashboard-title">🌧️ Rainfall Prediction in India</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="dashboard-subtitle">'
    'Historical Rainfall Analysis, Regional Patterns and Next-Year Prediction'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# OVERVIEW
# ============================================================

if page == "Overview":

    st.markdown(
        '<div class="section-title">Executive Summary</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Long-Term Average Rainfall</div>
                <div class="kpi-value">{long_term_mean:,.1f} mm</div>
                <div class="kpi-note">1901–2015 average</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Latest Historical Year</div>
                <div class="kpi-value">{latest_year}</div>
                <div class="kpi-note">{latest_value:,.1f} mm</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Meteorological Subdivisions</div>
                <div class="kpi-value">{subdivision_count}</div>
                <div class="kpi-note">Regional coverage</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c4:
        missing = historical.isna().sum().sum()

        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Missing Data Cells</div>
                <div class="kpi-value">{missing:,}</div>
                <div class="kpi-note">Across historical dataset</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")

    # Annual trend
    st.markdown(
        '<div class="section-title">📈 India Annual Rainfall Trend</div>',
        unsafe_allow_html=True
    )

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=annual_data["YEAR"],
            y=annual_data["ANNUAL"],
            name="Annual Rainfall",
            marker_color="#3182ce"
        )
    )

    rolling = annual_data["ANNUAL"].rolling(
        window=10,
        center=True
    ).mean()

    fig.add_trace(
        go.Scatter(
            x=annual_data["YEAR"],
            y=rolling,
            name="10-Year Moving Average",
            mode="lines",
            line=dict(
                color="#f59e0b",
                width=3
            )
        )
    )

    fig.add_hline(
        y=long_term_mean,
        line_dash="dash",
        line_color="#ef4444",
        annotation_text=f"Long-term mean: {long_term_mean:,.0f} mm"
    )

    fig.update_layout(
        height=500,
        template="plotly_dark",
        xaxis_title="Year",
        yaxis_title="Average Annual Rainfall (mm)",
        hovermode="x unified"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # Regional averages
    st.markdown(
        '<div class="section-title">🌍 Average Rainfall by Subdivision</div>',
        unsafe_allow_html=True
    )

    regional = (
        historical
        .groupby("SUBDIVISION")["ANNUAL"]
        .mean()
        .dropna()
        .sort_values(ascending=False)
        .reset_index()
    )

    fig2 = px.bar(
        regional,
        x="ANNUAL",
        y="SUBDIVISION",
        orientation="h",
        color="ANNUAL",
        color_continuous_scale="Blues",
        labels={
            "ANNUAL": "Average Annual Rainfall (mm)",
            "SUBDIVISION": "Subdivision"
        },
        title="Average Annual Rainfall by Meteorological Subdivision"
    )

    fig2.update_layout(
        height=800,
        template="plotly_dark"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )


# ============================================================
# HISTORICAL TRENDS
# ============================================================

elif page == "Historical Trends":

    st.markdown(
        '<div class="section-title">📈 Historical Rainfall Trends</div>',
        unsafe_allow_html=True
    )

    selected_subdivision = st.selectbox(
        "Select Meteorological Subdivision",
        sorted(historical["SUBDIVISION"].dropna().unique())
    )

    sub_data = historical[
        historical["SUBDIVISION"] == selected_subdivision
    ].copy()

    sub_data = sub_data.sort_values("YEAR")

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=sub_data["YEAR"],
            y=sub_data["ANNUAL"],
            name="Annual Rainfall",
            marker_color="#2563eb"
        )
    )

    rolling = sub_data["ANNUAL"].rolling(
        10,
        center=True
    ).mean()

    fig.add_trace(
        go.Scatter(
            x=sub_data["YEAR"],
            y=rolling,
            name="10-Year Moving Average",
            mode="lines",
            line=dict(
                color="#f59e0b",
                width=3
            )
        )
    )

    fig.update_layout(
        template="plotly_dark",
        height=550,
        title=f"Rainfall Trend — {selected_subdivision}",
        xaxis_title="Year",
        yaxis_title="Annual Rainfall (mm)",
        hovermode="x unified"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # Anomaly chart
    sub_data["ANOMALY"] = (
        sub_data["ANNUAL"] -
        sub_data["ANNUAL"].mean()
    )

    fig2 = px.bar(
        sub_data,
        x="YEAR",
        y="ANOMALY",
        color="ANOMALY",
        color_continuous_scale="RdBu",
        title="Rainfall Anomaly Relative to Subdivision Average",
        labels={
            "ANOMALY": "Rainfall Anomaly (mm)"
        }
    )

    fig2.update_layout(
        template="plotly_dark",
        height=450
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )


# ============================================================
# MONTHLY & SEASONAL
# ============================================================

elif page == "Seasonal & Monthly Analysis":

    st.markdown(
        '<div class="section-title">🌦️ Monthly & Seasonal Rainfall</div>',
        unsafe_allow_html=True
    )

    monthly_means = historical[months].mean()

    month_df = pd.DataFrame({
        "Month": monthly_means.index,
        "Rainfall": monthly_means.values
    })

    fig = px.bar(
        month_df,
        x="Month",
        y="Rainfall",
        color="Rainfall",
        color_continuous_scale="Blues",
        title="Average Monthly Rainfall Across India",
        labels={
            "Rainfall": "Average Rainfall (mm)"
        }
    )

    fig.update_layout(
        template="plotly_dark",
        height=450
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # Seasonal data
    seasons = [
        "Jan-Feb",
        "Mar-May",
        "Jun-Sep",
        "Oct-Dec"
    ]

    available_seasons = [
        x for x in seasons
        if x in historical.columns
    ]

    season_means = historical[
        available_seasons
    ].mean()

    season_df = pd.DataFrame({
        "Season": season_means.index,
        "Rainfall": season_means.values
    })

    fig2 = px.pie(
        season_df,
        names="Season",
        values="Rainfall",
        hole=0.45,
        title="Seasonal Contribution to Rainfall"
    )

    fig2.update_layout(
        template="plotly_dark",
        height=500
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

    # Heatmap
    st.markdown(
        '<div class="section-title">🔥 Monthly Rainfall Heatmap</div>',
        unsafe_allow_html=True
    )

    heatmap_data = historical.pivot_table(
        index="SUBDIVISION",
        values=months,
        aggfunc="mean"
    )

    fig3 = px.imshow(
        heatmap_data,
        aspect="auto",
        color_continuous_scale="Blues",
        labels={
            "x": "Month",
            "y": "Subdivision",
            "color": "Rainfall (mm)"
        },
        title="Average Monthly Rainfall by Subdivision"
    )

    fig3.update_layout(
        height=850,
        template="plotly_dark"
    )

    st.plotly_chart(
        fig3,
        use_container_width=True
    )


# ============================================================
# REGIONAL ANALYSIS
# ============================================================

elif page == "Regional Analysis":

    st.markdown(
        '<div class="section-title">🗺️ Regional Rainfall Analysis</div>',
        unsafe_allow_html=True
    )

    regional = (
        historical
        .groupby("SUBDIVISION")
        .agg(
            Average_Rainfall=("ANNUAL", "mean"),
            Minimum_Rainfall=("ANNUAL", "min"),
            Maximum_Rainfall=("ANNUAL", "max"),
            Std_Dev=("ANNUAL", "std")
        )
        .reset_index()
    )

    regional["CV"] = (
        regional["Std_Dev"] /
        regional["Average_Rainfall"] *
        100
    )

    fig = px.scatter(
        regional,
        x="Average_Rainfall",
        y="CV",
        size="Maximum_Rainfall",
        color="Average_Rainfall",
        hover_name="SUBDIVISION",
        color_continuous_scale="Blues",
        title="Rainfall Amount vs Variability",
        labels={
            "Average_Rainfall": "Average Annual Rainfall (mm)",
            "CV": "Coefficient of Variation (%)"
        }
    )

    fig.update_layout(
        template="plotly_dark",
        height=600
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown(
        '<div class="section-title">Regional Summary</div>',
        unsafe_allow_html=True
    )

    st.dataframe(
        regional.sort_values(
            "Average_Rainfall",
            ascending=False
        ).round(2),
        use_container_width=True,
        hide_index=True
    )

    # District dataset
    st.markdown(
        '<div class="section-title">📍 District-Level Rainfall Data</div>',
        unsafe_allow_html=True
    )

    st.write(
        f"The district dataset contains {len(district):,} records."
    )

    st.dataframe(
        district.head(20),
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# RAINFALL PREDICTION
# ============================================================

elif page == "Rainfall Prediction":

    st.markdown(
        '<div class="section-title">🔮 Annual Rainfall Prediction</div>',
        unsafe_allow_html=True
    )

    st.info(
        "The prediction uses historical annual rainfall patterns "
        "for each meteorological subdivision. The forecast year "
        "is 2016 because the historical dataset ends in 2015."
    )

    def forecast_subdivision(data):

        data = data[
            ["YEAR", "ANNUAL"]
        ].dropna().sort_values("YEAR").copy()

        if len(data) < 10:
            return np.nan

        data["lag1"] = data["ANNUAL"].shift(1)
        data["lag2"] = data["ANNUAL"].shift(2)
        data["rolling3"] = (
            data["ANNUAL"]
            .shift(1)
            .rolling(3)
            .mean()
        )

        train = data.dropna()

        if len(train) < 8:
            return float(data["ANNUAL"].iloc[-1])

        X = train[
            ["YEAR", "lag1", "lag2", "rolling3"]
        ]

        y = train["ANNUAL"]

        model = RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            max_depth=8
        )

        model.fit(X, y)

        last = data.iloc[-1]

        previous_values = data["ANNUAL"].dropna()

        lag1 = previous_values.iloc[-1]

        lag2 = (
            previous_values.iloc[-2]
            if len(previous_values) >= 2
            else lag1
        )

        rolling3 = (
            previous_values
            .tail(3)
            .mean()
        )

        prediction = model.predict(
            pd.DataFrame({
                "YEAR": [2016],
                "lag1": [lag1],
                "lag2": [lag2],
                "rolling3": [rolling3]
            })
        )[0]

        return prediction


    predictions = []

    for subdivision in historical["SUBDIVISION"].dropna().unique():

        temp = historical[
            historical["SUBDIVISION"] == subdivision
        ]

        pred = forecast_subdivision(temp)

        historical_mean = temp["ANNUAL"].mean()

        predictions.append({
            "SUBDIVISION": subdivision,
            "Historical Average": historical_mean,
            "Predicted 2016": pred,
            "Difference": pred - historical_mean,
            "Difference %": (
                (pred - historical_mean) /
                historical_mean * 100
            )
            if historical_mean != 0
            else np.nan
        })

    prediction_df = pd.DataFrame(predictions)

    # National prediction
    national_prediction = prediction_df[
        "Predicted 2016"
    ].mean()

    p1, p2, p3 = st.columns(3)

    with p1:
        st.metric(
            "Predicted 2016 Rainfall",
            f"{national_prediction:,.1f} mm"
        )

    with p2:
        st.metric(
            "Historical Average",
            f"{long_term_mean:,.1f} mm"
        )

    with p3:

        change = (
            national_prediction -
            long_term_mean
        ) / long_term_mean * 100

        st.metric(
            "Predicted Difference",
            f"{change:+.2f}%"
        )

    st.markdown("---")

    # Prediction chart
    chart_data = prediction_df.sort_values(
        "Predicted 2016",
        ascending=False
    )

    fig = px.bar(
        chart_data,
        x="Predicted 2016",
        y="SUBDIVISION",
        orientation="h",
        color="Predicted 2016",
        color_continuous_scale="Blues",
        title="Predicted Annual Rainfall by Subdivision — 2016",
        labels={
            "Predicted 2016": "Predicted Rainfall (mm)"
        }
    )

    fig.update_layout(
        template="plotly_dark",
        height=850
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown(
        '<div class="section-title">Prediction Results</div>',
        unsafe_allow_html=True
    )

    st.dataframe(
        prediction_df.round(2),
        use_container_width=True,
        hide_index=True
    )

    st.warning(
        "This is a machine-learning forecast based on historical "
        "rainfall patterns. It should be interpreted as an analytical "
        "forecast rather than an official meteorological prediction."
    )


# ============================================================
# DATA EXPLORER
# ============================================================

elif page == "Data Explorer":

    st.markdown(
        '<div class="section-title">📋 Historical Dataset Explorer</div>',
        unsafe_allow_html=True
    )

    selected_year = st.slider(
        "Select Year",
        int(historical["YEAR"].min()),
        int(historical["YEAR"].max()),
        int(historical["YEAR"].max())
    )

    year_data = historical[
        historical["YEAR"] == selected_year
    ]

    st.write(
        f"Rainfall records for {selected_year}"
    )

    st.dataframe(
        year_data,
        use_container_width=True,
        hide_index=True
    )

    st.markdown(
        '<div class="section-title">Missing Values</div>',
        unsafe_allow_html=True
    )

    missing_df = (
        historical.isna()
        .sum()
        .reset_index()
    )

    missing_df.columns = [
        "Column",
        "Missing Values"
    ]

    missing_df = missing_df[
        missing_df["Missing Values"] > 0
    ].sort_values(
        "Missing Values",
        ascending=False
    )

    fig = px.bar(
        missing_df,
        x="Missing Values",
        y="Column",
        orientation="h",
        color="Missing Values",
        color_continuous_scale="Reds",
        title="Missing Values by Column"
    )

    fig.update_layout(
        template="plotly_dark",
        height=500
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Rainfall Prediction in India | Historical data: 1901–2015 | "
    "Interactive analytical dashboard"
)
