from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


DATA_PATH = Path(__file__).with_name("Economy.xlsx")
QUARTER_ORDER = ["Q1", "Q2", "Q3", "Q4"]
CHART_COLORS = ["#087f5b", "#e07a32", "#2677a8", "#b34747", "#708238"]


@st.cache_data
def load_data():
    data = pd.read_excel(DATA_PATH)
    data.columns = data.columns.str.strip()
    required_columns = {
        "Year", "Quarter", "Sector", "GDP_Lakh_Crore", "Growth_%",
        "Employment_Million", "Exports_Crore", "Imports_Crore",
    }
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        raise ValueError(f"Economy.xlsx is missing columns: {', '.join(sorted(missing_columns))}")
    data["Quarter"] = pd.Categorical(
        data["Quarter"], categories=QUARTER_ORDER, ordered=True
    )
    return data.drop_duplicates()


def style_axis(axis):
    axis.set_facecolor("#ffffff")
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.spines["bottom"].set_color("#dce5df")
    axis.tick_params(axis="both", colors="#53645c", labelsize=9, length=0, pad=7)
    axis.grid(axis="y", color="#e8efeb", linewidth=0.8)
    axis.set_axisbelow(True)


def format_chart(figure):
    figure.patch.set_facecolor("#ffffff")
    figure.tight_layout(pad=1.5)
    st.pyplot(figure, use_container_width=True)
    plt.close(figure)


st.set_page_config(
    page_title="Indian Economy Analytics (2015–2025)",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp { background: linear-gradient(135deg, #f5f8f5 0%, #f7f8f4 55%, #eef5f3 100%); }
    h1, h2, h3 { font-family: 'Manrope', sans-serif; color: #173d30; }
    .hero { padding: 1rem 0 1.15rem; border-bottom: 1px solid #dce7df; margin-bottom: 1.25rem; }
    .eyebrow { color: #087f5b; font-size: .75rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; }
    .hero h1 { font-size: 2.15rem; margin: .2rem 0 .25rem; }
    .hero p { color: #607067; margin: 0; font-size: .96rem; }
    [data-testid="stMetric"] { background: rgba(255,255,255,.86); border: 1px solid #e0e9e3; border-radius: 8px; padding: 14px 16px; }
    [data-testid="stMetricLabel"] { color: #617269; }
    [data-testid="stMetricValue"] { color: #173d30; }
    [data-testid="stSidebar"] { background: #edf3ee; border-right: 1px solid #dce7df; }
    div[data-testid="stTabs"] button { font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)

try:
    economy = load_data()
except (FileNotFoundError, ValueError, ImportError) as error:
    st.error(f"Unable to load the economy dataset: {error}")
    st.stop()

years = sorted(economy["Year"].dropna().astype(int).unique().tolist())
sectors = sorted(economy["Sector"].dropna().unique().tolist())
quarters = [quarter for quarter in QUARTER_ORDER if quarter in economy["Quarter"].astype(str).unique()]

st.sidebar.markdown("### Explore the data")
year_range = st.sidebar.slider(
    "Year range", min_value=years[0], max_value=years[-1], value=(years[0], years[-1])
)
selected_sectors = st.sidebar.multiselect("Sectors", sectors, default=sectors)
selected_quarters = st.sidebar.multiselect("Quarters", quarters, default=quarters)

if not selected_sectors or not selected_quarters:
    st.info("Choose at least one sector and one quarter in the sidebar.")
    st.stop()

filtered = economy[
    economy["Year"].between(year_range[0], year_range[1])
    & economy["Sector"].isin(selected_sectors)
    & economy["Quarter"].astype(str).isin(selected_quarters)
].copy()

if filtered.empty:
    st.warning("No observations match these filters.")
    st.stop()

latest_year = int(filtered["Year"].max())
previous_year = latest_year - 1
latest_gdp = filtered.loc[filtered["Year"] == latest_year, "GDP_Lakh_Crore"].sum()
previous_gdp = filtered.loc[filtered["Year"] == previous_year, "GDP_Lakh_Crore"].sum()
gdp_change = (latest_gdp / previous_gdp - 1) * 100 if previous_gdp else None
trade_balance = filtered["Exports_Crore"].sum() - filtered["Imports_Crore"].sum()

st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">Economic data · 2015–2025</div>
      <h1>India Economic Monitor</h1>
      <p>Sector output, trade, employment and growth across quarterly observations.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

metric_columns = st.columns(4)
metric_columns[0].metric(
    f"GDP · {latest_year}", f"₹{latest_gdp:,.1f} lakh crore",
    f"{gdp_change:+.1f}% vs {previous_year}" if gdp_change is not None else "No prior-year comparison",
)
metric_columns[1].metric("Average growth", f"{filtered['Growth_%'].mean():.2f}%")
metric_columns[2].metric("Imports", f"₹{filtered['Imports_Crore'].sum() / 1_000_000:,.2f} lakh crore")
metric_columns[3].metric(
    "Trade balance", f"₹{trade_balance / 1_000_000:+,.2f} lakh crore",
    "Exports minus imports",
)

overview_tab, forecast_tab, data_tab = st.tabs(["Overview", "Import outlook", "Data"])

with overview_tab:
    annual = filtered.groupby("Year", observed=True).agg(
        GDP=("GDP_Lakh_Crore", "sum"),
        Exports=("Exports_Crore", "sum"),
        Imports=("Imports_Crore", "sum"),
        Growth=("Growth_%", "mean"),
    ).sort_index()
    sector_gdp = filtered.groupby("Sector", observed=True)["GDP_Lakh_Crore"].sum().sort_values()

    left, right = st.columns([1.55, 1])
    with left:
        st.subheader("GDP by year")
        figure, axis = plt.subplots(figsize=(8, 3.6))
        style_axis(axis)
        for index, sector in enumerate(selected_sectors):
            series = filtered[filtered["Sector"] == sector].groupby("Year")["GDP_Lakh_Crore"].sum()
            axis.plot(series.index, series.values, color=CHART_COLORS[index % len(CHART_COLORS)],
                      linewidth=2, marker="o", markersize=3, label=sector)
        axis.set_ylabel("GDP (lakh crore)", color="#53645c", fontsize=9)
        axis.set_xticks(annual.index)
        axis.tick_params(axis="x", rotation=0)
        axis.legend(frameon=False, ncol=2, fontsize=8, loc="upper left")
        format_chart(figure)
    with right:
        st.subheader("GDP by sector")
        figure, axis = plt.subplots(figsize=(6, 3.6))
        style_axis(axis)
        axis.barh(sector_gdp.index, sector_gdp.values, color="#168567", height=0.68)
        axis.set_xlabel("GDP (lakh crore)", color="#53645c", fontsize=9)
        axis.grid(axis="x", color="#e8efeb", linewidth=0.8)
        axis.grid(axis="y", visible=False)
        format_chart(figure)

    left, right = st.columns(2)
    with left:
        st.subheader("Imports and exports")
        figure, axis = plt.subplots(figsize=(7, 3.4))
        style_axis(axis)
        axis.plot(annual.index, annual["Imports"] / 1_000_000, color="#e07a32", marker="o",
                  linewidth=2, label="Imports")
        axis.plot(annual.index, annual["Exports"] / 1_000_000, color="#2677a8", marker="o",
                  linewidth=2, label="Exports")
        axis.set_ylabel("₹ lakh crore", color="#53645c", fontsize=9)
        axis.set_xticks(annual.index)
        axis.legend(frameon=False, ncol=2)
        format_chart(figure)
    with right:
        st.subheader("Average growth by year")
        figure, axis = plt.subplots(figsize=(7, 3.4))
        style_axis(axis)
        growth_colors = ["#b34747" if value < 0 else "#087f5b" for value in annual["Growth"]]
        axis.bar(annual.index, annual["Growth"], color=growth_colors, width=0.68)
        axis.axhline(0, color="#9eada4", linewidth=0.8)
        axis.set_ylabel("Growth (%)", color="#53645c", fontsize=9)
        axis.set_xticks(annual.index)
        format_chart(figure)

with forecast_tab:
    st.subheader("Annual import trend")
    st.caption(
        "A simple linear trend fitted to annual import totals for the selected sectors and quarters. "
        "This is an illustrative extrapolation, not an official economic forecast."
    )
    horizon = st.slider("Years to project", min_value=1, max_value=5, value=2)
    annual_imports = filtered.groupby("Year")["Imports_Crore"].sum().sort_index()
    if len(annual_imports) < 2:
        st.info("Select at least two years to fit an import trend.")
    else:
        historical_years = annual_imports.index.to_numpy(dtype=float)
        projected_years = np.arange(int(historical_years[-1]) + 1,
                                    int(historical_years[-1]) + horizon + 1)
        coefficients = np.polyfit(historical_years, annual_imports.to_numpy(dtype=float), 1)
        projected_imports = np.polyval(coefficients, projected_years)

        figure, axis = plt.subplots(figsize=(10, 4.2))
        style_axis(axis)
        axis.plot(historical_years, annual_imports.values / 1_000_000, color="#087f5b",
                  marker="o", linewidth=2.4, label="Historical imports")
        axis.plot(projected_years, projected_imports / 1_000_000, color="#e07a32",
                  marker="o", linewidth=2.2, linestyle="--", label="Trend projection")
        axis.axvline(historical_years[-1], color="#cbd7cf", linestyle=":", linewidth=1.2)
        axis.set_ylabel("Imports (₹ lakh crore)", color="#53645c", fontsize=9)
        axis.legend(frameon=False)
        format_chart(figure)

        forecast_table = pd.DataFrame({
            "Year": projected_years.astype(int),
            "Projected imports (crore)": np.maximum(projected_imports, 0).round(0).astype(int),
        })
        st.dataframe(forecast_table, hide_index=True, use_container_width=True)

with data_tab:
    st.subheader("Filtered observations")
    st.caption(f"{len(filtered):,} quarterly-sector records")
    st.dataframe(
        filtered.sort_values(["Year", "Quarter", "Sector"]).reset_index(drop=True),
        hide_index=True,
        use_container_width=True,
    )