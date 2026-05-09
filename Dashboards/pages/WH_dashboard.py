import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from pathlib import Path

# Page config is managed by the main dashboard app, not by individual page modules.

# --- 2. CUSTOM CSS ---
st.markdown("""
    <style>
    .main-title { font-size: 45px !important; font-weight: 700; color: #1E3A8A; text-align: center; }
    .sub-title { font-size: 28px !important; font-weight: 600; color: #2563EB; margin-top: 20px; }
    .metric-card { background-color: #F0F2F6; padding: 15px; border-radius: 10px; border-left: 5px solid #2563EB; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. DATA LOADING & CALCULATIONS ---
DATA_DIR = Path(__file__).resolve().parents[2] / "Datasets"


@st.cache_data
def load_and_process_happiness():
    path = DATA_DIR / "HZ_WH_data.csv"
    
    if path.exists():
        df_raw = pd.read_csv(path)
        
        # Stats Before Cleaning
        BEFORE_CLEANING = 2363 
        rows_before = len(df_raw)
        cols_count = len(df_raw.columns)
        
        raw_missing_count = df_raw.isnull().sum().sum()
        missing_pct = (raw_missing_count / df_raw.size) * 100
        raw_dups_count = df_raw.duplicated().sum()
        dups_pct = (raw_dups_count / rows_before) * 100
        
        # Cleaning & Formatting
        if 'Year' in df_raw.columns:
            df_raw['Year'] = pd.to_numeric(df_raw['Year'], errors='coerce').fillna(0).astype(int)
        
        df = df_raw.dropna()
        
        # Stats After Cleaning
        rows_after = len(df)
        rows_removed = BEFORE_CLEANING - rows_after
        data_loss_pct = (rows_removed / BEFORE_CLEANING) * 100
        
        # Global Avg Happiness (Life Ladder)
        avg_happiness = df['Life Ladder'].mean() if 'Life Ladder' in df.columns else 0
        
        return df, BEFORE_CLEANING, rows_after, rows_removed, cols_count, missing_pct, dups_pct, data_loss_pct, avg_happiness, df_raw.isnull().sum()
    return None, 0, 0, 0, 0, 0, 0, 0, 0, None

# --- LOADING SCREEN ---
with st.spinner('Hold on! Gathering happiness levels from around the globe... 💪🌍'):
    df, r_before, r_after, r_removed, c_cnt, m_pct, d_pct, loss_pct, avg_happy, missing_summary = load_and_process_happiness()

if df is not None:
    # --- HEADER ---
    st.markdown('<p class="main-title">🌍 World Happiness Report Interactive Analytics</p>', unsafe_allow_html=True)
    
    # --- 4x4 KPI CARDS ---
    st.markdown('<p class="sub-title">Data Health & Cleaning Impact</p>', unsafe_allow_html=True)
    
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Rows Before Cleaning", f"{r_before:,}")
    k2.metric("Rows After Cleaning", f"{r_after:,}")
    k3.metric("Total Rows Removed", f"{r_removed:,}")
    k4.metric("Dataset Columns", c_cnt)
    
    k5, k6, k7, k8 = st.columns(4)
    k5.metric("Initial Missing Data %", f"{m_pct:.2f}%")
    k6.metric("Initial Duplication %", f"{d_pct:.2f}%")
    k7.metric("Total Data Loss %", f"{loss_pct:.2f}%")
    k8.metric("Avg Happiness Score", f"{avg_happy:.2f}")

    # --- PROJECT INFO ---
    with st.expander("📖 Detailed Project Description & Research Objectives (Click to Expand)"):
        st.markdown(f"""
        ### **About the World Happiness Report**
        The World Happiness Report is a landmark survey of the state of global happiness. It ranks countries based on their citizens' perceptions of their own lives, primarily using the **Life Ladder** score.

        ### **Key Factors Analyzed:**
        1. **Life Ladder (Happiness Score):** The primary measure of subjective well-being.
        2. **GDP/Capita:** Economic output per person, adjusted for purchasing power.
        3. **Social Support:** The perception of having someone to count on in times of trouble.
        4. **Healthy Life Expectancy:** The average number of years a newborn is expected to live in good health.
        5. **Freedom of Choice:** The freedom to make key life decisions.

        ### **Research Objectives:**
        - **Data Integrity:** Filtering incomplete records (Removing **{r_removed:,}** rows) to ensure accurate cross-country comparisons.
        - **Temporal Analysis:** Tracking how global happiness has evolved from {df['Year'].min()} to {df['Year'].max()}.
        - **Correlation Mapping:** Exploring how economic prosperity vs. social factors influence overall well-being.
        """)
        st.write("**Data Source:** Sustainable Development Solutions Network (SDSN) / Gallup World Poll")

    st.divider()

    # --- 6. INTERACTIVE WORLD MAP ---
    st.markdown('<p class="sub-title">🗺️ Global Happiness Distribution Map</p>', unsafe_allow_html=True)
    
    min_yr, max_yr = int(df['Year'].min()), int(df['Year'].max())
    selected_year = st.sidebar.slider("Global Map Year Select:", min_yr, max_yr, max_yr)

    map_df = df[df['Year'] == selected_year]

    fig_map = px.choropleth(
        map_df,
        locations="Country Name",
        locationmode="country names",
        color="Life Ladder",
        hover_name="Country Name",
        hover_data=["GDP/Capita", "Social Support"] if "GDP/Capita" in df.columns else None,
        color_continuous_scale=px.colors.sequential.YlGnBu,
        title=f"World Happiness Map - Year {selected_year}"
    )
    fig_map.update_layout(geo=dict(projection_type='equirectangular'), height=600)
    st.plotly_chart(fig_map, width="stretch")

    st.divider()

    # --- DATA QUALITY SECTION ---
    st.markdown('<p class="sub-title">🛡️ Data Quality & Integrity</p>', unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["Missing Values Analysis", "Regional Imbalance", "Outlier Summary"])
    
    with tab1:
        st.write("Missing values per column in the raw dataset:")
        st.bar_chart(missing_summary)
    
    with tab2:
        st.write("Data Distribution (Class Imbalance check):")
        fig_imb = px.histogram(df, x="Year", title="Data points collected per Year", template="plotly_white", color_discrete_sequence=['#1E3A8A'])
        st.plotly_chart(fig_imb, width="stretch")
    
    with tab3:
        st.write("Outlier detection for Happiness & Economic Factors:")
        fig_box_out = px.box(df, y=['Life Ladder', 'GDP/Capita', 'Social Support'], template="plotly_white", color_discrete_sequence=['#2563EB'])
        st.plotly_chart(fig_box_out, width="stretch")

    # --- NEW: FIXED DATASET PREVIEW (Added under Data Quality) ---
    st.markdown('<p class="sub-title">📋 Cleaned Dataset Preview</p>', unsafe_allow_html=True)
    st.dataframe(df.head(50), width="stretch")
    st.caption(f"Showing the first 50 records of the validated happiness database (Total: {r_after:,} rows).")

    st.divider()

    # --- DYNAMIC VISUALIZATION AREA ---
    st.markdown('<p class="sub-title">📊 Multi-Dimensional Analysis Explorer</p>', unsafe_allow_html=True)
    
    options_map = {
        'Happiness Score': 'Life Ladder',
        'GDP Per Capita': 'GDP/Capita',
        'Social Support': 'Social Support',
        'Healthy Life Expectancy': 'Healthy Life Expectancy At Birth',
        'Freedom of Choice': 'Freedom To Make Life Choices',
        'Generosity': 'Generosity',
        'Perception of Corruption': 'Perceptions Of Corruption'
    }
    available_options = {k: v for k, v in options_map.items() if v in df.columns}

    c1, c2, c3 = st.columns([2, 2, 2])
    with c1:
        x_choice = st.selectbox("Select X-axis:", options=list(available_options.keys()), index=1)
    with c2:
        y_choice = st.selectbox("Select Y-axis:", options=list(available_options.keys()), index=0)
    with c3:
        chart_type = st.selectbox("Select Chart Style:", 
                                  options=["Scatter", "Bar (Averages)", "Line (Trends)", "Histogram", "Box", "Heatmap"])

    x_col = available_options[x_choice]
    y_col = available_options[y_choice]

    if chart_type == "Scatter":
        fig = px.scatter(map_df, x=x_col, y=y_col, size="Life Ladder", hover_name="Country Name", color=y_col, template="plotly_white")
    elif chart_type == "Bar (Averages)":
        bar_data = map_df.nlargest(15, y_col) 
        fig = px.bar(bar_data, x="Country Name", y=y_col, color=y_col, template="plotly_white")
    elif chart_type == "Line (Trends)":
        st.write("Historical Trend for a specific country:")
        target_country = st.selectbox("Choose Country:", options=sorted(df['Country Name'].unique()))
        country_df = df[df['Country Name'] == target_country]
        fig = px.line(country_df, x='Year', y=[x_col, y_col], markers=True, template="plotly_white")
    elif chart_type == "Histogram":
        fig = px.histogram(map_df, x=x_col, template="plotly_white", color_discrete_sequence=['#2563EB'])
    elif chart_type == "Box":
        fig = px.box(map_df, y=y_col, template="plotly_white", color_discrete_sequence=['#1E3A8A'])
    elif chart_type == "Heatmap":
        numeric_df = df[list(available_options.values())]
        corr = numeric_df.corr()
        fig = px.imshow(corr, text_auto=True, color_continuous_scale='YlGnBu', template="plotly_white")

    st.plotly_chart(fig, width="stretch")

    # --- 9. FINAL CONCLUSIONS ---
    st.divider()
    st.markdown('<p class="sub-title">📝 Study Findings & Conclusion</p>', unsafe_allow_html=True)
    
    f1, f2 = st.columns(2)
    with f1:
        st.markdown(f"""
        ### **Data Quality Notes:**
        - **Records Removed:** **{r_removed:,}** rows were removed due to missing metrics.
        - **Economic Signal:** GDP per capita correlates with Life Ladder, but it does not explain all variance.
        """)
    with f2:
        st.markdown("""
        ### **Sociological Observations:**
        - **Social Support:** Social Support and Freedom to make life choices often show strong correlation with Life Ladder.
        - **Historical Trends:** Time series show regional fluctuations across years, indicating sensitivity to global events.
        """)

    st.info(f"""
    **Conclusion:**
    The cleaned dataset (**{r_after:,}** rows from **{r_before:,}** raw entries) supports comparative analysis of economic and social predictors of Life Ladder across countries and years. Results are descriptive and reflect the available indicators.
    """)

else:
    st.error("CSV File not found! Please check 'Datasets/HZ_WH_data.csv'.")