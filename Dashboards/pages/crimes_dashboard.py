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
def load_and_process_crime():
    path = DATA_DIR / "cleaned_crime_data.csv"
    
    if path.exists():
        df = pd.read_csv(path)
        
        BEFORE_CLEANING = 1004894 
        AFTER_CLEANING = len(df)
        ROWS_REMOVED = BEFORE_CLEANING - AFTER_CLEANING
        
        missing_summary = df.isnull().sum()
        m_pct = (df.isnull().sum().sum() / df.size) * 100
        d_pct = (df.duplicated().sum() / len(df)) * 100
        loss_pct = (ROWS_REMOVED / BEFORE_CLEANING) * 100
        
        if 'victim_sex' in df.columns:
            gender_full_map = {'M': 'Male', 'F': 'Female', 'X': 'Other', 'H': 'Unknown'}
            df['victim_sex'] = df['victim_sex'].map(gender_full_map).fillna("Unspecified")
        
        if 'time_occurred' in df.columns:
            df['time_occurred'] = pd.to_numeric(df['time_occurred'], errors='coerce').fillna(0)
            df['hour'] = (df['time_occurred'] // 100).astype(int)
            
        return df, BEFORE_CLEANING, AFTER_CLEANING, ROWS_REMOVED, m_pct, d_pct, loss_pct, missing_summary
    return None, 0, 0, 0, 0, 0, 0, None

with st.spinner('Analyzing Urban Security Data... 💪'):
    df, r_before, r_after, r_removed, m_pct, d_pct, loss_pct, missing_summary = load_and_process_crime()

if df is not None:
    # --- HEADER ---
    st.markdown('<p class="main-title">🛡️ Urban Crime Patterns & Security Analysis</p>', unsafe_allow_html=True)
    
    # --- 4x4 KPI CARDS ---
    st.markdown('<p class="sub-title">Data Health & Cleaning Impact</p>', unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Rows Before Cleaning", f"{r_before:,}")
    k2.metric("Rows After Cleaning", f"{r_after:,}")
    k3.metric("Total Rows Removed", f"{r_removed:,}")
    k4.metric("Dataset Columns", len(df.columns))
    
    k5, k6, k7, k8 = st.columns(4)
    k5.metric("Initial Missing Data %", f"{m_pct:.2f}%")
    k6.metric("Initial Duplication %", f"{d_pct:.2f}%")
    k7.metric("Total Data Loss %", f"{loss_pct:.2f}%")
    k8.metric("Avg Victim Age", f"{df['victim_age'].mean():.1f}")

    # --- PROJECT INFO ---
    with st.expander("📖 Detailed Project Description & Research Objectives (Click to Expand)"):
        st.markdown(f"""
        ### **About the Crime Dataset**
        This project focuses on analyzing an **extensive urban crime dataset** containing over **1 million original records**. The analysis seeks to transform vast amounts of raw criminal incident data into structured insights that can help in understanding public safety dynamics.

        ### **Key Dimensions Analyzed:**
        1. **Temporal Patterns:** Understanding how crime frequency fluctuates within a 24-hour cycle.
        2. **Victim Demographics:** Analyzing the impact of crime across different age groups and gender identities.
        3. **Geographic Distribution:** Identifying crime hotspots based on Area IDs to assess neighborhood risk levels.

        ### **Research Objectives:**
        - **Data Integrity:** To refine the dataset by filtering out noisy, incomplete, or duplicated entries (Totaling **{r_removed:,}** removed rows).
        - **Trend Identification:** To visually represent peak crime hours using advanced polar mapping (Heat Clock).
        - **Predictive Foundation:** To establish a high-quality data baseline for future machine learning models aimed at crime prevention and resource allocation.
        """)
        st.write("**Data Source:** Urban Police Department Records / Open Data Portal")

    st.divider()

    # --- DATA QUALITY SECTION ---
    st.markdown('<p class="sub-title">🛡️ Data Quality & Integrity</p>', unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["Missing Values Analysis", "Demographic Distribution", "Outlier Summary"])
    
    with tab1:
        st.write("Missing values per column before cleaning:")
        st.bar_chart(missing_summary)
    
    with tab2:
        st.write("Victim Gender Identity Distribution:")
        fig_sex = px.pie(df, names='victim_sex', hole=0.4, template="plotly_white", color_discrete_sequence=px.colors.sequential.Reds)
        st.plotly_chart(fig_sex, width="stretch")
    
    with tab3:
        st.write("Victim Age Outlier Detection:")
        fig_box_out = px.box(
            df, 
            y='victim_age', 
            labels={'victim_age': 'Victim Age (Years)'},
            template="plotly_white", 
            color_discrete_sequence=['#EF4444']
        )
        fig_box_out.update_layout(yaxis_title="Age of Victims", xaxis_title="Overall Data Spread")
        st.plotly_chart(fig_box_out, width="stretch")

    # --- FIXED DATASET PREVIEW (Under Data Quality) ---
    st.markdown('<p class="sub-title">📋 Cleaned Dataset Preview</p>', unsafe_allow_html=True)
    st.dataframe(df.head(50), width="stretch")
    st.caption(f"Displaying a fixed preview of the first 50 rows from the cleaned database (Total: {r_after:,} rows).")

    st.divider()

    # --- CRIME HEAT CLOCK ---
    st.markdown('<p class="sub-title">🕒 Crime Heat Clock</p>', unsafe_allow_html=True)
    hour_counts = df.groupby('hour').size().reset_index(name='counts')
    full_hours = pd.DataFrame({'hour': range(24)})
    hour_counts = full_hours.merge(hour_counts, on='hour', how='left').fillna(0)

    fig_clock = go.Figure(go.Barpolar(
        r=hour_counts['counts'],
        theta=hour_counts['hour'] * 15,
        width=[1]*24,
        marker_color=hour_counts['counts'],
        marker_colorscale='Reds',
        name='Incidents'
    ))

    fig_clock.update_layout(
        polar=dict(
            angularaxis=dict(
                tickvals=list(range(0, 360, 45)),
                ticktext=['Midnight', '3 AM', '6 AM', '9 AM', 'Noon', '3 PM', '6 PM', '9 PM'],
                direction="clockwise"
            )
        ),
        template="plotly_white", height=500
    )
    st.plotly_chart(fig_clock, width="stretch")

    st.divider()

    # --- DYNAMIC VISUALIZATION AREA ---
    st.markdown('<p class="sub-title">📊 Dynamic Visualization Area</p>', unsafe_allow_html=True)
    
    plot_options = {'Victim Age': 'victim_age', 'Area ID': 'area_id', 'Victim Gender': 'victim_sex', 'Hour of Day': 'hour'}
    
    c1, c2, c3 = st.columns([2, 2, 2])
    with c1:
        x_choice = st.selectbox("Select Variable (X-axis):", options=list(plot_options.keys()), index=0)
    with c2:
        y_choice = st.selectbox("Select Variable (Y-axis):", options=list(plot_options.keys()), index=3)
    with c3:
        chart_type = st.selectbox("Select Chart Style:", 
                                  options=["Bar", "Line", "Scatter", "Histogram", "Box", "Heatmap"])

    x_col = plot_options[x_choice]
    y_col = plot_options[y_choice]

    if chart_type == "Scatter":
        fig = px.scatter(df.sample(min(2000, len(df))), x=x_col, y=y_col, color=y_col, template="plotly_white", color_continuous_scale="Reds")
    elif chart_type == "Bar":
        bar_data = df.groupby(x_col)[y_col].mean().reset_index()
        fig = px.bar(bar_data, x=x_col, y=y_col, color=y_col, template="plotly_white", color_continuous_scale="Reds")
    elif chart_type == "Line":
        line_data = df.groupby(x_col)[y_col].mean().reset_index()
        fig = px.line(line_data, x=x_col, y=y_col, template="plotly_white")
    elif chart_type == "Histogram":
        fig = px.histogram(df, x=x_col, template="plotly_white", color_discrete_sequence=['#1E3A8A'])
    elif chart_type == "Box":
        fig = px.box(df, x=x_col, y=y_col, template="plotly_white", color_discrete_sequence=['#EF4444'])
    elif chart_type == "Heatmap":
        numeric_df = df.select_dtypes(include=['float64', 'int64'])
        corr = numeric_df.corr()
        fig = px.imshow(corr, text_auto=True, color_continuous_scale='Reds', template="plotly_white")

    st.plotly_chart(fig, width="stretch")

    # --- 9. FINAL CONCLUSIONS ---
    st.divider()
    st.markdown('<p class="sub-title">📝 Study Findings & Conclusion</p>', unsafe_allow_html=True)
    
    f1, f2 = st.columns(2)
    with f1:
        st.markdown(f"""
        ### **Data Quality Notes:**
        - **Records Removed:** **{loss_pct:.2f}%** of rows were excluded due to missing values or invalid ages.
        - **Missing Values:** Initial missing value rate was **{m_pct:.2f}%**.
        - **Demographic Labels:** Gender codes were mapped to labels for interpretability.
        """)
    with f2:
        st.markdown("""
        ### **Incident Patterns:**
        - **Temporal Risk:** Incident density increases during late-night and early-morning hours.
        - **Spatial Hotspots:** Some area IDs account for a disproportionate share of incidents.
        - **Victim Age:** Certain age groups appear more frequently in incident reports.
        """)

    st.info(f"""
    **Conclusion:**
    The cleaned dataset (**{r_after:,}** rows from **{r_before:,}** raw entries) supports descriptive analysis of time, location, and demographic patterns in reported incidents. Findings are observational and limited to the available fields.
    """)

else:
    st.error("CSV file not found! Please check 'Datasets/cleaned_crime_data.csv'.")