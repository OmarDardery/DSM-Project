import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from pathlib import Path

# --- 1. PAGE CONFIG ---
# Page config is set by the main dashboard app.

# --- 2. CUSTOM CSS ---
st.markdown("""
    <style>
    .main-title { font-size: 45px !important; font-weight: 700; color: #1E3A8A; text-align: center; }
    .sub-title { font-size: 28px !important; font-weight: 600; color: #2563EB; margin-top: 20px; }
    .metric-label { font-size: 20px !important; font-weight: 700; color: #1E3A8A; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. DATA LOADING & CLEANING ---
@st.cache_data
def load_crime_data():
    project_root = Path(__file__).resolve().parents[2]
    path = project_root / "Datasets" / "cleaned_crime_data.csv"

    if path.exists():
        df = pd.read_csv(path)
        initial = 1_004_894  # Raw row count before cleaning (from report Section 2.1)

        # Ensure numeric types to prevent calculation errors
        if 'time_occurred' in df.columns:
            df['time_occurred'] = pd.to_numeric(df['time_occurred'], errors='coerce').fillna(0)
            df['hour'] = (df['time_occurred'] // 100).astype(int)

        # Parse dates and derive temporal features
        if 'date_occurred' in df.columns:
            df['date_occurred'] = pd.to_datetime(df['date_occurred'], errors='coerce')
            df['month'] = df['date_occurred'].dt.month
            df['day_of_week'] = df['date_occurred'].dt.day_name()

        # Time period bins (from report Section 2.4)
        if 'hour' in df.columns:
            def time_bucket(h):
                if h < 6:   return 'Night (00–05)'
                if h < 12:  return 'Morning (06–11)'
                if h < 17:  return 'Afternoon (12–16)'
                return 'Evening (17–23)'
            df['time_period'] = df['hour'].apply(time_bucket)

        cleaned = len(df)
        return df, initial, cleaned

    return None, 1_004_894, 0


with st.spinner('⏳ Loading crime data — 424k+ records, hang tight...'):
    df, initial_count, cleaned_count = load_crime_data()


if df is not None:

    # --- HEADER ---
    st.markdown('<p class="main-title">🛡️ Urban Crime Patterns & Security Analysis</p>', unsafe_allow_html=True)

    # --- 4. TOP METRICS ---
    m1, m2, m3 = st.columns(3)
    m1.metric("Original Rows",  f"{initial_count:,}")
    m2.metric("Cleaned Rows",   f"{cleaned_count:,}")
    m3.metric("Rows Dropped",   f"{initial_count - cleaned_count:,}")

    # --- 5. DATA HEALTH & CLEANING IMPACT ---
    st.markdown('<p class="sub-title">🧹 Data Health & Cleaning Impact</p>', unsafe_allow_html=True)

    # Cleaning steps with before/after row counts (from report Section 2.3)
    cleaning_steps = [
        ("Raw Dataset (2020–2024)",          1_004_894, "Starting point — full LAPD dataset"),
        ("After filtering to 2022–2024",       595_171, "Excludes COVID-era anomalies (2020–2021)"),
        ("After dropping irrelevant columns",  595_171, "Removed Crm Cd 2/3/4, Cross Street, Mocodes"),
        ("After removing age ≤ 0",             570_000, "Data-entry errors — victims must have a positive age"),
        ("After valid gender filter (F/M/X)",  555_000, "Unintelligible gender codes removed"),
        ("After date logic check",             424_634, "Dropped rows where date_occurred > date_reported"),
    ]
    steps_df = pd.DataFrame(cleaning_steps, columns=["Cleaning Step", "Rows Remaining", "Justification"])

    fig_funnel = px.funnel(
        steps_df,
        x="Rows Remaining",
        y="Cleaning Step",
        color_discrete_sequence=["#1E3A8A"],
        template="plotly_white",
    )
    fig_funnel.update_layout(height=400, margin=dict(l=20, r=20, t=20, b=20), showlegend=False)
    st.plotly_chart(fig_funnel, use_container_width=True)

    # Cleaning summary table
    st.markdown("**Step-by-step justification:**")
    st.dataframe(
        steps_df[["Cleaning Step", "Rows Remaining", "Justification"]],
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        f"The cleaning pipeline reduced {initial_count:,} raw records to {cleaned_count:,} clean rows "
        f"({initial_count - cleaned_count:,} dropped — {(initial_count - cleaned_count) / initial_count * 100:.1f}% of raw data). "
        "Missing values in optional fields (weapon, ethnicity, premise) were filled with informative defaults "
        "rather than dropped, preserving all 424k rows in the final clean set."
    )

    st.divider()

    # --- 6. PROJECT OVERVIEW ---
    st.markdown('<p class="sub-title">Project Overview</p>', unsafe_allow_html=True)

    st.info(f"""
    This project analyses an **extensive LAPD crime dataset (2022–2024)** sourced from the Los Angeles Open Data Portal,
    covering incidents across all 21 policing divisions. After removing **{initial_count - cleaned_count:,}** invalid
    or irrelevant records, we work with **{cleaned_count:,} clean incidents** to uncover patterns that impact public safety.

    Two predictive models were built: **(1)** a **Random Forest classifier** predicting whether a crime results in an arrest
    — achieving **72.82% CV accuracy and AUC 0.730** — and **(2)** a **multi-class holiday classifier** predicting crime type,
    reaching **23.71% accuracy** versus a 16.7% random baseline (1.42× better than chance).

    Key findings confirm that **time of occurrence** and **crime type** are the dominant predictors of arrest likelihood,
    while victim gender carries near-zero predictive signal and can be safely excluded from the model.
    """)

    # --- 7. DETAILED PROJECT DESCRIPTION (EXPANDABLE) ---
    with st.expander("📖 Detailed Project Description & Research Objectives — Click to Expand"):

        st.markdown("#### 🎯 Research Objectives")
        st.markdown("""
        1. **Data Engineering** — Clean and reduce 1,004,894 raw LAPD records to a reliable 424,634-row dataset
           in a dedicated cleaning notebook, mirroring professional data science workflows.
        2. **Exploratory Data Analysis (EDA)** — Understand distributions, class balance, temporal trends,
           and geographic crime density across all 21 LAPD policing divisions.
        3. **Arrest Prediction (Primary Task)** — Build and evaluate a regularised Random Forest classifier
           to predict whether a crime incident results in an arrest, with Decision Tree and Logistic Regression
           trained purely as comparison baselines.
        4. **Rigorous Validation** — Apply stratified 5-fold cross-validation, learning curve analysis,
           confusion matrices, and ROC curve analysis for a full performance picture.
        5. **Crime Type Classification** — Build a multi-class holiday-aware classifier to predict
           crime group from contextual features and diagnose its structural limitations through root cause analysis.
        6. **Real-World Interpretation** — Translate all results into actionable recommendations for
           patrol scheduling and resource allocation.
        """)

        st.markdown("#### 📂 Dataset Description")
        st.markdown("""
        **Source:** LAPD Crime Data 2020–2024, Los Angeles Open Data Portal (`data.lacity.org`)
        **Scope after cleaning:** 424,634 incidents across 21 LAPD divisions, 2022–2024
        **Excluded period:** 2020–2021 deliberately excluded to avoid COVID-19 pandemic reporting distortions.
        """)

        dataset_cols = [
            ("incident_id",         "Integer",  "Unique DR Number — primary key only, not a model feature"),
            ("date_occurred",       "Date",     "Actual crime date — used for holiday tagging & temporal features"),
            ("time_occurred",       "Int HHMM", "Time of crime (0–2359) — the single most important model feature"),
            ("area_id / location",  "Int/Str",  "LAPD division ID and name (e.g. Hollywood, Newton)"),
            ("crime_code",          "Integer",  "Specific crime code — used to engineer crime_group"),
            ("crime_description",   "String",   "Text description — reference only, not fed to models"),
            ("victim_age",          "Integer",  "Victim age — invalid values (≤0) removed in cleaning"),
            ("victim_sex",          "String",   "F / M / X — near-zero importance in arrest model"),
            ("victim_ethnicity",    "String",   "Ethnic descent code — missing filled with 'unknown'"),
            ("premise_description", "String",   "Crime location type — missing filled with 'Unknown'"),
            ("weapon_description",  "String",   "Weapon used — missing filled with 'NO WEAPON'"),
            ("status_outcome",      "String",   "Investigative status — source for the arrest_made target"),
            ("latitude / longitude","Float",    "Block-level coordinates — used in geographic analysis"),
            ("arrest_made",         "Yes/No",   "Engineered target: 'Yes' if status_outcome contains 'Arrest'"),
            ("crime_group",         "6 classes","Engineered: crime_code bucketed into 6 meaningful categories"),
        ]
        cols_df = pd.DataFrame(dataset_cols, columns=["Column", "Type", "Role & Relevance"])
        st.dataframe(cols_df, use_container_width=True, hide_index=True)

        st.markdown("#### ⚙️ Feature Engineering")
        st.markdown("""
        | Feature | Description |
        |---|---|
        | `crime_group` | crime_code bucketed into 6 categories: Homicide & Sexual Assault, Robbery & Aggravated Assault, Burglary, Theft, Simple Assault & Battery, Vandalism/Fraud/Other |
        | `arrest_made` | Binary Yes/No derived from status_outcome — the primary classification target |
        | `time_period` | Hour binned into Night (00–05), Morning (06–11), Afternoon (12–16), Evening (17–23) |
        | `is_holiday` | Binary 0/1 flag for US Federal Holidays + Halloween, New Year's Eve, Valentine's Day, St. Patrick's Day |
        | `holiday_name` | Named holiday or 'None' for non-holiday dates |
        """)

        st.markdown("#### 📐 Modelling Strategy")
        st.markdown("""
        **Data split:** 70% Train (297,243 rows) / 15% Validation (63,695) / 15% Test (63,696) — all stratified on `arrest_made`.
        A dedicated validation set absorbs all model-selection bias so the test set remains a genuinely unseen final estimate.

        | Model | Role | Key Parameters |
        |---|---|---|
        | Random Forest | **Primary** | 100 trees, max_depth=20, min_samples_leaf=10, class_weight='balanced' |
        | Decision Tree | Comparison baseline | max_depth=8, class_weight='balanced' |
        | Logistic Regression | Linear baseline | class_weight='balanced', max_iter=1000 |

        All models use `class_weight='balanced'` because only 8.4% of incidents result in an arrest —
        without balancing, any model achieves 91.6% accuracy by always predicting "No arrest", learning nothing.
        """)

    st.divider()

    # --- 6. CRIME HEAT CLOCK ---
    st.markdown('<p class="sub-title">🕒 Crime Heat Clock — Incidents by Hour of Day</p>', unsafe_allow_html=True)

    hour_counts = df.groupby('hour').size().reset_index(name='counts')
    full_hours  = pd.DataFrame({'hour': range(24)})
    hour_counts = full_hours.merge(hour_counts, on='hour', how='left').fillna(0)

    fig_clock = go.Figure(go.Barpolar(
        r=hour_counts['counts'],
        theta=hour_counts['hour'] * 15,   # 360° / 24 hours = 15° per hour
        width=[14.5] * 24,
        marker_color=hour_counts['counts'],
        marker_colorscale='Reds',
        marker_line_width=0.5,
        marker_line_color='white',
        name='Incidents',
        hovertemplate="Hour %{customdata}:00<br>Incidents: %{r:,.0f}<extra></extra>",
        customdata=hour_counts['hour'],
    ))
    fig_clock.update_layout(
        polar=dict(
            angularaxis=dict(
                tickvals=list(range(0, 360, 45)),
                ticktext=['Midnight', '3 AM', '6 AM', '9 AM', 'Noon', '3 PM', '6 PM', '9 PM'],
                direction="clockwise",
                rotation=90,
            ),
        ),
        template="plotly_white",
        height=500,
        margin=dict(l=50, r=50, t=20, b=20),
    )
    st.plotly_chart(fig_clock, use_container_width=True)
    st.caption(
        "Crime peaks sharply in the evening (6–11 PM) with a secondary spike around noon. "
        "Time of occurrence is the strongest single predictor in the arrest model (feature importance ≈ 0.33). "
        "Early morning hours (2–5 AM) show the lowest incident counts across all divisions."
    )

    st.divider()

    # --- 7. RANKED CRIME GROUPS BAR ---
    st.markdown('<p class="sub-title">📊 Crime Group Distribution (Ranked)</p>', unsafe_allow_html=True)

    if 'crime_group' in df.columns:
        cg_data = df['crime_group'].value_counts().reset_index()
        cg_data.columns = ['Crime Group', 'Incident Count']
        cg_data = cg_data.sort_values('Incident Count', ascending=True)
    else:
        # Fallback to report constants (Section 3.1.3)
        cg_data = pd.DataFrame({
            'Crime Group': [
                'Homicide & Sexual Assault', 'Robbery & Aggravated Assault',
                'Theft (Personal & Vehicle)', 'Simple Assault & Battery',
                'Vandalism, Fraud & Other', 'Burglary'
            ],
            'Incident Count': [2_881, 37_335, 65_000, 78_859, 90_092, 149_458]
        })

    fig_cg = px.bar(
        cg_data,
        x='Incident Count',
        y='Crime Group',
        orientation='h',
        text_auto=',d',
        color='Incident Count',
        color_continuous_scale='Reds',
        template='plotly_white',
    )
    fig_cg.update_layout(showlegend=False, height=400, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig_cg, use_container_width=True)
    st.caption(
        "Burglary dominates with 149k+ incidents — 51.8× more frequent than Homicide & Sexual Assault (2,881). "
        "This chart identifies which crime categories are most characteristic of LA, and the severe 52:1 imbalance "
        "is the primary structural challenge for the multi-class crime type classifier."
    )

    st.divider()

    # --- 8. INTERACTIVE DATA DISCOVERY ---
    # Section for user-driven data discovery and custom chart building
    st.markdown('<p class="sub-title">📊 Advanced Interactive Data Explorer</p>', unsafe_allow_html=True)
    st.markdown("Customize your analysis by mixing variables, chart types, and color groupings from the cleaned dataset.")

    # Categorize columns by data type for better selection control
    num_cols_dv = df.select_dtypes(include='number').columns.tolist()
    cat_cols_dv = df.select_dtypes(include='object').columns.tolist()
    all_cols_dv = num_cols_dv + cat_cols_dv

    # Mapping technical database column names to user-friendly display labels
    friendly_dv = {
        'victim_age': 'Victim Age', 'hour': 'Hour of Day', 'area_id': 'Area ID',
        'time_occurred': 'Time Occurred', 'month': 'Month',
        'victim_sex': 'Victim Gender', 'crime_group': 'Crime Group',
        'time_period': 'Time Period', 'arrest_made': 'Arrest Made',
        'location': 'Division', 'premise_description': 'Premise',
        'weapon_description': 'Weapon', 'victim_ethnicity': 'Ethnicity',
        'day_of_week': 'Day of Week',
    }
    
    # Filter available columns and create a list for the dropdown menus
    avail_dv = {friendly_dv.get(c, c): c for c in all_cols_dv if c in df.columns}
    avail_dv_list = sorted(list(avail_dv.keys()))

    # UI Layout: Create 4 columns for interactive widgets/controls
    dv_c1, dv_c2, dv_c3, dv_c4 = st.columns([2, 2, 2, 1.5])

    with dv_c1:
        # Dropdown for X-Axis selection
        dv_x = st.selectbox("🎯 X-Axis (Variable):", avail_dv_list, 
                            index=avail_dv_list.index('Victim Age') if 'Victim Age' in avail_dv_list else 0)
    with dv_c2:
        # Dropdown for Y-Axis selection
        dv_y = st.selectbox("🎯 Y-Axis (Variable):", avail_dv_list, 
                            index=avail_dv_list.index('Hour of Day') if 'Hour of Day' in avail_dv_list else 1)
    with dv_c3:
        # Dropdown for optional color grouping (Legend)
        color_options = ["None"] + avail_dv_list
        dv_color = st.selectbox("🎨 Colour By:", color_options, index=0)
    with dv_c4:
        # Radio buttons to choose the preferred visualization style
        dv_chart = st.radio("📈 Chart Type:", ["Bar", "Scatter", "Histogram", "Box", "Line"])

    # Retrieve actual dataframe column names based on user friendly-selection
    x_raw = avail_dv[dv_x]
    y_raw = avail_dv[dv_y]
    color_raw = avail_dv[dv_color] if dv_color != "None" else None

    # Chart Generation Logic based on user choice
    if dv_chart == "Scatter":
        # Random sampling to optimize browser performance (prevents lag with large datasets)
        sample_df = df.sample(min(5000, len(df)))
        fig_dv = px.scatter(sample_df, x=x_raw, y=y_raw, color=color_raw,
                            labels={x_raw: dv_x, y_raw: dv_y},
                            template="plotly_white", color_continuous_scale="Reds",
                            title=f"{dv_x} vs {dv_y} (Sampled 5k points)")
    
    elif dv_chart == "Histogram":
        # Visualizing frequency distribution of a single variable
        fig_dv = px.histogram(df, x=x_raw, color=color_raw,
                              labels={x_raw: dv_x},
                              template="plotly_white", color_discrete_sequence=['#1E3A8A'],
                              title=f"Distribution of {dv_x}")
    
    elif dv_chart == "Box":
        # Box plot for statistical distribution and outlier detection
        fig_dv = px.box(df.sample(min(10000, len(df))), x=x_raw, y=y_raw, color=color_raw,
                        template="plotly_white", title=f"{dv_y} by {dv_x}")
    
    elif dv_chart == "Line":
        # Line chart showing volume trends based on the selected X-axis
        line_data = df.groupby(x_raw).size().reset_index(name="Count")
        fig_dv = px.line(line_data, x=x_raw, y="Count", markers=True,
                         template="plotly_white", title=f"Incident Count over {dv_x}")
    
    else:  # Bar Chart (Default)
        # Aggregating data to show total counts per category
        bar_agg = df.groupby(x_raw).size().reset_index(name="Count")
        fig_dv = px.bar(bar_agg, x=x_raw, y="Count", 
                        color="Count" if color_raw is None else None,
                        color_continuous_scale="Reds",
                        template="plotly_white", title=f"Incident Count by {dv_x}")

    # Display the final generated chart with dynamic width adjustment
    st.plotly_chart(fig_dv, use_container_width=True)
    st.caption(f"Visualizing data from {len(df):,} cleaned records.")

    # --- 9. MODEL RESULTS SUMMARY ---
    st.divider()
    st.markdown('<p class="sub-title">🤖 Predictive Model Results</p>', unsafe_allow_html=True)

    # 3-model comparison bar chart (from report Section 3.8, Table 4)
    models     = ["Random Forest\n(Primary)", "Decision Tree\n(Comparison)", "Logistic Regression\n(Comparison)"]
    cv_scores  = [72.82, 60.37, 53.34]
    val_scores = [72.13, 60.39, 53.34]
    tst_scores = [72.08, 59.86, 52.94]

    fig_models = go.Figure()
    for label, vals, color in [
        ("CV Accuracy",         cv_scores,  "#1E3A8A"),
        ("Validation Accuracy", val_scores, "#2563EB"),
        ("Test Accuracy",       tst_scores, "#60A5FA"),
    ]:
        fig_models.add_trace(go.Bar(
            name=label, x=models, y=vals,
            marker_color=color,
            text=[f"{v:.2f}%" for v in vals],
            textposition='outside',
        ))
    fig_models.update_layout(
        barmode='group',
        template="plotly_white",
        height=420,
        yaxis=dict(range=[40, 82], title="Accuracy (%)"),
        legend=dict(orientation="h", y=1.08),
        margin=dict(l=20, r=20, t=20, b=20),
    )
    st.plotly_chart(fig_models, use_container_width=True)
    st.caption(
        "Logistic Regression at 53.3% barely exceeds random chance — proving the arrest–feature relationship "
        "is non-linear and directly validating Random Forest as the primary model. The ~12pp gap between "
        "Random Forest and Decision Tree quantifies the value of ensembling 100 trees. All models stay within "
        "<1pp across CV, Validation, and Test sets, confirming there is no data leakage."
    )

    st.divider()

        # --- 9.5 HOLIDAY CRIME TYPE CLASSIFIER ANALYSIS ---
    st.divider()
    st.markdown('<p class="sub-title">🎄 Why the Holiday Crime Type Classifier Performed Poorly</p>', unsafe_allow_html=True)

    st.warning("""
    The **Holiday Crime Type Classifier** attempted to predict which of the 6 crime groups would occur
    using contextual features such as:
    
    - Time of occurrence
    - Victim age
    - Victim gender
    - Holiday indicators (`holiday_name`, `is_holiday`)
    
    Unlike the arrest prediction model, this classifier achieved only **23.71% accuracy**
    (compared to a random baseline of 16.7%).
    """)

    # Root cause metrics
    root_causes = pd.DataFrame({
        "Problem": [
            "Extreme Class Imbalance",
            "Sparse Holiday Data",
            "Weak Feature Signal"
        ],
        "Severity Score": [51.8, 5, 1]
    })

    fig_root = px.bar(
        root_causes,
        x="Problem",
        y="Severity Score",
        color="Severity Score",
        color_continuous_scale="Reds",
        template="plotly_white",
        text_auto=".1f",
        title="Root Causes Behind Poor Holiday Classifier Performance"
    )

    fig_root.update_layout(
        height=420,
        margin=dict(l=20, r=20, t=50, b=20),
        coloraxis_showscale=False
    )

    st.plotly_chart(fig_root, use_container_width=True)

    # Detailed explanation
    exp1, exp2, exp3 = st.columns(3)

    with exp1:
        st.error("""
        ### ① Extreme Class Imbalance
        
        Burglary had over **149k incidents** while
        Homicide & Sexual Assault had only **2.8k**.
        
        This creates a **51.8 : 1 imbalance ratio**.
        
        The model therefore learns the majority
        classes far more effectively than the rare ones.
        """)

    with exp2:
        st.warning("""
        ### ② Sparse Holiday Data
        
        Most rows were NOT holidays.
        
        Holiday-related incidents represented
        only a tiny fraction of the dataset.
        
        Many holiday/crime combinations appeared
        only a few dozen times — far too little
        for stable learning.
        """)

    with exp3:
        st.info("""
        ### ③ Weak Predictive Features
        
        The selected features simply do not contain
        enough information to accurately determine
        crime category.
        
        Missing high-signal features include:
        
        - Weapon type
        - Premise/location type
        - Area/division
        - Suspect information
        """)

    st.markdown("### 📉 Per-Class Performance")

    holiday_results = pd.DataFrame({
        "Crime Group": [
            "Burglary",
            "Homicide & Sexual Assault",
            "Robbery & Agg. Assault",
            "Simple Assault & Battery",
            "Theft",
            "Vandalism / Fraud / Other"
        ],
        "F1-Score": [0.34, 0.04, 0.21, 0.22, 0.19, 0.21]
    })

    fig_f1 = px.bar(
        holiday_results,
        x="Crime Group",
        y="F1-Score",
        color="F1-Score",
        color_continuous_scale="Blues",
        template="plotly_white",
        text_auto=".2f",
        title="Holiday Classifier F1-Score by Crime Group"
    )

    fig_f1.update_layout(
        height=420,
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis_tickangle=-15,
        coloraxis_showscale=False
    )

    st.plotly_chart(fig_f1, use_container_width=True)

    st.caption(
        "The classifier performed inconsistently across crime categories. "
        "Rare classes such as Homicide & Sexual Assault achieved extremely low precision and F1-scores "
        "because the model generated many false positives. "
        "Even the best-performing class (Burglary) achieved only 0.34 F1-score, confirming that "
        "holiday information and basic demographics alone are insufficient for reliable crime type prediction."
    )

    st.success("""
    ### Final Interpretation
    
    Despite the low accuracy, the model was still valuable analytically.
    
    It proved that:
    
    - Crime type prediction is significantly harder than arrest prediction.
    - Holiday-related features carry almost no predictive power.
    - Temporal and demographic information alone cannot reliably distinguish crime categories.
    
    Most importantly, the experiment identified WHICH missing features are likely necessary
    for future improvement — especially weapon type, premise type, and geographic information.
    """)

    # --- 10. FEATURE IMPORTANCE ---
    st.markdown('<p class="sub-title">🔑 What Drives Arrest Prediction? (Feature Importance)</p>', unsafe_allow_html=True)

    # From report Section 3.7
    fi_data = {
        'Feature': [
            'time_occurred',
            'crime_group: Robbery & Agg. Assault',
            'victim_age',
            'crime_group: Vandalism / Other',
            'crime_group: Simple Assault',
            'crime_group: Theft',
            'crime_group: Homicide',
            'victim_sex (M)',
            'victim_sex (X)',
        ],
        'Importance': [0.33, 0.22, 0.18, 0.09, 0.08, 0.06, 0.03, 0.01, 0.00]
    }
    fi_df = pd.DataFrame(fi_data).sort_values('Importance', ascending=True)

    fig_fi = px.bar(
        fi_df,
        x='Importance',
        y='Feature',
        orientation='h',
        text_auto='.2f',
        color='Importance',
        color_continuous_scale='Blues',
        template='plotly_white',
    )
    fig_fi.update_layout(showlegend=False, height=400, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig_fi, use_container_width=True)
    st.caption(
        "time_occurred dominates (0.33) but regularisation reduced its grip from 0.54 in the initial model — "
        "a sign of a healthier, more balanced learner. Robbery & Aggravated Assault is second (0.22) "
        "because it has significantly higher arrest rates than property crimes. "
        "victim_sex carries near-zero importance and can be safely removed with negligible accuracy impact."
    )

    # --- 12. CLEANED DATA PREVIEW ---
    st.divider()
    st.markdown('<p class="sub-title">🗂️ Cleaned Data Preview</p>', unsafe_allow_html=True)

    # Column selector
    all_cols = df.columns.tolist()
    friendly_col_map = {
        'incident_id': 'Incident ID', 'date_reported': 'Date Reported',
        'date_occurred': 'Date Occurred', 'time_occurred': 'Time Occurred',
        'hour': 'Hour', 'area_id': 'Area ID', 'location': 'Division',
        'crime_code': 'Crime Code', 'crime_description': 'Crime Description',
        'victim_age': 'Victim Age', 'victim_sex': 'Victim Gender',
        'victim_ethnicity': 'Victim Ethnicity', 'premise_description': 'Premise',
        'weapon_description': 'Weapon', 'status_outcome': 'Status Outcome',
        'arrest_made': 'Arrest Made', 'crime_group': 'Crime Group',
        'time_period': 'Time Period', 'latitude': 'Latitude', 'longitude': 'Longitude',
    }
    default_cols = [c for c in ['date_occurred', 'hour', 'location', 'crime_group',
                                 'victim_age', 'victim_sex', 'arrest_made'] if c in all_cols]

    prev_c1, prev_c2 = st.columns([3, 1])
    with prev_c1:
        selected_preview_cols = st.multiselect(
            "Choose columns to display:",
            options=all_cols,
            default=default_cols,
            format_func=lambda c: friendly_col_map.get(c, c),
            key="preview_cols",
        )
    with prev_c2:
        n_rows = st.selectbox("Rows to show:", [10, 25, 50, 100], index=1, key="preview_rows")

    if selected_preview_cols:
        preview_df = df[selected_preview_cols].head(n_rows).copy()
        # Rename columns to friendly names for display
        preview_df.columns = [friendly_col_map.get(c, c) for c in preview_df.columns]
        st.dataframe(preview_df, use_container_width=True, hide_index=True)
    else:
        st.info("Select at least one column above to preview the data.")

    # Summary stats toggle
    if st.checkbox("📈 Show numeric column statistics", key="show_stats"):
        num_cols = df.select_dtypes(include='number').columns.tolist()
        st.dataframe(df[num_cols].describe().round(2), use_container_width=True)

    st.caption(
        f"Showing the first {n_rows} rows of the {cleaned_count:,}-row cleaned dataset. "
        "All rows passed the full cleaning pipeline: valid age (>0), valid gender (F/M/X), "
        "logical date order (date_occurred ≤ date_reported), and filled missing optional fields."
    )


    # --- 14. CONCLUSIONS ---
    st.divider()

    col_inf1, col_inf2 = st.columns(2)
    with col_inf1:
        st.info(
            "**🔍 Statistical Significance:** Analysis confirms that crime behaviour follows a non-random "
            "distribution. Time of occurrence and crime group show significant correlations with arrest "
            "outcomes. The Random Forest achieves AUC 0.730 — meaningfully above the 0.5 random baseline "
            "and a major improvement over the unregularised model's AUC of 0.636."
        )
    with col_inf2:
        st.success(
            f"**✅ Final Conclusion:** By refining {initial_count:,} raw records down to {cleaned_count:,} "
            "clean incidents, we established a reliable analytical baseline. The regularised Random Forest "
            "(72.82% CV accuracy, 58% arrest recall) is suitable for patrol scheduling and resource allocation. "
            "Adding weapon type, premise type, and area ID as features would likely push AUC above 0.80."
        )

else:
    st.error("CSV file not found! Please ensure 'cleaned_crime_data.csv' is located in the 'Datasets' folder.")
