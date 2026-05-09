import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

# Page config is managed by the main dashboard app, not by individual page modules.

# --- 2. CUSTOM CSS ---
st.markdown("""
    <style>
    .main-title { font-size: 45px !important; font-weight: 700; color: #1E3A8A; text-align: center; }
    .sub-title { font-size: 28px !important; font-weight: 600; color: #2563EB; margin-top: 20px; }
    .metric-card { background-color: #F0F2F6; padding: 15px; border-radius: 10px; border-left: 5px solid #2563EB; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. MAPPING DICTIONARY (Short names for UI, Full for Reference) ---
# رجعنا الرموز الأصلية عشان الشكل يكون مظبوط
friendly_names = {
    'age': 'Age',
    'gender': 'Gender',
    'A_total': 'A_total',
    'O_total': 'O_total',
    'C_total': 'C_total',
    'E_total': 'E_total',
    'N_total': 'N_total'
}

# ده قاموس مرجعي بس عشان لو حد عايز يعرف الرمز معناه إيه
trait_reference = {
    'A_total': 'Agreeableness',
    'O_total': 'Openness',
    'C_total': 'Conscientiousness',
    'E_total': 'Extraversion',
    'N_total': 'Neuroticism'
}

gender_map = {1: "Male", 2: "Female", 3: "Other", 0: "Unspecified"}
reverse_names = {v: k for k, v in friendly_names.items()}

# --- 4. DATA LOADING & CLEANING ---
DATA_DIR = Path(__file__).resolve().parents[2] / "Datasets"


@st.cache_data
def load_and_clean():
    path = DATA_DIR / "cleaned_big5_totals.csv"
    
    if path.exists():
        df_raw = pd.read_csv(path)
        
        rows_before = len(df_raw)
        cols_count = len(df_raw.columns)
        raw_missing_count = df_raw.isnull().sum().sum()
        missing_pct = (raw_missing_count / df_raw.size) * 100
        raw_dups_count = df_raw.duplicated().sum()
        dups_pct = (raw_dups_count / rows_before) * 100
        
        if 'gender' in df_raw.columns:
            df_raw['gender'] = df_raw['gender'].map(gender_map).fillna("Unknown")
        
        trait_cols = ['A_total', 'O_total', 'C_total', 'E_total', 'N_total']
        all_numeric = ['age'] + trait_cols
        for col in all_numeric:
            if col in df_raw.columns:
                df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')
        
        df = df_raw.dropna()
        df = df[(df['age'] > 5) & (df['age'] < 100)] 
        
        rows_after = len(df)
        rows_removed = rows_before - rows_after
        removed_data_pct = (rows_removed / rows_before) * 100
        avg_score = df[trait_cols].mean().mean()
        
        return df, rows_before, rows_after, rows_removed, cols_count, missing_pct, dups_pct, removed_data_pct, avg_score, df_raw.isnull().sum()
    return None, 0, 0, 0, 0, 0, 0, 0, 0, None

df, r_before, r_after, r_removed, c_count, m_pct, d_pct, rem_pct, global_avg, missing_summary = load_and_clean()

if df is not None:
    # --- 5. SIDEBAR FILTERS ---
    st.sidebar.header("🔍 Global Filters")
    
    # إضافة دليل الرموز في الجنب عشان اليوزر ميتلخبطش
    with st.sidebar.expander("📝 Traits Reference Guide"):
        for key, val in trait_reference.items():
            st.write(f"**{key}**: {val}")

    age_range = st.sidebar.slider("Select Age Range", 
                                  int(df['age'].min()), int(df['age'].max()), 
                                  (int(df['age'].min()), int(df['age'].max())))
    
    gender_list = ["All Participants"] + list(df['gender'].unique())
    gender_choice = st.sidebar.selectbox("Filter by Gender Identity", options=gender_list)

    filtered_df = df[(df['age'] >= age_range[0]) & (df['age'] <= age_range[1])]
    if gender_choice != "All Participants":
        filtered_df = filtered_df[filtered_df['gender'] == gender_choice]

    # --- HEADER ---
    st.markdown('<p class="main-title">🧠 Big Five Personality Traits Analysis</p>', unsafe_allow_html=True)
    
    # --- 4x4 KPI CARDS ---
    st.markdown('<p class="sub-title">Data Health & Cleaning Impact</p>', unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Rows Before Cleaning", f"{r_before:,}")
    k2.metric("Rows After Cleaning", f"{r_after:,}")
    k3.metric("Total Rows Removed", f"{r_removed:,}")
    k4.metric("Dataset Columns", c_count)
    
    k5, k6, k7, k8 = st.columns(4)
    k5.metric("Initial Missing Data %", f"{m_pct:.2f}%")
    k6.metric("Initial Duplication %", f"{d_pct:.2f}%")
    k7.metric("Total Data Loss %", f"{rem_pct:.2f}%")
    k8.metric("Avg Personality Score", f"{global_avg:.2f}")

    # --- PROJECT INFO (Back to original text) ---
    with st.expander("📖 Detailed Project Description & Research Objectives (Click to Expand)"):
        st.markdown("""
        ### **About the Dataset**
        This dataset is derived from the **Big Five Personality Test**, an internationally recognized psychological model (OCEAN) that evaluates five broad domains of human personality. The data was collected through an open-source assessment platform, capturing responses from thousands of individuals globally.

        ### **The Five Dimensions Analyzed:**
        1. **Openness (O_total):** Measures intellectual curiosity, imagination, and a preference for novelty.
        2. **Conscientiousness (C_total):** Evaluates organization, dependability, and self-discipline.
        3. **Extraversion (E_total):** Assesses sociability, assertiveness, and emotional expression.
        4. **Agreeableness (A_total):** Reflects trustworthiness, altruism, and kindness.
        5. **Neuroticism (N_total):** Analyzes emotional stability and the tendency to experience negative emotions.

        ### **Research Objectives:**
        - **Data Integrity:** To transform raw, "noisy" survey data into a high-quality dataset by filtering out invalid ages and incomplete responses.
        - **Demographic Interaction:** To explore how personality traits shift across different age groups and genders.
        - **Correlation Mapping:** To identify how specific traits interact with each other (e.g., does high Neuroticism correlate with lower Agreeableness?).
        - **Statistical Modeling:** To provide a visual foundation for modeling human behavior based on structured psychometric data.
        """)
        st.write("**Dataset Source:** Kaggle / Open-Source Psychometrics Project")

    st.divider()

    # --- 6. DATA QUALITY SECTION ---
    st.markdown('<p class="sub-title">🛡️ Data Quality & Integrity</p>', unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["Missing Values Analysis", "Demographic Imbalance", "Outlier Summary"])
    
    with tab1:
        st.write("Missing values frequency before cleaning:")
        st.bar_chart(missing_summary)
    
    with tab2:
        st.write("Gender Identity Distribution (Labels as words):")
        fig_imb = px.pie(filtered_df, names='gender', hole=0.4, template="plotly_white")
        st.plotly_chart(fig_imb, width="stretch")
    
    with tab3:
        st.write("Outlier detection (Trait Codes):")
        fig_box_out = px.box(filtered_df, y=['A_total', 'O_total', 'C_total', 'E_total', 'N_total'], template="plotly_white")
        st.plotly_chart(fig_box_out, width="stretch")

    st.divider()

    # --- 7. DATA PREVIEW & DOWNLOAD ---
    st.markdown('<p class="sub-title">📋 Cleaned Data Preview</p>', unsafe_allow_html=True)
    st.write("Dataset Table (A_total, O_total, C_total, E_total, N_total)")
    st.dataframe(filtered_df.head(10), width="stretch")
    
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Final Filtered Dataset", data=csv, file_name="cleaned_personality_data.csv", mime="text/csv")

    # --- 8. CHART CONTROLS ---
    st.markdown('<p class="sub-title">📊 Dynamic Visualization Area</p>', unsafe_allow_html=True)
    
    display_list = list(friendly_names.values())

    c1, c2, c3 = st.columns([2, 2, 2])
    with c1:
        x_choice = st.selectbox("Select Variable (X-axis):", options=display_list, index=0)
    with c2:
        y_choice = st.selectbox("Select Variable (Y-axis):", options=display_list, index=3)
    with c3:
        chart_type = st.selectbox("Select Chart Style:", 
                                  options=["Bar", "Line", "Scatter", "Histogram", "Box", "Heatmap"])

    x_col = reverse_names[x_choice]
    y_col = reverse_names[y_choice]

    if chart_type == "Scatter":
        fig = px.scatter(filtered_df.sample(min(2000, len(filtered_df))), x=x_col, y=y_col, color=y_col, template="plotly_white")
    elif chart_type == "Bar":
        bar_data = filtered_df.groupby(x_col)[y_col].mean().reset_index()
        fig = px.bar(bar_data, x=x_col, y=y_col, color=y_col, template="plotly_white")
    elif chart_type == "Line":
        line_data = filtered_df.groupby(x_col)[y_col].mean().reset_index()
        fig = px.line(line_data, x=x_col, y=y_col, template="plotly_white")
    elif chart_type == "Histogram":
        fig = px.histogram(filtered_df, x=x_col, template="plotly_white")
    elif chart_type == "Box":
        fig = px.box(filtered_df, x=x_col, y=y_col, template="plotly_white")
    elif chart_type == "Heatmap":
        corr = filtered_df[['age', 'A_total', 'O_total', 'C_total', 'E_total', 'N_total']].corr()
        fig = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', template="plotly_white")

    st.plotly_chart(fig, width="stretch")

    # --- 9. FINDINGS & CONCLUSIONS ---
    st.divider()
    st.markdown('<p class="sub-title">📝 Study Findings & Conclusion</p>', unsafe_allow_html=True)
    
    f1, f2 = st.columns(2)
    with f1:
        st.markdown(f"""
        ### **Data Quality Notes:**
        - **Records Removed:** **{r_removed:,}** rows were removed during cleaning based on missing values and age bounds.
        - **Missing Values:** Approximately **{m_pct:.2f}%** of the initial data was incomplete, largely in personality responses.
        - **Demographic Balance:** Gender participation is uneven, which affects group-level averages.
        """)
    with f2:
        st.markdown("""
        ### **Trait Patterns:**
        - **Core Correlations:** Some traits (e.g., Neuroticism and Extraversion) show inverse relationships in this sample.
        - **Age Patterns:** Openness remains relatively stable across adult ages, while Conscientiousness tends to increase with age.
        - **Age Bounds:** The analysis excludes ages under 5 and over 100 to reduce outlier impact.
        """)

    st.info(f"""
    **Conclusion:**
    The cleaned dataset (**{r_after:,}** rows from **{r_before:,}** raw entries) supports descriptive analysis of Big Five traits across age and gender. Observed relationships reflect sample-level patterns and should be interpreted in that context.
    """)

    st.divider()

    # --- 10. OPTIONAL INTERACTIVE IPIP TEST ---
    st.markdown('<p class="sub-title">🧪 Optional: Take the Big Five Test</p>', unsafe_allow_html=True)
    st.write("If you want, answer the 50 statements below. Your totals are compared to the filtered dataset distribution.")

    ipip_items = [
        ("I am the life of the party in social gatherings.", "E_total", False),
        ("I feel little concern for others when they are in need.", "A_total", True),
        ("I am always prepared and plan ahead.", "C_total", False),
        ("I get stressed out easily when things go wrong.", "N_total", False),
        ("I have a rich vocabulary and use precise words.", "O_total", False),
        ("I don't talk a lot, even with familiar people.", "E_total", True),
        ("I am interested in people and their stories.", "A_total", False),
        ("I leave my belongings around and misplace things.", "C_total", True),
        ("I am relaxed most of the time, even under pressure.", "N_total", True),
        ("I have difficulty understanding abstract ideas like theories.", "O_total", True),
        ("I feel comfortable around people, even new ones.", "E_total", False),
        ("I insult people when I'm frustrated.", "A_total", True),
        ("I pay attention to details in my work.", "C_total", False),
        ("I worry about things more than I should.", "N_total", False),
        ("I have a vivid imagination and picture scenes easily.", "O_total", False),
        ("I keep in the background in groups.", "E_total", True),
        ("I sympathize with others' feelings when they are upset.", "A_total", False),
        ("I make a mess of things at home or work.", "C_total", True),
        ("I seldom feel blue or down for long.", "N_total", True),
        ("I am not interested in abstract ideas like philosophy.", "O_total", True),
        ("I start conversations with people I don't know.", "E_total", False),
        ("I am not interested in other people's problems.", "A_total", True),
        ("I get chores done right away instead of putting them off.", "C_total", False),
        ("I am easily disturbed by small problems.", "N_total", False),
        ("I have excellent ideas and enjoy brainstorming.", "O_total", False),
        ("I have little to say in group discussions.", "E_total", True),
        ("I have a soft heart and forgive easily.", "A_total", False),
        ("I often forget to put things back in their proper place.", "C_total", True),
        ("I get upset easily and show it.", "N_total", False),
        ("I do not have a good imagination for stories or images.", "O_total", True),
        ("I talk to a lot of different people at parties.", "E_total", False),
        ("I am not really interested in others and keep to myself.", "A_total", True),
        ("I like order and keeping things organized.", "C_total", False),
        ("I change my mood a lot over the day.", "N_total", False),
        ("I am quick to understand things when learning.", "O_total", False),
        ("I don't like to draw attention to myself.", "E_total", True),
        ("I take time out for others when they need help.", "A_total", False),
        ("I shirk my duties when tasks are boring.", "C_total", True),
        ("I have frequent mood swings without much warning.", "N_total", False),
        ("I use difficult words when I speak or write.", "O_total", False),
        ("I don't mind being the center of attention.", "E_total", False),
        ("I feel others' emotions and react to them.", "A_total", False),
        ("I follow a schedule and stick to it.", "C_total", False),
        ("I get irritated easily by small annoyances.", "N_total", False),
        ("I spend time reflecting on things and ideas.", "O_total", False),
        ("I am quiet around strangers until I know them.", "E_total", True),
        ("I make people feel at ease in social settings.", "A_total", False),
        ("I am exacting in my work and double-check it.", "C_total", False),
        ("I often feel blue or down.", "N_total", False),
        ("I am full of ideas and like sharing them.", "O_total", False)
    ]

    response_options = {
        1: "Very Inaccurate",
        2: "Moderately Inaccurate",
        3: "Neither Accurate Nor Inaccurate",
        4: "Moderately Accurate",
        5: "Very Accurate"
    }

    def score_response(value, reverse_scored):
        return 6 - value if reverse_scored else value

    def normal_percentile(z_value):
        return 0.5 * (1 + math.erf(z_value / math.sqrt(2)))

    with st.form("ipip_form"):
        st.write("Select one option for each item.")
        responses = []
        for idx, (text, trait, reverse_scored) in enumerate(ipip_items, start=1):
            label = f"{idx}. {text}"
            choice = st.radio(
                label,
                options=list(response_options.keys()),
                format_func=lambda v: response_options[v],
                horizontal=True,
                key=f"ipip_{idx}"
            )
            responses.append((trait, score_response(choice, reverse_scored)))

        submitted = st.form_submit_button("Calculate My Scores")

    if submitted:
        user_totals = {"A_total": 0, "O_total": 0, "C_total": 0, "E_total": 0, "N_total": 0}
        for trait, score in responses:
            user_totals[trait] += score

        totals_df = pd.DataFrame([user_totals])
        st.markdown("**Your trait totals (range 10-50):**")
        st.dataframe(totals_df, width="stretch")

        st.markdown("**Your position vs. the dataset distribution:**")
        dist_cols = ["A_total", "O_total", "C_total", "E_total", "N_total"]
        curve_cols = st.columns(2)
        for i, trait in enumerate(dist_cols):
            mean_val = filtered_df[trait].mean()
            std_val = filtered_df[trait].std()
            user_val = user_totals[trait]

            if std_val == 0 or np.isnan(std_val):
                st.warning(f"Not enough variance to chart {trait}.")
                continue

            x_min = min(filtered_df[trait].min(), user_val) - 2
            x_max = max(filtered_df[trait].max(), user_val) + 2
            x_vals = np.linspace(x_min, x_max, 200)
            y_vals = (1 / (std_val * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_vals - mean_val) / std_val) ** 2)

            z_score = (user_val - mean_val) / std_val
            percentile = normal_percentile(z_score) * 100

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode="lines", name="Distribution"))
            fig.add_trace(go.Scatter(
                x=[user_val],
                y=[(1 / (std_val * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((user_val - mean_val) / std_val) ** 2)],
                mode="markers",
                marker=dict(size=10, color="#2563EB"),
                name="Your Score"
            ))
            fig.update_layout(
                title=f"{trait_reference[trait]} ({trait})",
                xaxis_title="Total Score",
                yaxis_title="Density",
                template="plotly_white",
                height=300,
                margin=dict(l=20, r=20, t=40, b=20)
            )

            with curve_cols[i % 2]:
                st.plotly_chart(fig, width="stretch")
                st.caption(f"$z$ = {z_score:.2f}, percentile ~ {percentile:.1f}%")

else:
    st.error("CSV File not found! Please check 'Datasets/big5-dataset.csv'.")