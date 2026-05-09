import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Doing the Dirty Work for Your Data!")

# =========================================================
# SESSION STATE INITIALIZATION
# =========================================================
if "df" not in st.session_state:
    st.session_state.df = None
if "cleaning_df" not in st.session_state:
    st.session_state.cleaning_df = None

# =========================================================
# FILE UPLOAD
# =========================================================
uploaded_file = st.file_uploader(
    "Upload your file",
    type=["csv", "xlsx", "xls", "json", "parquet"]
)

if uploaded_file is not None:
    st.subheader("Choose File Type")
    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)

    if col1.button("CSV"):
        st.session_state.df = pd.read_csv(uploaded_file)
        st.session_state.cleaning_df = st.session_state.df.copy()
    if col2.button("Excel"):
        st.session_state.df = pd.read_excel(uploaded_file)
        st.session_state.cleaning_df = st.session_state.df.copy()
    if col3.button("JSON"):
        st.session_state.df = pd.read_json(uploaded_file)
        st.session_state.cleaning_df = st.session_state.df.copy()
    if col4.button("Parquet"):
        st.session_state.df = pd.read_parquet(uploaded_file)
        st.session_state.cleaning_df = st.session_state.df.copy()

# =========================================================
# RETRIEVE FROM SESSION STATE
# =========================================================
df = st.session_state.df
cleaning_df = st.session_state.cleaning_df

# =========================================================
# MAIN APP — only runs if data is loaded
# =========================================================
if df is not None:

    st.success("File loaded successfully!")
    st.write(f"Dataframe shape: {df.shape[0]} rows and {df.shape[1]} columns")

    # ---------------------------------------------------------
    # DATA PREVIEW
    # ---------------------------------------------------------
    st.subheader("Here is a data snippet:")
    row_num = st.number_input(
        "Number of rows to preview:",
        min_value=1,
        max_value=len(df),
        value=min(25, len(df)),
        step=1
    )
    st.dataframe(df.head(row_num))

    st.subheader("Data Summary")
    st.write(df.describe())

    st.subheader("Data Types")
    st.write(df.dtypes)

    st.subheader("Null Value Percentage")
    null_percent = (df.isnull().sum() / len(df)) * 100
    null_df = pd.DataFrame(null_percent, columns=["Null Percentage"]).T
    st.dataframe(null_df)

    # =========================================================
    # 🧹 DATA CLEANING SECTION
    # =========================================================
    st.divider()
    st.subheader("🧹 Data Cleaning")
    st.info(f"Working on a clean copy of your data. Current shape: **{cleaning_df.shape[0]} rows × {cleaning_df.shape[1]} columns**")

    # ---------------------------------------------------------
    # STEP 1 — RENAME COLUMNS
    # ---------------------------------------------------------
    st.markdown("#### ✏️ Step 1: Rename Columns")
    st.write("Edit any column name below and click Apply:")

    new_names = {}
    rename_cols = list(cleaning_df.columns)
    num_cols_display = st.number_input("How many columns to rename?", min_value=1, max_value=len(rename_cols), value=1, step=1, key="rename_count")

    for i in range(int(num_cols_display)):
        r1, r2 = st.columns(2)
        with r1:
            old_name = st.selectbox(f"Column to rename #{i+1}:", options=rename_cols, key=f"old_name_{i}")
        with r2:
            new_name = st.text_input(f"New name #{i+1}:", value=old_name, key=f"new_name_{i}")
        if old_name and new_name and old_name != new_name:
            new_names[old_name] = new_name

    if st.button("✅ Apply Renaming"):
        if new_names:
            st.session_state.cleaning_df = st.session_state.cleaning_df.rename(columns=new_names)
            cleaning_df = st.session_state.cleaning_df
            st.success(f"✅ Renamed: {new_names}")
        else:
            st.info("No changes to apply.")

    # ---------------------------------------------------------
    # STEP 2 — CHANGE DATA TYPES
    # ---------------------------------------------------------
    st.markdown("#### 🔢 Step 2: Change Data Types")
    st.write("Cast columns to the correct type before any string cleaning or analysis:")

    dtype_options = ["Keep current", "int", "float", "str", "datetime", "bool", "category"]
    dtype_actions = {}

    for col in cleaning_df.columns:
        d1, d2 = st.columns([2, 2])
        with d1:
            st.markdown(f"`{col}` — current type: `{cleaning_df[col].dtype}`")
        with d2:
            chosen = st.selectbox("Cast to:", dtype_options, key=f"dtype_{col}")
        if chosen != "Keep current":
            dtype_actions[col] = chosen

    if st.button("✅ Apply Type Conversions"):
        errors = []
        for col, target_type in dtype_actions.items():
            try:
                if target_type == "int":
                    st.session_state.cleaning_df[col] = st.session_state.cleaning_df[col].astype(int)
                elif target_type == "float":
                    st.session_state.cleaning_df[col] = st.session_state.cleaning_df[col].astype(float)
                elif target_type == "str":
                    st.session_state.cleaning_df[col] = st.session_state.cleaning_df[col].astype(str)
                elif target_type == "datetime":
                    st.session_state.cleaning_df[col] = pd.to_datetime(st.session_state.cleaning_df[col], errors='coerce')
                elif target_type == "bool":
                    st.session_state.cleaning_df[col] = st.session_state.cleaning_df[col].astype(bool)
                elif target_type == "category":
                    st.session_state.cleaning_df[col] = st.session_state.cleaning_df[col].astype("category")
            except Exception as e:
                errors.append(f"`{col}`: {e}")

        cleaning_df = st.session_state.cleaning_df
        if errors:
            for err in errors:
                st.warning(f"⚠️ Could not convert {err}")
        else:
            st.success("✅ Type conversions applied!")

    # ---------------------------------------------------------
    # STEP 3 — STRIP WHITESPACE
    # ---------------------------------------------------------
    st.markdown("#### ✂️ Step 3: Strip Whitespace from String Columns")

    str_cols = cleaning_df.select_dtypes(include="object").columns.tolist()

    if not str_cols:
        st.info("No string columns found.")
    else:
        selected_strip_cols = st.multiselect(
            "Select columns to strip whitespace from:",
            options=str_cols,
            default=str_cols,
            key="strip_cols"
        )
        if st.button("✅ Strip Whitespace"):
            for col in selected_strip_cols:
                st.session_state.cleaning_df[col] = st.session_state.cleaning_df[col].str.strip()
            cleaning_df = st.session_state.cleaning_df
            st.success(f"✅ Stripped whitespace from: {selected_strip_cols}")

    # ---------------------------------------------------------
    # STEP 4 — STANDARDIZE CASING
    # ---------------------------------------------------------
    st.markdown("#### 🔡 Step 4: Standardize Text Casing")

    str_cols_now = cleaning_df.select_dtypes(include="object").columns.tolist()

    if not str_cols_now:
        st.info("No string columns found.")
    else:
        casing_actions = {}
        casing_options = ["Do nothing", "UPPER", "lower", "Title Case"]

        for col in str_cols_now:
            ca1, ca2 = st.columns([2, 2])
            with ca1:
                st.markdown(f"`{col}`")
            with ca2:
                casing = st.selectbox("Casing:", casing_options, key=f"casing_{col}")
            if casing != "Do nothing":
                casing_actions[col] = casing

        if st.button("✅ Apply Casing"):
            for col, case in casing_actions.items():
                if case == "UPPER":
                    st.session_state.cleaning_df[col] = st.session_state.cleaning_df[col].str.upper()
                elif case == "lower":
                    st.session_state.cleaning_df[col] = st.session_state.cleaning_df[col].str.lower()
                elif case == "Title Case":
                    st.session_state.cleaning_df[col] = st.session_state.cleaning_df[col].str.title()
            cleaning_df = st.session_state.cleaning_df
            st.success("✅ Casing applied!")

    # ---------------------------------------------------------
    # STEP 5 — REPLACE VALUES
    # ---------------------------------------------------------
    st.markdown("#### 🔄 Step 5: Find & Replace Values")

    replace_col = st.selectbox("Select column to apply find & replace:", options=cleaning_df.columns.tolist(), key="replace_col")
    rp1, rp2 = st.columns(2)
    with rp1:
        find_val = st.text_input("Find value:", key="find_val")
    with rp2:
        replace_val = st.text_input("Replace with:", key="replace_val")

    if st.button("✅ Apply Replace"):
        if find_val != "":
            try:
                st.session_state.cleaning_df[replace_col] = st.session_state.cleaning_df[replace_col].replace(find_val, replace_val)
                cleaning_df = st.session_state.cleaning_df
                st.success(f"✅ Replaced `{find_val}` with `{replace_val}` in `{replace_col}`")
            except Exception as e:
                st.error(f"❌ Could not apply replace: {e}")
        else:
            st.warning("⚠️ Please enter a value to find.")

    # ---------------------------------------------------------
    # STEP 6 — DROP DUPLICATES
    # ---------------------------------------------------------
    st.markdown("#### 🔁 Step 6: Duplicate Rows")
    n_dupes = cleaning_df.duplicated().sum()
    st.write(f"Found **{n_dupes}** duplicate row(s).")

    if n_dupes > 0:
        if st.button("Drop Duplicate Rows"):
            st.session_state.cleaning_df = st.session_state.cleaning_df.drop_duplicates()
            cleaning_df = st.session_state.cleaning_df
            st.success(f"✅ Dropped {n_dupes} duplicate rows.")

    # ---------------------------------------------------------
    # STEP 7 — DROP COLUMNS BY NULL THRESHOLD
    # ---------------------------------------------------------
    st.markdown("#### 🗑️ Step 7: Drop Columns by Null Threshold")
    threshold = st.slider(
        "Drop columns where null % is greater than:",
        min_value=0, max_value=100, value=50, step=5, format="%d%%"
    )

    null_pct = (cleaning_df.isnull().sum() / len(cleaning_df)) * 100
    cols_to_drop = null_pct[null_pct > threshold].index.tolist()

    if cols_to_drop:
        st.warning(f"These columns exceed the threshold: **{', '.join(cols_to_drop)}**")
        if st.button("Drop These Columns"):
            st.session_state.cleaning_df = st.session_state.cleaning_df.drop(columns=cols_to_drop)
            cleaning_df = st.session_state.cleaning_df
            st.success(f"✅ Dropped: {', '.join(cols_to_drop)}")
    else:
        st.info("No columns exceed the selected threshold.")

    # ---------------------------------------------------------
    # STEP 8 — PER-COLUMN NULL HANDLING
    # ---------------------------------------------------------
    st.markdown("#### 🔧 Step 8: Handle Nulls Per Column")

    cols_with_nulls = cleaning_df.columns[cleaning_df.isnull().any()].tolist()

    if not cols_with_nulls:
        st.info("✅ No columns with missing values found.")
    else:
        per_column_actions = {}

        for col in cols_with_nulls:
            col_null_count = cleaning_df[col].isnull().sum()
            col_dtype = cleaning_df[col].dtype
            is_numeric = pd.api.types.is_numeric_dtype(cleaning_df[col])

            options = ["Do nothing", "Drop rows with null", "Drop entire column"]
            if is_numeric:
                options += ["Fill with Mean", "Fill with Median"]
            options += ["Fill with Mode", "Fill with Custom Value"]

            with st.expander(f"📌 `{col}` — {col_null_count} nulls ({null_pct[col]:.1f}%) | dtype: {col_dtype}"):
                action = st.selectbox(f"Action for `{col}`:", options=options, key=f"action_{col}")
                custom_val = None
                if action == "Fill with Custom Value":
                    custom_val = st.text_input(f"Enter custom value for `{col}`:", key=f"custom_{col}")
                per_column_actions[col] = (action, custom_val)

        if st.button("✅ Apply All Column Actions"):
            for col, (action, custom_val) in per_column_actions.items():
                if action == "Drop rows with null":
                    st.session_state.cleaning_df = st.session_state.cleaning_df.dropna(subset=[col])
                elif action == "Drop entire column":
                    st.session_state.cleaning_df = st.session_state.cleaning_df.drop(columns=[col])
                elif action == "Fill with Mean":
                    st.session_state.cleaning_df[col] = st.session_state.cleaning_df[col].fillna(st.session_state.cleaning_df[col].mean())
                elif action == "Fill with Median":
                    st.session_state.cleaning_df[col] = st.session_state.cleaning_df[col].fillna(st.session_state.cleaning_df[col].median())
                elif action == "Fill with Mode":
                    st.session_state.cleaning_df[col] = st.session_state.cleaning_df[col].fillna(st.session_state.cleaning_df[col].mode()[0])
                elif action == "Fill with Custom Value":
                    if custom_val:
                        try:
                            if pd.api.types.is_numeric_dtype(st.session_state.cleaning_df[col]):
                                custom_val = float(custom_val)
                        except ValueError:
                            pass
                        st.session_state.cleaning_df[col] = st.session_state.cleaning_df[col].fillna(custom_val)
                    else:
                        st.warning(f"⚠️ No custom value entered for `{col}`, skipping.")
            cleaning_df = st.session_state.cleaning_df
            st.success("✅ All actions applied!")

    # ---------------------------------------------------------
    # STEP 9 — GLOBAL DROP NULL ROWS
    # ---------------------------------------------------------
    st.markdown("#### 🧨 Step 9: Drop ALL Remaining Null Rows")
    remaining_nulls = cleaning_df.isnull().sum().sum()
    st.write(f"Remaining null values: **{remaining_nulls}**")

    if remaining_nulls > 0:
        if st.button("Drop All Remaining Null Rows"):
            before = len(cleaning_df)
            st.session_state.cleaning_df = st.session_state.cleaning_df.dropna()
            cleaning_df = st.session_state.cleaning_df
            st.success(f"✅ Dropped {before - len(cleaning_df)} rows. {len(cleaning_df)} remaining.")

    # ---------------------------------------------------------
    # STEP 10 — OUTLIER DETECTION & HANDLING
    # ---------------------------------------------------------
    st.divider()
    st.markdown("#### 🔍 Step 10: Outlier Detection & Handling (IQR)")

    numeric_cols = cleaning_df.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_cols:
        st.info("No numeric columns found for outlier detection.")
    else:
        st.write("Outliers detected using **IQR method** (Q1 - 1.5×IQR, Q3 + 1.5×IQR).")

        outlier_summary = []
        for col in numeric_cols:
            Q1 = cleaning_df[col].quantile(0.25)
            Q3 = cleaning_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            n_outliers = ((cleaning_df[col] < lower) | (cleaning_df[col] > upper)).sum()
            outlier_summary.append({
                "Column": col, "Q1": round(Q1, 3), "Q3": round(Q3, 3),
                "IQR": round(IQR, 3), "Lower Bound": round(lower, 3),
                "Upper Bound": round(upper, 3), "Outlier Count": n_outliers
            })

        summary_df = pd.DataFrame(outlier_summary)
        st.dataframe(summary_df, use_container_width=True)

        cols_with_outliers = summary_df[summary_df["Outlier Count"] > 0]["Column"].tolist()

        if not cols_with_outliers:
            st.success("✅ No outliers detected!")
        else:
            outlier_actions = {}
            for col in cols_with_outliers:
                row = summary_df[summary_df["Column"] == col].iloc[0]
                with st.expander(f"📌 `{col}` — {int(row['Outlier Count'])} outliers | bounds: [{row['Lower Bound']}, {row['Upper Bound']}]"):
                    action = st.selectbox(
                        f"Action for `{col}`:",
                        options=["Do nothing", "Delete outlier rows", "Cap to bounds (Winsorize)"],
                        key=f"outlier_action_{col}"
                    )
                    outlier_actions[col] = {"action": action, "lower": row["Lower Bound"], "upper": row["Upper Bound"]}

            if st.button("✅ Apply Outlier Actions"):
                rows_before = len(cleaning_df)
                for col, info in outlier_actions.items():
                    if info["action"] == "Delete outlier rows":
                        st.session_state.cleaning_df = st.session_state.cleaning_df[
                            (st.session_state.cleaning_df[col] >= info["lower"]) &
                            (st.session_state.cleaning_df[col] <= info["upper"])
                        ]
                    elif info["action"] == "Cap to bounds (Winsorize)":
                        st.session_state.cleaning_df[col] = st.session_state.cleaning_df[col].clip(
                            lower=info["lower"], upper=info["upper"]
                        )
                cleaning_df = st.session_state.cleaning_df
                st.success(f"✅ Done! {rows_before} → {len(cleaning_df)} rows ({rows_before - len(cleaning_df)} removed)")

    # ---------------------------------------------------------
    # FINAL PREVIEW & SAVE
    # ---------------------------------------------------------
    st.divider()
    st.subheader("👁️ Final Cleaned Data Preview")
    st.write(f"Shape: **{cleaning_df.shape[0]} rows × {cleaning_df.shape[1]} columns**")
    st.dataframe(cleaning_df.head(25))

    if st.button("💾 Save Final Cleaned Data to Session"):
        st.session_state.df = cleaning_df
        # st.success("✅ Saved! This version will be used for visualizations.")

    # ---------------------------------------------------------
    # FINAL SAVE & EXPORT
    # ---------------------------------------------------------
    st.divider()
    st.subheader("💾 Save & Export Cleaned Data")

    col_save1, col_save2 = st.columns(2)

    # --- Save to Session (for visualizations) ---
    with col_save1:
        st.markdown("#### 🧠 Save to Session")
        st.write("Keep the cleaned data in memory to use in the visualization steps.")
        if st.button("💾 Save to Session"):
            st.session_state.df = cleaning_df
            st.success("✅ Saved! Visualizations will use this cleaned version.")

    # --- Download as File ---
    with col_save2:
        st.markdown("#### 📥 Download as File")
        st.write("Export and download the cleaned dataset to your machine.")

        export_format = st.radio(
            "Choose export format:",
            options=["CSV", "Excel", "JSON"],
            horizontal=True,
            key="export_format"
        )

        if export_format == "CSV":
            data = cleaning_df.to_csv(index=False).encode("utf-8")
            mime = "text/csv"
            filename = "cleaned_data.csv"

        elif export_format == "Excel":
            import io
            buffer = io.BytesIO()
            cleaning_df.to_excel(buffer, index=False, engine="openpyxl")
            data = buffer.getvalue()
            mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            filename = "cleaned_data.xlsx"

        elif export_format == "JSON":
            data = cleaning_df.to_json(orient="records", indent=2).encode("utf-8")
            mime = "application/json"
            filename = "cleaned_data.json"

        st.download_button(
            label=f"⬇️ Download as {export_format}",
            data=data,
            file_name=filename,
            mime=mime
        )

    # =========================================================
    # 📊 VISUALIZATION SECTION
    # =========================================================
    st.divider()
    st.subheader("📊 Data Visualizations")

    viz_df = st.session_state.df  # uses the saved cleaned data

    if viz_df is None:
        st.warning("⚠️ Please save your cleaned data to session first before visualizing.")
    else:
        numeric_cols   = viz_df.select_dtypes(include=np.number).columns.tolist()
        categoric_cols = viz_df.select_dtypes(include=["object", "category"]).columns.tolist()
        all_cols       = viz_df.columns.tolist()

        plot_type = st.selectbox(
            "🎨 Choose a plot type:",
            options=["Histogram", "Bar Chart", "Scatter Plot", "Line Chart", "Box Plot", "Correlation Heatmap", "Pie Chart"],
            key="plot_type"
        )

        st.markdown("---")

        # ---------------------------------------------------------
        # HISTOGRAM
        # ---------------------------------------------------------
        if plot_type == "Histogram":
            st.markdown("#### 📊 Histogram")
            col = st.selectbox("Select column:", options=all_cols, key="hist_col")
            bins = st.slider("Number of bins:", min_value=5, max_value=100, value=30, step=5)
            color = st.color_picker("Bar color:", value="#1E3A8A")

            try:
                if not pd.api.types.is_numeric_dtype(viz_df[col]):
                    st.error("❌ Invalid: Histogram requires a numeric column.")
                else:
                    fig, ax = plt.subplots()
                    ax.hist(viz_df[col].dropna(), bins=bins, color=color, edgecolor="white")
                    ax.set_xlabel(col)
                    ax.set_ylabel("Frequency")
                    ax.set_title(f"Distribution of {col}")
                    sns.despine()
                    st.pyplot(fig)
            except Exception as e:
                st.error(f"❌ Invalid: Could not render plot. {e}")

        # ---------------------------------------------------------
        # BAR CHART
        # ---------------------------------------------------------
        elif plot_type == "Bar Chart":
            st.markdown("#### 📊 Bar Chart")

            if not categoric_cols:
                st.error("❌ Invalid: No categorical columns found for Bar Chart.")
            else:
                x_col = st.selectbox("X-axis (categorical):", options=categoric_cols, key="bar_x")
                y_col = st.selectbox("Y-axis (numeric):", options=all_cols, key="bar_y")
                top_n = st.slider("Show top N categories:", min_value=2, max_value=30, value=10)
                color = st.color_picker("Bar color:", value="#2563EB")

                try:
                    if not pd.api.types.is_numeric_dtype(viz_df[y_col]):
                        st.error("❌ Invalid: Y-axis must be a numeric column.")
                    else:
                        bar_data = viz_df.groupby(x_col)[y_col].mean().nlargest(top_n).reset_index()
                        fig, ax = plt.subplots(figsize=(10, 5))
                        ax.bar(bar_data[x_col].astype(str), bar_data[y_col], color=color, edgecolor="white")
                        ax.set_xlabel(x_col)
                        ax.set_ylabel(f"Mean of {y_col}")
                        ax.set_title(f"Mean {y_col} by {x_col} (Top {top_n})")
                        plt.xticks(rotation=45, ha="right")
                        sns.despine()
                        st.pyplot(fig)
                except Exception as e:
                    st.error(f"❌ Invalid: Could not render plot. {e}")

        # ---------------------------------------------------------
        # SCATTER PLOT
        # ---------------------------------------------------------
        elif plot_type == "Scatter Plot":
            st.markdown("#### 🔵 Scatter Plot")

            if len(numeric_cols) < 2:
                st.error("❌ Invalid: Scatter plot requires at least 2 numeric columns.")
            else:
                x_col = st.selectbox("X-axis:", options=numeric_cols, key="scatter_x")
                y_col = st.selectbox("Y-axis:", options=numeric_cols, key="scatter_y")
                hue_col = st.selectbox("Color by (optional):", options=["None"] + categoric_cols, key="scatter_hue")
                alpha = st.slider("Point opacity:", min_value=0.1, max_value=1.0, value=0.6, step=0.1)

                try:
                    fig, ax = plt.subplots(figsize=(9, 5))
                    if hue_col == "None":
                        ax.scatter(viz_df[x_col], viz_df[y_col], alpha=alpha, color="#1E3A8A")
                    else:
                        categories = viz_df[hue_col].dropna().unique()
                        palette = sns.color_palette("tab10", len(categories))
                        for i, cat in enumerate(categories):
                            mask = viz_df[hue_col] == cat
                            ax.scatter(viz_df.loc[mask, x_col], viz_df.loc[mask, y_col],
                                       alpha=alpha, label=str(cat), color=palette[i])
                        ax.legend(title=hue_col, bbox_to_anchor=(1.05, 1), loc="upper left")
                    ax.set_xlabel(x_col)
                    ax.set_ylabel(y_col)
                    ax.set_title(f"{x_col} vs {y_col}")
                    sns.despine()
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"❌ Invalid: Could not render plot. {e}")

        # ---------------------------------------------------------
        # LINE CHART
        # ---------------------------------------------------------
        elif plot_type == "Line Chart":
            st.markdown("#### 📈 Line Chart")

            x_col = st.selectbox("X-axis:", options=all_cols, key="line_x")
            y_col = st.selectbox("Y-axis (numeric):", options=all_cols, key="line_y")
            color = st.color_picker("Line color:", value="#2563EB")

            try:
                if not pd.api.types.is_numeric_dtype(viz_df[y_col]):
                    st.error("❌ Invalid: Y-axis must be numeric for a Line Chart.")
                elif viz_df[x_col].nunique() > 500:
                    st.error("❌ Invalid: X-axis has too many unique values (>500). Consider a different column.")
                else:
                    line_data = viz_df[[x_col, y_col]].dropna().sort_values(by=x_col)
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.plot(line_data[x_col].astype(str), line_data[y_col], color=color, linewidth=2)
                    ax.set_xlabel(x_col)
                    ax.set_ylabel(y_col)
                    ax.set_title(f"{y_col} over {x_col}")
                    plt.xticks(rotation=45, ha="right")
                    sns.despine()
                    st.pyplot(fig)
            except Exception as e:
                st.error(f"❌ Invalid: Could not render plot. {e}")

        # ---------------------------------------------------------
        # BOX PLOT
        # ---------------------------------------------------------
        elif plot_type == "Box Plot":
            st.markdown("#### 📦 Box Plot")

            y_col = st.selectbox("Numeric column:", options=all_cols, key="box_y")
            group_col = st.selectbox("Group by (optional):", options=["None"] + categoric_cols, key="box_group")
            color = st.color_picker("Box color:", value="#1E3A8A")

            try:
                if not pd.api.types.is_numeric_dtype(viz_df[y_col]):
                    st.error("❌ Invalid: Box plot requires a numeric column.")
                else:
                    fig, ax = plt.subplots(figsize=(10, 5))
                    if group_col == "None":
                        ax.boxplot(viz_df[y_col].dropna(), patch_artist=True,
                                   boxprops=dict(facecolor=color, color="white"),
                                   medianprops=dict(color="white", linewidth=2))
                        ax.set_ylabel(y_col)
                        ax.set_title(f"Box Plot of {y_col}")
                    else:
                        top_cats = viz_df[group_col].value_counts().nlargest(10).index.tolist()
                        filtered = viz_df[viz_df[group_col].isin(top_cats)]
                        groups = [filtered[filtered[group_col] == cat][y_col].dropna().values for cat in top_cats]
                        ax.boxplot(groups, labels=top_cats, patch_artist=True,
                                   boxprops=dict(facecolor=color, color="white"),
                                   medianprops=dict(color="white", linewidth=2))
                        ax.set_xlabel(group_col)
                        ax.set_ylabel(y_col)
                        ax.set_title(f"{y_col} by {group_col} (Top 10 categories)")
                        plt.xticks(rotation=45, ha="right")
                    sns.despine()
                    st.pyplot(fig)
            except Exception as e:
                st.error(f"❌ Invalid: Could not render plot. {e}")

        # ---------------------------------------------------------
        # CORRELATION HEATMAP
        # ---------------------------------------------------------
        elif plot_type == "Correlation Heatmap":
            st.markdown("#### 🌡️ Correlation Heatmap")

            if len(numeric_cols) < 2:
                st.error("❌ Invalid: Need at least 2 numeric columns for a heatmap.")
            else:
                selected_cols = st.multiselect(
                    "Select numeric columns to include:",
                    options=numeric_cols,
                    default=numeric_cols[:min(8, len(numeric_cols))],
                    key="heatmap_cols"
                )

                try:
                    if len(selected_cols) < 2:
                        st.error("❌ Invalid: Please select at least 2 columns.")
                    else:
                        corr = viz_df[selected_cols].corr()
                        fig, ax = plt.subplots(figsize=(10, 7))
                        sns.heatmap(
                            corr, annot=True, fmt=".2f", cmap="coolwarm",
                            center=0, linewidths=0.5, ax=ax,
                            annot_kws={"size": 9}
                        )
                        ax.set_title("Correlation Matrix")
                        st.pyplot(fig)
                except Exception as e:
                    st.error(f"❌ Invalid: Could not render heatmap. {e}")

        # ---------------------------------------------------------
        # PIE CHART
        # ---------------------------------------------------------
        elif plot_type == "Pie Chart":
            st.markdown("#### 🥧 Pie Chart")

            if not categoric_cols:
                st.error("❌ Invalid: No categorical columns found for Pie Chart.")
            else:
                col = st.selectbox("Select categorical column:", options=categoric_cols, key="pie_col")
                top_n = st.slider("Show top N categories:", min_value=2, max_value=15, value=7)

                try:
                    pie_data = viz_df[col].value_counts().nlargest(top_n)
                    if len(pie_data) < 2:
                        st.error("❌ Invalid: Need at least 2 categories to render a Pie Chart.")
                    else:
                        fig, ax = plt.subplots(figsize=(8, 8))
                        ax.pie(
                            pie_data.values,
                            labels=pie_data.index.astype(str),
                            autopct="%1.1f%%",
                            colors=sns.color_palette("Blues_r", len(pie_data)),
                            startangle=140,
                            wedgeprops={"edgecolor": "white", "linewidth": 1.5}
                        )
                        ax.set_title(f"Distribution of {col} (Top {top_n})")
                        st.pyplot(fig)
                except Exception as e:
                    st.error(f"❌ Invalid: Could not render plot. {e}")

    # =========================================================
    # ⚙️ PREPROCESSING SECTION
    # =========================================================
    st.divider()
    st.subheader("⚙️ Preprocessing")

    # Always work on the latest saved version
    if "preprocessed_df" not in st.session_state:
        st.session_state.preprocessed_df = st.session_state.df.copy() if st.session_state.df is not None else None

    pre_df = st.session_state.preprocessed_df

    if pre_df is None:
        st.warning("⚠️ Please save your cleaned data to session first before preprocessing.")
    else:
        st.info(f"Working on: **{pre_df.shape[0]} rows × {pre_df.shape[1]} columns**")

        numeric_cols   = pre_df.select_dtypes(include=np.number).columns.tolist()
        categoric_cols = pre_df.select_dtypes(include=["object", "category"]).columns.tolist()

        # ---------------------------------------------------------
        # STEP 1 — MIN-MAX SCALING
        # ---------------------------------------------------------
        st.markdown("#### 📏 Step 1: Min-Max Scaling")
        st.write("Scales values to a [0, 1] range. Best for algorithms sensitive to magnitude like KNN or Neural Networks.")

        if not numeric_cols:
            st.info("No numeric columns available.")
        else:
            minmax_cols = st.multiselect(
                "Select columns to scale:",
                options=numeric_cols,
                key="minmax_cols"
            )

            if st.button("✅ Apply Min-Max Scaling"):
                if not minmax_cols:
                    st.warning("⚠️ No columns selected.")
                else:
                    try:
                        for col in minmax_cols:
                            col_min = st.session_state.preprocessed_df[col].min()
                            col_max = st.session_state.preprocessed_df[col].max()
                            if col_max == col_min:
                                st.warning(f"⚠️ `{col}` has no range (min == max), skipping.")
                            else:
                                st.session_state.preprocessed_df[col] = (
                                    st.session_state.preprocessed_df[col] - col_min
                                ) / (col_max - col_min)
                        pre_df = st.session_state.preprocessed_df
                        st.success(f"✅ Min-Max Scaling applied to: {minmax_cols}")
                    except Exception as e:
                        st.error(f"❌ Could not apply scaling: {e}")

        # ---------------------------------------------------------
        # STEP 2 — STANDARD SCALING (Z-SCORE)
        # ---------------------------------------------------------
        st.markdown("#### 📐 Step 2: Standard Scaling (Z-Score)")
        st.write("Transforms values to have mean=0 and std=1. Best for algorithms like Linear Regression or SVM.")

        if not numeric_cols:
            st.info("No numeric columns available.")
        else:
            standard_cols = st.multiselect(
                "Select columns to standardize:",
                options=numeric_cols,
                key="standard_cols"
            )

            if st.button("✅ Apply Standard Scaling"):
                if not standard_cols:
                    st.warning("⚠️ No columns selected.")
                else:
                    try:
                        for col in standard_cols:
                            col_mean = st.session_state.preprocessed_df[col].mean()
                            col_std  = st.session_state.preprocessed_df[col].std()
                            if col_std == 0:
                                st.warning(f"⚠️ `{col}` has std=0, skipping.")
                            else:
                                st.session_state.preprocessed_df[col] = (
                                    st.session_state.preprocessed_df[col] - col_mean
                                ) / col_std
                        pre_df = st.session_state.preprocessed_df
                        st.success(f"✅ Standard Scaling applied to: {standard_cols}")
                    except Exception as e:
                        st.error(f"❌ Could not apply standardization: {e}")

        # ---------------------------------------------------------
        # STEP 3 — ONE-HOT ENCODING
        # ---------------------------------------------------------
        st.markdown("#### 🔳 Step 3: One-Hot Encoding")
        st.write("Creates a binary column for each category. Best for nominal data with no order (e.g. gender, city).")

        if not categoric_cols:
            st.info("No categorical columns available.")
        else:
            ohe_cols = st.multiselect(
                "Select columns to one-hot encode:",
                options=categoric_cols,
                key="ohe_cols"
            )

            drop_first = st.checkbox(
                "Drop first dummy column (avoids multicollinearity)",
                value=True,
                key="drop_first"
            )

            if st.button("✅ Apply One-Hot Encoding"):
                if not ohe_cols:
                    st.warning("⚠️ No columns selected.")
                else:
                    try:
                        # Warn if any column has too many unique values
                        skipped = []
                        to_encode = []
                        for col in ohe_cols:
                            n_unique = st.session_state.preprocessed_df[col].nunique()
                            if n_unique > 50:
                                skipped.append(f"`{col}` ({n_unique} unique values)")
                            else:
                                to_encode.append(col)

                        if skipped:
                            st.warning(f"⚠️ Skipped high-cardinality columns: {', '.join(skipped)}. Consider Label Encoding instead.")

                        if to_encode:
                            st.session_state.preprocessed_df = pd.get_dummies(
                                st.session_state.preprocessed_df,
                                columns=to_encode,
                                drop_first=drop_first,
                                dtype=int
                            )
                            pre_df = st.session_state.preprocessed_df
                            st.success(f"✅ One-Hot Encoding applied to: {to_encode}")
                            st.write(f"New shape: **{pre_df.shape[0]} rows × {pre_df.shape[1]} columns**")
                    except Exception as e:
                        st.error(f"❌ Could not apply encoding: {e}")

        # ---------------------------------------------------------
        # STEP 4 — LABEL ENCODING
        # ---------------------------------------------------------
        st.markdown("#### 🏷️ Step 4: Label Encoding")
        st.write("Maps each category to an integer. Best for ordinal data with a natural order (e.g. low/medium/high).")

        # Refresh categoric cols after OHE may have removed some
        categoric_cols_now = st.session_state.preprocessed_df.select_dtypes(include=["object", "category"]).columns.tolist()

        if not categoric_cols_now:
            st.info("No categorical columns remaining.")
        else:
            label_cols = st.multiselect(
                "Select columns to label encode:",
                options=categoric_cols_now,
                key="label_cols"
            )

            if st.button("✅ Apply Label Encoding"):
                if not label_cols:
                    st.warning("⚠️ No columns selected.")
                else:
                    try:
                        mapping_log = {}
                        for col in label_cols:
                            categories = st.session_state.preprocessed_df[col].dropna().unique()
                            mapping = {cat: i for i, cat in enumerate(sorted(categories, key=str))}
                            st.session_state.preprocessed_df[col] = st.session_state.preprocessed_df[col].map(mapping)
                            mapping_log[col] = mapping

                        pre_df = st.session_state.preprocessed_df

                        st.success(f"✅ Label Encoding applied to: {label_cols}")

                        # Show the mappings so the user knows what each number means
                        with st.expander("📖 View encoding mappings"):
                            for col, mapping in mapping_log.items():
                                st.write(f"**`{col}`:** {mapping}")

                    except Exception as e:
                        st.error(f"❌ Could not apply label encoding: {e}")

        # ---------------------------------------------------------
        # PREVIEW & SAVE
        # ---------------------------------------------------------
        st.divider()
        st.subheader("👁️ Preprocessed Data Preview")
        st.write(f"Shape: **{pre_df.shape[0]} rows × {pre_df.shape[1]} columns**")
        st.dataframe(pre_df.head(25))

        col_s1, col_s2 = st.columns(2)

        with col_s1:
            if st.button("💾 Save Preprocessed Data to Session"):
                st.session_state.df = st.session_state.preprocessed_df.copy()
                st.success("✅ Saved! Visualizations will use this preprocessed version.")

        with col_s2:
            export_format = st.radio(
                "Export format:",
                options=["CSV", "Excel", "JSON"],
                horizontal=True,
                key="pre_export_format"
            )

            if export_format == "CSV":
                data     = pre_df.to_csv(index=False).encode("utf-8")
                mime     = "text/csv"
                filename = "preprocessed_data.csv"
            elif export_format == "Excel":
                import io
                buffer = io.BytesIO()
                pre_df.to_excel(buffer, index=False, engine="openpyxl")
                data     = buffer.getvalue()
                mime     = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                filename = "preprocessed_data.xlsx"
            elif export_format == "JSON":
                data     = pre_df.to_json(orient="records", indent=2).encode("utf-8")
                mime     = "application/json"
                filename = "preprocessed_data.json"

            st.download_button(
                label=f"⬇️ Download as {export_format}",
                data=data,
                file_name=filename,
                mime=mime
            )
    
    # =========================================================
    # ⚖️ CLASS IMBALANCE HANDLING
    # =========================================================
    st.divider()
    st.subheader("⚖️ Class Imbalance Handling")

    if "balanced_df" not in st.session_state:
        st.session_state.balanced_df = st.session_state.df.copy() if st.session_state.df is not None else None

    bal_df = st.session_state.balanced_df

    if bal_df is None:
        st.warning("⚠️ Please save your preprocessed data to session first.")
    else:
        st.info(f"Working on: **{bal_df.shape[0]} rows × {bal_df.shape[1]} columns**")

        # ---------------------------------------------------------
        # STEP 1 — SELECT TARGET COLUMN & SHOW CLASS DISTRIBUTION
        # ---------------------------------------------------------
        st.markdown("#### 🎯 Step 1: Select Target Column")

        target_col = st.selectbox(
            "Select your target (label) column:",
            options=bal_df.columns.tolist(),
            key="balance_target"
        )

        # Show class distribution
        class_counts = bal_df[target_col].value_counts()
        st.write("**Current class distribution:**")

        dist_col1, dist_col2 = st.columns([2, 1])
        with dist_col1:
            fig, ax = plt.subplots(figsize=(7, 3))
            ax.bar(class_counts.index.astype(str), class_counts.values,
                   color=sns.color_palette("Blues_r", len(class_counts)),
                   edgecolor="white")
            ax.set_xlabel(target_col)
            ax.set_ylabel("Count")
            ax.set_title("Class Distribution")
            sns.despine()
            st.pyplot(fig)
        with dist_col2:
            st.dataframe(
                pd.DataFrame({
                    "Class": class_counts.index.astype(str),
                    "Count": class_counts.values,
                    "Percentage": (class_counts.values / len(bal_df) * 100).round(2)
                }).set_index("Class")
            )

        # Imbalance check
        imbalance_ratio = class_counts.max() / class_counts.min()
        if imbalance_ratio < 1.5:
            st.success(f"✅ Dataset looks balanced (ratio: {imbalance_ratio:.2f}). No action needed.")
        else:
            st.warning(f"⚠️ Imbalance detected! Majority/minority ratio: **{imbalance_ratio:.2f}**")

        st.divider()

        # ---------------------------------------------------------
        # STEP 2 — CHOOSE TECHNIQUE
        # ---------------------------------------------------------
        st.markdown("#### 🔧 Step 2: Choose Balancing Technique")

        technique = st.radio(
            "Select technique:",
            options=["None", "Random Undersampling", "SMOTE (Oversampling)"],
            horizontal=True,
            key="balance_technique"
        )

        # ---------------------------------------------------------
        # RANDOM UNDERSAMPLING
        # ---------------------------------------------------------
        if technique == "Random Undersampling":
            st.write("Randomly removes rows from majority classes to match the minority class count.")
            st.warning("⚠️ This reduces your dataset size. Make sure you have enough data.")

            minority_count = int(class_counts.min())
            target_count = st.slider(
                "Target count per class:",
                min_value=10,
                max_value=minority_count,
                value=minority_count,
                step=10,
                key="undersample_count"
            )

            if st.button("✅ Apply Random Undersampling"):
                try:
                    balanced_frames = []
                    for cls in bal_df[target_col].unique():
                        cls_df = bal_df[bal_df[target_col] == cls]
                        sampled = cls_df.sample(
                            n=min(target_count, len(cls_df)),
                            random_state=42
                        )
                        balanced_frames.append(sampled)

                    st.session_state.balanced_df = pd.concat(balanced_frames).sample(frac=1, random_state=42).reset_index(drop=True)
                    bal_df = st.session_state.balanced_df

                    st.success(f"✅ Undersampling applied! New shape: **{bal_df.shape[0]} rows × {bal_df.shape[1]} columns**")
                    st.write("**New class distribution:**")
                    st.dataframe(bal_df[target_col].value_counts().rename("Count"))
                except Exception as e:
                    st.error(f"❌ Could not apply undersampling: {e}")

        # ---------------------------------------------------------
        # SMOTE OVERSAMPLING
        # ---------------------------------------------------------
        elif technique == "SMOTE (Oversampling)":
            st.write("Generates synthetic samples for minority classes using interpolation between existing points.")
            st.info("ℹ️ SMOTE only works on **numeric columns**. Make sure encoding is done before this step.")

            try:
                from imblearn.over_sampling import SMOTE
                smote_available = True
            except ImportError:
                smote_available = False
                st.error("❌ `imbalanced-learn` is not installed. Add `imbalanced-learn` to your requirements.txt and restart.")

            if smote_available:
                feature_cols = [c for c in bal_df.columns if c != target_col]
                numeric_features = bal_df[feature_cols].select_dtypes(include=np.number).columns.tolist()
                non_numeric = [c for c in feature_cols if c not in numeric_features]

                if non_numeric:
                    st.warning(f"⚠️ These non-numeric columns will be excluded from SMOTE: {non_numeric}")

                if not numeric_features:
                    st.error("❌ No numeric feature columns found. Please apply encoding first.")
                else:
                    k_neighbors = st.slider(
                        "K neighbors (SMOTE sensitivity):",
                        min_value=1,
                        max_value=10,
                        value=5,
                        key="smote_k",
                        help="Higher = smoother synthetic samples. Lower = closer to real data points."
                    )

                    if st.button("✅ Apply SMOTE"):
                        try:
                            X = bal_df[numeric_features]
                            y = bal_df[target_col]

                            # Check minimum class size vs k_neighbors
                            min_class_size = class_counts.min()
                            if min_class_size <= k_neighbors:
                                st.error(f"❌ Minority class has only {min_class_size} samples, which is ≤ k_neighbors ({k_neighbors}). Reduce k or get more data.")
                            else:
                                smote = SMOTE(k_neighbors=k_neighbors, random_state=42)
                                X_res, y_res = smote.fit_resample(X, y)

                                resampled_df = pd.DataFrame(X_res, columns=numeric_features)
                                resampled_df[target_col] = y_res

                                st.session_state.balanced_df = resampled_df
                                bal_df = st.session_state.balanced_df

                                st.success(f"✅ SMOTE applied! New shape: **{bal_df.shape[0]} rows × {bal_df.shape[1]} columns**")
                                st.write("**New class distribution:**")
                                st.dataframe(bal_df[target_col].value_counts().rename("Count"))
                        except Exception as e:
                            st.error(f"❌ Could not apply SMOTE: {e}")

        # ---------------------------------------------------------
        # PREVIEW & SAVE
        # ---------------------------------------------------------
        st.divider()
        st.subheader("👁️ Balanced Data Preview")
        st.write(f"Shape: **{bal_df.shape[0]} rows × {bal_df.shape[1]} columns**")
        st.dataframe(bal_df.head(25))

        col_b1, col_b2 = st.columns(2)

        with col_b1:
            if st.button("💾 Save Balanced Data to Session"):
                st.session_state.df = st.session_state.balanced_df.copy()
                st.success("✅ Saved! This version will be used for modeling.")

        with col_b2:
            export_format = st.radio(
                "Export format:",
                options=["CSV", "Excel", "JSON"],
                horizontal=True,
                key="bal_export_format"
            )

            if export_format == "CSV":
                data     = bal_df.to_csv(index=False).encode("utf-8")
                mime     = "text/csv"
                filename = "balanced_data.csv"
            elif export_format == "Excel":
                import io
                buffer = io.BytesIO()
                bal_df.to_excel(buffer, index=False, engine="openpyxl")
                data     = buffer.getvalue()
                mime     = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                filename = "balanced_data.xlsx"
            elif export_format == "JSON":
                data     = bal_df.to_json(orient="records", indent=2).encode("utf-8")
                mime     = "application/json"
                filename = "balanced_data.json"

            st.download_button(
                label=f"⬇️ Download as {export_format}",
                data=data,
                file_name=filename,
                mime=mime
            )

    
