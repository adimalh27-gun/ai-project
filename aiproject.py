import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==========================================================================
# PAGE CONFIGURATION
# ==========================================================================
st.set_page_config(
    page_title="AI Job Market Analysis Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
<style>
...
</style>
""", unsafe_allow_html=True)

st.markdown("""
<h1 style="font-size:70px;font-weight:800;">
AI Job Market Analysis Dashboard
</h1>
""", unsafe_allow_html=True)
st.markdown("---")

# ==========================================================================
# EXECUTIVE SUMMARY
# ==========================================================================
st.markdown("# 📋Executive Summary")
st.markdown("""
This project analyzes a global **AI Job Market dataset** covering thousands of AI, Machine Learning, and Data 
Science job postings across the world. The analysis explores salary trends, experience requirements, industry 
demand, company characteristics, and hiring patterns over time in the rapidly growing field of Artificial 
Intelligence employment.

 #### **🎯Key Objectives:**
- Identify which AI job roles command the highest salaries
- Understand how experience level, employment type, and education affect compensation
- Explore how AI jobs are distributed across industries and company sizes
- Examine the relationship between years of experience and salary
- Discover which countries offer the most AI job opportunities
- Track how AI job postings have trended over time

#### **Expected Deliverables:**
- Interactive Plotly visualizations covering distribution, comparison, hierarchy, correlation, and trend analysis
- Statistical summaries of key salary and experience indicators
- Actionable insights for job seekers, employers, and workforce researchers
""")

st.markdown("---")

# ==========================================================================
# PROJECT DESCRIPTION
# ==========================================================================
st.markdown("# Project Description")

st.markdown("### ⚠️Problem Statement")
st.markdown("""
The Artificial Intelligence job market is expanding rapidly, with new roles, skill requirements, and compensation 
structures emerging across industries and regions. This project leverages a comprehensive AI jobs dataset to 
provide data-driven insights that can help:

- **Job Seekers**: Understand which roles, skills, and experience levels lead to higher salaries
- **Employers & HR Teams**: Benchmark compensation and understand competitive hiring markets
- **Workforce Researchers**: Study global trends in AI talent demand and remote work adoption
- **Policy Makers & Educators**: Understand skill and education demands shaping the AI workforce
""")

st.markdown("### 📂 Dataset Overview")
st.markdown("""
The AI Job Market dataset contains detailed information about AI-related job postings, including:

**Job Information:**
- Job title, industry, and required skills
- Experience level (Entry, Mid, Senior, Executive) and employment type (Full-time, Part-time, Contract, Freelance)

**Compensation Data:**
- Salary in USD and original currency
- Benefits score and remote work ratio

**Company Information:**
- Company name, size, and location
- Employee residence (for remote-work comparison)

**Candidate Requirements:**
- Years of experience required
- Education level required

**Posting Timeline:**
- Job posting date and application deadline
- Job description length
""")

st.markdown("---")

# ==========================================================================
# DATA OVERVIEW SECTION
# ==========================================================================
st.markdown("# Data Overview")


@st.cache_data
def load_data():
    """Load the AI Job Market dataset with caching"""
    try:
        df = pd.read_excel('ai_job_dataset.xlsx')
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None


@st.cache_data
def prepare_dataframe_for_display(df, max_string_length=100):
    """Prepare dataframe/series for display, avoiding Arrow serialization issues"""
    if df is None:
        return df

    if isinstance(df, pd.Series):
        df = df.to_frame()

    if df.empty:
        return df

    try:
        df_display = df.copy()
        for col in df_display.columns:
            col_dtype = df_display[col].dtype

            if col_dtype == 'object' or col_dtype == 'O' or pd.api.types.is_object_dtype(df_display[col]):
                df_display[col] = df_display[col].astype(str)
                df_display[col] = df_display[col].replace(['nan', 'None', 'NaN', '<NA>', 'null'], '')

                mask = df_display[col].str.len() > max_string_length
                if mask.any():
                    df_display.loc[mask, col] = df_display.loc[mask, col].str[:max_string_length] + '...'

            elif pd.api.types.is_numeric_dtype(df_display[col]):
                if df_display[col].dtype == 'object':
                    df_display[col] = pd.to_numeric(df_display[col], errors='coerce')

        return df_display

    except Exception as e:
        st.warning(f"DataFrame conversion warning: {str(e)}. Using fallback conversion.")
        try:
            df_fallback = df.copy()
            for col in df_fallback.columns:
                df_fallback[col] = df_fallback[col].astype(str)
            return df_fallback
        except Exception:
            return pd.DataFrame(columns=[str(col) for col in df.columns]).astype(str)


def safe_dataframe_display(df, use_container_width=True, **kwargs):
    if df is not None and not df.empty:
        prepared_df = prepare_dataframe_for_display(df)
        return st.dataframe(prepared_df, use_container_width=use_container_width, **kwargs)
    else:
        return st.info("No data to display")


# Load the data
df = load_data()

if df is not None:
    st.subheader("📊 Dataset Basic Information")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Records", f"{df.shape[0]:,}")
    with col2:
        st.metric("Total Columns", df.shape[1])
    with col3:
        st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    with col4:
        st.metric("Unique Job Titles", df['job_title'].nunique())

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📋 Column Info",
        "❌ Missing Values",
        "👀 Sample Data",
        "📊 Statistics",
        "🔤 Categorical Data",
        "✅ Data Quality"
    ])

    with tab1:
        st.subheader("📋 Column Information")
        col_info = pd.DataFrame({
            'Column Name': df.columns,
            'Data Type': df.dtypes.astype(str),
            'Non-Null Count': df.count(),
            'Null Count': df.isnull().sum(),
            'Null Percentage': (df.isnull().sum() / len(df) * 100).round(2)
        })
        st.dataframe(prepare_dataframe_for_display(col_info), use_container_width=True)

        st.subheader("Data Types Summary")
        dtype_summary = df.dtypes.value_counts()
        dtype_summary_df = pd.DataFrame({
            'Data Type': dtype_summary.index.astype(str),
            'Count': dtype_summary.values
        })
        st.dataframe(prepare_dataframe_for_display(dtype_summary_df), use_container_width=True)

        st.subheader("Detailed Column Descriptions")
        st.markdown("""
        **Column Descriptions:**
        - **job_id**: Unique identifier for each job posting
        - **job_title**: Title of the AI-related job role
        - **salary_usd**: Annual salary converted to USD
        - **salary_currency**: Original currency of the salary
        - **experience_level**: EN (Entry), MI (Mid), SE (Senior), EX (Executive)
        - **employment_type**: FT (Full-time), PT (Part-time), CT (Contract), FL (Freelance)
        - **company_location**: Country where the hiring company is based
        - **company_size**: S (Small), M (Medium), L (Large)
        - **employee_residence**: Country where the employee resides
        - **remote_ratio**: Percentage of remote work (0, 50, 100)
        - **required_skills**: Comma-separated list of required skills
        - **education_required**: Minimum education required for the role
        - **years_experience**: Years of professional experience required
        - **industry**: Industry sector of the hiring company
        - **posting_date**: Date the job was posted
        - **application_deadline**: Deadline to apply for the job
        - **job_description_length**: Character length of the job description
        - **benefits_score**: Score reflecting quality of benefits offered (0-10)
        - **company_name**: Name of the hiring company
        """)

    with tab2:
        st.subheader(" ❌ Missing Values Analysis")
        missing_data = df.isnull().sum().sort_values(ascending=False)
        missing_data = missing_data[missing_data > 0]

        if not missing_data.empty:
            st.write("Columns with missing values:")
            missing_df = pd.DataFrame({
                'Column': missing_data.index,
                'Missing Count': missing_data.values,
                'Missing Percentage': (missing_data.values / len(df) * 100).round(2)
            })
            st.dataframe(prepare_dataframe_for_display(missing_df), use_container_width=True)
            st.subheader("Missing Values Visualization")
            st.bar_chart(missing_df.set_index('Column')['Missing Percentage'])
        else:
            st.success("No missing values found in the dataset! The data is fully complete.")

    with tab3:
        st.subheader(" 👀 Sample Data")
        st.write("First 10 rows of the dataset:")
        st.dataframe(prepare_dataframe_for_display(df.head(10)), use_container_width=True)

        st.subheader("Random Sample")
        st.write("10 random rows from the dataset:")
        st.dataframe(prepare_dataframe_for_display(df.sample(10, random_state=42)), use_container_width=True)

        st.subheader("Last 5 Rows")
        st.dataframe(prepare_dataframe_for_display(df.tail(5)), use_container_width=True)

    with tab4:
        st.subheader(" 📊 Statistical Summary")

        numerical_cols = df.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            st.write("**Numerical Columns Statistics:**")
            st.dataframe(prepare_dataframe_for_display(df[numerical_cols].describe()), use_container_width=True)

        object_cols = df.select_dtypes(include=['object']).columns
        if len(object_cols) > 0:
            st.write("**Text/Object Columns Statistics:**")
            st.dataframe(prepare_dataframe_for_display(df[object_cols].describe()), use_container_width=True)

    with tab5:
        st.subheader(" 🔤 Categorical Data Analysis")
        categorical_cols = ['experience_level', 'employment_type', 'company_size',
                             'education_required', 'industry', 'company_location']

        for col in categorical_cols:
            if col in df.columns:
                unique_count = df[col].nunique()
                st.write(f"**{col}**: {unique_count} unique values")

                if unique_count <= 20:
                    value_counts = df[col].value_counts()
                    st.dataframe(prepare_dataframe_for_display(value_counts.reset_index()), use_container_width=True)
                else:
                    value_counts = df[col].value_counts().head(10)
                    st.write("Top 10 most frequent values:")
                    st.dataframe(prepare_dataframe_for_display(value_counts.reset_index()), use_container_width=True)
                st.write("---")

    with tab6:
        st.subheader("✅ Data Quality Assessment")

        duplicate_count = df.duplicated().sum()
        st.metric("Duplicate Rows", duplicate_count)

        issues = []
        if df['salary_usd'].isna().sum() > 0:
            issues.append(f"Salary column has {df['salary_usd'].isna().sum()} missing values")
        if df['years_experience'].isna().sum() > 0:
            issues.append(f"Years of experience column has {df['years_experience'].isna().sum()} missing values")

        negative_salaries = (df['salary_usd'] < 0).sum()
        if negative_salaries > 0:
            issues.append(f"{negative_salaries} rows have negative salary values")

        st.subheader("Data Quality Issues")
        if issues:
            st.write("**Potential data quality issues identified:**")
            for issue in issues:
                st.write(f"- {issue}")
        else:
            st.success("No major data quality issues detected! The dataset is clean and analysis-ready.")

        st.subheader("Additional Quality Metrics")
        col1, col2, col3 = st.columns(3)
        with col1:
            total_nulls = df.isnull().sum().sum()
            st.metric("Total Null Values", total_nulls)
        with col2:
            completeness = ((df.size - total_nulls) / df.size * 100)
            st.metric("Data Completeness", f"{completeness:.2f}%")
        with col3:
            unique_rows = df.shape[0] - duplicate_count
            st.metric("Unique Rows", unique_rows)

else:
    st.error("Could not load the dataset. Please ensure 'ai_job_dataset.xlsx' is in the correct directory.")

st.markdown("---")

# ==========================================================================
# DATA CLEANING & PREPROCESSING
# ==========================================================================
st.markdown("#  Data Cleaning & Preprocessing")

if df is not None:
    st.markdown("""
    This section handles missing values, standardizes categorical labels for readability, and prepares
    date fields for time-based analysis. Data cleaning ensures the quality and reliability of our analysis.
    """)

    @st.cache_data
    def clean_data(df):
        """Clean and enrich the AI job dataset"""
        df_cleaned = df.copy()

        # Standardize date columns
        for date_col in ['posting_date', 'application_deadline']:
            if date_col in df_cleaned.columns:
                df_cleaned[date_col] = pd.to_datetime(df_cleaned[date_col], errors='coerce')

        # Fill missing numeric values with median
        numeric_cols = ['salary_usd', 'years_experience', 'benefits_score', 'job_description_length', 'remote_ratio']
        for col in numeric_cols:
            if col in df_cleaned.columns and df_cleaned[col].isnull().sum() > 0:
                df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())

        # Fill missing categorical values with 'Unknown'
        categorical_cols = ['job_title', 'experience_level', 'employment_type', 'company_location',
                             'company_size', 'education_required', 'industry', 'company_name']
        for col in categorical_cols:
            if col in df_cleaned.columns:
                df_cleaned[col] = df_cleaned[col].fillna('Unknown')

        # Human-readable labels for coded columns
        experience_map = {'EN': 'Entry-level', 'MI': 'Mid-level', 'SE': 'Senior-level', 'EX': 'Executive-level'}
        employment_map = {'FT': 'Full-time', 'PT': 'Part-time', 'CT': 'Contract', 'FL': 'Freelance'}
        company_size_map = {'S': 'Small', 'M': 'Medium', 'L': 'Large'}

        df_cleaned['experience_level_label'] = df_cleaned['experience_level'].map(experience_map).fillna(df_cleaned['experience_level'])
        df_cleaned['employment_type_label'] = df_cleaned['employment_type'].map(employment_map).fillna(df_cleaned['employment_type'])
        df_cleaned['company_size_label'] = df_cleaned['company_size'].map(company_size_map).fillna(df_cleaned['company_size'])

        # Posting month for trend analysis
        df_cleaned['posting_month'] = df_cleaned['posting_date'].dt.to_period('M').dt.to_timestamp()

        # Remove exact duplicate rows
        df_cleaned = df_cleaned.drop_duplicates()

        return df_cleaned

    df_cleaned = clean_data(df)

    st.subheader("Missing Values Treatment")
    st.write("""
    - Numeric fields (salary, years of experience, benefits score) → filled with the column median
    - Categorical fields → filled with 'Unknown' where necessary
    - Duplicate rows were removed
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Rows Before Cleaning", f"{df.shape[0]:,}")
    with col2:
        st.metric("Rows After Cleaning", f"{df_cleaned.shape[0]:,}")

    st.subheader("Readable Category Labels Added")
    st.write("""
    To make charts easier to read, coded columns were mapped to descriptive labels:
    - `experience_level` (EN/MI/SE/EX) → `experience_level_label` (Entry/Mid/Senior/Executive)
    - `employment_type` (FT/PT/CT/FL) → `employment_type_label` (Full-time/Part-time/Contract/Freelance)
    - `company_size` (S/M/L) → `company_size_label` (Small/Medium/Large)
    """)
    st.dataframe(
        prepare_dataframe_for_display(
            df_cleaned[['experience_level', 'experience_level_label', 'employment_type',
                        'employment_type_label', 'company_size', 'company_size_label']].head(5)
        ),
        use_container_width=True
    )

    st.subheader("Outlier Detection — Salary")
    q1 = df_cleaned['salary_usd'].quantile(0.25)
    q3 = df_cleaned['salary_usd'].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = df_cleaned[(df_cleaned['salary_usd'] < lower_bound) | (df_cleaned['salary_usd'] > upper_bound)]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Salary IQR Range", f"${lower_bound:,.0f} – ${upper_bound:,.0f}")
    with col2:
        st.metric("Detected Outliers", f"{len(outliers):,}")
    with col3:
        st.metric("Outlier Percentage", f"{len(outliers) / len(df_cleaned) * 100:.2f}%")

    st.write("""
    Outliers are retained in the dataset since very high or low salaries are legitimate for executive roles
    or entry-level/freelance positions, rather than data entry errors.
    """)

    st.subheader("Final Processed Dataset Summary")
    st.dataframe(prepare_dataframe_for_display(df_cleaned.describe(include='all').transpose().head(20)), use_container_width=True)

else:
    st.error("Dataset not available for cleaning.")

st.markdown("---")
# ==========================================================
# DATA VISUALIZATION & INSIGHTS
# ==========================================================

st.title("📊 Data Visualization & Insights")

@st.cache_data
def load_clean_data():
    try:
        df = pd.read_excel("ai_job_dataset.xlsx")
        return df
    except Exception as e:
        st.error(e)
        return None


cleaned_df = load_clean_data()

if cleaned_df is not None:

    experience_map = {
        "EN": "Entry Level",
        "MI": "Mid Level",
        "SE": "Senior Level",
        "EX": "Executive Level"
    }

    employment_map = {
        "FT": "Full-Time",
        "PT": "Part-Time",
        "CT": "Contract",
        "FL": "Freelance"
    }


    company_size_map = {
        "S": "Small",
        "M": "Medium",
        "L": "Large"
    }

    st.sidebar.title("AI Job Data Filters")
    st.sidebar.markdown("""
    ---
    Use the filters below to explore the AI job market.

    💡 **Tip:** Select your filters and click "Apply Filters" to update the dashboard.
    """)
    st.sidebar.markdown("---")

    if "filtered_df" not in st.session_state:
        st.session_state.filtered_df = cleaned_df.copy()

    if "form_id" not in st.session_state:
        st.session_state.form_id = 0

    with st.sidebar.form(f"filter_form_{st.session_state.form_id}"):
        experience = st.multiselect(
            "Experience Level",
            options=list(experience_map.keys()),
            format_func=lambda x: experience_map[x]
        )

        employment = st.multiselect(
            "Employment Type",
            options=list(employment_map.keys()),
            format_func=lambda x: employment_map[x]
        )

        industry = st.multiselect(
            "Industry",
            options=sorted(cleaned_df["industry"].dropna().unique())
        )

        company_size = st.multiselect(
            "Company Size",
            options=list(company_size_map.keys()),
            format_func=lambda x: company_size_map[x]
        )

        location = st.multiselect(
            "Company Location",
            options=sorted(cleaned_df["company_location"].dropna().unique())
        )
        salary_range = st.slider(
        "Salary Range (USD)",
        min_value=int(cleaned_df["salary_usd"].min()),
        max_value=int(cleaned_df["salary_usd"].max()),
        value=(
            int(cleaned_df["salary_usd"].min()),
            int(cleaned_df["salary_usd"].max())
        )
    )
            
        col1, col2 = st.columns(2)

        with col1:
            apply = st.form_submit_button("✅ Apply Filters", use_container_width=True)

        with col2:
            reset = st.form_submit_button("🔄 Reset Filters", use_container_width=True)

    if apply:

        filtered_df = cleaned_df.copy()

        if experience:
            filtered_df = filtered_df[
                filtered_df["experience_level"].isin(experience)
            ]

        if employment:
            filtered_df = filtered_df[
                filtered_df["employment_type"].isin(employment)
            ]
       

        if industry:
            filtered_df = filtered_df[
                filtered_df["industry"].isin(industry)
            ]

        if company_size:
            filtered_df = filtered_df[
                filtered_df["company_size"].isin(company_size)
            ]

        if location:
            filtered_df = filtered_df[
                filtered_df["company_location"].isin(location)
            ]
        filtered_df = filtered_df[
        (filtered_df["salary_usd"] >= salary_range[0]) &
        (filtered_df["salary_usd"] <= salary_range[1])
    ]

        st.session_state.filtered_df = filtered_df

    if reset:
        st.session_state.filtered_df = cleaned_df.copy()
        st.session_state.form_id += 1
        st.rerun()

    filtered_df = st.session_state.filtered_df
   

    if filtered_df.empty:
        st.warning("No data available after applying filters.")
        st.stop()

    filtered_df = filtered_df.copy()

    filtered_df["experience_level_label"] = filtered_df["experience_level"].map(experience_map)
    filtered_df["employment_type_label"] = filtered_df["employment_type"].map(employment_map)
    filtered_df["company_size_label"] = filtered_df["company_size"].map(company_size_map)

 
    # ================= Current Status =================
    st.sidebar.markdown("---")
    st.sidebar.subheader("Current Status")

    st.sidebar.metric("Total Jobs", f"{len(cleaned_df):,}")

    if len(filtered_df) == len(cleaned_df):
        st.sidebar.warning("Filters not applied yet")
        st.sidebar.info("Click '✅ Apply Filters' to update charts.")
    else:
        st.sidebar.metric("Jobs Showing", f"{len(filtered_df):,}")

        percentage = (len(filtered_df) / len(cleaned_df)) * 100
        st.sidebar.metric("Percentage Shown", f"{percentage:.1f}%")

        active_filters = sum([
            bool(experience),
            bool(employment),
            bool(industry),
            bool(company_size),
            bool(location)
        ])

        st.sidebar.success(f"✅ {active_filters} filter(s) active")
    
    # ==================================================================
    # CHART 1: BAR CHART — Job Title vs Average Salary
    # ==================================================================
    st.subheader("1. Which AI Job Roles Pay the Most?")
    st.write("**Chart Type:** Bar Chart | `px.bar()`")
    

    avg_salary_by_title = (
        filtered_df.groupby('job_title')['salary_usd']
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )

    fig1 = px.bar(
        avg_salary_by_title,
        x='job_title',
        y='salary_usd',
        title="Average Salary (USD) by AI Job Title",
        labels={'job_title': 'Job Title', 'salary_usd': 'Average Salary (USD)'},
        color='salary_usd',
        color_continuous_scale='viridis'
    )

    fig1.update_layout(height=550, xaxis_tickangle=-45, showlegend=False,yaxis_title=" Average Salary (USD)",
    yaxis_title_font=dict(size=20),
    xaxis_title="Job title",
    xaxis_title_font=dict(size=20),
    title_x=0.3,
    title_font=dict(size=22),)
    st.plotly_chart(fig1, use_container_width=True)
    st.markdown("#### **Key Insights:**")
    top_role = avg_salary_by_title.iloc[0]
    bottom_role = avg_salary_by_title.iloc[-1]
    st.write(f"- **{top_role['job_title']}** is the highest-paying role, averaging **${top_role['salary_usd']:,.0f}**")
    st.write(f"- **{bottom_role['job_title']}** has the lowest average salary at **${bottom_role['salary_usd']:,.0f}**")
    st.write(f"- The salary gap between the highest and lowest paying roles is **${top_role['salary_usd'] - bottom_role['salary_usd']:,.0f}**")

    st.markdown("---")


    # ==================================================================
    # CHART 2: HISTOGRAM — Salary Distribution
    # ==================================================================
    st.subheader("2. Distribution of Salaries Across AI Jobs")
    st.write("**Chart Type:** Histogram | `px.histogram()`")

    fig2 = px.histogram(
        filtered_df,
        x='salary_usd',
        nbins=40,
        title="Distribution of Salary (USD)",
        labels={'salary_usd': 'Salary (USD)', 'count': 'Count'},
        color_discrete_sequence=['#636EFA']
    )
    fig2.update_layout(height=500, bargap=0.05,xaxis_title="Salary (USD)",
     xaxis_title_font=dict(size=20),yaxis_title="Count",yaxis_title_font=dict(size=20),title_x=0.3,
    title_font=dict(size=24))
   
    fig2.add_vline(x=filtered_df['salary_usd'].mean(), line_dash="dash", line_color="red",
                    annotation_text="Mean", annotation_position="top")
    fig2.add_vline(x=filtered_df['salary_usd'].median(), line_dash="dash", line_color="green",
                    annotation_text="Median", annotation_position="bottom")
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown(" #### **Key Insights:**")
    st.write("- Most AI job salaries are concentrated between **$50K and $100K**.")
    st.write("- The salary distribution is right-skewed, with a few high-paying outliers.")
    st.write("- The mean salary is higher than the median, indicating the impact of high salaries.")
    

    st.markdown("---")

    # ==================================================================
    # CHART 3: BOX PLOT — Experience Level vs Salary
    # ==================================================================
    st.subheader("3. Salary Spread and Outliers by Experience Level")
    st.write("**Chart Type:** Box Plot | `px.box()`")

    exp_order = ['Entry-level', 'Mid-level', 'Senior-level', 'Executive-level']
    exp_order = [e for e in exp_order if e in filtered_df['experience_level'].unique()]

    fig3 = px.box(
        filtered_df,
        x='experience_level',
        y='salary_usd',
        title="Salary Distribution by Experience Level",
        labels={'experience_level': 'Experience Level', 'salary_usd': 'Salary (USD)'},
        category_orders={'experience_level': exp_order},
        color='experience_level',
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig3.update_layout(height=550, showlegend=False,
    xaxis_title="Experience Level",
    xaxis_title_font=dict(size=20),
    yaxis_title="Salary (USD)",
    yaxis_title_font=dict(size=20),title_x=0.3,
    title_font=dict(size=22))
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown(" #### **Key Insights:**")
    st.write("- Executive-level (EX) professionals earn the highest salaries, while Entry-level (EN) professionals earn the lowest.")
    st.write("- Salary generally increases as experience level increases.")
    st.write("-  Executive and Senior roles show greater salary variation than Mid and Entry-level roles.")
    

    
    st.markdown("---")

    # ==================================================================
    # CHART 4: VIOLIN PLOT — Employment Type vs Salary
    # ==================================================================
    st.subheader("4. Salary Distribution by Employment Type")
    st.write("**Chart Type:** Violin Plot | `px.violin()`")

    fig4 = px.violin(
    filtered_df,
    x='employment_type_label',
    y='salary_usd',
    title="Salary Distribution: Full-time, Part-time, Contract, Freelance",
    labels={
        'employment_type_label': 'Employment Type',
        'salary_usd': 'Salary (USD)'
    },
    box=True,
    color='employment_type_label',
    color_discrete_sequence=px.colors.qualitative.Set3
)
    fig4.update_layout(height=550, showlegend=False,
    yaxis_title="Salary (USD)",
    yaxis_title_font=dict(size=20),
    xaxis_title="Employment Type",
    xaxis_title_font=dict(size=20),
    title_x=0.2,
    title_font=dict(size=22))
    st.plotly_chart(fig4, use_container_width=True)

    st.markdown("#### **Key Insights:**")
    median_by_emp = filtered_df.groupby('employment_type_label')['salary_usd'].median().sort_values(ascending=False)
    st.write(f"- **{median_by_emp.index[0]}** roles have the highest median salary at **${median_by_emp.iloc[0]:,.0f}**")
    st.write(f"- **{median_by_emp.index[-1]}** roles have the lowest median salary at **${median_by_emp.iloc[-1]:,.0f}**")
    st.write(" - **Contract, freelance, and part-time roles have comparable salary distributions with moderate variation.**")


    st.markdown("---")

    # ==================================================================
    # CHART 5: TREEMAP — Industry -> Experience Level
    # ==================================================================
    st.subheader("5. Which Industries Have the Most AI Jobs?")
    st.write("**Chart Type:** Treemap | `px.treemap()`")
    treemap_df = (filtered_df.groupby(["industry", "experience_level"], as_index=False).agg(Count=("salary_usd", "size"),Average_Salary=("salary_usd", "mean") ))  
    fig5 = px.treemap(
    treemap_df,
    path=["industry", "experience_level"],
    values="Count",
    color="Average_Salary",
    title="Job Distribution across Industries and Experience Levels",
   )
    fig5.update_layout(
        width=1000,
        height=800,
        margin=dict(t=60, l=10, r=10, b=10),
        title_x=0.2,
        title_font=dict(size=22)
    
    )
    fig5.update_traces( marker=dict(
            line=dict(color="white", width=2) ))
    st.plotly_chart(fig5,use_container_width=True)

    st.markdown("#### **Key Insights:**")
    st.write(" - **Executive (EX) roles generally have the highest average salaries across most industries.**")
    st.write("- **Average salaries increase with experience level, from Entry (EN) to Executive (EX).**")
    st.write(" - **Salary differences exist across industries, showing that both industry and experience influence AI job salaries.**")

    st.markdown("---")

    # ==================================================================
    # CHART 6: SUNBURST — Industry -> Experience Level -> Employment Type
    # ==================================================================
    st.subheader("6. Hierarchical View of AI Job Distribution")
    st.write("**Chart Type:** Sunburst Chart | `px.sunburst()`")

    # Limit to the top 8 industries by job count so the chart stays readable
    top_industries = filtered_df['industry'].value_counts().head(8).index.tolist()
    sunburst_source =filtered_df[filtered_df['industry'].isin(top_industries)]

    sunburst_data = sunburst_source.groupby(
        ['industry', 'experience_level_label', 'employment_type_label']
    ).size().reset_index(name='count')

    fig6 = px.sunburst(
        sunburst_data,
        path=['industry', 'experience_level_label', 'employment_type_label'],
        values='count',
        title="AI Jobs: Industry → Experience Level → Employment Type (Top 8 Industries)",
        color='industry',
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig6.update_layout(
    title_font=dict(size=22),
    height=650,
    width=850,
    title_x=0.2,
    margin=dict(t=80, l=20, r=20, b=20)
     )

    fig6.update_traces(
     textinfo="label+percent parent",
    insidetextorientation="radial",
    maxdepth=2
)  
    st.plotly_chart(fig6,use_container_width=True)
    st.write(" #### **Key Insights:**")
    st.write("- Each ring represents a deeper level of segmentation: industry, then experience level, then employment type")
    top_industry_seg = sunburst_data.groupby('industry')['count'].sum().idxmax()
    st.write(f"- **{top_industry_seg}** contains the largest share of job postings in this hierarchical breakdown")
    st.write("- This view helps identify which experience/employment combinations dominate within each industry")

    st.markdown("---")

    # ==================================================================
    # CHART 7: SCATTER PLOT — Years of Experience vs Salary
    # ==================================================================
    st.subheader("7. Relationship Between Experience and Salary")
    st.write("**Chart Type:** Scatter Plot | `px.scatter()`")

    fig7 = px.scatter(
        filtered_df,
        x='years_experience',
        y='salary_usd',
        color='experience_level_label',
        title="Years of Experience vs Salary (USD)",
        labels={'years_experience': 'Years of Experience', 'salary_usd': 'Salary (USD)'},
        opacity=0.6,
        color_discrete_sequence=px.colors.qualitative.Set1
    )
    fig7.update_layout(height=600,
    yaxis_title="Salary (USD)",
    yaxis_title_font=dict(size=20),
    xaxis_title="Years of Experience",
    xaxis_title_font=dict(size=20),
    title_x=0.3,
    title_font=dict(size=22))
    st.plotly_chart(fig7, use_container_width=True)

    st.write("#### **Key Insights:**")

    correlation = filtered_df["years_experience"].corr(filtered_df["salary_usd"])

    st.write(f"- Correlation between years of experience and salary is **{correlation:.2f}**")

    corr_strength = (
        "strong" if abs(correlation) > 0.6
        else "moderate" if abs(correlation) > 0.3
        else "weak"
    )

    st.write(f"- This indicates a **{corr_strength}** positive relationship between experience and pay")
    st.write("- **Salary variation becomes wider with higher experience, indicating greater earning potential for experienced professionals.**")
    st.markdown("---")

    # ==================================================================
    # CHART 8: DENSITY HEATMAP — Experience Level vs Company Size
    # ==================================================================
    st.subheader("8. Experience Levels Across Company Sizes")
    st.write("**Chart Type:** Density Heatmap | `px.density_heatmap()`")

    size_order = ['Small', 'Medium', 'Large']
    size_order = [s for s in size_order if s in filtered_df['company_size_label'].unique()]

    fig8 = px.density_heatmap(
        filtered_df,
        x='experience_level_label',
        y='company_size_label',
        title="Experience Level Distribution by Company Size",
        labels={'experience_level_label': 'Experience Level', 'company_size_label': 'Company Size'},
        category_orders={'experience_level_label': exp_order, 'company_size_label': size_order},
        color_continuous_scale='Viridis',
        text_auto=True
    )
    fig8.update_layout(height=500,
    yaxis_title="Company Size",
    yaxis_title_font=dict(size=20),
    xaxis_title="Experience Level",
    xaxis_title_font=dict(size=20),
    title_x=0.3,
    title_font=dict(size=22))
    st.plotly_chart(fig8, use_container_width=True)

    st.write("#### **Key Insights:**")
    crosstab = pd.crosstab(filtered_df['experience_level_label'],filtered_df['company_size_label'])
    max_combo = crosstab.stack().idxmax()
    st.write(f"- The most common combination is **{max_combo[0]}** experience at **{max_combo[1]}** companies (**{crosstab.stack().max():,}** postings)")
    st.write("- This heatmap highlights which experience levels are prioritized by companies of different sizes")
    st.write("- **Hiring demand is relatively balanced across company sizes, with only small differences in the number of job postings.**")

    st.markdown("---")

    # ==================================================================
    # CHART 9: CHOROPLETH MAP — Company Location vs Number of Jobs
    # ==================================================================
    st.subheader("9. Countries with the Highest AI Job Opportunities")
    st.write("**Chart Type:** Choropleth Map | `px.choropleth()`")

    jobs_by_country = filtered_df['company_location'].value_counts().reset_index()
    jobs_by_country.columns = ['company_location', 'job_count']

    # Explicit ISO-3 country code mapping ensures every country renders correctly on the map
    iso3_map = {
        'Australia': 'AUS', 'Austria': 'AUT', 'Canada': 'CAN', 'China': 'CHN',
        'Denmark': 'DNK', 'Finland': 'FIN', 'France': 'FRA', 'Germany': 'DEU',
        'India': 'IND', 'Ireland': 'IRL', 'Israel': 'ISR', 'Japan': 'JPN',
        'Netherlands': 'NLD', 'Norway': 'NOR', 'Singapore': 'SGP', 'South Korea': 'KOR',
        'Sweden': 'SWE', 'Switzerland': 'CHE', 'United Kingdom': 'GBR', 'United States': 'USA'
    }
    jobs_by_country['iso_alpha'] = jobs_by_country['company_location'].map(iso3_map)

    fig9 = px.choropleth(
        jobs_by_country,
        locations='iso_alpha',
        color='job_count',
        hover_name='company_location',
        title="Number of AI Job Postings by Company Location",
        labels={'job_count': 'Number of Jobs'},
        color_continuous_scale='Plasma'
    )
    fig9.update_layout(height=550, geo=dict(showframe=False, showcoastlines=True, projection_type='natural earth'),title_x=0.2,
    title_font=dict(size=22))
    st.plotly_chart(fig9, use_container_width=True)

    st.write("#### **Key Insights:**")
    st.write(f"- **{jobs_by_country.iloc[0]['company_location']}** has the most AI job postings with **{jobs_by_country.iloc[0]['job_count']:,}** jobs")
    st.write(f"- AI job postings span **{jobs_by_country.shape[0]}** countries in this dataset")
    top5_share = jobs_by_country.head(5)['job_count'].sum() / jobs_by_country['job_count'].sum() * 100
    st.write(f"- The top 5 countries account for **{top5_share:.1f}%** of all AI job postings")

    st.markdown("---")

    # ==================================================================
    # CHART 10: LINE CHART — Job Postings Over Time
    # ==================================================================
    st.subheader("10. Trend of AI Job Postings Over Time")
    st.write("**Chart Type:** Timeline (Line Chart) | `px.line()`")

    # Ensure date columns are in datetime format
    filtered_df["posting_date"] = pd.to_datetime(
        filtered_df["posting_date"],
        errors="coerce"
    )

    # Create posting_month if it doesn't exist
    filtered_df["posting_month"] = (
        filtered_df["posting_date"]
        .dt.to_period("M")
        .dt.to_timestamp()
    )

    postings_over_time = (
        filtered_df.groupby("posting_month")
        .size()
        .reset_index(name="job_postings")
        .sort_values("posting_month")
    )

    fig10 = px.line(
        postings_over_time,
        x="posting_month",
        y="job_postings",
        title="Number of AI Job Postings Over Time",
        labels={
            "posting_month": "Posting Date (Month)",
            "job_postings": "Number of Job Postings"
        },
        markers=True
    )

    fig10.update_layout(height=500,
    yaxis_title="Number of Job Postings",
    yaxis_title_font=dict(size=20),
    xaxis_title="Posting Date (Month)",
    xaxis_title_font=dict(size=20),
    title_x=0.3,
    title_font=dict(size=22))

    st.plotly_chart(fig10, use_container_width=True)

    st.write("#### **Key Insights:**")

    if len(postings_over_time) > 1:
        peak_month = postings_over_time.loc[
            postings_over_time["job_postings"].idxmax()
        ]

        st.write(
            f"- Job postings peaked in **{peak_month['posting_month'].strftime('%B %Y')}** "
            f"with **{peak_month['job_postings']:,}** postings"
        )

        trend_direction = (
            "increased"
            if postings_over_time["job_postings"].iloc[-1]
            > postings_over_time["job_postings"].iloc[0]
            else "decreased"
        )

        st.write(
            f"- Overall, monthly job postings have **{trend_direction}** across the observed time period"
        )

    st.write(
        f"- Data spans from **{filtered_df['posting_date'].min().strftime('%B %Y')}** "
        f"to **{filtered_df['posting_date'].max().strftime('%B %Y')}**"
    )

    st.markdown("---")

# ==========================================================================
# PROJECT CONCLUSION & KEY INSIGHTS
# ==========================================================================
if df is not None:
    st.markdown("# Project Conclusion & Key Insights")
    st.markdown("---")

    st.subheader("Data Analysis Summary")
    st.markdown("""
    This comprehensive analysis of the global AI Job Market dataset has revealed significant insights into 
    compensation trends, hiring patterns, and workforce demand across the AI industry. Through 10 diverse 
    Plotly Express visualizations, we uncovered valuable patterns that can guide career decisions, hiring 
    strategies, and workforce planning.
    """)

    st.subheader("Major Findings & Insights")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### **Compensation Trends**
        - **Role-Based Pay Gaps**: Specialized and leadership AI roles command the highest salaries
        - **Experience Premium**: Salary increases substantially with experience level
        - **Employment Type Impact**: Compensation varies notably between full-time, contract, and freelance roles
        - **Wide Salary Range**: The market spans from entry-level to executive-level compensation

        ### **Industry & Role Demand**
        - **Industry Concentration**: Certain industries drive a disproportionate share of AI hiring
        - **Role Diversity**: A wide variety of specialized AI roles exist beyond generic "Data Scientist" titles
        - **Hierarchical Patterns**: Industry, experience, and employment type interact in distinct combinations
        """)

    with col2:
        st.markdown("""
        ### **Global Distribution**
        - **Geographic Concentration**: A handful of countries account for the majority of AI job postings
        - **Company Size Patterns**: Experience-level hiring differs across small, medium, and large companies

        ### **Time Trends**
        - **Posting Volume Trends**: AI job postings show identifiable monthly patterns
        - **Growing Field**: The volume of AI-related postings reflects the field's continued expansion

        ### **Experience-Salary Relationship**
        - **Positive Correlation**: More years of experience is associated with higher pay
        - **Diminishing Variance**: Salary spread differs meaningfully across experience tiers
        """)

    st.markdown("---")

    st.subheader("Business Implications & Recommendations")

    with st.expander("**For Job Seekers**", expanded=True):
        st.markdown("""
        1. **Target High-Paying Roles**: Focus skill development toward roles shown to command top salaries
        2. **Build Experience Strategically**: Years of experience meaningfully affects earning potential
        3. **Consider Employment Type**: Full-time vs. contract/freelance roles carry different compensation profiles
        4. **Look Beyond Borders**: Consider opportunities in countries with the strongest AI hiring activity
        """)

    with st.expander("**For Employers & HR Teams**"):
        st.markdown("""
        1. **Benchmark Compensation**: Use salary distributions by role and experience level to stay competitive
        2. **Understand Regional Demand**: Identify which countries have the deepest AI talent pools
        3. **Structure Career Ladders**: Align pay bands with experience-level salary patterns
        4. **Monitor Hiring Trends**: Track posting volume trends to plan recruitment timing
        """)

    with st.expander("**For Workforce Researchers & Educators**"):
        st.markdown("""
        1. **Curriculum Alignment**: Align training programs with in-demand, high-paying AI roles
        2. **Track Market Growth**: Monitor posting trends over time to understand field maturity
        3. **Study Regional Gaps**: Investigate why certain countries lead in AI job creation
        4. **Industry Focus**: Understand which industries are driving AI adoption and hiring
        """)

    st.markdown("---")
    st.markdown("# Final Conclusion")

    if 'df_viz' in locals() and df_viz is not None:
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Countries Analyzed", df_viz['company_location'].nunique())
        with col2:
            st.metric("Total Job Postings", f"{df_viz.shape[0]:,}")
        with col3:
            st.metric("Avg Salary (USD)", f"${df_viz['salary_usd'].mean():,.0f}")
        with col4:
            st.metric("Unique Job Titles", df_viz['job_title'].nunique())
        with col5:
            st.metric("Industries Covered", df_viz['industry'].nunique())

    st.markdown("""
    ### **Project Impact**

    This analysis provides a comprehensive understanding of the global AI job market through data-driven insights. 
    The findings demonstrate the complex interplay between experience, role specialization, geography, and company 
    characteristics in determining AI career compensation and opportunity.

    **Key Takeaways:**

    1. **Experience Pays**: Career progression in AI is closely tied to meaningful salary growth
    2. **Role Specialization Matters**: Not all "AI jobs" are equal — specific titles command very different pay
    3. **Geography Shapes Opportunity**: A small number of countries dominate AI hiring activity
    4. **Industry Drives Demand**: AI adoption is concentrated in specific industry sectors
    5. **The Field is Growing**: Posting trends reflect the continued expansion of AI as a career path
    """)

else:
    st.error("Dataset not available for visualization.")