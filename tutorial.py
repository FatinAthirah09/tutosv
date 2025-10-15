import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

st.set_page_config(
    page_title="Scientific Visualization"
)

st.header("Scientific Visualization", divider="gray")

# --- 1. CONFIGURATION AND DATA LOADING ---

st.set_page_config(layout="wide", page_title="Academic Performance and Survey Analysis")

# Define the full list of semester columns for reuse
ALL_SEMESTERS = [
    '1st Year Semester 1', '1st Year Semester 2', '1st Year Semester 3',
    '2nd Year Semester 1', '2nd Year Semester 2', '2nd Year Semester 3',
    '3rd Year Semester 1', '3rd Year Semester 2', '3rd Year Semester 3'
]

# Define the year-semester mapping
SEMESTER_COLS_BY_YEAR = {
    '1st Year': ['1st Year Semester 1', '1st Year Semester 2', '1st Year Semester 3'],
    '2nd Year': ['2nd Year Semester 1', '2nd Year Semester 2', '2nd Year Semester 3'],
    '3rd Year': ['3rd Year Semester 1', '3rd Year Semester 2', '3rd Year Semester 3']
}

@st.cache_data
def load_and_prepare_data(url):
    """Loads, cleans, and filters the dataset."""
    try:
        df = pd.read_csv(url)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

    # --- Data Filtering (Assuming 'Arts' Faculty filter) ---
    FACULTY_NAME = 'Arts' 
    if 'Faculty' in df.columns:
        # Filter rows where the 'Faculty' column contains 'Arts' (case-insensitive)
        arts_df = df[df['Faculty'].str.contains(FACULTY_NAME, case=False, na=False)].copy()
    else:
        st.warning(f"The 'Faculty' column was not found. Using full dataset as '{FACULTY_NAME}' Faculty for demonstration.")
        arts_df = df.copy()

    # Data Cleaning: Coerce semester columns to numeric, non-numeric become NaN
    for col in ALL_SEMESTERS:
        if col in arts_df.columns:
            arts_df[col] = pd.to_numeric(arts_df[col], errors='coerce')
            
    # Remove rows where all semester data is NaN (if any)
    arts_df.dropna(subset=[col for col in ALL_SEMESTERS if col in arts_df.columns], how='all', inplace=True)

    return arts_df

# Load the data
URL = 'https://raw.githubusercontent.com/FatinAthirah09/tutosv/refs/heads/main/student_survey_exported%20(1).csv'
arts_df = load_and_prepare_data(URL)

if arts_df.empty:
    st.stop()

st.title("📊 Student Survey Analysis (Arts Faculty)")
st.markdown("---")

# --- Helper Function for Bar Chart Data Processing ---

def get_semester_counts(df, condition_type='above_3'):
    """Calculates counts of students based on score condition."""
    existing_semesters = [col for col in ALL_SEMESTERS if col in df.columns]
    
    if not existing_semesters:
        return pd.DataFrame({'Semester': [], 'Count': []})

    counts = {}
    for semester in existing_semesters:
        if condition_type == 'above_3':
            counts[semester] = df[df[semester] > 3.00].shape[0]
        elif condition_type == 'below_2_5':
            counts[semester] = df[df[semester] < 2.50].shape[0]

    counts_df = pd.DataFrame.from_dict(counts, orient='index', columns=['Count'])
    counts_df = counts_df.reset_index().rename(columns={'index': 'Semester'})
    
    # Ensure semester order is respected for plotting
    counts_df['Semester'] = pd.Categorical(counts_df['Semester'], categories=ALL_SEMESTERS, ordered=True)
    counts_df = counts_df.sort_values('Semester')
    
    return counts_df

# --- 2. GENDER DISTRIBUTION (Pie Chart and Bar Chart) ---
st.header("1. Gender Distribution")
col1, col2 = st.columns(2)

if 'Gender' in arts_df.columns:
    gender_counts = arts_df['Gender'].value_counts().reset_index()
    gender_counts.columns = ['Gender', 'Count']
    
    with col1:
        st.subheader("Comparison (Pie Chart)")
        fig_pie = px.pie(
            gender_counts,
            values='Count',
            names='Gender',
            title='Distribution of Gender (Percentage)',
            hole=0.3,
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        st.subheader("Quantity (Bar Chart)")
        fig_bar = px.bar(
            gender_counts,
            x='Gender',
            y='Count',
            title='Distribution of Gender (Count)',
            text='Count',
            color='Gender',
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_bar.update_traces(textposition='outside')
        fig_bar.update_layout(showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)
else:
    st.info("The 'Gender' column is missing for this analysis.")

st.markdown("---")

# --- 3. ACADEMIC PERFORMANCE AND RISK ---

st.header("2. Academic Performance Trends")

# --- 3a. Students Above 3.00 ---
st.subheader("Students with Score Above 3.00 per Semester")
counts_above_3_df = get_semester_counts(arts_df, 'above_3')

if not counts_above_3_df.empty:
    fig_above_3 = px.bar(
        counts_above_3_df,
        x='Semester',
        y='Count',
        title='Number of Students with Score Above 3.00 in Each Semester',
        text='Count',
        color_discrete_sequence=px.colors.sequential.Viridis
    )
    fig_above_3.update_traces(textposition='outside')
    fig_above_3.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_above_3, use_container_width=True)
else:
    st.info("Semester columns not found for the Above 3.00 analysis.")

# --- 3b. Average Ratings by Academic Year ---
st.subheader("Overall Average Ratings by Academic Year")

average_ratings = {}
for year, cols in SEMESTER_COLS_BY_YEAR.items():
    existing_cols = [col for col in cols if col in arts_df.columns]
    if existing_cols:
        average_ratings[year] = arts_df[existing_cols].mean(axis=1, numeric_only=True)

average_ratings_df = pd.DataFrame(average_ratings)
overall_average_ratings = average_ratings_df.mean().reset_index(name='Average Rating')
overall_average_ratings.rename(columns={'index': 'Academic Year'}, inplace=True)


if not overall_average_ratings.empty:
    fig_line = px.line(
        overall_average_ratings,
        x='Academic Year',
        y='Average Rating',
        title='Overall Average Ratings by Academic Year',
        markers=True,
        line_shape='spline',
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    fig_line.update_traces(text=[f'{r:.2f}' for r in overall_average_ratings['Average Rating']], textposition="top center")
    fig_line.update_layout(yaxis_title='Overall Average Rating', xaxis_title='Academic Year')
    st.plotly_chart(fig_line, use_container_width=True)
else:
    st.info("Data for academic year average ratings not found.")
    
# --- 3c. Students Below 2.50 (Academic Risk) ---
st.subheader("Students with Score Below 2.50 per Semester (Academic Risk)")
counts_below_2_5_df = get_semester_counts(arts_df, 'below_2_5')

if not counts_below_2_5_df.empty:
    fig_below_2_5 = px.bar(
        counts_below_2_5_df,
        x='Semester',
        y='Count',
        title='Number of Students with Score Below 2.50 in Each Semester',
        text='Count',
        color_discrete_sequence=px.colors.sequential.Reds_r
    )
    fig_below_2_5.update_traces(textposition='outside')
    fig_below_2_5.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_below_2_5, use_container_width=True)
else:
    st.info("Semester columns not found for the Below 2.50 analysis.")

st.markdown("---")

# --- 4. GENDER COMPARISON FOR HIGH ACHIEVERS (1st Year Sem 1) ---
st.header("3. Gender Breakdown of High Achievers")

SEMESTER_COL_GENDER = '1st Year Semester 1'

if SEMESTER_COL_GENDER in arts_df.columns and 'Gender' in arts_df.columns:
    above_3_semester = arts_df[arts_df[SEMESTER_COL_GENDER] > 3.00].copy()
    gender_counts_above_3 = above_3_semester['Gender'].value_counts().reset_index()
    gender_counts_above_3.columns = ['Gender', 'Count']

    fig_gender_comp = px.bar(
        gender_counts_above_3,
        x='Gender',
        y='Count',
        color='Gender',
        title=f'Students with Score Above 3.00 in {SEMESTER_COL_GENDER} by Gender',
        text='Count',
        color_discrete_sequence=px.colors.qualitative.Vivid
    )
    fig_gender_comp.update_traces(textposition='outside')
    fig_gender_comp.update_layout(xaxis_title='Gender', yaxis_title='Number of Students', showlegend=False)
    st.plotly_chart(fig_gender_comp, use_container_width=True)
else:
    st.info(f"Required columns ('{SEMESTER_COL_GENDER}' or 'Gender') not found for this analysis.")
    
st.markdown("---")

# --- 5. DIVERGING STACKED BAR CHART (Survey Questions) ---

st.header("4. Survey Feedback Analysis (Diverging Stacked Bar)")

# Define the columns for the diverging chart
Q1_COL = 'Area of Evaluation [Department provides comprehensive guidelines to the students in advance by means of a brochure/handbook    ]'
Q2_COL = 'Item [Lesson plans/course outlines are provided in advance to the students ]'

if Q1_COL in arts_df.columns and Q2_COL in arts_df.columns:
    df_selected = arts_df[[Q1_COL, Q2_COL]].copy()
    
    # Count percentages for ratings 1–5
    counts = df_selected.apply(lambda x: x.value_counts(normalize=True) * 100).fillna(0)
    # Ensure integer indices (1.0 to 5.0) for percentage lookup
    counts = counts.reindex([1.0, 2.0, 3.0, 4.0, 5.0], fill_value=0)

    # Prepare summary for diverging chart
    summary = pd.DataFrame({
        'Question': counts.columns,
        # Negative is -(Disagree 1 + 2)
        'Negative': -(counts.loc[1.0] + counts.loc[2.0]), 
        # Neutral (3)
        'Neutral': counts.loc[3.0], 
        # Positive (Agree 4 + 5)
        'Positive': counts.loc[4.0] + counts.loc[5.0]
    }).reset_index(drop=True)

    # Use Plotly Graph Objects for fine control over stacked bars
    fig_diverging = go.Figure()

    neutral_left = summary['Negative']
    positive_left = summary['Negative'] + summary['Neutral']

    # 1. Negative (Disagree) Bars
    fig_diverging.add_trace(go.Bar(
        y=summary['Question'],
        x=summary['Negative'],
        name='Disagree (1–2)',
        orientation='h',
        marker_color='#d73027' # Red
    ))

    # 2. Neutral Bars
    fig_diverging.add_trace(go.Bar(
        y=summary['Question'],
        x=summary['Neutral'],
        base=neutral_left, # Starts where Negative ends
        name='Neutral (3)',
        orientation='h',
        marker_color='#fdae61' # Orange
    ))

    # 3. Positive (Agree) Bars
    fig_diverging.add_trace(go.Bar(
        y=summary['Question'],
        x=summary['Positive'],
        base=positive_left, # Starts where Neutral ends
        name='Agree (4–5)',
        orientation='h',
        marker_color='#1a9850' # Green
    ))

    # Layout Customizations
    fig_diverging.update_layout(
        barmode='stack',
        title='Diverging Stacked Bar Chart (Selected Questions)',
        xaxis_title='Percentage of Responses (%)',
        yaxis_title='Survey Question',
        legend_title='Response Rating',
        # Add a line at x=0
        shapes=[
            dict(
                type='line',
                xref='x',
                yref='paper',
                x0=0,
                y0=0,
                x1=0,
                y1=1,
                line=dict(color='Black', width=1)
            )
        ]
    )
    st.plotly_chart(fig_diverging, use_container_width=True)
else:
    st.info(f"Survey columns for diverging chart not found.")
