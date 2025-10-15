import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

st.set_page_config(
    page_title="Scientific Visualization"
)

st.header("Scientific Visualization", divider="gray")

# 1. Configuration and Data Loading
st.set_page_config(layout="wide", page_title="Gender Distribution in Arts Faculty")

@st.cache_data
def load_data(url):
    """Loads the dataset from a URL."""
    try:
        df = pd.read_csv(url)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# URL provided in the prompt
url = 'https://raw.githubusercontent.com/FatinAthirah09/tutosv/refs/heads/main/student_survey_exported%20(1).csv'
df_url = load_data(url)

if df_url.empty:
    st.stop()

# --- Filtering the DataFrame (Necessary step based on the prompt's context) ---
# Assuming there is a 'Faculty' column and we are interested in 'Arts'.
# If the column name is different, this line must be adjusted.
FACULTY_NAME = 'Arts' # Replace with the actual faculty name if needed

# Attempt to filter if the column exists, otherwise use the full DataFrame for demonstration
if 'Faculty' in df_url.columns:
    arts_df = df_url[df_url['Faculty'].str.contains(FACULTY_NAME, case=False, na=False)]
else:
    # Fallback if 'Faculty' column is missing or to show data head
    st.warning("The 'Faculty' column was not found. Showing data head and proceeding with full dataset for demonstration.")
    arts_df = df_url

st.title("Student Survey Data Visualization")
st.markdown("### Gender Distribution Analysis (Arts Faculty)")

# Display a sample of the data
st.subheader("Data Preview (First 5 Rows)")
st.dataframe(arts_df.head(), use_container_width=True)

# 2. Data Preparation for Plotting
if 'Gender' not in arts_df.columns:
    st.error("The 'Gender' column is missing from the dataset. Cannot generate plots.")
    st.stop()

gender_counts = arts_df['Gender'].value_counts().reset_index()
gender_counts.columns = ['Gender', 'Count']

# 3. Plotly Visualizations

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Gender Distribution: Comparison (Pie Chart)")
    # Create the Plotly Pie Chart
    fig_pie = px.pie(
        gender_counts,
        values='Count',
        names='Gender',
        title='Distribution of Gender in Arts Faculty (Percentage)',
        hole=0.3, # Optional: makes it a donut chart
        color_discrete_sequence=px.colors.qualitative.Pastel # Custom color palette
    )
    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig_pie, use_container_width=True)

with col2:
    st.subheader("Gender Distribution: Quantity (Bar Chart)")
    # Create the Plotly Bar Chart
    fig_bar = px.bar(
        gender_counts,
        x='Gender',
        y='Count',
        title='Distribution of Gender in Arts Faculty (Count)',
        color='Gender',
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    # Customize the layout
    fig_bar.update_layout(
        xaxis_title='Gender',
        yaxis_title='Count',
        showlegend=False
    )
    st.plotly_chart(fig_bar, use_container_width=True)

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

    # --- Data Filtering (Reused from previous step logic) ---
    FACULTY_NAME = 'Arts' 
    if 'Faculty' in df.columns:
        arts_df = df[df['Faculty'].str.contains(FACULTY_NAME, case=False, na=False)].copy()
    else:
        st.warning(f"The 'Faculty' column was not found. Using full dataset as '{FACULTY_NAME}' Faculty.")
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

st.title("📊 Academic Performance and Survey Analysis (Arts Faculty)")
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
            # Count students with scores above 3.00
            counts[semester] = df[df[semester] > 3.00].shape[0]
        elif condition_type == 'below_2_5':
            # Count students with scores below 2.50
            counts[semester] = df[df[semester] < 2.50].shape[0]

    counts_df = pd.DataFrame.from_dict(counts, orient='index', columns=['Count'])
    counts_df = counts_df.reset_index().rename(columns={'index': 'Semester'})
    
    # Ensure semester order is respected for plotting
    counts_df['Semester'] = pd.Categorical(counts_df['Semester'], categories=ALL_SEMESTERS, ordered=True)
    counts_df = counts_df.sort_values('Semester')
    
    return counts_df

# --- 2. PLOT 1: STUDENTS ABOVE 3.00 PER SEMESTER (Bar Chart) ---

st.header("1. Performance: Students with Score Above 3.00")
st.markdown("The chart below displays the number of students who achieved a score greater than 3.00 in each semester.")
counts_above_3_df = get_semester_counts(arts_df, 'above_3')

if not counts_above_3_df.empty:
    fig1 = px.bar(
        counts_above_3_df,
        x='Semester',
        y='Count',
        title='Number of Students with Score Above 3.00 in Each Semester',
        text='Count',
        color_discrete_sequence=px.colors.sequential.Viridis
    )
    fig1.update_traces(textposition='outside')
    fig1.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig1, use_container_width=True)
else:
    st.info("Semester columns not found for this analysis.")

st.markdown("---")

# --- 3. PLOT 2: AVERAGE RATINGS BY ACADEMIC YEAR (Line Chart) ---

st.header("2. Trend Analysis: Average Ratings by Academic Year")

average_ratings = {}
for year, cols in SEMESTER_COLS_BY_YEAR.items():
    existing_cols = [col for col in cols if col in arts_df.columns]
    if existing_cols:
        # Calculate the mean across the semester columns for each student
        average_ratings[year] = arts_df[existing_cols].mean(axis=1, numeric_only=True)

# Create a DataFrame from the average ratings
average_ratings_df = pd.DataFrame(average_ratings)
# Calculate the overall average for each year across all students
overall_average_ratings = average_ratings_df.mean().reset_index(name='Average Rating')
overall_average_ratings.rename(columns={'index': 'Academic Year'}, inplace=True)


if not overall_average_ratings.empty:
    fig2 = px.line(
        overall_average_ratings,
        x='Academic Year',
        y='Average Rating',
        title='Overall Average Ratings by Academic Year',
        markers=True,
        line_shape='spline', # Optional: smooths the line
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    # Add count values as annotations
    fig2.update_traces(text=[f'{r:.2f}' for r in overall_average_ratings['Average Rating']], textposition="top center")
    fig2.update_layout(yaxis_title='Overall Average Rating', xaxis_title='Academic Year')
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("Data for academic year average ratings not found.")

st.markdown("---")

# --- 4. PLOT 3: DIVERGING STACKED BAR CHART (Survey Questions) ---

st.header("3. Survey Feedback: Diverging Stacked Bar Chart")
st.markdown("This chart visualizes the distribution of responses (Disagree, Neutral, Agree) for selected survey questions.")

# Define the columns for the diverging chart
Q1_COL = 'Area of Evaluation [Department provides comprehensive guidelines to the students in advance by means of a brochure/handbook    ]'
Q2_COL = 'Item [Lesson plans/course outlines are provided in advance to the students ]'

df_selected = arts_df[[Q1_COL, Q2_COL]]

# Count percentages for ratings 1–5
counts = df_selected.apply(lambda x: x.value_counts(normalize=True) * 100).fillna(0)
counts = counts.reindex([1.0, 2.0, 3.0, 4.0, 5.0], fill_value=0)

# Prepare summary for diverging chart
summary = pd.DataFrame({
    'Question': counts.columns,
    # Negative = Disagree (1 + 2) - made negative for diverging chart plotting
    'Negative': -(counts.loc[1.0] + counts.loc[2.0]), 
    # Neutral (3) - used as the 'left' start point for Positive
    'Neutral': counts.loc[3.0], 
    # Positive = Agree (4 + 5)
    'Positive': counts.loc[4.0] + counts.loc[5.0]
}).reset_index(drop=True)

if not summary.empty:
    # Use Plotly Graph Objects for fine control over stacked bars
    fig3 = go.Figure()

    # Calculate the left position for the Neutral bars (Negative values)
    neutral_left = summary['Negative']

    # Calculate the left position for the Positive bars (Negative + Neutral values)
    positive_left = summary['Negative'] + summary['Neutral']

    # Add Negative (Disagree) Bars
    fig3.add_trace(go.Bar(
        y=summary['Question'],
        x=summary['Negative'],
        name='Disagree (1–2)',
        orientation='h',
        marker_color='#d73027' # Red
    ))

    # Add Neutral Bars
    fig3.add_trace(go.Bar(
        y=summary['Question'],
        x=summary['Neutral'],
        base=neutral_left, # Starts where Negative ends
        name='Neutral (3)',
        orientation='h',
        marker_color='#fdae61' # Orange
    ))

    # Add Positive (Agree) Bars
    fig3.add_trace(go.Bar(
        y=summary['Question'],
        x=summary['Positive'],
        base=positive_left, # Starts where Neutral ends
        name='Agree (4–5)',
        orientation='h',
        marker_color='#1a9850' # Green
    ))

    # Layout Customizations
    fig3.update_layout(
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
    st.plotly_chart(fig3, use_container_width=True)
else:
    st.info(f"Survey columns '{Q1_COL}' or '{Q2_COL}' not found for this analysis.")


st.markdown("---")

# --- 5. PLOT 4: STUDENTS BELOW 2.50 PER SEMESTER (Bar Chart) ---

st.header("4. Risk Assessment: Students with Score Below 2.50")
st.markdown("This chart identifies the number of students who scored below 2.50 in each semester (potential academic risk).")
counts_below_2_5_df = get_semester_counts(arts_df, 'below_2_5')

if not counts_below_2_5_df.empty:
    fig4 = px.bar(
        counts_below_2_5_df,
        x='Semester',
        y='Count',
        title='Number of Students with Score Below 2.50 in Each Semester',
        text='Count',
        color_discrete_sequence=px.colors.sequential.Reds_r # Use a red palette for 'risk'
    )
    fig4.update_traces(textposition='outside')
    fig4.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig4, use_container_width=True)
else:
    st.info("Semester columns not found for this analysis.")

st.markdown("---")

# --- 6. PLOT 5: GENDER BREAKDOWN (Above 3.00 in 1st Year Sem 1) (Bar Chart) ---

st.header("5. Gender Comparison: High Achievers in '1st Year Semester 1'")
st.markdown("The chart compares the count of high-achieving students (score > 3.00) in the first semester, broken down by gender.")

SEMESTER_COL_GENDER = '1st Year Semester 1'

if SEMESTER_COL_GENDER in arts_df.columns and 'Gender' in arts_df.columns:
    # Filter for students with a score above 3.00 in the specified semester
    above_3_semester = arts_df[arts_df[SEMESTER_COL_GENDER] > 3.00].copy()

    # Count the number of students above 3.00 by gender
    gender_counts_above_3 = above_3_semester['Gender'].value_counts().reset_index()
    gender_counts_above_3.columns = ['Gender', 'Count']

    fig5 = px.bar(
        gender_counts_above_3,
        x='Gender',
        y='Count',
        color='Gender',
        title=f'Students with Score Above 3.00 in {SEMESTER_COL_GENDER} by Gender',
        text='Count',
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig5.update_traces(textposition='outside')
    fig5.update_layout(xaxis_title='Gender', yaxis_title='Number of Students')
    st.plotly_chart(fig5, use_container_width=True)
else:
    st.info(f"Required columns ('{SEMESTER_COL_GENDER}' or 'Gender') not found for this analysis.")
