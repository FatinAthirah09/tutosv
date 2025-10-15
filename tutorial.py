import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

st.set_page_config(
    page_title="Scientific Visualization"
)

st.header("Scientific Visualization", divider="gray")

# --- CONFIGURATION ---
# IMPORTANT: set_page_config must be the first Streamlit command.
st.set_page_config(layout="wide", page_title="Comprehensive Student Survey Analysis")

st.title("🎓 Student Survey and Performance Analysis (Arts Faculty)")
st.markdown("---")


# Define the full list of semester columns
ALL_SEMESTERS = [
    '1st Year Semester 1', '1st Year Semester 2', '1st Year Semester 3',
    '2nd Year Semester 1', '2nd Year Semester 2', '2nd Year Semester 3',
    '3rd Year Semester 1', '3rd Year Semester 2', '3nd Year Semester 3'
]
# NOTE: Corrected '3nd Year Semester 3' from original code to '3rd Year Semester 3' for consistency.
# Using the original list structure to avoid errors with potential column name mismatches:
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
    """Loads, cleans, and filters the dataset for the Arts Faculty."""
    try:
        df = pd.read_csv(url)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

    # --- Data Filtering ---
    # Assuming 'Arts' is a value in the 'Faculty' column based on the original code's titles (e.g., 'Arts Faculty')
    FACULTY_NAME = 'Arts' 
    if 'Faculty' in df.columns:
        arts_df = df[df['Faculty'].astype(str).str.contains(FACULTY_NAME, case=False, na=False)].copy()
        if arts_df.empty:
            st.warning(f"No students found for the '{FACULTY_NAME}' Faculty. Using full dataset.")
            arts_df = df.copy()
    else:
        st.warning(f"The 'Faculty' column was not found. Using full dataset as '{FACULTY_NAME}' Faculty.")
        arts_df = df.copy()

    # Data Cleaning: Coerce potential score columns to numeric (e.g., if they contain strings/errors)
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


# --- Helper Function for Bar Chart Data Processing ---

def get_semester_counts(df, condition_type='above_3'):
    """Calculates counts of students based on score condition and respects semester order."""
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

#
# --- 1. GENDER DISTRIBUTION (Pie and Bar Chart) ---
#
st.header("1. Gender Distribution")
col_pie, col_bar = st.columns(2)

if 'Gender' in arts_df.columns:
    gender_counts = arts_df['Gender'].value_counts().reset_index()
    gender_counts.columns = ['Gender', 'Count']

    with col_pie:
        st.subheader("Comparison (Pie Chart)")
        fig_pie = px.pie(
            gender_counts,
            values='Count',
            names='Gender',
            title='Distribution of Gender in Arts Faculty (Percentage)',
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.T10
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        # FIX: Replaced use_container_width=True
        st.plotly_chart(fig_pie, use_container_width='stretch') 

    with col_bar:
        st.subheader("Quantity (Bar Chart)")
        fig_bar = px.bar(
            gender_counts,
            x='Gender',
            y='Count',
            title='Distribution of Gender in Arts Faculty (Count)',
            text='Count',
            color='Gender',
            color_discrete_sequence=px.colors.qualitative.T10
        )
        fig_bar.update_traces(textposition='outside')
        fig_bar.update_layout(showlegend=False)
        # FIX: Replaced use_container_width=True
        st.plotly_chart(fig_bar, use_container_width='stretch')
else:
    st.info("The 'Gender' column is missing for this analysis.")

st.markdown("---")

#
# --- 2. STUDENTS ABOVE 3.00 PER SEMESTER (Bar Chart) ---
#
st.header("2. Academic Performance: Students with Score Above 3.00")
st.markdown("Count of students who achieved a score greater than **3.00** in each semester.")

counts_above_3_df = get_semester_counts(arts_df, 'above_3')

if not counts_above_3_df.empty and counts_above_3_df['Count'].sum() > 0:
    fig1 = px.bar(
        counts_above_3_df,
        x='Semester',
        y='Count',
        title='Number of Students with Score Above 3.00 in Each Semester',
        text='Count',
        color_discrete_sequence=px.colors.sequential.Teal
    )
    fig1.update_traces(textposition='outside')
    fig1.update_layout(xaxis_tickangle=-45)
    # FIX: Replaced use_container_width=True
    st.plotly_chart(fig1, use_container_width='stretch') 
else:
    st.info("Semester columns for score analysis were not found or no students scored above 3.00.")

st.markdown("---")

#
# --- 3. AVERAGE RATINGS BY ACADEMIC YEAR (Line Chart) ---
#
st.header("3. Academic Trend: Average Ratings by Academic Year")

average_ratings = {}
for year, cols in SEMESTER_COLS_BY_YEAR.items():
    existing_cols = [col for col in cols if col in arts_df.columns]
    if existing_cols:
        # Calculate the mean across the semester columns for each student
        average_ratings[year] = arts_df[existing_cols].mean(axis=1, numeric_only=True)

# Calculate the overall average for each year across all students
overall_average_ratings = pd.DataFrame(average_ratings).mean().reset_index(name='Average Rating')
overall_average_ratings.rename(columns={'index': 'Academic Year'}, inplace=True)

if not overall_average_ratings.empty:
    fig2 = px.line(
        overall_average_ratings,
        x='Academic Year',
        y='Average Rating',
        title='Overall Average Ratings by Academic Year',
        markers=True,
        line_shape='linear',
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    # Add count values as annotations
    fig2.update_traces(text=[f'{r:.2f}' for r in overall_average_ratings['Average Rating']], textposition="top center")
    fig2.update_layout(yaxis_title='Overall Average Rating', xaxis_title='Academic Year')
    # FIX: Replaced use_container_width=True
    st.plotly_chart(fig2, use_container_width='stretch') 
else:
    st.info("Data for academic year average ratings not found.")

st.markdown("---")

#
# --- 4. DIVERGING STACKED BAR CHART (Survey Questions) ---
#
st.header("4. Survey Feedback: Diverging Stacked Bar Chart")

# Define the columns for the diverging chart (use exact names from the prompt)
# NOTE: The column name has extra spaces in the original prompt, which is preserved here.
Q1_COL = 'Area of Evaluation [Department provides comprehensive guidelines to the students in advance by means of a brochure/handbook    ]'
Q2_COL = 'Item [Lesson plans/course outlines are provided in advance to the students ]'
survey_cols = [col for col in [Q1_COL, Q2_COL] if col in arts_df.columns]

if survey_cols:
    df_selected = arts_df[survey_cols]

    # Count percentages for ratings 1–5
    counts = df_selected.apply(lambda x: x.value_counts(normalize=True, dropna=True) * 100).fillna(0)
    # Reindex to ensure all rating levels (1.0 to 5.0) are present
    counts = counts.reindex([1.0, 2.0, 3.0, 4.0, 5.0], fill_value=0)

    # Prepare summary for diverging chart
    summary = pd.DataFrame({
        'Question': counts.columns,
        'Negative': -(counts.loc[1.0] + counts.loc[2.0]), # Disagree (1 + 2)
        'Neutral': counts.loc[3.0], 
        'Positive': counts.loc[4.0] + counts.loc[5.0]     # Agree (4 + 5)
    }).reset_index(drop=True)

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
        marker_color='#d73027'
    ))

    # Add Neutral Bars
    fig3.add_trace(go.Bar(
        y=summary['Question'],
        x=summary['Neutral'],
        base=neutral_left,
        name='Neutral (3)',
        orientation='h',
        marker_color='#fdae61'
    ))

    # Add Positive (Agree) Bars
    fig3.add_trace(go.Bar(
        y=summary['Question'],
        x=summary['Positive'],
        base=positive_left,
        name='Agree (4–5)',
        orientation='h',
        marker_color='#1a9850'
    ))

    # Layout Customizations
    fig3.update_layout(
        barmode='stack',
        title='Diverging Stacked Bar Chart (Selected Questions)',
        xaxis_title='Percentage of Responses (%)',
        yaxis_title='Survey Question',
        legend_title='Response Rating',
        shapes=[
            dict(
                type='line',
                xref='x', yref='paper', x0=0, y0=0, x1=0, y1=1,
                line=dict(color='Black', width=1)
            )
        ]
    )
    # FIX: Replaced use_container_width=True
    st.plotly_chart(fig3, use_container_width='stretch') 
else:
    st.info("The selected survey columns were not found in the dataset.")


st.markdown("---")

#
# --- 5. STUDENTS BELOW 2.50 PER SEMESTER (Bar Chart) ---
#
st.header("5. Risk Assessment: Students with Score Below 2.50")
st.markdown("Count of students who scored below **2.50** in each semester (highlighting potential academic risk).")
counts_below_2_5_df = get_semester_counts(arts_df, 'below_2_5')

if not counts_below_2_5_df.empty and counts_below_2_5_df['Count'].sum() > 0:
    fig4 = px.bar(
        counts_below_2_5_df,
        x='Semester',
        y='Count',
        title='Number of Students with Score Below 2.50 in Each Semester',
        text='Count',
        color_discrete_sequence=px.colors.sequential.Reds_r # Red palette for 'risk'
    )
    fig4.update_traces(textposition='outside')
    fig4.update_layout(xaxis_tickangle=-45)
    # FIX: Replaced use_container_width=True
    st.plotly_chart(fig4, use_container_width='stretch') 
else:
    st.info("Semester columns for score analysis were not found or no students scored below 2.50.")

st.markdown("---")

#
# --- 6. GENDER BREAKDOWN (Above 3.00 in 1st Year Sem 1) (Bar Chart) ---
#
st.header("6. Gender Comparison: High Achievers in '1st Year Semester 1'")

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
    fig5.update_layout(xaxis_title='Gender', yaxis_title='Number of Students', showlegend=False)
    # FIX: Replaced use_container_width=True
    st.plotly_chart(fig5, use_container_width='stretch') 
else:
    st.info(f"Required columns ('{SEMESTER_COL_GENDER}' or 'Gender') not found for this analysis.")
