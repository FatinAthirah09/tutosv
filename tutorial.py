import streamlit as st
import pandas as pd
import plotly.express as px

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
