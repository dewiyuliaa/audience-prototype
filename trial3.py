import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

# Set page config
st.set_page_config(page_title="Audience Insight Dashboard", layout="wide")

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0;
    }
    .metric-card {
        background-color: #0d1118ff;
        padding: 5px;
        border-radius: 5px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 5px;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 10px 0;
    }s
    .metric-label {
        font-family: Arial, sans-serif;
        font-size: 14px;
        color: white;
    }
    .per-day {
        text-align: right;
        float: right;
        margin-top: -40px;
        color: white;
    }
    .estimates-text {
        font-family: Arial, sans-serif;
        font-size: 12px;
        font-style: italic;
        color: #cccccc;
    }
    .tab-container {
        display: flex;
        gap: 10px;
        margin-bottom: 20px;
    }
    .tab-button {
        background-color: #f0f2f6;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        cursor: pointer;
        font-weight: bold;
    }
    .tab-button.active {
        background-color: #4c78e0;
        color: white;
    }
    .stMultiSelect [data-baseweb=select] span {
        max-width: 200px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar configuration
st.sidebar.title("Custom Audiences")

# Date range selector with date picker
st.sidebar.markdown("### Select date range")
# Default date range
default_start_date = datetime.date(2025, 1, 1)
default_end_date = datetime.date(2025, 3, 31)

# Create a date range picker
start_date, end_date = st.sidebar.date_input(
    "",
    [default_start_date, default_end_date],
    format="YYYY/MM/DD",
    label_visibility="collapsed"
)

# City selector with multiselect
st.sidebar.markdown("### Select city")
cities = ["Jakarta", "Bandung", "Surabaya", "Denpasar", "Bogor", "Depok", "Bekasi", "Tangerang", "Semarang", "Medan"]
selected_cities = st.sidebar.multiselect(
    "",
    cities,
    default=["Jakarta"],
    label_visibility="collapsed"
)

# Age selector with checkboxes
st.sidebar.markdown("### Select age")
age_groups = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
selected_age = st.sidebar.selectbox("", age_groups, label_visibility="collapsed")

# Gender selector
st.sidebar.markdown("### Select gender")
genders = ["female", "male"]
selected_genders = []
col1, col2 = st.sidebar.columns(2)
with col1:
    if st.checkbox("female", value=True):
        selected_genders.append("female")
with col2:
    if st.checkbox("male", value=True):
        selected_genders.append("male")

# Kanal selector with search and checkboxes
st.sidebar.markdown("### Select kanal")
kanals = ["Nasional", "Ekonomi", "Olahraga", "Internasional", "Gaya Hidup", "Hiburan", "Teknologi"]
selected_kanal = st.sidebar.selectbox("", kanals, label_visibility="collapsed")

# Device selector 
st.sidebar.markdown("### Select device")
devices = ["Desktop", "Mobile", "Tablet"]
selected_device = st.sidebar.selectbox("", devices, label_visibility="collapsed")

# Category selector with multiselect
st.sidebar.markdown("### Select category")
categories = [
    "null", "politik nasional", "kriminal", "korupsi", 
    "politik internasional", "lalu lintas dan transportasi", 
    "bencana alam", "selebriti lokal", "buku dan karya tulis", 
    "pendidikan", "kesehatan"
]

# Add a search box for categories
category_search = st.sidebar.text_input("Type to search", key="category_search")
filtered_categories = [cat for cat in categories if category_search.lower() in cat.lower()]

# Create a container for the categories with checkboxes
selected_categories = st.sidebar.multiselect(
    "",
    filtered_categories,
    default=["politik nasional", "kriminal"],
    label_visibility="collapsed"
)

# Main content
st.markdown("<h1 class='main-header'>Audience Insight Dashboard</h1>", unsafe_allow_html=True)

# Create tabs
tab1, tab2 = st.columns(2)
user_login = True  # Default tab

with tab1:
    if st.button("User Login", use_container_width=True):
        user_login = True

with tab2:
    if st.button("User Non Login", use_container_width=True):
        user_login = False

# Generate sample data based on the selected date range
def generate_data():
    # Calculate date range
    if isinstance(start_date, datetime.date) and isinstance(end_date, datetime.date):
        date_range_days = (end_date - start_date).days + 1
        dates = pd.date_range(start=start_date, end=end_date)
    else:
        # Fallback to default range
        dates = pd.date_range('2025-01-01', periods=100, freq='D')
    
    # Generate data
    data = pd.DataFrame({
        'date': dates,
        'views': np.random.randint(100, 1000, size=len(dates)),
        'audiences': np.random.randint(50, 500, size=len(dates)),
        'items': np.random.randint(1, 10, size=len(dates))
    })
    return data

# Generate audience data
def generate_audience_data():
    return {
        "estimated_audience": 7684,
        "email_contacts": 5727,
        "phone_contacts": 5414,
        "total_audience": 36499,
        "views": 3117426,
        "sessions_per_user": 9.05,
        "avg": 2513.17,
        "views_per_user": 85.41
    }

data = generate_data()
audience_data = generate_audience_data()

# Get the last 30 days of data
filtered_data = data.tail(30)

# Display sections based on selected tab
col1, col2 = st.columns([2, 3])

with col1:
    st.subheader("Audience Size")
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-label'>Estimated Audience Size:</div>
        <div class='metric-value'>{audience_data['estimated_audience']:,}</div>
        <div class='estimates-text per-day'>(a day)</div>
    </div>
    <div class='estimates-text'>Estimates may vary significantly over time based on your targeting selections and available data.</div>
    """, unsafe_allow_html=True)
    
    # Add space before subheader
    st.markdown("<div style='margin-top: 50px;'></div>", unsafe_allow_html=True)
    
    if user_login:  # Only show for User Login tab
        st.subheader("Reachable Audience")
        col_email, col_phone = st.columns(2)
        
        with col_email:
            st.markdown("<div class='metric-card'><div class='metric-label'>Email</div><div class='metric-value'>5,727</div></div>", unsafe_allow_html=True)
        
        with col_phone:
            st.markdown("<div class='metric-card'><div class='metric-label'>Phone Number</div><div class='metric-value'>5,414</div></div>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class='estimates-text'>This is based on available audience data and reflects the estimated count of individuals within your selected audience who have provided valid contact information (email or phone number). These are provided to give you an idea of how many users may be contactable through direct outreach.</div>
        """, unsafe_allow_html=True)

with col2:
    st.subheader("Trend: Last 30 Days")
    selected_chart = st.selectbox("", ["Area Chart", "Bar Chart", "Line Chart", "Scatter Plot"], label_visibility="collapsed")
    
    # Use your suggested chart code with x-axis labeled 1-30
    if selected_chart == "Bar Chart":
        # Prepare data for bar chart - x-axis as 1-30
        chart_data = filtered_data[['views', 'audiences']].copy()
        chart_data.index = range(1, 31)  # Set index to 1-30
        st.bar_chart(chart_data)
    elif selected_chart == "Line Chart":
        # Prepare data for line chart - x-axis as 1-30
        chart_data = filtered_data[['views', 'audiences']].copy()
        chart_data.index = range(1, 31)  # Set index to 1-30
        st.line_chart(chart_data)
    elif selected_chart == "Area Chart":
        # Prepare data for area chart - x-axis as 1-30
        chart_data = filtered_data[['views', 'audiences']].copy()
        chart_data.index = range(1, 31)  # Set index to 1-30
        st.area_chart(chart_data)
    else:  # Scatter Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(filtered_data['audiences'], filtered_data['views'])
        ax.set_xlabel('Audience')
        ax.set_ylabel('Views')
        st.pyplot(fig)

# Add space before subheader
    st.markdown("<div style='margin-top: 120px;'></div>", unsafe_allow_html=True)

# Key Metrics section
st.subheader("Key Metrics")
metric_cols = st.columns(5)

with metric_cols[0]:
    st.markdown("<div class='metric-card'><div class='metric-label'>Total Audience:</div><div class='metric-value'>36,499</div></div>", unsafe_allow_html=True)

with metric_cols[1]:
    st.markdown("<div class='metric-card'><div class='metric-label'>Views:</div><div class='metric-value'>3,117,426</div></div>", unsafe_allow_html=True)

with metric_cols[2]:
    st.markdown("<div class='metric-card'><div class='metric-label'>Views per user:</div><div class='metric-value'>85.41</div></div>", unsafe_allow_html=True)
    
with metric_cols[3]:
    st.markdown("<div class='metric-card'><div class='metric-label'>Avg Session Duration:</div><div class='metric-value'>2,513.17</div></div>", unsafe_allow_html=True)

with metric_cols[4]:
    st.markdown("<div class='metric-card'><div class='metric-label'>Sessions per user:</div><div class='metric-value'>9.05</div></div>", unsafe_allow_html=True)