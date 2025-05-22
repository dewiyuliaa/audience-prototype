import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from datetime import timedelta

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
    }
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

# Load and process data
@st.cache_data
def load_data():
    try:
        # Read the CSV file
        df = pd.read_csv("~/Downloads/cnbc.csv", encoding='utf-8')
        
        # Convert date format to string (exactly like in your original code)
        df['date'] = pd.to_datetime(df['date'], format='%Y%m%d').dt.strftime('%Y-%m-%d')
        
        # Add age_group column based on age ranges
        df['age_group'] = None
        df.loc[(df['age'] >= 18) & (df['age'] <= 24), 'age_group'] = "18-24"
        df.loc[(df['age'] >= 25) & (df['age'] <= 34), 'age_group'] = "25-34"
        df.loc[(df['age'] >= 35) & (df['age'] <= 44), 'age_group'] = "35-44"
        df.loc[(df['age'] >= 45) & (df['age'] <= 54), 'age_group'] = "45-54"
        df.loc[(df['age'] >= 55) & (df['age'] <= 64), 'age_group'] = "55-64"
        df.loc[df['age'] >= 65, 'age_group'] = "65+"
        
        # Define kanal_group function
        def categorize_kanal(kanalid):
            if pd.isna(kanalid):
                return "Other"
            
            kanalid_str = str(kanalid)
            
            if kanalid_str.startswith("2-3"):
                return "News"
            elif kanalid_str.startswith("2-5"):
                return "Market"
            elif kanalid_str.startswith("2-9"):
                return "Entrepreneur"
            elif kanalid_str.startswith("2-12"):
                return "Tech"
            elif kanalid_str.startswith("2-11"):
                return "Lifestyle"
            elif kanalid_str.startswith("2-10"):
                return "Syariah"
            elif kanalid_str.startswith("2-13"):
                return "Opini"
            elif kanalid_str.startswith("2-71"):
                return "MyMoney"
            elif kanalid_str.startswith("2-78"):
                return "Cuap Cuap Cuan"
            elif kanalid_str.startswith("2-127"):
                return "Research"
            else:
                return "Other"
        
        # Apply kanal categorization
        df['kanal_group'] = df['kanalid'].apply(categorize_kanal)
        
        return df
    
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def calculate_metrics(df):
    """Calculate all metrics from the dataframe"""
    if df.empty:
        return {
            "unique_users": 0,
            "unique_email": 0,
            "unique_phone": 0,
            "total_page_views": 0,
            "views_per_user": 0,
            "average_session_duration": 0,
            "sessions_per_user": 0
        }
    
    unique_users = df['user_id'].nunique()
    unique_email = df['email'].dropna().nunique()
    unique_phone = df['phone_number'].dropna().nunique()
    total_page_views = df['page_views'].sum()
    views_per_user = round(total_page_views / unique_users, 2) if unique_users > 0 else 0
    
    total_session_time = df['session_length_in_seconds'].sum()
    unique_sessions_count = df['session_id'].nunique()
    average_session_duration = round(total_session_time / unique_sessions_count, 2) if unique_sessions_count > 0 else 0
    
    # Create user_session for counting unique user sessions
    df_temp = df.copy()
    df_temp['user_session'] = df_temp['user_id'].astype(str) + '_' + df_temp['session_id'].astype(str)
    unique_user_sessions = df_temp['user_session'].nunique()
    sessions_per_user = round(unique_user_sessions / unique_users, 2) if unique_users > 0 else 0
    
    return {
        "unique_users": unique_users,
        "unique_email": unique_email,
        "unique_phone": unique_phone,
        "total_page_views": total_page_views,
        "views_per_user": views_per_user,
        "average_session_duration": average_session_duration,
        "sessions_per_user": sessions_per_user
    }

def predict_users_combined(daily_unique_users, days_to_predict=1, use_last_n_days=30):
    """Predict next day's users based on historical data - exact copy from your original code"""
    if len(daily_unique_users) == 0:
        return 0
    
    # Convert Series to DataFrame for easier manipulation
    daily_users_df = daily_unique_users.reset_index()
    daily_users_df.columns = ['date', 'unique_users']
    
    # Convert to datetime for calculations (if not already)
    date_format = '%Y-%m-%d'
    daily_users_df['date'] = pd.to_datetime(daily_users_df['date'])
    
    # Sort by date
    daily_users_df = daily_users_df.sort_values('date').copy()
    
    # Get the last n days of data
    if len(daily_users_df) > use_last_n_days:
        df_last_n = daily_users_df.iloc[-use_last_n_days:].copy()
    else:
        df_last_n = daily_users_df.copy()
    
    if len(df_last_n) == 0:
        return 0
    
    # Calculate 7-day moving average (40% weight)
    moving_avg_days = min(7, len(df_last_n))
    last_7_avg = df_last_n['unique_users'].iloc[-moving_avg_days:].mean()
    
    # Calculate weighted average (40% weight)
    last_7_days = df_last_n['unique_users'].iloc[-moving_avg_days:].reset_index(drop=True)
    
    # Adjust weights based on available days
    if moving_avg_days == 7:
        weights = np.array([0.05, 0.1, 0.1, 0.15, 0.15, 0.2, 0.25])
    else:
        weights = np.linspace(0.5, 1.5, moving_avg_days)
        weights = weights / weights.sum()  # Normalize to sum to 1
    
    weighted_avg = (last_7_days * weights).sum()
    
    # Calculate median (20% weight)
    last_7_median = df_last_n['unique_users'].iloc[-moving_avg_days:].median()
    
    # Combine the methods
    combined_prediction = (last_7_avg * 0.4) + (weighted_avg * 0.4) + (last_7_median * 0.2)
    
    # Generate future dates in datetime format
    last_date = daily_users_df['date'].iloc[-1]
    future_dates_dt = [last_date + timedelta(days=i+1) for i in range(days_to_predict)]
    
    # Convert future dates to YYYY-MM-DD format
    future_dates = [dt.strftime(date_format) for dt in future_dates_dt]
    
    # Create prediction DataFrame
    predictions_df = pd.DataFrame({
        'date': future_dates,
        'moving_avg': [round(last_7_avg)] * days_to_predict,
        'weighted_avg': [round(weighted_avg)] * days_to_predict,
        'median': [round(last_7_median)] * days_to_predict,
        'estimated_users': [round(combined_prediction)] * days_to_predict
    })
    
    return round(combined_prediction)

def get_daily_metrics(df, last_n_days=30):
    """Get daily metrics for chart visualization"""
    if df.empty:
        return pd.DataFrame(columns=['date', 'audiences', 'views'])
    
    # Group by date to get daily metrics (date is already in string format)
    daily_metrics = df.groupby('date').agg({
        'user_id': 'nunique',  # This gives us daily unique users (audiences)
        'page_views': 'sum'    # This gives us daily total page views
    }).reset_index()
    daily_metrics.columns = ['date', 'audiences', 'views']
    
    # Convert date to datetime for sorting
    daily_metrics['date_dt'] = pd.to_datetime(daily_metrics['date'])
    daily_metrics = daily_metrics.sort_values('date_dt')
    daily_metrics = daily_metrics.drop('date_dt', axis=1)
    
    # Get available last days (up to last_n_days)
    if len(daily_metrics) > last_n_days:
        return daily_metrics.tail(last_n_days)
    else:
        return daily_metrics

# Load data
df = load_data()
if df is None:
    st.stop()

# Get unique values for selectors from the actual data
all_cities = sorted(df['city'].dropna().unique().tolist())
all_age_groups = sorted(df['age_group'].dropna().unique().tolist())
all_genders = sorted(df['sex'].dropna().unique().tolist())
all_kanals = sorted(df['kanal_group'].dropna().unique().tolist())
all_devices = sorted(df['device_category'].dropna().unique().tolist())

# Handle categoryauto_new_rank1 - more strict filtering to exclude column names
all_categories = []
if 'categoryauto_new_rank1' in df.columns:
    categories_data = df['categoryauto_new_rank1'].dropna()
    # Remove empty strings and any values that look like column names
    categories_data = categories_data[categories_data != '']
    categories_data = categories_data[categories_data != 'categoryauto_new_rank1']
    categories_data = categories_data[~categories_data.str.contains('age|articleid|birthdate|city|complete_dc|contenttype|date', case=False, na=False)]
    
    # Only keep if we have actual meaningful data
    if len(categories_data) > 0:
        unique_categories = categories_data.unique()
        # Further filter to exclude single letter or very short strings that look like column artifacts
        meaningful_categories = [cat for cat in unique_categories if len(str(cat)) > 2 and not str(cat).islower()]
        if len(meaningful_categories) > 0:
            all_categories = sorted(meaningful_categories)

# Get date range from actual data
min_date_str = df['date'].min()
max_date_str = df['date'].max()
min_date = datetime.datetime.strptime(min_date_str, '%Y-%m-%d').date()
max_date = datetime.datetime.strptime(max_date_str, '%Y-%m-%d').date()

# Sidebar configuration
st.sidebar.title("Custom Audiences")

# Date range selector with actual data range
st.sidebar.markdown("### Select date range")
date_range = st.sidebar.date_input(
    "",
    [min_date, max_date],
    min_value=min_date,
    max_value=max_date,
    format="YYYY/MM/DD",
    label_visibility="collapsed"
)

start_date = date_range[0] if len(date_range) > 0 else min_date
end_date = date_range[1] if len(date_range) > 1 else max_date

# City selector with no default
st.sidebar.markdown("### Select city")
selected_cities = st.sidebar.multiselect(
    "",
    all_cities,
    default=[],
    label_visibility="collapsed"
)

# Age selector with no default - single select
st.sidebar.markdown("### Select age")
selected_age = st.sidebar.selectbox(
    "",
    [""] + all_age_groups,  # Add empty option as first item
    index=0,  # Select the empty option
    label_visibility="collapsed"
)

# Gender selector - NO default values
st.sidebar.markdown("### Select gender")
selected_genders = []
if len(all_genders) >= 2:
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if "female" in all_genders and st.checkbox("female", value=False):  # Changed to False
            selected_genders.append("female")
    with col2:
        if "male" in all_genders and st.checkbox("male", value=False):  # Changed to False
            selected_genders.append("male")
else:
    # Handle case where there might be different gender values
    for gender in all_genders:
        if st.sidebar.checkbox(gender, value=False):  # Changed to False
            selected_genders.append(gender)

# Kanal selector with no default
st.sidebar.markdown("### Select kanal")
selected_kanal = st.sidebar.multiselect(
    "",
    all_kanals,
    default=[],
    label_visibility="collapsed"
)

# Device selector with no default - single select
st.sidebar.markdown("### Select device")
selected_device = st.sidebar.selectbox(
    "",
    [""] + all_devices,  # Add empty option as first item
    index=0,  # Select the empty option
    label_visibility="collapsed"
)

# Category selector - always show selector even if no categories
st.sidebar.markdown("### Select category")
selected_categories = st.sidebar.multiselect(
    "",
    all_categories,  # Will be empty list if no categories, but selector still shows
    default=[],
    label_visibility="collapsed"
)

# Apply filters to the dataframe
filtered_df = df.copy()

# Apply date filter (convert selected dates to string format for comparison)
start_date_str = start_date.strftime('%Y-%m-%d')
end_date_str = end_date.strftime('%Y-%m-%d')
filtered_df = filtered_df[(filtered_df['date'] >= start_date_str) & 
                          (filtered_df['date'] <= end_date_str)]

# Apply other filters if selections are made
if selected_cities:
    filtered_df = filtered_df[filtered_df['city'].isin(selected_cities)]

if selected_age and selected_age != "":  # Check if not empty selection
    filtered_df = filtered_df[filtered_df['age_group'] == selected_age]

if selected_genders:
    filtered_df = filtered_df[filtered_df['sex'].isin(selected_genders)]

if selected_kanal:
    filtered_df = filtered_df[filtered_df['kanal_group'].isin(selected_kanal)]

if selected_device and selected_device != "":  # Check if not empty selection
    filtered_df = filtered_df[filtered_df['device_category'] == selected_device]

if selected_categories and 'categoryauto_new_rank1' in df.columns:
    filtered_df = filtered_df[filtered_df['categoryauto_new_rank1'].isin(selected_categories)]

# Calculate metrics based on filtered data
filtered_metrics = calculate_metrics(filtered_df)

# Calculate estimated audience from filtered data (using the exact same method as your original code)
if not filtered_df.empty:
    daily_unique_users = filtered_df.groupby('date')['user_id'].nunique()
    estimated_audience = predict_users_combined(daily_unique_users)
else:
    estimated_audience = 0

# Get daily metrics for chart (last n days based on date range)
daily_chart_data = get_daily_metrics(filtered_df, 30)
num_days = len(daily_chart_data)

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

# Display sections based on selected tab
col1, col2 = st.columns([2, 3])

with col1:
    st.subheader("Audience Size")
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-label'>Estimated Audience Size:</div>
        <div class='metric-value'>{estimated_audience:,}</div>
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
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Email</div>
                <div class='metric-value'>{filtered_metrics['unique_email']:,}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_phone:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Phone Number</div>
                <div class='metric-value'>{filtered_metrics['unique_phone']:,}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='estimates-text'>This is based on available audience data and reflects the estimated count of individuals within your selected audience who have provided valid contact information (email or phone number). These are provided to give you an idea of how many users may be contactable through direct outreach.</div>
        """, unsafe_allow_html=True)

        # Add download button
        st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
        
        # Get actual contact data for download (full_name, email, phone_number)
        if not filtered_df.empty:
            # Select the columns we want to download
            download_columns = []
            if 'full_name' in filtered_df.columns:
                download_columns.append('full_name')
            if 'email' in filtered_df.columns:
                download_columns.append('email')
            if 'phone_number' in filtered_df.columns:
                download_columns.append('phone_number')
            
            if download_columns:
                contact_data = filtered_df[download_columns].dropna(subset=['email'])  # Only rows with email
                contact_data = contact_data.drop_duplicates(subset=['email'])  # Remove duplicate emails
                contact_csv = contact_data.to_csv(index=False)
            else:
                contact_csv = pd.DataFrame(columns=["Email"]).to_csv(index=False)
        else:
            contact_csv = pd.DataFrame(columns=["Email"]).to_csv(index=False)
        
        # Make download button
        st.download_button(
            label="Download Contact List",
            data=contact_csv,
            file_name="contact_list.csv",
            mime="text/csv",
            use_container_width=True
        )

with col2:
    st.subheader(f"Trend: Last {num_days} Days")
    selected_chart = st.selectbox("", ["Area Chart", "Bar Chart", "Line Chart", "Scatter Plot"], label_visibility="collapsed")
    
    # Handle chart display
    if daily_chart_data.empty:
        st.info("No data available for the selected filters and date range.")
    else:
        # Prepare chart data with actual dates as labels
        chart_data = daily_chart_data[['audiences', 'views']].copy()
        
        # Format dates for better display (14 May 2025 format) and set as index
        chart_data['formatted_date'] = pd.to_datetime(daily_chart_data['date']).dt.strftime('%d %b %Y')
        chart_data = chart_data.set_index('formatted_date')
        
        # Display the chart
        if selected_chart == "Bar Chart":
            st.bar_chart(chart_data[['audiences', 'views']])
        elif selected_chart == "Line Chart":
            st.line_chart(chart_data[['audiences', 'views']])
        elif selected_chart == "Area Chart":
            st.area_chart(chart_data[['audiences', 'views']])
        else:  # Scatter Plot
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(chart_data['audiences'], chart_data['views'])
            ax.set_xlabel('Audience')
            ax.set_ylabel('Views')
            # Set x-axis labels to show dates in new format without rotation
            dates = pd.to_datetime(daily_chart_data['date'])
            ax.set_xticks(range(len(dates)))
            ax.set_xticklabels([d.strftime('%d %b %Y') for d in dates], rotation=0)
            st.pyplot(fig)

# Add space before subheader
st.markdown("<div style='margin-top: 180px;'></div>", unsafe_allow_html=True)

# Key Metrics section
st.subheader("Key Metrics")
metric_cols = st.columns(5)

with metric_cols[0]:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-label'>Total Audience:</div>
        <div class='metric-value'>{filtered_metrics['unique_users']:,}</div>
    </div>
    """, unsafe_allow_html=True)

with metric_cols[1]:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-label'>Views:</div>
        <div class='metric-value'>{filtered_metrics['total_page_views']:,}</div>
    </div>
    """, unsafe_allow_html=True)

with metric_cols[2]:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-label'>Views per user:</div>
        <div class='metric-value'>{filtered_metrics['views_per_user']}</div>
    </div>
    """, unsafe_allow_html=True)
    
with metric_cols[3]:
    # Format average session duration with comma for thousands and 2 decimal places
    formatted_avg_duration = f"{filtered_metrics['average_session_duration']:,.2f}"
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-label'>Avg Session Duration:</div>
        <div class='metric-value'>{formatted_avg_duration}</div>
    </div>
    """, unsafe_allow_html=True)

with metric_cols[4]:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-label'>Sessions per user:</div>
        <div class='metric-value'>{filtered_metrics['sessions_per_user']}</div>
    </div>
    """, unsafe_allow_html=True)
