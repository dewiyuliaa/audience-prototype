import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from datetime import timedelta
import base64
import os

# Function to load and encode logo
def get_logo_base64():
    """Load and encode the CNBC logo"""
    try:
        logo_path = "CNBC_logo.svg.png"
        expanded_path = os.path.expanduser(logo_path)
        
        if os.path.exists(expanded_path):
            with open(expanded_path, "rb") as f:
                logo_data = f.read()
            return base64.b64encode(logo_data).decode()
        
        # If logo not found, return empty string (no logo will be displayed)
        return ""
    except Exception as e:
        # If there's any error loading the logo, return empty string
        return ""

# Set page config
st.set_page_config(page_title="Audience Insight Dashboard", layout="wide")

# Custom CSS for styling
st.markdown("""
<style>
    /* Hide Streamlit default elements */
    .stDeployButton {display:none;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main styling */
    .stApp {
        background-color: #f8fafc;
    }
    
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        margin-bottom: 0;
        color: #1e293b;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 20px;
    }
    
    .logo-img {
        height: 60px;
        width: auto;
    }
    
    /* Enhanced metric cards */
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        padding: 24px 16px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
        border: 1px solid #e2e8f0;
        margin-bottom: 16px;
        text-align: center;
        height: 140px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        line-height: 1.2;
        word-break: break-word;
        color: #1e293b;
    }
    
    .metric-unit {
        font-size: 13px;
        font-weight: 500;
        color: #64748b;
        margin-top: 6px;
        display: block;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .metric-growth {
        font-size: 12px;
        font-weight: 600;
        margin-top: 4px;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 4px;
    }
    
    .growth-positive {
        color: #10b981;
    }
    
    .growth-negative {
        color: #ef4444;
    }
    
    .growth-neutral {
        color: #6b7280;
    }
    
    .growth-icon {
        font-size: 10px;
        display: inline-block;
    }
    
    .metric-label {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        font-size: 15px;
        font-weight: 600;
        color: #374151;
        margin-bottom: 8px;
    }
    
    /* Audience size card - special styling */
    .audience-size-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 32px 24px;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.3);
        margin-bottom: 24px;
        text-align: center;
    }
    
    .audience-size-label {
        font-size: 16px;
        font-weight: 600;
        color: rgba(255, 255, 255, 0.9);
        margin-bottom: 12px;
    }
    
    .audience-size-value {
        font-size: 3rem;
        font-weight: 700;
        color: white;
        margin: 0;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    /* Reachable audience cards */
    .reachable-card {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        color: white;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 3px 12px rgba(79, 70, 229, 0.3);
        margin-bottom: 16px;
    }
    
    .reachable-label {
        font-size: 14px;
        font-weight: 500;
        color: rgba(255, 255, 255, 0.8);
        margin-bottom: 8px;
    }
    
    .reachable-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: white;
    }
    
    .estimates-text {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        font-size: 13px;
        font-style: normal;
        color: #6b7280;
        line-height: 1.5;
        margin-top: 16px;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f1f5f9;
        padding: 24px 16px;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 24px;
        margin-top: 40px;
    }
    
    /* Chart container - removed white background and padding */
    .chart-container {
        margin-bottom: 24px;
    }
    
    /* Download button styling */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 12px 24px;
        transition: all 0.2s ease;
    }
    
    .stDownloadButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
    }
    
    /* Notes styling */
    .notes-container {
        background: #f8fafc;
        border-left: 4px solid #3b82f6;
        padding: 16px 20px;
        border-radius: 0 8px 8px 0;
        margin-top: 24px;
    }
    
    .notes-container strong {
        color: #1e293b;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Load and process data
@st.cache_data
def load_data():
    try:
        # Read the CSV files
        df1 = pd.read_csv("cnbc(updated)2.csv", encoding='utf-8')
        df2 = pd.read_csv("cnbc2(updated).csv", encoding='utf-8')
        
        # Process df1 (User Login data)
        df1['date'] = pd.to_datetime(df1['date'], format='%Y%m%d').dt.strftime('%Y-%m-%d')
        
        # Add age_group column
        def safe_age_to_group(age_value):
            if pd.isna(age_value) or age_value == 'unknown' or age_value == '':
                return "Unknown"
            
            try:
                age_int = int(float(age_value))
                if 18 <= age_int <= 24:
                    return "18-24"
                elif 25 <= age_int <= 34:
                    return "25-34"
                elif 35 <= age_int <= 44:
                    return "35-44"
                elif 45 <= age_int <= 54:
                    return "45-54"
                elif 55 <= age_int <= 64:
                    return "55-64"
                elif age_int >= 65:
                    return "65+"
                else:
                    return "Other"
            except (ValueError, TypeError):
                return "Unknown"
        
        df1['age_group'] = df1['age'].apply(safe_age_to_group)
        
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
        
        df1['kanal_group'] = df1['kanalid'].apply(categorize_kanal)
        
        # Process df2 (User Non Login data)
        df2['date'] = pd.to_datetime(df2['date'], format='%Y%m%d').dt.strftime('%Y-%m-%d')
        
        if 'Gender' in df2.columns:
            df2['sex'] = df2['Gender'].str.lower()
        
        if 'Age' in df2.columns:
            df2['age_group'] = df2['Age']
        
        if 'Device category' in df2.columns:
            df2['device_category'] = df2['Device category']
        
        if 'City' in df2.columns:
            df2['city'] = df2['City']
        
        if 'Kanal ID' in df2.columns:
            df2['kanal_group'] = df2['Kanal ID'].apply(categorize_kanal)
        
        return df1, df2
    
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

def calculate_metrics(df, user_login=True):
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
    
    if user_login:
        unique_users = df['user_id'].nunique()
        unique_email = df['email'].dropna().nunique()
        unique_phone = df['phone_number'].dropna().nunique()
        total_page_views = df['page_views'].sum()
        views_per_user = round(total_page_views / unique_users, 2) if unique_users > 0 else 0
        
        total_session_time = df['session_length_in_seconds'].sum()
        unique_sessions_count = df['session_id'].nunique()
        average_session_duration = round(total_session_time / unique_sessions_count, 2) if unique_sessions_count > 0 else 0
        
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
    else:
        total_audience = df['Total users'].sum() if 'Total users' in df.columns else 0
        total_views = df['Views'].sum() if 'Views' in df.columns else 0
        views_per_user = round(total_views / total_audience, 2) if total_audience > 0 else 0
        
        if 'Average session duration' in df.columns and 'Total users' in df.columns:
            weighted_duration = (df['Average session duration'] * df['Total users']).sum()
            average_session_duration = round(weighted_duration / total_audience, 2) if total_audience > 0 else 0
        else:
            average_session_duration = 0
        
        total_sessions = df['Sessions'].sum() if 'Sessions' in df.columns else 0
        sessions_per_user = round(total_sessions / total_audience, 2) if total_audience > 0 else 0
        
        return {
            "unique_users": total_audience,
            "unique_email": 0,
            "unique_phone": 0,
            "total_page_views": total_views,
            "views_per_user": views_per_user,
            "average_session_duration": average_session_duration,
            "sessions_per_user": sessions_per_user
        }

def format_audience_range(estimated_value):
    """Convert estimated audience to a range format with M/K formatting"""
    if estimated_value == 0:
        return "0"
    
    def format_single_value(value):
        if value >= 1000000:
            return f"{value/1000000:.1f}M"
        elif value >= 1000:
            return f"{value/1000:.1f}K"
        else:
            return f"{value:,}"
    
    if estimated_value < 10:
        return f"{estimated_value}"
    elif estimated_value < 100:
        base = round(estimated_value / 10) * 10
        lower = max(0, base - 5)
        upper = base + 5
        return f"{lower} - {upper}"
    elif estimated_value < 1000:
        base = round(estimated_value / 50) * 50
        lower = max(0, base - 25)
        upper = base + 25
        return f"{lower:,} - {upper:,}"
    elif estimated_value < 10000:
        base = round(estimated_value / 100) * 100
        lower = max(0, base - 100)
        upper = base + 100
        return f"{format_single_value(lower)} - {format_single_value(upper)}"
    elif estimated_value < 100000:
        base = round(estimated_value / 1000) * 1000
        lower = max(0, base - 500)
        upper = base + 500
        return f"{format_single_value(lower)} - {format_single_value(upper)}"
    elif estimated_value < 1000000:
        base = round(estimated_value / 5000) * 5000
        lower = max(0, base - 2500)
        upper = base + 2500
        return f"{format_single_value(lower)} - {format_single_value(upper)}"
    else:
        value_in_millions = estimated_value / 1000000
        
        if value_in_millions < 10:
            base = round(value_in_millions * 10) / 10
            lower = max(0, base - 0.1)
            upper = base + 0.1
            lower_formatted = f"{lower:.1f}M"
            upper_formatted = f"{upper:.1f}M"
        else:
            base = round(value_in_millions * 2) / 2
            lower = max(0, base - 0.5)
            upper = base + 0.5
            lower_formatted = f"{lower:.1f}M"
            upper_formatted = f"{upper:.1f}M"
        
        return f"{lower_formatted} - {upper_formatted}"

def calculate_growth_metrics(current_df, previous_df, user_login=True):
    """Calculate growth metrics compared to previous period"""
    current_metrics = calculate_metrics(current_df, user_login)
    previous_metrics = calculate_metrics(previous_df, user_login)
    
    growth_metrics = {}
    
    # Calculate growth for each metric
    metrics_to_compare = ['unique_users', 'total_page_views', 'views_per_user', 'average_session_duration', 'sessions_per_user']
    
    for metric in metrics_to_compare:
        current_value = current_metrics[metric]
        previous_value = previous_metrics[metric]
        
        if previous_value == 0:
            # If previous value is 0, show 100% growth if current > 0, else 0%
            growth_percentage = 100.0 if current_value > 0 else 0.0
        else:
            growth_percentage = ((current_value - previous_value) / previous_value) * 100
        
        growth_metrics[metric] = round(growth_percentage, 1)
    
    return growth_metrics

def get_previous_period_data(current_df, start_date, end_date, selected_cities, selected_age, 
                           selected_genders, selected_kanal, selected_device, selected_categories,
                           selected_aws, selected_paylater, user_login):
    """Get data for the previous period with same length and filters"""
    
    # Calculate the length of current period
    current_period_days = (end_date - start_date).days + 1
    
    # Calculate previous period dates
    previous_end_date = start_date - timedelta(days=1)
    previous_start_date = previous_end_date - timedelta(days=current_period_days - 1)
    
    # Convert to string format for filtering
    previous_start_str = previous_start_date.strftime('%Y-%m-%d')
    previous_end_str = previous_end_date.strftime('%Y-%m-%d')
    
    # Filter for previous period
    previous_df = current_df.copy()
    previous_df = previous_df[(previous_df['date'] >= previous_start_str) & 
                             (previous_df['date'] <= previous_end_str)]
    
    # Apply same filters as current period
    if selected_cities:
        previous_df = previous_df[previous_df['city'].isin(selected_cities)]
    
    if selected_age:
        previous_df = previous_df[previous_df['age_group'].isin(selected_age)]
    
    if selected_genders:
        previous_df = previous_df[previous_df['sex'].isin(selected_genders)]
    
    if selected_kanal:
        previous_df = previous_df[previous_df['kanal_group'].isin(selected_kanal)]
    
    if selected_device:
        previous_df = previous_df[previous_df['device_category'].isin(selected_device)]
    
    if selected_categories and 'categoryauto_new_rank1' in current_df.columns:
        previous_df = previous_df[previous_df['categoryauto_new_rank1'].isin(selected_categories)]
    
    # Apply User Login specific filters
    if user_login:
        if selected_aws and 'aws' in current_df.columns:
            previous_df = previous_df[previous_df['aws'].isin(selected_aws)]
        
        if selected_paylater and 'paylater_status' in current_df.columns:
            previous_df = previous_df[previous_df['paylater_status'].isin(selected_paylater)]
    
    return previous_df
    """Format numbers for display - show millions as M, thousands as K"""
    if value == 0:
        return "0"
    elif value >= 1000000:
        return f"{value/1000000:.1f}M"
    elif value >= 1000:
        return f"{value/1000:.1f}K"
    else:
        return f"{value:,}"

def predict_users_combined(daily_data, days_to_predict=1, use_last_n_days=30, user_login=True):
    """Predict users for multiple days based on historical data"""
    if len(daily_data) == 0:
        return 0
    
    daily_users_df = daily_data.reset_index()
    if user_login:
        daily_users_df.columns = ['date', 'unique_users']
    else:
        daily_users_df.columns = ['date', 'total_users']
        daily_users_df['unique_users'] = daily_users_df['total_users']
    
    daily_users_df['date'] = pd.to_datetime(daily_users_df['date'])
    daily_users_df = daily_users_df.sort_values('date').copy()
    
    if len(daily_users_df) > use_last_n_days:
        df_last_n = daily_users_df.iloc[-use_last_n_days:].copy()
    else:
        df_last_n = daily_users_df.copy()
    
    if len(df_last_n) == 0:
        return 0
    
    if days_to_predict == 1:
        moving_avg_days = min(7, len(df_last_n))
        last_7_avg = df_last_n['unique_users'].iloc[-moving_avg_days:].mean()
        
        last_7_days = df_last_n['unique_users'].iloc[-moving_avg_days:].reset_index(drop=True)
        
        if moving_avg_days == 7:
            weights = np.array([0.05, 0.1, 0.1, 0.15, 0.15, 0.2, 0.25])
        else:
            weights = np.linspace(0.5, 1.5, moving_avg_days)
            weights = weights / weights.sum()
        
        weighted_avg = (last_7_days * weights).sum()
        last_7_median = df_last_n['unique_users'].iloc[-moving_avg_days:].median()
        
        daily_prediction = (last_7_avg * 0.4) + (weighted_avg * 0.4) + (last_7_median * 0.2)
        
        return round(daily_prediction)
    
    else:
        if user_login:
            recent_avg = df_last_n['unique_users'].mean()
            
            if days_to_predict == 2:
                overlap_factor = 0.80
            elif days_to_predict == 3:
                overlap_factor = 0.70
            elif days_to_predict <= 5:
                overlap_factor = 0.60
            elif days_to_predict <= 7:
                overlap_factor = 0.55
            else:
                overlap_factor = 0.50
            
            conservative_estimate = recent_avg * days_to_predict * overlap_factor
            
            if len(daily_users_df) >= days_to_predict:
                similar_periods = []
                for i in range(len(daily_users_df) - days_to_predict + 1):
                    period_sum = daily_users_df['unique_users'].iloc[i:i+days_to_predict].sum()
                    similar_periods.append(period_sum)
                
                if similar_periods:
                    pattern_estimate = np.median(similar_periods) * overlap_factor
                else:
                    pattern_estimate = conservative_estimate
            else:
                pattern_estimate = conservative_estimate
            
            min_daily = df_last_n['unique_users'].min()
            min_estimate = min_daily * days_to_predict * (overlap_factor + 0.1)
            
            if len(df_last_n) >= 3:
                recent_3_avg = df_last_n['unique_users'].iloc[-3:].mean()
                trend_estimate = recent_3_avg * days_to_predict * overlap_factor
            else:
                trend_estimate = conservative_estimate
            
            final_prediction = (
                conservative_estimate * 0.25 +
                pattern_estimate * 0.35 +
                min_estimate * 0.15 +
                trend_estimate * 0.25
            )
            
            max_reasonable = recent_avg * days_to_predict * 0.90
            min_reasonable = min_daily * days_to_predict * 0.40
            
            final_prediction = min(final_prediction, max_reasonable)
            final_prediction = max(final_prediction, min_reasonable)
            
            return round(final_prediction)
        
        else:
            recent_avg = df_last_n['unique_users'].mean()
            avg_based_estimate = recent_avg * days_to_predict
            
            if len(daily_users_df) >= days_to_predict:
                similar_periods = []
                for i in range(len(daily_users_df) - days_to_predict + 1):
                    period_sum = daily_users_df['unique_users'].iloc[i:i+days_to_predict].sum()
                    similar_periods.append(period_sum)
                
                if similar_periods:
                    pattern_estimate = np.median(similar_periods)
                else:
                    pattern_estimate = avg_based_estimate
            else:
                pattern_estimate = avg_based_estimate
            
            if len(df_last_n) >= 3:
                recent_3_avg = df_last_n['unique_users'].iloc[-3:].mean()
                trend_estimate = recent_3_avg * days_to_predict
            else:
                trend_estimate = avg_based_estimate
            
            max_daily = df_last_n['unique_users'].max()
            optimistic_estimate = max_daily * days_to_predict * 0.85
            
            if len(df_last_n) >= 7:
                recent_7_days = df_last_n['unique_users'].iloc[-7:].reset_index(drop=True)
                weights = np.array([0.05, 0.1, 0.1, 0.15, 0.15, 0.2, 0.25])
                weighted_recent_avg = (recent_7_days * weights).sum()
                weighted_estimate = weighted_recent_avg * days_to_predict
            else:
                weighted_estimate = avg_based_estimate
            
            final_prediction = (
                avg_based_estimate * 0.15 +
                pattern_estimate * 0.35 +
                trend_estimate * 0.20 +
                optimistic_estimate * 0.15 +
                weighted_estimate * 0.15
            )
            
            min_daily = df_last_n['unique_users'].min()
            max_daily = df_last_n['unique_users'].max()
            
            min_reasonable = min_daily * days_to_predict * 0.8
            max_reasonable = max_daily * days_to_predict * 1.1
            
            avg_min_bound = recent_avg * days_to_predict * 0.85
            avg_max_bound = recent_avg * days_to_predict * 1.15
            
            final_min = min(min_reasonable, avg_min_bound)
            final_max = max(max_reasonable, avg_max_bound)
            
            final_prediction = max(final_prediction, final_min)
            final_prediction = min(final_prediction, final_max)
            
            return round(final_prediction)

def get_daily_metrics(df, last_n_days=30, user_login=True):
    """Get daily metrics for chart visualization"""
    if df.empty:
        return pd.DataFrame(columns=['date', 'audiences', 'views'])
    
    if user_login:
        daily_metrics = df.groupby('date').agg({
            'user_id': 'nunique',
            'page_views': 'sum'
        }).reset_index()
        daily_metrics.columns = ['date', 'audiences', 'views']
    else:
        daily_metrics = df.groupby('date').agg({
            'Total users': 'sum' if 'Total users' in df.columns else lambda x: 0,
            'Views': 'sum' if 'Views' in df.columns else lambda x: 0
        }).reset_index()
        daily_metrics.columns = ['date', 'audiences', 'views']
    
    daily_metrics['date_dt'] = pd.to_datetime(daily_metrics['date'])
    daily_metrics = daily_metrics.sort_values('date_dt')
    daily_metrics = daily_metrics.drop('date_dt', axis=1)
    
    if len(daily_metrics) > last_n_days:
        return daily_metrics.tail(last_n_days)
    else:
        return daily_metrics

# Load data
df1, df2 = load_data()
if df1 is None or df2 is None:
    st.stop()

# Initialize user_login state
if 'user_login' not in st.session_state:
    st.session_state.user_login = True

# Get unique values for selectors from the appropriate dataframe
def get_filter_options(user_login):
    current_df = df1 if user_login else df2
    
    all_cities = sorted(current_df['city'].dropna().unique().tolist()) if 'city' in current_df.columns else []
    all_age_groups = sorted(current_df['age_group'].dropna().unique().tolist()) if 'age_group' in current_df.columns else []
    all_genders = sorted(current_df['sex'].dropna().unique().tolist()) if 'sex' in current_df.columns else []
    all_kanals = sorted(current_df['kanal_group'].dropna().unique().tolist()) if 'kanal_group' in current_df.columns else []
    all_devices = sorted(current_df['device_category'].dropna().unique().tolist()) if 'device_category' in current_df.columns else []
    
    all_aws = []
    all_ages_raw = []
    all_paylater_status = []
    
    if user_login:
        if 'aws' in current_df.columns:
            all_aws = sorted(current_df['aws'].dropna().unique().tolist())
        
        if 'age' in current_df.columns:
            all_ages_raw = sorted([str(x) for x in current_df['age'].dropna().unique().tolist()])
        
        if 'paylater_status' in current_df.columns:
            all_paylater_status = sorted(current_df['paylater_status'].dropna().unique().tolist())
    
    all_categories = []
    if 'categoryauto_new_rank1' in current_df.columns:
        categories_data = current_df['categoryauto_new_rank1'].dropna().unique()
        if len(categories_data) > 0:
            all_categories = sorted(categories_data.tolist())
    
    min_date_str = current_df['date'].min()
    max_date_str = current_df['date'].max()
    min_date = datetime.datetime.strptime(min_date_str, '%Y-%m-%d').date()
    max_date = datetime.datetime.strptime(max_date_str, '%Y-%m-%d').date()
    
    return {
        'cities': all_cities,
        'age_groups': all_age_groups,
        'genders': all_genders,
        'kanals': all_kanals,
        'devices': all_devices,
        'categories': all_categories,
        'aws': all_aws,
        'ages_raw': all_ages_raw,
        'paylater_status': all_paylater_status,
        'min_date': min_date,
        'max_date': max_date
    }

# Get initial filter options
filter_options = get_filter_options(st.session_state.user_login)

# Update filter options when switching tabs
def update_filter_options_on_tab_switch():
    new_filter_options = get_filter_options(st.session_state.user_login)
    
    if "city_selector" in st.session_state:
        current_cities = st.session_state["city_selector"]
        valid_cities = [city for city in current_cities if city in new_filter_options['cities']]
        st.session_state["city_selector"] = valid_cities
    
    if "age_selector" in st.session_state:
        current_ages = st.session_state["age_selector"]
        valid_ages = [age for age in current_ages if age in new_filter_options['age_groups']]
        st.session_state["age_selector"] = valid_ages
    
    if "kanal_selector" in st.session_state:
        current_kanals = st.session_state["kanal_selector"]
        valid_kanals = [kanal for kanal in current_kanals if kanal in new_filter_options['kanals']]
        st.session_state["kanal_selector"] = valid_kanals
    
    if "device_selector" in st.session_state:
        current_devices = st.session_state["device_selector"]
        valid_devices = [device for device in current_devices if device in new_filter_options['devices']]
        st.session_state["device_selector"] = valid_devices
    
    if "category_selector" in st.session_state:
        current_categories = st.session_state["category_selector"]
        valid_categories = [cat for cat in current_categories if cat in new_filter_options['categories']]
        st.session_state["category_selector"] = valid_categories
    
    if not st.session_state.user_login:
        filter_keys_to_clear = ["aws_selector", "paylater_selector"]
        for key in filter_keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
    
    if "date_range_selector" in st.session_state:
        current_range = st.session_state["date_range_selector"]
        if len(current_range) == 2:
            start_date_current = current_range[0]
            end_date_current = current_range[1]
            
            if (start_date_current < new_filter_options['min_date'] or 
                end_date_current > new_filter_options['max_date']):
                st.session_state["date_range_selector"] = [new_filter_options['min_date'], new_filter_options['max_date']]
    
    return new_filter_options

# Initialize or update filter options
if 'last_user_login_state' not in st.session_state:
    st.session_state.last_user_login_state = st.session_state.user_login

# Check if tab was switched
if st.session_state.last_user_login_state != st.session_state.user_login:
    filter_options = update_filter_options_on_tab_switch()
    st.session_state.last_user_login_state = st.session_state.user_login
else:
    filter_options = get_filter_options(st.session_state.user_login)

# Sidebar configuration
st.sidebar.title("Custom Audiences")

# Date range selector
st.sidebar.markdown("### Select date range")
date_range = st.sidebar.date_input(
    "",
    [filter_options['min_date'], filter_options['max_date']],
    min_value=filter_options['min_date'],
    max_value=filter_options['max_date'],
    format="YYYY/MM/DD",
    label_visibility="collapsed",
    key="date_range_selector"
)

start_date = date_range[0] if len(date_range) > 0 else filter_options['min_date']
end_date = date_range[1] if len(date_range) > 1 else filter_options['max_date']

# City selector
st.sidebar.markdown("### Select city")
selected_cities = st.sidebar.multiselect(
    "",
    filter_options['cities'],
    default=[],
    label_visibility="collapsed",
    key="city_selector",
    placeholder="Choose options"
)

# Age selector
st.sidebar.markdown("### Select age")
selected_age = st.sidebar.multiselect(
    "",
    filter_options['age_groups'],
    default=[],
    label_visibility="collapsed",
    key="age_selector",
    placeholder="Choose options"
)

# Gender selector
st.sidebar.markdown("### Select gender")
selected_genders = []
if len(filter_options['genders']) >= 2:
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if "female" in filter_options['genders'] and st.checkbox("female", value=False, key="female_checkbox"):
            selected_genders.append("female")
    with col2:
        if "male" in filter_options['genders'] and st.checkbox("male", value=False, key="male_checkbox"):
            selected_genders.append("male")
else:
    for i, gender in enumerate(filter_options['genders']):
        if st.sidebar.checkbox(gender, value=False, key=f"gender_checkbox_{i}"):
            selected_genders.append(gender)

# Kanal selector
st.sidebar.markdown("### Select kanal")
selected_kanal = st.sidebar.multiselect(
    "",
    filter_options['kanals'],
    default=[],
    label_visibility="collapsed",
    key="kanal_selector",
    placeholder="Choose options"
)

# Device selector
st.sidebar.markdown("### Select device")
selected_device = st.sidebar.multiselect(
    "",
    filter_options['devices'],
    default=[],
    label_visibility="collapsed",
    key="device_selector",
    placeholder="Choose options"
)

# AWS selector (User Login only)
selected_aws = []
if st.session_state.user_login and filter_options['aws']:
    st.sidebar.markdown("### Select Allo Wallet Status")
    selected_aws = st.sidebar.multiselect(
        "",
        filter_options['aws'],
        default=[],
        label_visibility="collapsed",
        key="aws_selector",
        placeholder="Choose options"
    )

# Paylater Status selector (User Login only)
selected_paylater = []
if st.session_state.user_login and filter_options['paylater_status']:
    st.sidebar.markdown("### Select Paylater Status")
    selected_paylater = st.sidebar.multiselect(
        "",
        filter_options['paylater_status'],
        default=[],
        label_visibility="collapsed",
        key="paylater_selector",
        placeholder="Choose options"
    )

# Category selector
st.sidebar.markdown("### Select category")
selected_categories = st.sidebar.multiselect(
    "",
    filter_options['categories'],
    default=[],
    label_visibility="collapsed",
    key="category_selector",
    placeholder="Choose options"
)

# Reset Filters button
st.sidebar.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)
if st.sidebar.button("🔄 Reset Filters", use_container_width=True, type="secondary"):
    filter_keys = [
        "date_range_selector", "city_selector", "age_selector", 
        "female_checkbox", "male_checkbox", "kanal_selector", 
        "device_selector", "category_selector", "aws_selector", "paylater_selector"
    ]
    
    for i in range(len(filter_options['genders'])):
        filter_keys.append(f"gender_checkbox_{i}")
    
    for key in filter_keys:
        if key in st.session_state:
            del st.session_state[key]
    
    st.session_state["date_range_selector"] = [filter_options['min_date'], filter_options['max_date']]
    st.session_state["city_selector"] = []
    st.session_state["age_selector"] = []
    st.session_state["female_checkbox"] = False
    st.session_state["male_checkbox"] = False
    st.session_state["kanal_selector"] = []
    st.session_state["device_selector"] = []
    st.session_state["category_selector"] = []
    st.session_state["aws_selector"] = []
    st.session_state["paylater_selector"] = []
    
    for i in range(len(filter_options['genders'])):
        st.session_state[f"gender_checkbox_{i}"] = False
        
    st.rerun()

# Apply filters to the dataframe
current_df = df1 if st.session_state.user_login else df2
filtered_df = current_df.copy()

# Apply date filter
start_date_str = start_date.strftime('%Y-%m-%d')
end_date_str = end_date.strftime('%Y-%m-%d')
filtered_df = filtered_df[(filtered_df['date'] >= start_date_str) & 
                          (filtered_df['date'] <= end_date_str)]

# Apply other filters
if selected_cities:
    filtered_df = filtered_df[filtered_df['city'].isin(selected_cities)]

if selected_age:
    filtered_df = filtered_df[filtered_df['age_group'].isin(selected_age)]

if selected_genders:
    filtered_df = filtered_df[filtered_df['sex'].isin(selected_genders)]

if selected_kanal:
    filtered_df = filtered_df[filtered_df['kanal_group'].isin(selected_kanal)]

if selected_device:
    filtered_df = filtered_df[filtered_df['device_category'].isin(selected_device)]

if selected_categories and 'categoryauto_new_rank1' in current_df.columns:
    filtered_df = filtered_df[filtered_df['categoryauto_new_rank1'].isin(selected_categories)]

# Apply User Login specific filters
if st.session_state.user_login:
    if selected_aws and 'aws' in current_df.columns:
        filtered_df = filtered_df[filtered_df['aws'].isin(selected_aws)]
    
    if selected_paylater and 'paylater_status' in current_df.columns:
        filtered_df = filtered_df[filtered_df['paylater_status'].isin(selected_paylater)]

# Calculate metrics and growth
filtered_metrics = calculate_metrics(filtered_df, st.session_state.user_login)

# Get previous period data for growth calculation
previous_period_df = get_previous_period_data(
    current_df, start_date, end_date, selected_cities, selected_age,
    selected_genders, selected_kanal, selected_device, selected_categories,
    selected_aws, selected_paylater, st.session_state.user_login
)

# Calculate growth metrics
growth_metrics = calculate_growth_metrics(filtered_df, previous_period_df, st.session_state.user_login)

# Calculate estimated audience
if not filtered_df.empty:
    prediction_df = current_df.copy()
    
    if selected_cities:
        prediction_df = prediction_df[prediction_df['city'].isin(selected_cities)]
    
    if selected_age:
        prediction_df = prediction_df[prediction_df['age_group'].isin(selected_age)]
    
    if selected_genders:
        prediction_df = prediction_df[prediction_df['sex'].isin(selected_genders)]
    
    if selected_kanal:
        prediction_df = prediction_df[prediction_df['kanal_group'].isin(selected_kanal)]
    
    if selected_device:
        prediction_df = prediction_df[prediction_df['device_category'].isin(selected_device)]
    
    if selected_categories and 'categoryauto_new_rank1' in current_df.columns:
        prediction_df = prediction_df[prediction_df['categoryauto_new_rank1'].isin(selected_categories)]
    
    if st.session_state.user_login:
        if selected_aws and 'aws' in current_df.columns:
            prediction_df = prediction_df[prediction_df['aws'].isin(selected_aws)]
        
        if selected_paylater and 'paylater_status' in current_df.columns:
            prediction_df = prediction_df[prediction_df['paylater_status'].isin(selected_paylater)]
    
    if not prediction_df.empty:
        if st.session_state.user_login:
            all_daily_data = prediction_df.groupby('date')['user_id'].nunique()
        else:
            all_daily_data = prediction_df.groupby('date')['Total users'].sum()
        
        days_in_range = (end_date - start_date).days + 1
        estimated_audience = predict_users_combined(all_daily_data, days_to_predict=days_in_range, user_login=st.session_state.user_login)
    else:
        estimated_audience = 0
else:
    estimated_audience = 0

# Get daily metrics for chart
daily_chart_data = get_daily_metrics(filtered_df, 30, st.session_state.user_login)
num_days = len(daily_chart_data)

# Main content
try:
    logo_base64 = get_logo_base64()
    if logo_base64:
        st.markdown(f"""
        <div class='main-header'>
            <img src="data:image/png;base64,{logo_base64}" class="logo-img" alt="CNBC Logo">
            <span>Audience Insight Dashboard</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <h1 class='main-header'>
            <span>Audience Insight Dashboard</span>
        </h1>
        """, unsafe_allow_html=True)
except:
    # Fallback if logo loading fails
    st.markdown("""
    <h1 class='main-header'>
        <span>Audience Insight Dashboard</span>
    </h1>
    """, unsafe_allow_html=True)

st.markdown("<div style='margin-bottom: 32px;'></div>", unsafe_allow_html=True)

# Create tabs
tab1, tab2 = st.columns(2)

with tab1:
    if st.button("User Login", use_container_width=True, type="primary" if st.session_state.user_login else "secondary"):
        if not st.session_state.user_login:
            st.session_state.user_login = True
            st.rerun()

with tab2:
    if st.button("User Non Login", use_container_width=True, type="primary" if not st.session_state.user_login else "secondary"):
        if st.session_state.user_login:
            st.session_state.user_login = False
            st.rerun()

# Display sections
col1, col2 = st.columns([2, 3])

with col1:
    st.subheader("Audience Size")
    
    days_in_range = (end_date - start_date).days + 1
    period_text = f"({days_in_range} days)" if days_in_range > 1 else "(a day)"
    
    audience_range = format_audience_range(estimated_audience)
    
    st.markdown(f"""
    <div class='audience-size-card'>
        <div class='audience-size-label'>Estimated Audience Size {period_text}</div>
        <div class='audience-size-value'>{audience_range}</div>
    </div>
    <div class='estimates-text'>Estimates may vary significantly over time based on your targeting selections and available data.</div>
    """, unsafe_allow_html=True)
    
    st.markdown("<div style='margin-top: 50px;'></div>", unsafe_allow_html=True)
    
    if st.session_state.user_login:
        st.subheader("Reachable Audience")
        col_email, col_phone = st.columns(2)
        
        with col_email:
            st.markdown(f"""
            <div class='reachable-card'>
                <div class='reachable-label'>Email</div>
                <div class='reachable-value'>{filtered_metrics['unique_email']:,}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_phone:
            st.markdown(f"""
            <div class='reachable-card'>
                <div class='reachable-label'>Phone Number</div>
                <div class='reachable-value'>{filtered_metrics['unique_phone']:,}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='estimates-text'>This is based on available audience data and reflects the estimated count of individuals within your selected audience who have provided valid contact information (email or phone number). These are provided to give you an idea of how many users may be contactable through direct outreach.</div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
        
        if not filtered_df.empty:
            download_columns = []
            if 'full_name' in filtered_df.columns:
                download_columns.append('full_name')
            if 'email' in filtered_df.columns:
                download_columns.append('email')
            if 'phone_number' in filtered_df.columns:
                download_columns.append('phone_number')
            
            if download_columns:
                contact_data = filtered_df[download_columns].dropna(subset=['email'])
                contact_data = contact_data.drop_duplicates(subset=['email'])
                
                if 'phone_number' in contact_data.columns:
                    contact_data = contact_data.copy()
                    contact_data['phone_number'] = contact_data['phone_number'].astype('Int64').astype(str)
                
                contact_csv = contact_data.to_csv(index=False)
            else:
                contact_csv = pd.DataFrame(columns=["Email"]).to_csv(index=False)
        else:
            contact_csv = pd.DataFrame(columns=["Email"]).to_csv(index=False)
        
        st.download_button(
            label="Download Contact List",
            data=contact_csv,
            file_name="contact_list.csv",
            mime="text/csv",
            use_container_width=True
        )

with col2:
    st.markdown(f"<h2 class='section-header'>Trend: Last {num_days} Days</h2>", unsafe_allow_html=True)
    
    # Display chart without white container or selector
    if daily_chart_data.empty:
        st.info("No data available for the selected filters and date range.")
    else:
        st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
        
        chart_data = daily_chart_data[['audiences', 'views']].copy()
        chart_data['formatted_date'] = pd.to_datetime(daily_chart_data['date']).dt.strftime('%d %b %Y')
        chart_data = chart_data.set_index('formatted_date')
        
        # Default to line chart (changed from area chart)
        st.line_chart(chart_data[['audiences', 'views']])
        
        st.markdown("</div>", unsafe_allow_html=True)

# Key Metrics section
days_in_range_for_metrics = (end_date - start_date).days + 1
metrics_period_text = f"({days_in_range_for_metrics} days)" if days_in_range_for_metrics > 1 else "(last day)"

st.markdown(f"<h2 class='section-header'>Key Metrics {metrics_period_text}</h2>", unsafe_allow_html=True)
metric_cols = st.columns(5)

with metric_cols[0]:
    if st.session_state.user_login:
        display_value = f"{filtered_metrics['unique_users']:,}"
    else:
        display_value = format_number_display(filtered_metrics['unique_users'])
    
    # Growth indicator
    growth_value = growth_metrics['unique_users']
    if growth_value > 0:
        growth_class = "growth-positive"
        growth_icon = "▲"
    elif growth_value < 0:
        growth_class = "growth-negative"
        growth_icon = "▼"
    else:
        growth_class = "growth-neutral"
        growth_icon = "●"
    
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-value'>{display_value}</div>
        <span class='metric-unit'>audiences</span>
        <div class='metric-growth {growth_class}'>
            <span class='growth-icon'>{growth_icon}</span>
            <span>{abs(growth_value):.1f}%</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

with metric_cols[1]:
    if st.session_state.user_login:
        display_value = f"{filtered_metrics['total_page_views']:,}"
    else:
        display_value = format_number_display(filtered_metrics['total_page_views'])
    
    # Growth indicator
    growth_value = growth_metrics['total_page_views']
    if growth_value > 0:
        growth_class = "growth-positive"
        growth_icon = "▲"
    elif growth_value < 0:
        growth_class = "growth-negative"
        growth_icon = "▼"
    else:
        growth_class = "growth-neutral"
        growth_icon = "●"
    
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-value'>{display_value}</div>
        <span class='metric-unit'>views</span>
        <div class='metric-growth {growth_class}'>
            <span class='growth-icon'>{growth_icon}</span>
            <span>{abs(growth_value):.1f}%</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

with metric_cols[2]:
    # Growth indicator
    growth_value = growth_metrics['views_per_user']
    if growth_value > 0:
        growth_class = "growth-positive"
        growth_icon = "▲"
    elif growth_value < 0:
        growth_class = "growth-negative"
        growth_icon = "▼"
    else:
        growth_class = "growth-neutral"
        growth_icon = "●"
    
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-value'>{filtered_metrics['views_per_user']}</div>
        <span class='metric-unit'>views/user</span>
        <div class='metric-growth {growth_class}'>
            <span class='growth-icon'>{growth_icon}</span>
            <span>{abs(growth_value):.1f}%</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
with metric_cols[3]:
    formatted_avg_duration = f"{filtered_metrics['average_session_duration']:,.2f}"
    
    # Growth indicator
    growth_value = growth_metrics['average_session_duration']
    if growth_value > 0:
        growth_class = "growth-positive"
        growth_icon = "▲"
    elif growth_value < 0:
        growth_class = "growth-negative"
        growth_icon = "▼"
    else:
        growth_class = "growth-neutral"
        growth_icon = "●"
    
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-value'>{formatted_avg_duration}</div>
        <span class='metric-unit'>seconds</span>
        <div class='metric-growth {growth_class}'>
            <span class='growth-icon'>{growth_icon}</span>
            <span>{abs(growth_value):.1f}%</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

with metric_cols[4]:
    # Growth indicator
    growth_value = growth_metrics['sessions_per_user']
    if growth_value > 0:
        growth_class = "growth-positive"
        growth_icon = "▲"
    elif growth_value < 0:
        growth_class = "growth-negative"
        growth_icon = "▼"
    else:
        growth_class = "growth-neutral"
        growth_icon = "●"
    
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-value'>{filtered_metrics['sessions_per_user']}</div>
        <span class='metric-unit'>sessions/user</span>
        <div class='metric-growth {growth_class}'>
            <span class='growth-icon'>{growth_icon}</span>
            <span>{abs(growth_value):.1f}%</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)

# Show User Login metrics below when on User Non Login tab
if not st.session_state.user_login:
    filtered_df1 = df1.copy()
    
    filtered_df1 = filtered_df1[(filtered_df1['date'] >= start_date_str) & 
                               (filtered_df1['date'] <= end_date_str)]
    
    if selected_cities:
        filtered_df1 = filtered_df1[filtered_df1['city'].isin(selected_cities)]
    
    if selected_age:
        filtered_df1 = filtered_df1[filtered_df1['age_group'].isin(selected_age)]
    
    if selected_genders:
        filtered_df1 = filtered_df1[filtered_df1['sex'].isin(selected_genders)]
    
    if selected_kanal:
        filtered_df1 = filtered_df1[filtered_df1['kanal_group'].isin(selected_kanal)]
    
    if selected_device:
        filtered_df1 = filtered_df1[filtered_df1['device_category'].isin(selected_device)]
    
    if selected_categories and 'categoryauto_new_rank1' in df1.columns:
        filtered_df1 = filtered_df1[filtered_df1['categoryauto_new_rank1'].isin(selected_categories)]
    
    user_login_metrics = calculate_metrics(filtered_df1, user_login=True)
    
    st.markdown(f"""
    <div class='metric-label'>Compared to User Login {metrics_period_text}</div>
    """, unsafe_allow_html=True)
    metric_cols_login = st.columns(5)
    
    with metric_cols_login[0]:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Total Audience:</div>
            <div class='metric-value'>{user_login_metrics['unique_users']:,}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with metric_cols_login[1]:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Views:</div>
            <div class='metric-value'>{user_login_metrics['total_page_views']:,}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with metric_cols_login[2]:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Views per user:</div>
            <div class='metric-value'>{user_login_metrics['views_per_user']}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with metric_cols_login[3]:
        formatted_avg_duration_login = f"{user_login_metrics['average_session_duration']:,.2f}"
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Avg Session Duration:</div>
            <div class='metric-value'>{formatted_avg_duration_login}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with metric_cols_login[4]:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Sessions per user:</div>
            <div class='metric-value'>{user_login_metrics['sessions_per_user']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)

# Notes section
if st.session_state.user_login:
    notes_text = """
    <div class='notes-container'>
    <strong>Notes:</strong><br>
    • <strong>Total audience</strong> is the number of unique users who have logged in to MPC during the selected period<br>
    • <strong>Views</strong> is the total number of page views generated by all users during the selected period<br>
    • <strong>Views per user</strong> is the average number of pages viewed by each user (Total Views ÷ Total Users)<br>
    • <strong>Average session duration (in seconds)</strong> is the average time users spend in a single session on the platform<br>
    • <strong>Sessions per user</strong> is the average number of separate sessions each user has during the selected period
    </div>
    """
else:
    notes_text = """
    <div class='notes-container'>
    <strong>Notes:</strong><br>
    • <strong>Total audience</strong> is the sum of total users who haven't logged in to MPC during the selected period<br>
    • <strong>Views</strong> is the sum of all page views generated during the selected period<br>
    • <strong>Views per user</strong> is calculated as Total Views ÷ Total Users<br>
    • <strong>Average session duration (in seconds)</strong> is the weighted average session duration across all user segments<br>
    • <strong>Sessions per user</strong> is calculated as Total Sessions ÷ Total Users
    </div>
    """

st.markdown(notes_text, unsafe_allow_html=True)
