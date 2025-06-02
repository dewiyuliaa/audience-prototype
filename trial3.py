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
        background-color: #f0f2f6ff;
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
        color: #31333fff;
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
        color: #31333fff;
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
        # Read the CSV files
        df1 = pd.read_csv("cnbc.csv", encoding='utf-8')
        df2 = pd.read_csv("cnbc2.csv", encoding='utf-8')
        
        # Process df1 (User Login data)
        # Convert date format to string (exactly like in your original code)
        df1['date'] = pd.to_datetime(df1['date'], format='%Y%m%d').dt.strftime('%Y-%m-%d')
        
        # Add age_group column based on age ranges
        df1['age_group'] = None
        df1.loc[(df1['age'] >= 18) & (df1['age'] <= 24), 'age_group'] = "18-24"
        df1.loc[(df1['age'] >= 25) & (df1['age'] <= 34), 'age_group'] = "25-34"
        df1.loc[(df1['age'] >= 35) & (df1['age'] <= 44), 'age_group'] = "35-44"
        df1.loc[(df1['age'] >= 45) & (df1['age'] <= 54), 'age_group'] = "45-54"
        df1.loc[(df1['age'] >= 55) & (df1['age'] <= 64), 'age_group'] = "55-64"
        df1.loc[df1['age'] >= 65, 'age_group'] = "65+"
        
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
        
        # Apply kanal categorization to df1
        df1['kanal_group'] = df1['kanalid'].apply(categorize_kanal)
        
        # Process df2 (User Non Login data)
        # Convert date format to string if needed
        df2['date'] = pd.to_datetime(df2['date'], format='%Y%m%d').dt.strftime('%Y-%m-%d')
        
        # Process df2 columns to match df1 structure
        if 'Gender' in df2.columns:
            df2['sex'] = df2['Gender'].str.lower()  # Normalize to lowercase
        
        if 'Age' in df2.columns:
            df2['age_group'] = df2['Age']  # Age is already in format like 18-24
        
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
        # Original calculation for df1 (User Login)
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
    else:
        # New calculation for df2 (User Non Login)
        total_audience = df['Total users'].sum() if 'Total users' in df.columns else 0
        total_views = df['Views'].sum() if 'Views' in df.columns else 0
        views_per_user = round(total_views / total_audience, 2) if total_audience > 0 else 0
        
        # For average session duration - we take the weighted average
        if 'Average session duration' in df.columns and 'Total users' in df.columns:
            weighted_duration = (df['Average session duration'] * df['Total users']).sum()
            average_session_duration = round(weighted_duration / total_audience, 2) if total_audience > 0 else 0
        else:
            average_session_duration = 0
        
        total_sessions = df['Sessions'].sum() if 'Sessions' in df.columns else 0
        sessions_per_user = round(total_sessions / total_audience, 2) if total_audience > 0 else 0
        
        return {
            "unique_users": total_audience,
            "unique_email": 0,  # Not available for non-login users
            "unique_phone": 0,  # Not available for non-login users
            "total_page_views": total_views,
            "views_per_user": views_per_user,
            "average_session_duration": average_session_duration,
            "sessions_per_user": sessions_per_user
        }

def format_audience_range(estimated_value):
    """Convert estimated audience to a range format with M/K formatting"""
    if estimated_value == 0:
        return "0"
    
    # Helper function to format individual values
    def format_single_value(value):
        if value >= 1000000:
            return f"{value/1000000:.1f}M"
        elif value >= 1000:
            return f"{value/1000:.1f}K"
        else:
            return f"{value:,}"
    
    # Determine range based on the magnitude of the number
    if estimated_value < 10:
        # For very small numbers, show exact value
        return f"{estimated_value}"
    elif estimated_value < 100:
        # For numbers < 100, round to nearest 10 and create ±5 range
        base = round(estimated_value / 10) * 10
        lower = max(0, base - 5)
        upper = base + 5
        return f"{lower} - {upper}"
    elif estimated_value < 1000:
        # For numbers < 1000, round to nearest 50 and create ±25 range
        base = round(estimated_value / 50) * 50
        lower = max(0, base - 25)
        upper = base + 25
        return f"{lower:,} - {upper:,}"
    elif estimated_value < 10000:
        # For numbers 1000-9999, use K format
        base = round(estimated_value / 100) * 100
        lower = max(0, base - 100)
        upper = base + 100
        return f"{format_single_value(lower)} - {format_single_value(upper)}"
    elif estimated_value < 100000:
        # For numbers 10K-99K, round to nearest 1000 and create ±500 range
        base = round(estimated_value / 1000) * 1000
        lower = max(0, base - 500)
        upper = base + 500
        return f"{format_single_value(lower)} - {format_single_value(upper)}"
    elif estimated_value < 1000000:
        # For numbers 100K-999K, round to nearest 5000 and create ±2500 range
        base = round(estimated_value / 5000) * 5000
        lower = max(0, base - 2500)
        upper = base + 2500
        return f"{format_single_value(lower)} - {format_single_value(upper)}"
    else:
        # For numbers 1M+, format in millions with wider range
        value_in_millions = estimated_value / 1000000
        
        if value_in_millions < 10:
            # For 1M-9.9M, round to nearest 0.1M and create ±0.1M range
            base = round(value_in_millions * 10) / 10
            lower = max(0, base - 0.1)
            upper = base + 0.1
            lower_formatted = f"{lower:.1f}M"
            upper_formatted = f"{upper:.1f}M"
        else:
            # For 10M+, round to nearest 0.5M and create ±0.5M range
            base = round(value_in_millions * 2) / 2
            lower = max(0, base - 0.5)
            upper = base + 0.5
            lower_formatted = f"{lower:.1f}M"
            upper_formatted = f"{upper:.1f}M"
        
        return f"{lower_formatted} - {upper_formatted}"

def format_number_display(value):
    """Format numbers for display - show millions as M, thousands as K"""
    if value == 0:
        return "0"
    elif value >= 1000000:
        # Show in millions with 1 decimal place
        return f"{value/1000000:.1f}M"
    elif value >= 1000:
        # Show in thousands with 1 decimal place  
        return f"{value/1000:.1f}K"
    else:
        # Show exact number with comma formatting
        return f"{value:,}"

def predict_users_combined(daily_data, days_to_predict=1, use_last_n_days=30, user_login=True):
    """Predict users for multiple days based on historical data - supports both login and non-login data"""
    if len(daily_data) == 0:
        return 0
    
    # For user_login=True, daily_data is daily unique users (Series)
    # For user_login=False, daily_data is already aggregated daily Total users (Series)
    
    # Convert Series to DataFrame for easier manipulation
    daily_users_df = daily_data.reset_index()
    if user_login:
        daily_users_df.columns = ['date', 'unique_users']
    else:
        daily_users_df.columns = ['date', 'total_users']
        # Rename for consistency with existing logic
        daily_users_df['unique_users'] = daily_users_df['total_users']
    
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
    
    # For single day prediction - use original logic
    if days_to_predict == 1:
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
        
        # Combine the methods for daily prediction
        daily_prediction = (last_7_avg * 0.4) + (weighted_avg * 0.4) + (last_7_median * 0.2)
        
        return round(daily_prediction)
    
    # For multiple days prediction
    else:
        if user_login:
            # Original logic for User Login (with overlap factors)
            # Calculate daily average from recent data
            recent_avg = df_last_n['unique_users'].mean()
            
            # More realistic overlap factors based on user behavior analysis
            if days_to_predict == 2:
                overlap_factor = 0.80  # 20% overlap
            elif days_to_predict == 3:
                overlap_factor = 0.70  # 30% overlap
            elif days_to_predict <= 5:
                overlap_factor = 0.60  # 40% overlap
            elif days_to_predict <= 7:
                overlap_factor = 0.55  # 45% overlap
            else:
                overlap_factor = 0.50  # 50% overlap for longer periods
            
            # Method 1: Conservative daily average approach
            conservative_estimate = recent_avg * days_to_predict * overlap_factor
            
            # Method 2: Historical period matching with better logic
            if len(daily_users_df) >= days_to_predict:
                # Get multiple similar periods for better average
                similar_periods = []
                for i in range(len(daily_users_df) - days_to_predict + 1):
                    period_sum = daily_users_df['unique_users'].iloc[i:i+days_to_predict].sum()
                    similar_periods.append(period_sum)
                
                if similar_periods:
                    # Use median instead of mean to reduce outlier impact
                    pattern_estimate = np.median(similar_periods) * overlap_factor
                else:
                    pattern_estimate = conservative_estimate
            else:
                pattern_estimate = conservative_estimate
            
            # Method 3: Minimum realistic estimate (for small audience segments)
            min_daily = df_last_n['unique_users'].min()
            min_estimate = min_daily * days_to_predict * (overlap_factor + 0.1)  # Slightly higher factor
            
            # Method 4: Recent trend-based estimate
            if len(df_last_n) >= 3:
                recent_3_avg = df_last_n['unique_users'].iloc[-3:].mean()
                trend_estimate = recent_3_avg * days_to_predict * overlap_factor
            else:
                trend_estimate = conservative_estimate
            
            # Combine methods with adjusted weights (more conservative)
            final_prediction = (
                conservative_estimate * 0.25 +
                pattern_estimate * 0.35 +
                min_estimate * 0.15 +
                trend_estimate * 0.25
            )
            
            # Additional safety check: don't exceed reasonable bounds
            max_reasonable = recent_avg * days_to_predict * 0.90  # Maximum 90% of theoretical max
            min_reasonable = min_daily * days_to_predict * 0.40   # Minimum 40% of theoretical min
            
            final_prediction = min(final_prediction, max_reasonable)
            final_prediction = max(final_prediction, min_reasonable)
            
            return round(final_prediction)
        
        else:
            # New logic for User Non Login (aggregated total users - no overlap factors)
            # For aggregated data, we sum the daily totals directly since they represent total activity
            
            # Method 1: Simple daily average * days
            recent_avg = df_last_n['unique_users'].mean()
            avg_based_estimate = recent_avg * days_to_predict
            
            # Method 2: Historical period matching (direct sum, no overlap)
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
            
            # Method 3: Recent trend-based (last 3 days average)
            if len(df_last_n) >= 3:
                recent_3_avg = df_last_n['unique_users'].iloc[-3:].mean()
                trend_estimate = recent_3_avg * days_to_predict
            else:
                trend_estimate = avg_based_estimate
            
            # Method 4: More optimistic estimate based on recent maximum
            max_daily = df_last_n['unique_users'].max()
            optimistic_estimate = max_daily * days_to_predict * 0.85  # 85% of theoretical max
            
            # Method 5: Weighted recent average (giving more weight to recent days)
            if len(df_last_n) >= 7:
                recent_7_days = df_last_n['unique_users'].iloc[-7:].reset_index(drop=True)
                weights = np.array([0.05, 0.1, 0.1, 0.15, 0.15, 0.2, 0.25])  # More weight to recent days
                weighted_recent_avg = (recent_7_days * weights).sum()
                weighted_estimate = weighted_recent_avg * days_to_predict
            else:
                weighted_estimate = avg_based_estimate
            
            # Combine methods with adjusted weights (less conservative, more realistic)
            final_prediction = (
                avg_based_estimate * 0.15 +      # Reduced weight
                pattern_estimate * 0.35 +        # Historical pattern (main weight)
                trend_estimate * 0.20 +          # Recent trend
                optimistic_estimate * 0.15 +     # Optimistic estimate
                weighted_estimate * 0.15         # Weighted recent average
            )
            
            # For aggregated data, adjust bounds to be less restrictive
            min_daily = df_last_n['unique_users'].min()
            max_daily = df_last_n['unique_users'].max()
            
            # More realistic bounds - not too restrictive
            min_reasonable = min_daily * days_to_predict * 0.8  # Allow going down to 80% of min
            max_reasonable = max_daily * days_to_predict * 1.1  # Allow going up to 110% of max
            
            # Also consider the average-based bounds
            avg_min_bound = recent_avg * days_to_predict * 0.85
            avg_max_bound = recent_avg * days_to_predict * 1.15
            
            # Use the less restrictive bounds
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
        # Original logic for df1 (User Login)
        # Group by date to get daily metrics (date is already in string format)
        daily_metrics = df.groupby('date').agg({
            'user_id': 'nunique',  # This gives us daily unique users (audiences)
            'page_views': 'sum'    # This gives us daily total page views
        }).reset_index()
        daily_metrics.columns = ['date', 'audiences', 'views']
    else:
        # New logic for df2 (User Non Login)
        # Group by date and sum the aggregated data
        daily_metrics = df.groupby('date').agg({
            'Total users': 'sum' if 'Total users' in df.columns else lambda x: 0,
            'Views': 'sum' if 'Views' in df.columns else lambda x: 0
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
    
    # Handle categoryauto_new_rank1 for both df1 and df2
    all_categories = []
    if 'categoryauto_new_rank1' in current_df.columns:
        categories_data = current_df['categoryauto_new_rank1'].dropna().unique()
        if len(categories_data) > 0:
            all_categories = sorted(categories_data.tolist())
    
    # Get date range from actual data
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
        'min_date': min_date,
        'max_date': max_date
    }

# Get initial filter options
filter_options = get_filter_options(st.session_state.user_login)

# Update filter options when switching tabs, but preserve existing filters
def update_filter_options_on_tab_switch():
    new_filter_options = get_filter_options(st.session_state.user_login)
    
    # Check if current filter values are still valid for the new dataset
    # If valid, keep them; if not, reset to empty
    
    # Update cities if current selection is valid
    if "city_selector" in st.session_state:
        current_cities = st.session_state["city_selector"]
        valid_cities = [city for city in current_cities if city in new_filter_options['cities']]
        st.session_state["city_selector"] = valid_cities
    
    # Update age groups if current selection is valid
    if "age_selector" in st.session_state:
        current_ages = st.session_state["age_selector"]
        valid_ages = [age for age in current_ages if age in new_filter_options['age_groups']]
        st.session_state["age_selector"] = valid_ages
    
    # Update kanal if current selection is valid
    if "kanal_selector" in st.session_state:
        current_kanals = st.session_state["kanal_selector"]
        valid_kanals = [kanal for kanal in current_kanals if kanal in new_filter_options['kanals']]
        st.session_state["kanal_selector"] = valid_kanals
    
    # Update devices if current selection is valid
    if "device_selector" in st.session_state:
        current_devices = st.session_state["device_selector"]
        valid_devices = [device for device in current_devices if device in new_filter_options['devices']]
        st.session_state["device_selector"] = valid_devices
    
    # Update categories if current selection is valid
    if "category_selector" in st.session_state:
        current_categories = st.session_state["category_selector"]
        valid_categories = [cat for cat in current_categories if cat in new_filter_options['categories']]
        st.session_state["category_selector"] = valid_categories
    
    # Update date range to match the new dataset's range if current range is outside
    if "date_range_selector" in st.session_state:
        current_range = st.session_state["date_range_selector"]
        if len(current_range) == 2:
            start_date_current = current_range[0]
            end_date_current = current_range[1]
            
            # Check if current date range is within new dataset's range
            if (start_date_current < new_filter_options['min_date'] or 
                end_date_current > new_filter_options['max_date']):
                # Reset to new dataset's full range
                st.session_state["date_range_selector"] = [new_filter_options['min_date'], new_filter_options['max_date']]
    
    # Keep chart type selection when switching tabs
    # (Chart type selector will automatically use the persisted value)
    
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

# Date range selector with actual data range
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

# City selector with no default
st.sidebar.markdown("### Select city")
selected_cities = st.sidebar.multiselect(
    "",
    filter_options['cities'],
    default=[],
    label_visibility="collapsed",
    key="city_selector",
    placeholder="Choose options"
)

# Age selector with no default - multi select
st.sidebar.markdown("### Select age")
selected_age = st.sidebar.multiselect(
    "",
    filter_options['age_groups'],
    default=[],
    label_visibility="collapsed",
    key="age_selector",
    placeholder="Choose options"
)

# Gender selector - NO default values
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
    # Handle case where there might be different gender values
    for i, gender in enumerate(filter_options['genders']):
        if st.sidebar.checkbox(gender, value=False, key=f"gender_checkbox_{i}"):
            selected_genders.append(gender)

# Kanal selector with no default
st.sidebar.markdown("### Select kanal")
selected_kanal = st.sidebar.multiselect(
    "",
    filter_options['kanals'],
    default=[],
    label_visibility="collapsed",
    key="kanal_selector",
    placeholder="Choose options"
)

# Device selector with no default - multi select
st.sidebar.markdown("### Select device")
selected_device = st.sidebar.multiselect(
    "",
    filter_options['devices'],
    default=[],
    label_visibility="collapsed",
    key="device_selector",
    placeholder="Choose options"
)

# Category selector - show for both User Login and User Non Login
st.sidebar.markdown("### Select category")
selected_categories = st.sidebar.multiselect(
    "",
    filter_options['categories'],
    default=[],
    label_visibility="collapsed",
    key="category_selector",
    placeholder="Choose options"
)

# Add Reset Filters button
st.sidebar.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)
if st.sidebar.button("🔄 Reset Filters", use_container_width=True, type="secondary"):
    # Clear specific session state keys for our filters
    filter_keys = [
        "date_range_selector", "city_selector", "age_selector", 
        "female_checkbox", "male_checkbox", "kanal_selector", 
        "device_selector", "category_selector"
    ]
    
    # Also clear any gender checkboxes that might have numbered keys
    for i in range(len(filter_options['genders'])):
        filter_keys.append(f"gender_checkbox_{i}")
    
    for key in filter_keys:
        if key in st.session_state:
            del st.session_state[key]
    
    # Set default values to reset state
    st.session_state["date_range_selector"] = [filter_options['min_date'], filter_options['max_date']]
    st.session_state["city_selector"] = []
    st.session_state["age_selector"] = []
    st.session_state["female_checkbox"] = False
    st.session_state["male_checkbox"] = False
    st.session_state["kanal_selector"] = []
    st.session_state["device_selector"] = []
    st.session_state["category_selector"] = []
    
    # Set gender checkboxes to False for dynamic genders
    for i in range(len(filter_options['genders'])):
        st.session_state[f"gender_checkbox_{i}"] = False
        
    st.rerun()

# Apply filters to the dataframe
current_df = df1 if st.session_state.user_login else df2
filtered_df = current_df.copy()

# Apply date filter (convert selected dates to string format for comparison)
start_date_str = start_date.strftime('%Y-%m-%d')
end_date_str = end_date.strftime('%Y-%m-%d')
filtered_df = filtered_df[(filtered_df['date'] >= start_date_str) & 
                          (filtered_df['date'] <= end_date_str)]

# Apply other filters if selections are made
if selected_cities:
    filtered_df = filtered_df[filtered_df['city'].isin(selected_cities)]

if selected_age:  # Check if any age groups are selected
    filtered_df = filtered_df[filtered_df['age_group'].isin(selected_age)]

if selected_genders:
    filtered_df = filtered_df[filtered_df['sex'].isin(selected_genders)]

if selected_kanal:
    filtered_df = filtered_df[filtered_df['kanal_group'].isin(selected_kanal)]

if selected_device:  # Check if any devices are selected
    filtered_df = filtered_df[filtered_df['device_category'].isin(selected_device)]

if selected_categories and 'categoryauto_new_rank1' in current_df.columns:
    filtered_df = filtered_df[filtered_df['categoryauto_new_rank1'].isin(selected_categories)]

# Calculate metrics based on filtered data
filtered_metrics = calculate_metrics(filtered_df, st.session_state.user_login)

# Calculate estimated audience using all historical data but apply current filters (except date range)
if not filtered_df.empty:
    # Create a dataset with all dates but apply current filters (except date range)
    prediction_df = current_df.copy()
    
    # Apply all filters EXCEPT date range
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
    
    # Use all historical dates for this filtered audience segment
    if not prediction_df.empty:
        if st.session_state.user_login:
            # For User Login: use unique users per day
            all_daily_data = prediction_df.groupby('date')['user_id'].nunique()
        else:
            # For User Non Login: use sum of Total users per day
            all_daily_data = prediction_df.groupby('date')['Total users'].sum()
        
        # Calculate the number of days in selected date range
        days_in_range = (end_date - start_date).days + 1
        
        # Predict users for the selected period length using filtered historical data
        estimated_audience = predict_users_combined(all_daily_data, days_to_predict=days_in_range, user_login=st.session_state.user_login)
    else:
        estimated_audience = 0
else:
    estimated_audience = 0

# Get daily metrics for chart (last n days based on date range)
daily_chart_data = get_daily_metrics(filtered_df, 30, st.session_state.user_login)
num_days = len(daily_chart_data)

# Main content
# Create header with logo
header_col1, header_col2 = st.columns([1, 6])

with header_col1:
    # Load and display the logo
    try:
        logo_path = "CNBC_logo.svg.png"
        # Expand the tilde to full path
        import os
        logo_path = os.path.expanduser(logo_path)
        st.image(logo_path, width=90)  # Adjust width to fit with header
    except Exception as e:
        st.write("")  # Silent fail if logo not found

with header_col2:
    # Add negative margin to pull the header closer to the logo
    st.markdown("""
    <h1 class='main-header' style='margin-left: -35px;'>Audience Insight Dashboard</h1>
    """, unsafe_allow_html=True)

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

# Display sections based on selected tab
col1, col2 = st.columns([2, 3])

with col1:
    st.subheader("Audience Size")
    
    # Calculate days in selected range for display
    days_in_range = (end_date - start_date).days + 1
    period_text = f"({days_in_range} days)" if days_in_range > 1 else "(a day)"
    
    # Format estimated audience as range
    audience_range = format_audience_range(estimated_audience)
    
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-label'>Estimated Audience Size:</div>
        <div class='metric-value'>{audience_range}</div>
        <div class='estimates-text per-day'>{period_text}</div>
    </div>
    <div class='estimates-text'>Estimates may vary significantly over time based on your targeting selections and available data.</div>
    """, unsafe_allow_html=True)
    
    # Add space before subheader
    st.markdown("<div style='margin-top: 50px;'></div>", unsafe_allow_html=True)
    
    if st.session_state.user_login:  # Only show for User Login tab
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

        # Add download button only for User Login
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
                
                # Fix phone number format - remove .0 at the end
                if 'phone_number' in contact_data.columns:
                    contact_data = contact_data.copy()
                    contact_data['phone_number'] = contact_data['phone_number'].astype('Int64').astype(str)
                
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
    
    # Get the current chart type selection (with proper default handling)
    if "chart_type_selector" not in st.session_state:
        st.session_state.chart_type_selector = "Area Chart"
    
    selected_chart = st.selectbox("", ["Area Chart", "Bar Chart", "Line Chart", "Scatter Plot"], 
                                 label_visibility="collapsed", key="chart_type_selector")
    
    # Handle chart display
    if daily_chart_data.empty:
        st.info("No data available for the selected filters and date range.")
    else:
        # Prepare chart data with actual dates as labels
        chart_data = daily_chart_data[['audiences', 'views']].copy()
        
        # Format dates for better display (14 May 2025 format) and set as index
        chart_data['formatted_date'] = pd.to_datetime(daily_chart_data['date']).dt.strftime('%d %b %Y')
        chart_data = chart_data.set_index('formatted_date')
        
        # Debug: Show what chart type is selected
        # st.write(f"Debug: Selected chart type is: {selected_chart}")
        
        # Display the chart based on selection
        if selected_chart == "Line Chart":
            st.line_chart(chart_data[['audiences', 'views']])
        elif selected_chart == "Bar Chart":
            st.bar_chart(chart_data[['audiences', 'views']])
        elif selected_chart == "Scatter Plot":
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(chart_data['audiences'], chart_data['views'])
            ax.set_xlabel('Audience')
            ax.set_ylabel('Views')
            # Set x-axis labels to show dates in new format without rotation
            dates = pd.to_datetime(daily_chart_data['date'])
            ax.set_xticks(range(len(dates)))
            ax.set_xticklabels([d.strftime('%d %b %Y') for d in dates], rotation=0)
            st.pyplot(fig)
        else:  # Default to Area Chart
            st.area_chart(chart_data[['audiences', 'views']])

# Add space before subheader
st.markdown("<div style='margin-top: 80px;'></div>", unsafe_allow_html=True)

# Key Metrics section
days_in_range_for_metrics = (end_date - start_date).days + 1
metrics_period_text = f"({days_in_range_for_metrics} days)" if days_in_range_for_metrics > 1 else "(last day)"

st.subheader(f"Key Metrics {metrics_period_text}")
metric_cols = st.columns(5)

with metric_cols[0]:
    # Use formatted display for User Non Login, regular formatting for User Login
    if st.session_state.user_login:
        display_value = f"{filtered_metrics['unique_users']:,}"
    else:
        display_value = format_number_display(filtered_metrics['unique_users'])
    
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-label'>Total Audience:</div>
        <div class='metric-value'>{display_value}</div>
    </div>
    """, unsafe_allow_html=True)

with metric_cols[1]:
    # Use formatted display for User Non Login, regular formatting for User Login
    if st.session_state.user_login:
        display_value = f"{filtered_metrics['total_page_views']:,}"
    else:
        display_value = format_number_display(filtered_metrics['total_page_views'])
    
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-label'>Views:</div>
        <div class='metric-value'>{display_value}</div>
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

# Add Notes section under Key Metrics
st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)

# If User Non Login is selected, show User Login metrics below
if not st.session_state.user_login:
    # Calculate User Login metrics with same filters
    filtered_df1 = df1.copy()
    
    # Apply same date filter
    filtered_df1 = filtered_df1[(filtered_df1['date'] >= start_date_str) & 
                               (filtered_df1['date'] <= end_date_str)]
    
    # Apply same filters if selections are made
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
    
    # Calculate User Login metrics
    user_login_metrics = calculate_metrics(filtered_df1, user_login=True)
    
    # Display User Login metrics
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
    
    # Add space before notes
    st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)

# Different notes based on login type
if st.session_state.user_login:
    notes_text = """
    <div class='estimates-text'>
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
    <div class='estimates-text'>
    <strong>Notes:</strong><br>
    • <strong>Total audience</strong> is the sum of total users who haven't logged in to MPC during the selected period<br>
    • <strong>Views</strong> is the sum of all page views generated during the selected period<br>
    • <strong>Views per user</strong> is calculated as Total Views ÷ Total Users<br>
    • <strong>Average session duration (in seconds)</strong> is the weighted average session duration across all user segments<br>
    • <strong>Sessions per user</strong> is calculated as Total Sessions ÷ Total Users
    </div>
    """

st.markdown(notes_text, unsafe_allow_html=True)
