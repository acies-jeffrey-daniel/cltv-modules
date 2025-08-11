import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Set Streamlit page configuration
st.set_page_config(layout="wide", page_title="Data Model EDA App (Revised)")

st.title("📊 Data Model Exploratory Data Analysis (EDA)")
st.markdown("Upload your CSV files for `Ads Spend`, `Customers`, `Orders`, `Products`, `Revenue`, `Transactions`, and `User Tracking` to perform interactive EDA.")

# --- File Uploaders ---
st.header("⬆️ Upload Your Data Files")

# Arrange uploaders in columns for better layout
upload_cols_1 = st.columns(4)
upload_cols_2 = st.columns(3) # One less file

with upload_cols_1[0]:
    ads_spend_file = st.file_uploader("Ads Spend CSV", type=["csv"], key="ads_spend_uploader")
with upload_cols_1[1]:
    customers_file = st.file_uploader("Customers CSV", type=["csv"], key="customers_uploader")
with upload_cols_1[2]:
    orders_file = st.file_uploader("Orders CSV", type=["csv"], key="orders_uploader")
with upload_cols_1[3]:
    products_file = st.file_uploader("Products CSV", type=["csv"], key="products_uploader")

with upload_cols_2[0]:
    revenue_file = st.file_uploader("Revenue CSV", type=["csv"], key="revenue_uploader")
with upload_cols_2[1]:
    transactions_file = st.file_uploader("Transactions CSV", type=["csv"], key="transactions_uploader")
with upload_cols_2[2]:
    user_tracking_file = st.file_uploader("User Tracking CSV", type=["csv"], key="user_tracking_uploader")


# --- Data Loading and Processing ---
df_ads_spend = None
df_customers = None
df_orders = None
df_products = None
df_revenue = None
df_transactions = None
df_user_tracking = None

# Helper function to load and convert dates
def load_csv_with_dates(file_uploader, date_cols_to_convert):
    if file_uploader:
        try:
            df = pd.read_csv(file_uploader)
            for col in date_cols_to_convert:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
            st.success(f"'{file_uploader.name}' loaded successfully!")
            return df
        except Exception as e:
            st.error(f"Error loading '{file_uploader.name}': {e}")
    return None

df_ads_spend = load_csv_with_dates(ads_spend_file, ['date']) # Assuming 'date' for ads_spend
df_customers = load_csv_with_dates(customers_file, ['sign_up_date', 'dob'])
df_orders = load_csv_with_dates(orders_file, ['order_datetime'])
df_products = load_csv_with_dates(products_file, [])
df_revenue = load_csv_with_dates(revenue_file, [])
df_transactions = load_csv_with_dates(transactions_file, ['purchase_timestamp'])
df_user_tracking = load_csv_with_dates(user_tracking_file, ['visit_timestamp']) # Assuming 'visit_timestamp' for user_tracking


# --- EDA Functions ---

def display_dataframe_info(df, df_name):
    """Displays basic info and descriptive statistics for a DataFrame."""
    st.subheader(f"Data Overview: {df_name}")
    st.write(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    st.write("First 5 Rows:")
    st.dataframe(df.head())

    st.write("Column Information:")
    buffer = pd.DataFrame(df.dtypes, columns=['Data Type'])
    buffer['Non-Null Count'] = df.count()
    st.dataframe(buffer)

    st.write("Descriptive Statistics (Numerical Columns):")
    st.dataframe(df.describe())

def display_missing_values(df):
    """Displays missing values count and percentage."""
    st.subheader("Missing Values Analysis")
    missing_data = df.isnull().sum()
    missing_percentage = (df.isnull().sum() / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing_data,
        'Missing Percentage (%)': missing_percentage
    })
    missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values(by='Missing Percentage (%)', ascending=False)
    if not missing_df.empty:
        st.dataframe(missing_df)
        fig = px.bar(missing_df, x=missing_df.index, y='Missing Percentage (%)',
                     title='Percentage of Missing Values per Column',
                     labels={'x': 'Column', 'y': 'Missing Percentage (%)'},
                     color='Missing Percentage (%)',
                     color_continuous_scale=px.colors.sequential.Plasma)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No missing values found in this dataset.")

def plot_numerical_distribution(df, column):
    """Plots histogram and box plot for a numerical column."""
    st.subheader(f"Distribution of {column}")
    fig_hist = px.histogram(df, x=column, nbins=50, title=f'Histogram of {column}',
                            marginal="box", # Adds box plot as marginal
                            color_discrete_sequence=px.colors.qualitative.Plotly)
    st.plotly_chart(fig_hist, use_container_width=True)

    # Box plot for outlier detection
    fig_box = px.box(df, y=column, title=f'Box Plot of {column} (Outlier Detection)',
                     color_discrete_sequence=px.colors.qualitative.Plotly)
    st.plotly_chart(fig_box, use_container_width=True)

def plot_categorical_distribution(df, column, top_n=10):
    """Plots bar chart for a categorical column."""
    st.subheader(f"Distribution of {column}")
    # Ensure the column is not empty before value_counts
    if not df[column].empty and not df[column].nunique() == 0: # Also check for 0 unique values
        value_counts = df[column].value_counts().nlargest(top_n).reset_index()
        value_counts.columns = [column, 'Count']
        fig = px.bar(value_counts, x=column, y='Count', title=f'Top {top_n} {column} Distribution',
                     color=column,
                     color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(f"Categorical column '{column}' is empty or contains no unique values, cannot plot distribution.")


def perform_eda(df, df_name):
    """Orchestrates EDA for a given DataFrame."""
    if df is None:
        st.warning(f"Please upload the {df_name} data first.")
        return

    display_dataframe_info(df, df_name)
    display_missing_values(df)

    st.subheader("Data Distribution & Outlier Analysis")
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    datetime_cols = df.select_dtypes(include='datetime64').columns.tolist()

    # Columns to explicitly exclude from plotting
    # These are typically unique identifiers, highly granular text, or dates better aggregated.
    excluded_columns_for_plots = [
        'customer_name', 'email', 'product_name', 'title', 'description', # From Customers, Product, Coupon
        'page_url', 'entry_page', 'exit_page', 'page_title', # From User_tracking
        'geo_location', # From Customers - often too granular for direct plotting
        'campaign_name' # From Ads Spend - could be unique per campaign or very high cardinality
    ]

    # Keywords to identify ID columns for exclusion (case-insensitive)
    id_keywords = [
        '_id', 'id', 'user_id', 'customer_id', 'transaction_id', 'order_item_id',
        'product_id', 'coupon_id', 'session_id', 'action_id',
        'revenue_listing_id', 'cost_id', 'campaign_id' # campaign_id added for Ads Spend
    ]

    # Filter out ID columns and other explicitly excluded columns from numerical and categorical lists
    filtered_numerical_cols = [
        col for col in numerical_cols
        if not any(keyword in col.lower() for keyword in id_keywords) and col.lower() not in [e.lower() for e in excluded_columns_for_plots]
    ]
    
    filtered_categorical_cols = [
        col for col in categorical_cols
        if not any(keyword in col.lower() for keyword in id_keywords) and col.lower() not in [e.lower() for e in excluded_columns_for_plots]
    ]

    # Numerical Columns
    if filtered_numerical_cols:
        st.markdown("##### Numerical Column Distributions")
        for col in filtered_numerical_cols:
            plot_numerical_distribution(df, col)
            st.markdown("---") # Separator
    else:
        st.info("No numerical columns (excluding IDs and non-insightful columns) found for distribution analysis.")

    # Categorical Columns
    if filtered_categorical_cols:
        st.markdown("##### Categorical Column Distributions")
        for col in filtered_categorical_cols:
            plot_categorical_distribution(df, col)
            st.markdown("---") # Separator
    else:
        st.info("No categorical columns (excluding IDs and non-insightful columns) found for distribution analysis.")

    # Datetime Columns (Time Series Trends)
    if datetime_cols:
        st.markdown("##### Time Series Trends")
        for col in datetime_cols:
            if not df[col].empty:
                # Example: Daily count of records
                df_time_series = df.set_index(col).resample('D').size().reset_index(name='Count')
                fig_time = px.line(df_time_series, x=col, y='Count', title=f'Daily Count of Records by {col}',
                                   labels={'Count': 'Number of Records', col: 'Date'},
                                   color_discrete_sequence=px.colors.qualitative.Dark24)
                st.plotly_chart(fig_time, use_container_width=True)
                st.markdown("---") # Separator
            else:
                st.info(f"Datetime column '{col}' is empty, cannot plot time series.")
    else:
        st.info("No datetime columns found for time series analysis.")

# --- Main EDA Section with Tabs ---
# Check if at least one DataFrame is loaded to show the tabs
if any([df_ads_spend is not None, df_customers is not None, df_orders is not None,
        df_products is not None, df_revenue is not None, df_transactions is not None,
        df_user_tracking is not None]):
    st.header("🔬 Exploratory Data Analysis Results")
    all_tabs = st.tabs([
        "Ads Spend EDA", "Customers EDA", "Orders EDA", "Products EDA",
        "Revenue EDA", "Transactions EDA", "User Tracking EDA"
    ])

    with all_tabs[0]:
        perform_eda(df_ads_spend, "Ads Spend")
    with all_tabs[1]:
        perform_eda(df_customers, "Customers")
    with all_tabs[2]:
        perform_eda(df_orders, "Orders")
    with all_tabs[3]:
        perform_eda(df_products, "Products")
    with all_tabs[4]:
        perform_eda(df_revenue, "Revenue")
    with all_tabs[5]:
        perform_eda(df_transactions, "Transactions")
    with all_tabs[6]:
        perform_eda(df_user_tracking, "User Tracking")
else:
    st.info("Please upload your CSV files to begin the EDA process. You can upload any combination of files.")

st.markdown("---")
st.markdown("Built with Streamlit and Plotly for interactive data exploration.")
