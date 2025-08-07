import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Set Streamlit page configuration
st.set_page_config(layout="wide", page_title="Data Model EDA App")

st.title("📊 Data Model Exploratory Data Analysis (EDA)")
st.markdown("Upload your `transactions`, `orders`, and `customers` CSV files to perform interactive EDA.")

# --- File Uploaders ---
st.header("⬆️ Upload Your Data Files")
col1, col2, col3 = st.columns(3)

with col1:
    transactions_file = st.file_uploader("Upload Transactions CSV", type=["csv"], key="transactions_uploader")
with col2:
    orders_file = st.file_uploader("Upload Orders CSV", type=["csv"], key="orders_uploader")
with col3:
    customers_file = st.file_uploader("Upload Customers CSV", type=["csv"], key="customers_uploader")

# --- Data Loading and Processing ---
df_transactions = None
df_orders = None
df_customers = None

if transactions_file:
    try:
        df_transactions = pd.read_csv(transactions_file)
        # Convert timestamp to datetime
        if 'purchase_timestamp' in df_transactions.columns:
            df_transactions['purchase_timestamp'] = pd.to_datetime(df_transactions['purchase_timestamp'], errors='coerce')
        st.success("Transactions data loaded successfully!")
    except Exception as e:
        st.error(f"Error loading Transactions data: {e}")

if orders_file:
    try:
        df_orders = pd.read_csv(orders_file)
        # Convert order_datetime to datetime if it exists (based on previous discussion)
        if 'order_datetime' in df_orders.columns:
            df_orders['order_datetime'] = pd.to_datetime(df_orders['order_datetime'], errors='coerce')
        st.success("Orders data loaded successfully!")
    except Exception as e:
        st.error(f"Error loading Orders data: {e}")

if customers_file:
    try:
        df_customers = pd.read_csv(customers_file)
        # Convert sign_up_date and dob to datetime
        if 'sign_up_date' in df_customers.columns:
            df_customers['sign_up_date'] = pd.to_datetime(df_customers['sign_up_date'], errors='coerce')
        if 'dob' in df_customers.columns:
            df_customers['dob'] = pd.to_datetime(df_customers['dob'], errors='coerce')
        st.success("Customers data loaded successfully!")
    except Exception as e:
        st.error(f"Error loading Customers data: {e}")

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
    if not df[column].empty:
        value_counts = df[column].value_counts().nlargest(top_n).reset_index()
        value_counts.columns = [column, 'Count']
        fig = px.bar(value_counts, x=column, y='Count', title=f'Top {top_n} {column} Distribution',
                     color=column,
                     color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(f"Categorical column '{column}' is empty, cannot plot distribution.")


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

    # Columns to explicitly exclude from plotting, beyond just 'id' keywords
    excluded_columns_for_plots = ['customer_name', 'email', 'dob']

    # Filter out ID columns and other explicitly excluded columns from numerical and categorical lists
    id_keywords = ['id', 'user_id', 'customer_id', 'transaction_id', 'order_item_id', 'visit_id', 'product_id', 'coupon_id']
    
    filtered_numerical_cols = [
        col for col in numerical_cols
        if not any(keyword in col.lower() for keyword in id_keywords) and col.lower() not in excluded_columns_for_plots
    ]
    
    filtered_categorical_cols = [
        col for col in categorical_cols
        if not any(keyword in col.lower() for keyword in id_keywords) and col.lower() not in excluded_columns_for_plots
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
if df_transactions is not None or df_orders is not None or df_customers is not None:
    st.header("🔬 Exploratory Data Analysis Results")
    tab1, tab2, tab3 = st.tabs(["Transactions EDA", "Orders EDA", "Customers EDA"])

    with tab1:
        perform_eda(df_transactions, "Transactions")
    with tab2:
        perform_eda(df_orders, "Orders")
    with tab3:
        perform_eda(df_customers, "Customers")
else:
    st.info("Please upload your CSV files to begin the EDA process.")

st.markdown("---")
st.markdown("Built with Streamlit and Plotly for interactive data exploration.")
