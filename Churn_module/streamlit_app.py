import streamlit as st
import pandas as pd
import numpy as np

# Set up the Streamlit page configuration
st.set_page_config(
    page_title="Cox PH Model Preprocessing",
    layout="centered", # Can be "wide" or "centered"
    initial_sidebar_state="auto"
)

st.title("📊 Cox PH Model Data Preprocessing")
st.markdown("---")

st.write(
    """
    Welcome! This app helps you preprocess your transaction-only ADS for Cox Proportional Hazards modeling.
    To begin, please upload your CSV file below.
    """
)

# --- File Upload Section ---
st.header("1. Upload Your Data 📂")
uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type="csv",
    help="Upload your dataset. It should be a transaction-only ADS (Analytical Data Set)."
)

df = None
if uploaded_file is not None:
    try:
        # Attempt to read CSV. We won't use parse_dates=True directly here in read_csv
        # because our custom get_column_types will handle parsing for identification.
        df = pd.read_csv(uploaded_file)
        st.success("CSV file uploaded successfully!")
        st.write("First 5 rows of your dataset:")
        st.dataframe(df.head())

        st.subheader("Dataset Information:")
        st.write(f"Number of rows: {df.shape[0]}")
        st.write(f"Number of columns: {df.shape[1]}")

    except Exception as e:
        st.error(f"Error loading file: {e}. Please ensure it's a valid CSV.")

st.markdown("---")

# --- Column Type Identification Logic (Your provided function) ---
def get_column_types(df):
    """
    Takes a pandas DataFrame and returns lists of numerical, categorical, and datetime columns.
    Automatically tries to detect datetime columns even if they are stored as object.
    """
    if df is None:
        return {"Numerical": [], "Categorical": [], "Datetime": [], "Other": []}

    # Detect numeric
    numerical_cols = df.select_dtypes(include=['number']).columns.tolist()

    # Detect datetime (both real datetime and object-like dates)
    datetime_cols = []
    # Create a temporary copy to avoid modifying the original DataFrame during detection
    df_temp = df.copy() 
    for col in df_temp.columns:
        if pd.api.types.is_datetime64_any_dtype(df_temp[col]):
            datetime_cols.append(col)
        elif df_temp[col].dtype == "object":
            # Try parsing a sample to see if it's datetime
            try:
                # Use errors='coerce' for detection to avoid stopping on first unparsable value
                # but append only if a significant portion can be converted
                # For robust detection, we might want to check a sample or a threshold of non-NaN values
                # For this specific function's logic, we stick to the 'raise' behavior for strictness.
                pd.to_datetime(df_temp[col], errors='raise') 
                datetime_cols.append(col)
            except Exception:
                pass # Not a datetime, move on

    # Categorical = objects that are not datetime + category dtype + boolean dtype
    categorical_cols = [
        col for col in df_temp.select_dtypes(include=['object', 'category', 'bool']).columns
        if col not in datetime_cols
    ]
    
    # Identify 'other' columns that aren't numerical, categorical, or datetime
    all_identified_cols = set(numerical_cols + categorical_cols + datetime_cols)
    other_cols = [col for col in df_temp.columns if col not in all_identified_cols]


    return {
        "Numerical": numerical_cols,
        "Categorical": categorical_cols,
        "Datetime": datetime_cols,
        "Other": other_cols # Include other columns in the returned dict
    }

# --- Column Type Identification Section ---
if df is not None:
    st.header("2. Identify Column Types 📋")
    st.write(
        """
        Based on your uploaded dataset, here are the identified numerical, categorical, and datetime columns.
        """
    )

    # Use your provided function to get column types
    col_types = get_column_types(df)
    numerical_cols = col_types["Numerical"]
    categorical_cols = col_types["Categorical"]
    datetime_cols = col_types["Datetime"]
    other_cols = col_types["Other"] # Retrieve other columns

    st.subheader("Numerical Columns:")
    if numerical_cols:
        st.markdown(f"- **Total: {len(numerical_cols)}**")
        for col in numerical_cols:
            st.code(col)
    else:
        st.info("No numerical columns found in the dataset.")

    st.subheader("Categorical Columns:")
    if categorical_cols:
        st.markdown(f"- **Total: {len(categorical_cols)}**")
        for col in categorical_cols:
            st.code(col)
    else:
        st.info("No categorical columns found in the dataset.")

    st.subheader("Datetime Columns:")
    if datetime_cols:
        st.markdown(f"- **Total: {len(datetime_cols)}**")
        for col in datetime_cols:
            st.code(col)
    else:
        st.info("No datetime columns found in the dataset.")

    if other_cols:
        st.subheader("Other Columns (Unclassified):")
        st.markdown(f"- **Total: {len(other_cols)}**")
        st.write("These columns did not fit into numerical, categorical, or datetime types. You might need to inspect them manually.")
        for col in other_cols:
            st.code(col)

    st.markdown("---")
    st.info(
        """
        **Note on Column Type Detection:**
        The detection now uses a more robust logic, attempting to convert 'object' columns to datetime.
        However, if date formats are highly inconsistent, manual conversion might still be needed.
        """
    )
else:
    st.info("Please upload a CSV file to see column types.")

st.sidebar.header("About This App")
st.sidebar.info(
    """
    This Streamlit application is designed to help you prepare your data for a
    Cox Proportional Hazards model. It's a step-by-step process, starting with
    data upload and basic column type identification.
    """
)
st.sidebar.markdown("---")
st.sidebar.write("Developed with ❤️ using Streamlit")
