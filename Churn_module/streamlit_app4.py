import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

# Set up the Streamlit page configuration
st.set_page_config(
    page_title="Cox PH Model Preprocessing",
    layout="centered",
    initial_sidebar_state="auto"
)

# Initialize session state for DataFrame and other processing states
if 'df' not in st.session_state:
    st.session_state.df = None
if 'uploaded_file_name' not in st.session_state:
    st.session_state.uploaded_file_name = None
if 'col_types' not in st.session_state:
    st.session_state.col_types = {"Numerical": [], "Categorical": [], "Datetime": [], "Other": []}
if 'df_processed' not in st.session_state: # This DataFrame will be modified through processing steps
    st.session_state.df_processed = None
if 'cols_to_remove' not in st.session_state: # Stores columns selected for removal
    st.session_state.cols_to_remove = []

st.title("📊 Cox PH Model Data Preprocessing")
st.markdown("---")

st.write(
    """
    Welcome! This app helps you preprocess your transaction-only ADS for Cox Proportional Hazards modeling.
    Navigate through the tabs below to perform different preprocessing steps.
    """
)

# --- Column Type Identification Logic (Moved to top for function access) ---
def get_column_types(df):
    """
    Takes a pandas DataFrame and returns lists of numerical, categorical, and datetime columns.
    Automatically tries to detect datetime columns even if they are stored as object.
    """
    if df is None:
        return {"Numerical": [], "Categorical": [], "Datetime": [], "Other": []}

    numerical_cols = df.select_dtypes(include=['number']).columns.tolist()

    datetime_cols = []
    df_temp = df.copy() # Operate on a copy to avoid SettingWithCopyWarning if types are converted
    for col in df_temp.columns:
        if pd.api.types.is_datetime64_any_dtype(df_temp[col]):
            datetime_cols.append(col)
        elif df_temp[col].dtype == "object":
            try:
                # Attempt conversion; if it fails, it's not a datetime
                pd.to_datetime(df_temp[col], errors='raise')
                datetime_cols.append(col)
            except Exception:
                pass

    categorical_cols = [
        col for col in df_temp.select_dtypes(include=['object', 'category', 'bool']).columns
        if col not in datetime_cols
    ]

    all_identified_cols = set(numerical_cols + categorical_cols + datetime_cols)
    other_cols = [col for col in df_temp.columns if col not in all_identified_cols]

    return {
        "Numerical": numerical_cols,
        "Categorical": categorical_cols,
        "Datetime": datetime_cols,
        "Other": other_cols
    }

# --- Collinearity and VIF Analysis Function (Moved to top for function access) ---
def check_collinearity_vif(df, numerical_cols, corr_threshold=0.8, vif_threshold=5.0):
    """
    Checks for high collinearity and VIF among numerical columns.
    Returns:
        - highly_correlated_pairs: list of tuples (col1, col2, correlation)
        - high_vif_cols: list of tuples (col, vif_value)
    """
    highly_correlated_pairs = []
    high_vif_cols = []

    if not numerical_cols:
        return highly_correlated_pairs, high_vif_cols

    # 1. Collinearity (Correlation)
    st.markdown("##### Correlation Analysis")
    st.write(f"Identifying pairs with absolute correlation greater than `{corr_threshold}`:")
    
    # Ensure numerical_cols actually exist in the DataFrame before attempting correlation
    numerical_cols_present = [col for col in numerical_cols if col in df.columns]
    if not numerical_cols_present:
        st.info("No numerical columns available for correlation analysis after previous steps.")
        return highly_correlated_pairs, high_vif_cols

    corr_matrix = df[numerical_cols_present].corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    found_corr_pairs = False
    for i in range(len(upper_tri.columns)):
        for j in range(i + 1, len(upper_tri.columns)):
            col1 = upper_tri.columns[i]
            col2 = upper_tri.columns[j]
            correlation = upper_tri.iloc[i, j]
            if pd.notna(correlation) and correlation > corr_threshold:
                highly_correlated_pairs.append((col1, col2, correlation))
                st.write(f"- `{col1}` and `{col2}`: Correlation = `{correlation:.3f}`")
                found_corr_pairs = True

    if not found_corr_pairs:
        st.info("No highly correlated numerical column pairs found above the threshold.")
    st.markdown("---")

    # 2. VIF (Variance Inflation Factor)
    st.markdown("##### Variance Inflation Factor (VIF) Analysis")
    st.write(f"Identifying columns with VIF greater than `{vif_threshold}`:")

    clean_numerical_cols_for_vif = []
    for col in numerical_cols_present: # Use numerical_cols_present to avoid errors for dropped columns
        if df[col].isnull().sum() == 0:
            clean_numerical_cols_for_vif.append(col)
        else:
            st.warning(f"Column `{col}` contains missing values and will be skipped for VIF calculation. Please handle NaNs first.")

    if not clean_numerical_cols_for_vif:
        st.info("No numerical columns without missing values to calculate VIF.")
        return highly_correlated_pairs, high_vif_cols

    # VIF requires at least 2 features apart from the constant
    if len(clean_numerical_cols_for_vif) < 2:
        st.info("Need at least two non-missing numerical columns to calculate VIF.")
        return highly_correlated_pairs, high_vif_cols

    try:
        X = add_constant(df[clean_numerical_cols_for_vif], has_constant='add') # Use has_constant='add' to prevent error if already added
        vif_data = pd.DataFrame()
        vif_data["feature"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

        vif_data = vif_data[vif_data['feature'] != 'const'] # Exclude the constant

        found_vif_cols = False
        for index, row in vif_data.iterrows():
            if row["VIF"] > vif_threshold:
                high_vif_cols.append((row["feature"], row["VIF"]))
                st.write(f"- `{row['feature']}`: VIF = `{row['VIF']:.3f}`")
                found_vif_cols = True

        if not found_vif_cols:
            st.info("No numerical columns found with VIF above the threshold.")

    except Exception as e:
        st.error(f"Error calculating VIF: {e}. This can happen if all selected numerical columns are perfectly correlated or if a column has zero variance (constant column).")
        st.info("Please ensure your numerical columns have sufficient variance and are not perfectly correlated before calculating VIF.")

    st.markdown("---")
    return highly_correlated_pairs, high_vif_cols


# --- Streamlit Tabs ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1. Upload Data 📁",
    "2. Identify Column Types 📋",
    "3. Outlier Capping (Revenue) 📈",
    "4. Collinearity & VIF 📉",
    "5. Apply Column Changes ✂️"
])

with tab1:
    st.header("1. Upload Your Data 📂")
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type="csv",
        help="Upload your dataset. It should be a transaction-only ADS (Analytical Data Set)."
    )

    if uploaded_file is not None:
        try:
            # Check if a new file is uploaded or if the existing DataFrame is different
            if st.session_state.df is None or uploaded_file.name != st.session_state.uploaded_file_name:
                df_loaded = pd.read_csv(uploaded_file)
                st.session_state.df = df_loaded.copy() # Store original for reset
                st.session_state.df_processed = df_loaded.copy() # This DF will be modified
                st.session_state.uploaded_file_name = uploaded_file.name # Store file name to check for new uploads
                st.success("CSV file uploaded successfully!")
                st.write("First 5 rows of your dataset:")
                st.dataframe(st.session_state.df.head())

                st.subheader("Dataset Information:")
                st.write(f"Number of rows: {st.session_state.df.shape[0]}")
                st.write(f"Number of columns: {st.session_state.df.shape[1]}")

                # Automatically identify column types on upload
                st.session_state.col_types = get_column_types(st.session_state.df_processed)

                # Clear previous state when a new file is uploaded
                st.session_state.cols_to_remove = [] # Clear columns to remove
                

            else:
                st.info("Using previously uploaded file. Upload a new file to change.")


        except Exception as e:
            st.error(f"Error loading file: {e}. Please ensure it's a valid CSV.")
    else:
        st.info("Please upload a CSV file to begin.")

with tab2:
    st.header("2. Identify Column Types 📋")
    st.write(
        """
        Based on your uploaded dataset, here are the identified numerical, categorical, and datetime columns.
        """
    )
    if st.session_state.df_processed is not None:
        col_types = st.session_state.col_types # Use the pre-identified types
        numerical_cols = col_types["Numerical"]
        categorical_cols = col_types["Categorical"]
        datetime_cols = col_types["Datetime"]
        other_cols = col_types["Other"]

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
            The detection uses a robust logic, attempting to convert 'object' columns to datetime.
            However, if date formats are highly inconsistent, manual conversion might still be needed.
            """
        )
    else:
        st.info("Please upload a CSV file in the 'Upload Data' tab to identify column types.")

with tab3:
    st.header("3. Outlier Capping (Revenue Columns) 📈")
    st.write("This step helps to mitigate the impact of extreme values in your numerical columns, especially 'revenue' related ones. This is done so the model can perform better. 📊")

    if st.session_state.df_processed is not None:
        numerical_cols_available = st.session_state.col_types["Numerical"]

        # Pre-select columns containing 'revenue' in their name (case-insensitive)
        default_selected_revenue_cols = [
            col for col in numerical_cols_available
            if 'revenue' in col.lower()
        ]

        selected_cols_for_capping = st.multiselect(
            "Select numerical columns to cap outliers (e.g., revenue columns):",
            options=numerical_cols_available,
            default=default_selected_revenue_cols,
            help="Select columns where you want to apply outlier capping. For revenue columns, this is highly recommended."
        )

        if st.button("Apply Outlier Capping"):
            if selected_cols_for_capping:
                modified_count = 0
                for col in selected_cols_for_capping:
                    # Calculate 75th percentile
                    q75 = st.session_state.df_processed[col].quantile(0.75)
                    # Calculate cap value
                    cap_value = 3 * q75

                    # Count values above cap before capping
                    values_above_cap = st.session_state.df_processed[st.session_state.df_processed[col] > cap_value].shape[0]

                    # Apply capping
                    st.session_state.df_processed[col] = st.session_state.df_processed[col].clip(upper=cap_value)

                    if values_above_cap > 0:
                        st.success(f"Capped **{values_above_cap}** outliers in column `{col}`. Values above **{cap_value:.2f}** were replaced by this value.")
                        modified_count += 1
                    else:
                        st.info(f"No outliers found above the cap of **{cap_value:.2f}** in column `{col}`. No changes applied.")

                if modified_count > 0:
                    st.write("DataFrame after capping (first 5 rows):")
                    st.dataframe(st.session_state.df_processed.head())
                    st.success("Outlier capping applied successfully to selected columns! 🎉")
                    # Re-identify column types in case capping changed anything (unlikely, but good practice)
                    st.session_state.col_types = get_column_types(st.session_state.df_processed)

                else:
                    st.warning("No columns were modified, either because no outliers were found or no columns were selected for capping.")
            else:
                st.warning("Please select at least one column to apply outlier capping.")
        st.info("The capping will replace values above `3 * 75th percentile` with this cap value.")
    else:
        st.info("Please upload a CSV file in the 'Upload Data' tab to perform outlier capping.")

with tab4:
    st.header("4. Collinearity and VIF Analysis 📉")
    st.write("Let's check for high collinearity and Variance Inflation Factor (VIF) among your numerical features.")

    if st.session_state.df_processed is not None:
        # Get numerical columns from the potentially modified DataFrame
        numerical_cols_for_vif = st.session_state.col_types["Numerical"]

        # Get the highly correlated and high VIF columns
        highly_correlated_pairs, high_vif_cols = check_collinearity_vif(
            st.session_state.df_processed,
            numerical_cols_for_vif,
            corr_threshold=0.8,
            vif_threshold=5.0
        )

        all_problematic_cols = set()
        for col1, col2, _ in highly_correlated_pairs:
            all_problematic_cols.add(col1)
            all_problematic_cols.add(col2)
        for col, _ in high_vif_cols:
            all_problematic_cols.add(col)

        if not all_problematic_cols:
            st.success("🎉 No numerical columns found with high collinearity (correlation > 0.8) or high VIF (> 5). You're good to go!")
            st.session_state.cols_to_remove = [] # Ensure it's cleared if no problems found
        else:
            st.warning("⚠️ **Potential Multicollinearity Detected!**")
            st.write("Consider removing or transforming the following columns to reduce multicollinearity, which can negatively impact model stability and interpretability:")
            for col in sorted(list(all_problematic_cols)):
                st.code(col)

            st.subheader("Choose Removal Strategy:")
            removal_option = st.radio(
                "How would you like to handle these columns?",
                ("Do nothing for now", "Automatically select columns for removal", "Manually select columns for removal"),
                key="removal_option" # Add a key to avoid duplicate widget warnings
            )

            if removal_option == "Automatically select columns for removal":
                st.write("The following columns would be automatically selected for removal:")
                cols_auto_remove = sorted(list(all_problematic_cols))
                if cols_auto_remove:
                    for col in cols_auto_remove:
                        st.code(col)
                    st.session_state.cols_to_remove = cols_auto_remove
                    st.info("These columns are marked for automatic removal. Proceed to 'Apply Column Changes' tab to confirm and remove.")
                else:
                    st.info("No columns were identified for automatic removal based on current thresholds.")
                    st.session_state.cols_to_remove = []

            elif removal_option == "Manually select columns for removal":
                st.write("Please select the columns you wish to remove:")
                manual_selection_options = sorted(list(all_problematic_cols))
                if manual_selection_options:
                    selected_for_removal = st.multiselect(
                        "Select columns to remove:",
                        options=manual_selection_options,
                        default=st.session_state.cols_to_remove, # Pre-fill with current selection if any
                        key="manual_col_selection" # Add a key
                    )
                    st.session_state.cols_to_remove = selected_for_removal
                    if selected_for_removal:
                        st.write("You have selected the following columns for removal:")
                        for col in selected_for_removal:
                            st.code(col)
                        st.info("Proceed to 'Apply Column Changes' tab to confirm and remove.")
                    else:
                        st.info("No columns selected for manual removal.")
                else:
                    st.info("No problematic columns found for manual selection.")
                    st.session_state.cols_to_remove = []
            else: # "Do nothing for now"
                st.info("No columns will be removed at this stage.")
                st.session_state.cols_to_remove = [] # Clear any previous selection
    else:
        st.info("Please upload a CSV file in the 'Upload Data' tab to perform collinearity and VIF analysis.")

with tab5:
    st.header("5. Apply Column Changes ✂️")
    st.write("This step will apply the selected column removals from the 'Collinearity & VIF' analysis.")

    if st.session_state.df_processed is not None:
        if st.session_state.cols_to_remove:
            st.warning("You have selected the following columns for removal:")
            for col in sorted(st.session_state.cols_to_remove):
                st.code(col)

            if st.button("Confirm and Remove Selected Columns"):
                # Filter out columns that might have been removed already or don't exist
                # This ensures we only try to drop existing columns
                cols_to_actually_drop = [
                    col for col in st.session_state.cols_to_remove
                    if col in st.session_state.df_processed.columns
                ]

                if cols_to_actually_drop:
                    original_cols_count = st.session_state.df_processed.shape[1]
                    st.session_state.df_processed = st.session_state.df_processed.drop(columns=cols_to_actually_drop)
                    st.success(f"Successfully removed **{len(cols_to_actually_drop)}** columns.")
                    st.write("DataFrame after removal (first 5 rows):")
                    st.dataframe(st.session_state.df_processed.head())
                    st.write(f"New number of columns: {st.session_state.df_processed.shape[1]} (was {original_cols_count})")

                    # Re-identify column types after removal
                    st.session_state.col_types = get_column_types(st.session_state.df_processed)
                    st.session_state.cols_to_remove = [] # Clear the list after applying
                    st.info("The DataFrame has been updated. You can now proceed with further steps or download the modified data.")
                else:
                    st.info("No valid columns were found to be removed from the current DataFrame.")
            else:
                st.info("Click the button above to apply the removal.")
        else:
            st.info("No columns have been marked for removal. Proceed to the 'Collinearity & VIF' tab to select columns.")
    else:
        st.info("Please upload a CSV file in the 'Upload Data' tab to begin data preprocessing.")


st.sidebar.header("About This App")
st.sidebar.info(
    """
    This Streamlit application is designed to help you prepare your data for a
    Cox Proportional Hazards model. It's a step-by-step process, now with
    tabs for better organization: data upload, column type identification,
    outlier capping, collinearity/VIF analysis, and applying column changes.
    """
)
st.sidebar.markdown("---")
st.sidebar.write("Developed with ❤️ using Streamlit")
