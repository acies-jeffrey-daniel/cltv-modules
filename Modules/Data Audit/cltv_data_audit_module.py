import streamlit as st
import pandas as pd
import io

# --- Configuration (Hardcoded for now, can be loaded from external source) ---

# Expected columns for different dataset types
EXPECTED_COLUMNS = {
    "Customer": {
        "user_id": ["user_id"],
        "gender": ["gender"],
        "dob": ["dob"],
        "location": ["location"],
        "device_type": ["device_type"],
        "preferred_language": ["preferred_language"],
        "signup_date": ["signup_date"],
        "registration_status": ["registration_status"],
        "loyalty_program_member": ["loyalty_program_member"],
        "account_status": ["account_status"],
        "acquisition_channel": ["acquisition_channel"]
    },
    "Transactional": {
        "transaction_id": ["transaction_id"],
        "visit_id": ["visit_id"],
        "user_id": ["user_id"],
        "order_id": ["order_id"],
        "purchase_date": ["purchase_date"],
        "payment_method": ["payment_method"],
        "Total_amount_per_transaction": ["Total_amount_per_transaction"]
    },
    "Orders": {
        "transaction_id": ["transaction_id"],
        "order_id": ["order_id"],
        "product_id": ["product_id"],
        "quantity": ["quantity"],
        "total_product_price": ["total_product_price"],
        "Unit_price": ["Unit_price"],
        "discount_code_used": ["discount_code_used"],
        "discount_value": ["discount_value"],
        "shipping_cost": ["shipping_cost"],
        "total_payable": ["total_payable"],
        "return_stat": ["return_stat"],
        "return_date": ["return_date"],
        "Order_date": ["Order_date"],
        "Total_order_amount": ["Total_order_amount"]
    },
    "Behavioral": {
        "session_id": ["session_id"],
        "visit_id": ["visit_id"],
        "user_id": ["user_id"],
        "key": ["key"],
        "device_id": ["device_id"],
        "cookie_id": ["cookie_id"],
        "device": ["device"],
        "entry_channel": ["entry_channel"],
        "user_country": ["user_country"],
        "entry_page": ["entry_page"],
        "number_of_page_viewed": ["number_of_page_viewed"],
        "visit_datetime": ["visit_datetime"],
        "sponsored_listing_viewed": ["sponsored_listing_viewed"],
        "banner_viewed": ["banner_viewed"],
        "homepage_promo_seen": ["homepage_promo_seen"],
        "product_search_view": ["product_search_view"],
        "session_total_cost": ["session_total_cost"]
    },
    "Campaign Costs": {
        "campaign_id": ["campaign_id"],
        "campaign_name": ["campaign_name"],
        "channel": ["channel"],
        "target_segment": ["target_segment"],
        "start_date": ["start_date"],
        "end_date": ["end_date"],
        "campaign_budget": ["campaign_budget"],
        "impressions": ["impressions"],
        "clicks": ["clicks"],
        "ctr": ["ctr"],
        "conversions": ["conversions"],
        "conversion_rate": ["conversion_rate"],
        "cost_per_click": ["cost_per_click"],
        "cost_per_conversion": ["cost_per_conversion"],
        "revenue_generated": ["revenue_generated"],
        "roi_campaign_level": ["roi_campaign_level"],
        "creative_type": ["creative_type"],
        "campaign_objective": ["campaign_objective"],
        "landing_page_url": ["landing_page_url"],
        "attribution_model": ["attribution_model"],
        "touchpoints_before_conversion": ["touchpoints_before_conversion"],
        "avg_time_to_convert": ["avg_time_to_convert"],
        "device_type": ["device_type"],
        "geo_location": ["geo_location"]
    },
    "Session Activity Revenue": {
        "session_id": ["session_id"],
        "user_id": ["user_id"],
        "page": ["page"],
        "sponsored_listing_viewed": ["sponsored_listing_viewed"],
        "banner_viewed": ["banner_viewed"],
        "homepage_promo_seen": ["homepage_promo_seen"],
        "product_search_view": ["product_search_view"],
        "sponsored_listing_viewed_cost": ["sponsored_listing_viewed_cost"],
        "banner_viewed_cost": ["banner_viewed_cost"],
        "homepage_promo_seen_cost": ["homepage_promo_seen_cost"],
        "product_search_view_cost": ["product_search_view_cost"],
        "session_total_cost": ["session_total_cost"]
    },
    "Marketing Touchpoints": {
        "campaign_id": ["campaign_id"],
        "channel": ["channel"],
        "touchpoint_type": ["touchpoint_type"],
        "touch_date": ["touch_date"],
        "campaign_medium": ["campaign_medium"],
        "utm_source": ["utm_source"],
        "utm_campaign": ["utm_campaign"],
        "touchpoint_status": ["touchpoint_status"],
        "conversion_flag": ["conversion_flag"],
        "time_to_conversion": ["time_to_conversion"],
        "creative_id": ["creative_id"],
        "segment_id": ["segment_id"],
        "frequency": ["frequency"],
        "reach": ["reach"],
        "click_through_rate": ["click_through_rate"],
        "cost_per_click": ["cost_per_click"],
        "landing_page_url": ["landing_page_url"]
    },
    "Product Master": {
        "product_id": ["product_id"],
        "product_name": ["product_name"],
        "product_category": ["product_category"],
        "subcategory": ["subcategory"],
        "brand_name": ["brand_name"],
        "price": ["price"],
        "available_stock": ["available_stock"],
        "rating": ["rating"],
        "is_active": ["is_active"],
        "launch_date": ["launch_date"],
        "discount_eligible": ["discount_eligible"],
        "avg_shipping_cost": ["avg_shipping_cost"],
        "product_weight_kg": ["product_weight_kg"]
    }
}

# Prioritized columns for each dataset type (must exist)
PRIORITIZED_COLUMNS = {
    "Customer": ["user_id", "dob", "location"],
    "Transactional": ["transaction_id", "user_id", "purchase_date", "Total_amount_per_transaction"],
    "Orders": ["transaction_id", "order_id", "product_id", "Order_date"],
    "Behavioral": ["session_id", "user_id", "visit_datetime"],
    "Campaign Costs": ["campaign_id", "cost_per_click", "start_date", "revenue_generated"],
    "Session Activity Revenue": ["session_id", "user_id", "session_total_cost"],
    "Marketing Touchpoints": ["campaign_id", "touchpoint_type", "touch_date", "conversion_flag"],
    "Product Master": ["product_id", "product_name", "price", "available_stock"]
}

# Minimum required columns for ingestion (threshold)
MIN_REQUIRED_COLUMNS = {
    "Customer": 3,
    "Transactional": 4,
    "Orders": 4,
    "Behavioral": 3,
    "Campaign Costs": 4,
    "Session Activity Revenue": 3,
    "Marketing Touchpoints": 4,
    "Product Master": 4
}

# --- Helper Functions ---

def load_data(uploaded_file):
    """Loads data from an uploaded file into a pandas DataFrame."""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file type. Please upload a CSV or Excel file.")
            return None
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

def get_automated_mapping(df_columns, expected_cols_aliases):
    """
    Attempts to automatically map DataFrame columns to expected internal names.
    Returns a dictionary of {internal_name: df_column_name} and a list of unmapped df columns.
    """
    mapped_cols = {}
    unmapped_df_cols = list(df_columns) # Start with all DataFrame columns as unmapped

    for internal_name, aliases in expected_cols_aliases.items():
        for alias in aliases:
            if alias.lower() in [col.lower() for col in df_columns]:
                # Found a match, add to mapped and remove from unmapped
                original_df_col_name = next(col for col in df_columns if col.lower() == alias.lower())
                mapped_cols[internal_name] = original_df_col_name
                if original_df_col_name in unmapped_df_cols:
                    unmapped_df_cols.remove(original_df_col_name)
                break # Move to the next internal_name once a match is found
    return mapped_cols, unmapped_df_cols

def validate_columns(df_columns, dataset_type, current_mapping):
    """
    Validates if prioritized and minimum required columns exist based on current mapping.
    Returns a dictionary of validation results and a compliance score.
    """
    validation_results = {
        "missing_prioritized": [],
        "mapped_count": 0,
        "total_expected_prioritized": len(PRIORITIZED_COLUMNS.get(dataset_type, []))
    }
    compliance_score = 0

    expected_prioritized = PRIORITIZED_COLUMNS.get(dataset_type, [])
    expected_aliases_for_type = EXPECTED_COLUMNS.get(dataset_type, {})

    # Check for missing prioritized columns
    for prio_col in expected_prioritized:
        if prio_col not in current_mapping:
            validation_results["missing_prioritized"].append(prio_col)

    # Count how many expected columns (from the broader expected list) are mapped
    mapped_expected_cols = set()
    for internal_name in expected_aliases_for_type.keys():
        if internal_name in current_mapping:
            mapped_expected_cols.add(internal_name)
    validation_results["mapped_count"] = len(mapped_expected_cols)

    # Calculate compliance score (simple example: percentage of expected columns mapped)
    total_expected = len(expected_aliases_for_type)
    if total_expected > 0:
        compliance_score = (validation_results["mapped_count"] / total_expected) * 100

    return validation_results, compliance_score

# --- Streamlit App Layout ---

st.set_page_config(layout="wide", page_title="Data Ingestion Pre-validation")

st.title("📊 Data Ingestion Pre-validation Module")
st.markdown("""
This module helps validate, map, and verify the structure and content of your uploaded datasets
before they are ingested into the data pipeline.
""")

# --- Session State Initialization ---
if 'uploaded_files_data' not in st.session_state:
    st.session_state['uploaded_files_data'] = [] # Stores {'name': str, 'df': pd.DataFrame, 'type': str, 'mapping': dict, 'unmapped_df_cols': list, 'validation_results': dict, 'compliance_score': float}
if 'current_step' not in st.session_state:
    st.session_state['current_step'] = 'upload' # 'upload', 'mapping', 'health_check', 'summary'
# No need for selected_file_index in session state for st.tabs, as tabs manage their own state


# --- Navigation ---
st.sidebar.header("Navigation")
if st.sidebar.button("Upload Datasets", key="nav_upload"):
    st.session_state['current_step'] = 'upload'
if st.sidebar.button("Review Mappings", key="nav_mapping", disabled=not st.session_state['uploaded_files_data']):
    st.session_state['current_step'] = 'mapping'
if st.sidebar.button("Health Check", key="nav_health", disabled=not st.session_state['uploaded_files_data']):
    st.session_state['current_step'] = 'health_check'
if st.sidebar.button("Final Summary", key="nav_summary", disabled=not st.session_state['uploaded_files_data']):
    st.session_state['current_step'] = 'summary'


# --- Step 1: Upload Datasets ---
if st.session_state['current_step'] == 'upload':
    st.header("Step 1: Upload Your Datasets")
    st.write("Please upload your datasets using the dedicated sections below.")

    # Updated list of dataset types for upload
    dataset_types_for_upload = [
        "Transactional", "Orders", "Behavioral", "Customer",
        "Campaign Costs", "Session Activity Revenue", "Marketing Touchpoints",
        "Product Master"
    ]
    uploaded_files_dict = {}

    # Create columns for the uploaders (e.g., 2 columns per row)
    num_cols_per_row = 2
    cols = st.columns(num_cols_per_row)

    for idx, dataset_type in enumerate(dataset_types_for_upload):
        with cols[idx % num_cols_per_row]: # Place in column 0 or 1 based on index
            st.markdown(f"**Upload {dataset_type} Dataset**")
            uploaded_file = st.file_uploader(
                f"Choose {dataset_type} file (CSV or Excel)",
                type=["csv", "xls", "xlsx"],
                key=f"file_uploader_{dataset_type.lower().replace(' ', '_').replace('-', '_')}" # Sanitize key
            )
            if uploaded_file:
                uploaded_files_dict[dataset_type] = uploaded_file
            # st.markdown("---") # Removed separator for cleaner card effect, columns provide separation

    # Process newly uploaded files
    new_files_data = []
    for dataset_type, uploaded_file in uploaded_files_dict.items():
        file_name = uploaded_file.name
        # Check if this file (by name and type) has already been processed in the current session state
        existing_file = next((f_data for f_data in st.session_state['uploaded_files_data']
                              if f_data['name'] == file_name and f_data['type'] == dataset_type), None)

        if not existing_file:
            df = load_data(uploaded_file)
            if df is not None:
                file_data_entry = {
                    'name': file_name,
                    'df': df,
                    'type': dataset_type, # Type is now set by the uploader
                    'mapping': {},
                    'unmapped_df_cols': [],
                    'validation_results': {},
                    'compliance_score': 0.0,
                    'user_decision_proceed': False,
                    'manual_changes_made': False
                }
                new_files_data.append(file_data_entry)
        else:
            st.info(f"File '{file_name}' for '{dataset_type}' already uploaded and processed.")

    # Add only new files to session state
    for new_file_data in new_files_data:
        # Before adding, remove any old entry for the same file name and type if it exists (e.g., if re-uploaded via the specific uploader)
        st.session_state['uploaded_files_data'] = [
            f_data for f_data in st.session_state['uploaded_files_data']
            if not (f_data['name'] == new_file_data['name'] and f_data['type'] == new_file_data['type'])
        ]
        st.session_state['uploaded_files_data'].append(new_file_data)


    if st.session_state['uploaded_files_data']:
        st.subheader("Currently Loaded Datasets")
        
        # Create tabs for each uploaded dataset in Step 1
        tab_titles = [f"{f['name']} ({f['type']})" for f in st.session_state['uploaded_files_data']]
        tabs = st.tabs(tab_titles)

        for i, tab in enumerate(tabs):
            with tab:
                file_data = st.session_state['uploaded_files_data'][i]
                st.write("Preview:")
                st.dataframe(file_data['df'].head())
                st.write("Columns:")
                st.write(file_data['df'].columns.tolist())
                st.info(f"Dataset type: **{file_data['type']}** (pre-defined)")


        # Check if all uploaded files have a recognized type
        can_proceed_to_mapping = all(
            f['type'] is not None and f['type'] in EXPECTED_COLUMNS
            for f in st.session_state['uploaded_files_data']
        )

        if st.button("Proceed to Column Mapping", key="proceed_mapping_btn", disabled=not can_proceed_to_mapping):
            st.session_state['current_step'] = 'mapping'
            st.rerun() # Rerun to switch to the next step
        elif not can_proceed_to_mapping and st.session_state['uploaded_files_data']:
            st.warning("Please ensure all uploaded datasets have a valid and recognized type to proceed.")


# --- Step 2: Column Matching (Automated & Manual) ---
elif st.session_state['current_step'] == 'mapping':
    st.header("Step 2: Column Mapping")
    st.write("Review the automated column mappings and perform manual mapping if needed.")

    if not st.session_state['uploaded_files_data']:
        st.warning("Please upload datasets first in the 'Upload Datasets' section.")
    else:
        # Create tabs for each uploaded dataset
        tab_titles = [f"{f['name']} ({f['type']})" for f in st.session_state['uploaded_files_data']]
        tabs = st.tabs(tab_titles)

        for i, tab in enumerate(tabs):
            with tab:
                file_data = st.session_state['uploaded_files_data'][i]

                st.subheader(f"Mapping for: {file_data['name']} ({file_data['type']})")
                if file_data['type'] is None or file_data['type'] not in EXPECTED_COLUMNS:
                    st.error(f"Dataset type '{file_data['type']}' is invalid or not selected for this file. Please go back to 'Upload Datasets' to correct it.")
                else:
                    df_cols = file_data['df'].columns.tolist()
                    expected_aliases = EXPECTED_COLUMNS.get(file_data['type'], {})

                    # Perform automated mapping if not already done or if re-upload/type change
                    if not file_data['mapping'] or (not file_data['user_decision_proceed'] and not file_data['manual_changes_made']):
                        auto_mapped, unmapped_df_cols = get_automated_mapping(df_cols, expected_aliases)
                        st.session_state['uploaded_files_data'][i]['mapping'] = auto_mapped
                        st.session_state['uploaded_files_data'][i]['unmapped_df_cols'] = unmapped_df_cols
                        st.session_state['uploaded_files_data'][i]['manual_changes_made'] = False # Reset flag

                    current_mapping = st.session_state['uploaded_files_data'][i]['mapping']
                    unmapped_df_cols = st.session_state['uploaded_files_data'][i]['unmapped_df_cols']

                    st.markdown("---")
                    st.markdown("#### Automated Mappings:")
                    if current_mapping:
                        for internal_name, df_col in current_mapping.items():
                            st.write(f"- **{internal_name}** (Internal) mapped to **'{df_col}'** (Uploaded)")
                    else:
                        st.info("No automated mappings found yet.")

                    # Manual Mapping section within an expander
                    with st.expander("Manual Column Mapping (Fallback)"):
                        st.info("If any expected columns are not automatically mapped, or if you want to change a mapping, use the dropdowns below.")

                        manual_mapping_changed = False
                        for internal_name in expected_aliases.keys(): # Iterate through expected internal names
                            current_mapped_val = current_mapping.get(internal_name)
                            
                            # Options for the selectbox: empty string, unmapped columns, and other df columns
                            options = [""] + sorted(unmapped_df_cols) + sorted([col for col in df_cols if col not in unmapped_df_cols and col != current_mapped_val])

                            # Set initial index for the selectbox
                            initial_index = 0
                            if current_mapped_val in options:
                                initial_index = options.index(current_mapped_val)
                            elif current_mapped_val: # If it's mapped but not in options (e.g., if it was manually removed from unmapped_df_cols)
                                options.insert(1, current_mapped_val) # Insert it after the empty string
                                initial_index = 1 # Set index to the newly inserted value

                            selected_df_col = st.selectbox(
                                f"Map **'{internal_name}'** (Expected) to:",
                                options=options,
                                index=initial_index,
                                key=f"manual_map_{i}_{internal_name}" # Use 'i' for unique key per tab
                            )

                            if selected_df_col != current_mapped_val: # Check if the selection actually changed
                                if selected_df_col == "": # User selected to unmap
                                    if internal_name in st.session_state['uploaded_files_data'][i]['mapping']:
                                        old_mapped_col = st.session_state['uploaded_files_data'][i]['mapping'].pop(internal_name)
                                        if old_mapped_col in df_cols and old_mapped_col not in st.session_state['uploaded_files_data'][i]['unmapped_df_cols']:
                                            st.session_state['uploaded_files_data'][i]['unmapped_df_cols'].append(old_mapped_col)
                                        manual_mapping_changed = True
                                        st.session_state['uploaded_files_data'][i]['manual_changes_made'] = True
                                elif selected_df_col: # User selected a new mapping
                                    # If there was an old mapping, return it to unmapped_df_cols
                                    if internal_name in st.session_state['uploaded_files_data'][i]['mapping']:
                                        old_mapped_col = st.session_state['uploaded_files_data'][i]['mapping'][internal_name]
                                        if old_mapped_col in df_cols and old_mapped_col not in st.session_state['uploaded_files_data'][i]['unmapped_df_cols']:
                                            st.session_state['uploaded_files_data'][i]['unmapped_df_cols'].append(old_mapped_col)

                                    st.session_state['uploaded_files_data'][i]['mapping'][internal_name] = selected_df_col
                                    # Remove the newly mapped column from unmapped_df_cols if it was there
                                    if selected_df_col in st.session_state['uploaded_files_data'][i]['unmapped_df_cols']:
                                        st.session_state['uploaded_files_data'][i]['unmapped_df_cols'].remove(selected_df_col)
                                    manual_mapping_changed = True
                                    st.session_state['uploaded_files_data'][i]['manual_changes_made'] = True
                        
                        if manual_mapping_changed:
                            st.info("Manual mapping updated. Please review.")
                            # Recalculate validation results immediately after manual change
                            val_res, comp_score = validate_columns(
                                df_cols,
                                file_data['type'],
                                st.session_state['uploaded_files_data'][i]['mapping']
                            )
                            st.session_state['uploaded_files_data'][i]['validation_results'] = val_res
                            st.session_state['uploaded_files_data'][i]['compliance_score'] = comp_score
                            st.rerun() # Rerun to refresh the UI with new mappings and unmapped columns

                    st.markdown("---")
                    st.write("Current Mapped Columns (after automated and manual adjustments):")
                    if st.session_state['uploaded_files_data'][i]['mapping']:
                        st.json(st.session_state['uploaded_files_data'][i]['mapping'])
                    else:
                        st.info("No columns mapped for this dataset yet.")

                    st.write("Remaining Unmapped Columns in your dataset:")
                    if st.session_state['uploaded_files_data'][i]['unmapped_df_cols']:
                        st.write(st.session_state['uploaded_files_data'][i]['unmapped_df_cols'])
                    else:
                        st.info("All columns in your dataset are mapped or not considered expected.")

        if st.button("Proceed to Health Check", key="proceed_health_btn"):
            # Before proceeding, run a final validation for all files
            for i, file_data in enumerate(st.session_state['uploaded_files_data']):
                df_cols = file_data['df'].columns.tolist()
                val_res, comp_score = validate_columns(
                    df_cols,
                    file_data['type'],
                    file_data['mapping']
                )
                st.session_state['uploaded_files_data'][i]['validation_results'] = val_res
                st.session_state['uploaded_files_data'][i]['compliance_score'] = comp_score

            st.session_state['current_step'] = 'health_check'
            st.rerun()

# --- Step 3: Health Check Page ---
elif st.session_state['current_step'] == 'health_check':
    st.header("Step 3: Data Health Check")
    st.write("Review the health of your datasets based on mapping and validation.")

    if not st.session_state['uploaded_files_data']:
        st.warning("Please upload and map datasets first.")
    else:
        # Create tabs for each uploaded dataset
        tab_titles = [f"{f['name']} ({f['type']})" for f in st.session_state['uploaded_files_data']]
        tabs = st.tabs(tab_titles)

        for i, tab in enumerate(tabs):
            with tab:
                file_data = st.session_state['uploaded_files_data'][i]

                st.subheader(f"Health Check for: {file_data['name']} ({file_data['type']})")

                if not file_data['type'] or file_data['type'] not in EXPECTED_COLUMNS:
                    st.error(f"Dataset type '{file_data['type']}' is invalid or not selected for this file. Please go back to 'Upload Datasets' to correct it.")
                else:
                    # Ensure validation results are up-to-date
                    df_cols = file_data['df'].columns.tolist()
                    val_res, comp_score = validate_columns(
                        df_cols,
                        file_data['type'],
                        file_data['mapping']
                    )
                    st.session_state['uploaded_files_data'][i]['validation_results'] = val_res
                    st.session_state['uploaded_files_data'][i]['compliance_score'] = comp_score

                    st.markdown("---")
                    st.markdown("#### Preview of Uploaded Data:")
                    st.dataframe(file_data['df'].head())

                    st.markdown("#### Validation Summary:")
                    if val_res["missing_prioritized"]:
                        st.error(f"🚨 Missing Prioritized Columns: {', '.join(val_res['missing_prioritized'])}")
                        st.info("Suggestion: Go back to 'Review Mappings' to manually map these columns or re-upload data.")
                    else:
                        st.success("✅ All prioritized columns are mapped!")

                    # Threshold Verification
                    min_required = MIN_REQUIRED_COLUMNS.get(file_data['type'], 0)
                    if val_res["mapped_count"] < min_required:
                        st.warning(f"⚠️ Threshold Alert: Only {val_res['mapped_count']} out of {min_required} required columns are mapped for ingestion.")
                    else:
                        st.success(f"👍 Threshold Met: {val_res['mapped_count']} columns mapped (min required: {min_required}).")

                    st.info(f"Compliance Score (Mapped Expected Columns): {comp_score:.2f}%")

                    st.markdown("#### Corrective Actions:")
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button(f"Re-upload Data for {file_data['name']}", key=f"re_upload_{i}"):
                            # Remove this file from session state to allow re-upload
                            st.session_state['uploaded_files_data'] = [f for f in st.session_state['uploaded_files_data'] if f['name'] != file_data['name'] or f['type'] != file_data['type']]
                            st.session_state['current_step'] = 'upload'
                            st.rerun()
                    with col2:
                        if st.button(f"Adjust Mappings for {file_data['name']}", key=f"adjust_map_{i}"):
                            st.session_state['current_step'] = 'mapping'
                            st.rerun()

                    st.markdown("---")
                    # User decision to proceed with this file
                    st.session_state['uploaded_files_data'][i]['user_decision_proceed'] = st.checkbox(
                        f"I accept the current mapping and health status for '{file_data['name']}' and wish to proceed.",
                        value=st.session_state['uploaded_files_data'][i]['user_decision_proceed'],
                        key=f"accept_proceed_{i}"
                    )
        
        # Check if all files have been accepted to enable the final summary button
        all_files_accepted = all(f['user_decision_proceed'] for f in st.session_state['uploaded_files_data'])

        if st.button("Generate Final Summary", key="generate_summary_btn", disabled=not all_files_accepted):
            st.session_state['current_step'] = 'summary'
            st.rerun()
        elif not all_files_accepted and st.session_state['uploaded_files_data']:
            st.warning("Please accept the health status for all files to proceed to the final summary.")


# --- Step 4: Final Summary ---
elif st.session_state['current_step'] == 'summary':
    st.header("Step 4: Final Summary & Output")
    st.write("Here's a summary of the validation process and generated outputs.")

    if not st.session_state['uploaded_files_data']:
        st.warning("No data processed yet. Please start from 'Upload Datasets'.")
    else:
        # Create tabs for each uploaded dataset
        tab_titles = [f"{f['name']} ({f['type']})" for f in st.session_state['uploaded_files_data']]
        tabs = st.tabs(tab_titles)

        for i, tab in enumerate(tabs):
            with tab:
                file_data = st.session_state['uploaded_files_data'][i]

                st.subheader(f"Summary for: {file_data['name']}")

                st.markdown("#### Mapped Column Dictionary:")
                if file_data['mapping']:
                    st.json(file_data['mapping'])
                else:
                    st.info("No columns were mapped for this dataset.")

                st.markdown("#### Health Check Report:")
                if file_data['validation_results']:
                    val_res = file_data['validation_results']
                    st.write(f"- **Missing Prioritized Columns:** {', '.join(val_res['missing_prioritized']) if val_res['missing_prioritized'] else 'None'}")
                    st.write(f"- **Mapped Expected Columns:** {val_res['mapped_count']}")
                    st.write(f"- **Compliance Score:** {file_data['compliance_score']:.2f}%")
                    min_required = MIN_REQUIRED_COLUMNS.get(file_data['type'], 0)
                    st.write(f"- **Threshold Met:** {'Yes' if val_res['mapped_count'] >= min_required else 'No'} (Min required: {min_required})")
                else:
                    st.info("Health check not performed or results not available.")

                st.markdown("#### User Decision Output:")
                st.write(f"- **Accepted Mappings & Proceeded:** {'Yes' if file_data['user_decision_proceed'] else 'No'}")
                st.write(f"- **Manual Changes Made:** {'Yes' if file_data['manual_changes_made'] else 'No'}")

                st.markdown("#### Alert/Notification Log:")
                alerts = []
                if file_data['validation_results'] and file_data['validation_results']['missing_prioritized']:
                    alerts.append(f"Missing prioritized columns: {', '.join(file_data['validation_results']['missing_prioritized'])}")
                min_required = MIN_REQUIRED_COLUMNS.get(file_data['type'], 0)
                if file_data['validation_results'] and file_data['validation_results']['mapped_count'] < min_required:
                    alerts.append(f"Threshold violation: Only {file_data['validation_results']['mapped_count']} of {min_required} required columns mapped.")
                if file_data['manual_changes_made']:
                    alerts.append("Manual column mapping was performed.")
                if not alerts:
                    alerts.append("No critical alerts generated.")
                for alert in alerts:
                    st.write(f"- {alert}")
                st.markdown("---")

        st.success("Data validation and mapping process completed!")
        st.info("You can now proceed with ingesting the validated data into your pipeline.")
