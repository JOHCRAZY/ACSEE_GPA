# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# import os
# import matplotlib.pyplot as plt
# import seaborn as sns
# import plotly.express as px
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# from sklearn.pipeline import Pipeline
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# from sklearn.linear_model import LinearRegression, Ridge, Lasso
# from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# from sklearn.inspection import permutation_importance
# import shap
# import pickle
# from datetime import datetime
# import warnings
# warnings.filterwarnings('ignore')

# # Set page configuration
# st.set_page_config(
#     page_title="GPA Prediction Dashboard",
#     page_icon="🎓",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Load data
# @st.cache_data
# def load_data():
#     """Load the sample data"""
#     try:
#         df = pd.read_csv('data/df.csv')
#         return df
#     except Exception as e:
#         st.error(f"Could not load data: {e}")
#         return None

# # Load model
# @st.cache_resource
# def load_linear_regression_model():
#     """Load the linear regression model"""
#     try:
#         model = joblib.load('models/linear_regression_pipeline.pkl')
#         return model
#     except Exception as e:
#         st.error(f"Could not load model: {e}")
#         return None

# # Feature engineering function
# def engineer_features(df):
#     """Add derived features to the dataframe"""
#     df = df.copy()
    
#     # Convert column names to lowercase for consistency with the model
#     column_mapping = {
#         'PREV_SAT': 'prev_sat',
#         'PREV_GPA': 'prev_gpa', 
#         'SAT': 'sat',
#         'STUDENT_TEACHER_RATIO': 'student_teacher_ratio'
#     }
    
#     # Rename columns to match expected format
#     for old_col, new_col in column_mapping.items():
#         if old_col in df.columns:
#             df[new_col] = df[old_col]
    
#     # Ensure we have the required base columns
#     required_cols = ['prev_sat', 'prev_gpa', 'sat', 'student_teacher_ratio']
#     for col in required_cols:
#         if col not in df.columns:
#             # Set default values if columns are missing
#             defaults = {'prev_sat': 500, 'prev_gpa': 3.0, 'sat': 500, 'student_teacher_ratio': 15}
#             df[col] = defaults[col]
    
#     # Create derived features safely
#     df['sat_improvement'] = df['sat'] - df['prev_sat']
#     df['relative_sat_change'] = df['sat_improvement'] / df['prev_sat'].replace(0, 1)
#     df['prev_gpa_sat_interaction'] = df['prev_gpa'] * df['prev_sat']
#     df['sat_retention'] = df['sat'] / df['prev_sat'].replace(0, 1)
#     df['prev_gpa_squared'] = df['prev_gpa'] ** 2
#     df['sat_squared'] = df['sat'] ** 2
    
#     return df

# def standardize_categorical_values(df, reference_df):
#     """
#     Standardize categorical values by converting to uppercase and mapping to reference values
#     """
#     df = df.copy()
#     categorical_columns = ['SCHOOL OWNERSHIP', 'SCHOOL CATEGORY', 'SCHOOL TYPE', 
#                           'COMBINATIONS CATEGORY', 'ACADEMIC LEVEL CATEGORY']
    
#     # Get unique values from reference data for mapping
#     reference_values = {}
#     for col in categorical_columns:
#         if col in reference_df.columns:
#             reference_values[col] = set(reference_df[col].unique())
    
#     mapping_report = {}
    
#     for col in categorical_columns:
#         if col in df.columns:
#             original_values = df[col].unique()
#             mapping_report[col] = {'original': list(original_values), 'mapped': [], 'unmapped': []}
            
#             # Convert to uppercase for matching
#             df[col] = df[col].astype(str).str.upper().str.strip()
            
#             # Try to map values to reference values
#             if col in reference_values:
#                 ref_upper = {val.upper(): val for val in reference_values[col]}
                
#                 def map_value(val):
#                     val_upper = str(val).upper().strip()
#                     if val_upper in ref_upper:
#                         mapping_report[col]['mapped'].append(f"{val} -> {ref_upper[val_upper]}")
#                         return ref_upper[val_upper]
#                     else:
#                         mapping_report[col]['unmapped'].append(val)
#                         # Return the most similar value or the first reference value as fallback
#                         return list(reference_values[col])[0]
                
#                 df[col] = df[col].apply(map_value)
    
#     return df, mapping_report

# def handle_missing_values(df, reference_df):
#     """
#     Handle missing values in the dataframe using various strategies
#     """
#     df = df.copy()
#     missing_report = {}
    
#     # Define numerical and categorical columns (case insensitive)
#     numerical_cols = ['PREV_SAT', 'PREV_GPA', 'SAT', 'STUDENT_TEACHER_RATIO']
#     categorical_cols = ['SCHOOL OWNERSHIP', 'SCHOOL CATEGORY', 'SCHOOL TYPE', 
#                        'COMBINATIONS CATEGORY', 'ACADEMIC LEVEL CATEGORY']
    
#     # Ensure we have the required columns, create them if missing
#     default_values = {
#         'PREV_SAT': 500,
#         'PREV_GPA': 3.0,
#         'SAT': 500,
#         'STUDENT_TEACHER_RATIO': 15
#     }
    
#     # Handle numerical missing values
#     for col in numerical_cols:
#         if col not in df.columns:
#             # Create missing column with default value
#             df[col] = default_values.get(col, 0)
#             missing_report[col] = f"Created missing column with default value: {default_values.get(col, 0)}"
#         else:
#             missing_count = df[col].isnull().sum()
#             if missing_count > 0:
#                 # Convert column names to match reference_df
#                 ref_col = col.lower()
#                 if ref_col in reference_df.columns:
#                     median_val = reference_df[ref_col].median()
#                     df[col].fillna(median_val, inplace=True)
#                     missing_report[col] = f"Filled {missing_count} missing values with median: {median_val:.2f}"
#                 else:
#                     default_val = default_values.get(col, 0)
#                     df[col].fillna(default_val, inplace=True)
#                     missing_report[col] = f"Filled {missing_count} missing values with default: {default_val}"
    
#     # Handle categorical missing values
#     categorical_defaults = {
#         'SCHOOL OWNERSHIP': 'PUBLIC',
#         'SCHOOL CATEGORY': 'SECONDARY',
#         'SCHOOL TYPE': 'MIXED',
#         'COMBINATIONS CATEGORY': 'GENERAL',
#         'ACADEMIC LEVEL CATEGORY': 'ORDINARY'
#     }
    
#     for col in categorical_cols:
#         if col not in df.columns:
#             # Create missing column with default value
#             df[col] = categorical_defaults.get(col, 'Unknown')
#             missing_report[col] = f"Created missing column with default value: {categorical_defaults.get(col, 'Unknown')}"
#         else:
#             missing_count = df[col].isnull().sum()
#             if missing_count > 0:
#                 if col in reference_df.columns:
#                     mode_val = reference_df[col].mode()[0] if len(reference_df[col].mode()) > 0 else categorical_defaults.get(col, 'Unknown')
#                     df[col].fillna(mode_val, inplace=True)
#                     missing_report[col] = f"Filled {missing_count} missing values with mode: {mode_val}"
#                 else:
#                     default_val = categorical_defaults.get(col, 'Unknown')
#                     df[col].fillna(default_val, inplace=True)
#                     missing_report[col] = f"Filled {missing_count} missing values with default: {default_val}"
    
#     return df, missing_report

# def validate_data_ranges(df, reference_df):
#     """
#     Validate that numerical values are within reasonable ranges
#     """
#     validation_report = {}
    
#     # Define expected ranges based on reference data
#     if not reference_df.empty:
#         ranges = {
#             'PREV_SAT': (reference_df['prev_sat'].min() * 0.5, reference_df['prev_sat'].max() * 1.5),
#             'PREV_GPA': (0.0, 5.0),
#             'SAT': (reference_df['sat'].min() * 0.5, reference_df['sat'].max() * 1.5),
#             'STUDENT_TEACHER_RATIO': (1, 100)
#         }
#     else:
#         ranges = {
#             'PREV_SAT': (200, 800),
#             'PREV_GPA': (0.0, 5.0),
#             'SAT': (200, 800),
#             'STUDENT_TEACHER_RATIO': (1, 100)
#         }
    
#     for col, (min_val, max_val) in ranges.items():
#         if col in df.columns:
#             out_of_range = df[(df[col] < min_val) | (df[col] > max_val)]
#             if len(out_of_range) > 0:
#                 validation_report[col] = f"Warning: {len(out_of_range)} values out of expected range ({min_val:.1f}-{max_val:.1f})"
#                 # Clip values to valid range
#                 df[col] = df[col].clip(lower=min_val, upper=max_val)
    
#     return df, validation_report

# # Load data and model
# df = load_data()
# if df is not None:
#     df = df.dropna().drop('STUDENT-TEACHER RATIO', axis=1, errors='ignore')
# model = load_linear_regression_model()

# if df is None or model is None:
#     st.error("Failed to load required data or model. Please check file paths.")
#     st.stop()

# # Engineer features for the loaded data
# enhanced_df = engineer_features(df).dropna()


# # Main content

# st.markdown('<div class="main-header">School GPA Prediction Tool</div>', unsafe_allow_html=True)

# st.markdown("""
# <div class="card">
#     Enter student and school information below to predict the expected GPA. The more accurate the input data, 
#     the more reliable the prediction will be. Required fields are marked with *
# </div>
# """, unsafe_allow_html=True)

# # Create tabs for different prediction approaches
# tabs = st.tabs(["📝 Standard Prediction", "📊 Batch Prediction"])

# # Standard Prediction Tab
# with tabs[0]:
#     col1, col2 = st.columns(2)
    
#     with col1:
#         st.markdown('<div class="sub-header">Academic Information</div>', unsafe_allow_html=True)
        
#         # Add tooltips to explain features
#         st.markdown("""
#         <div class="tooltip">Previous SAT 
#             <span class="tooltiptext">School's SAT from previous academic year</span>
#         </div>
#         """, unsafe_allow_html=True)
#         prev_sat = st.number_input("Previous SAT*", 
#                             min_value=float(1), 
#                             max_value=float(1000), 
#                             value=float(enhanced_df['prev_sat'].median()))
        
#         st.markdown("""
#         <div class="tooltip">Previous GPA
#             <span class="tooltiptext">School's GPA from previous academic year</span>
#         </div>
#         """, unsafe_allow_html=True)
#         prev_gpa = st.number_input("Previous GPA*", 
#                             min_value=1.0, 
#                             max_value= 5.0, 
#                             value=float(enhanced_df['prev_gpa'].median()))
        
#         st.markdown("""
#         <div class="tooltip">Current SAT
#             <span class="tooltiptext">Student's most recent SAT</span>
#         </div>
#         """, unsafe_allow_html=True)
#         sat = st.number_input("Current SAT*", 
#                         min_value=float(enhanced_df['sat'].min()), 
#                         max_value=float(enhanced_df['sat'].max()), 
#                         value=float(enhanced_df['sat'].median()))
#         student_teacher_ratio = st.number_input("Student-Teacher Ratio*", 
#                                         min_value=float(enhanced_df['student_teacher_ratio'].min()), 
#                                         max_value=float(enhanced_df['student_teacher_ratio'].max()), 
#                                         value=float(enhanced_df['student_teacher_ratio'].median()))
        
#     with col2:
#         st.markdown('<div class="sub-header">School Information</div>', unsafe_allow_html=True)
        
#         school_ownership = st.selectbox("School Ownership*", 
#                                         options=enhanced_df['SCHOOL OWNERSHIP'].unique())
        
#         school_category = st.selectbox("School Category*", 
#                                         options=enhanced_df['SCHOOL CATEGORY'].unique())
        
#         school_type = st.selectbox("School Type*", 
#                                     options=enhanced_df['SCHOOL TYPE'].unique())
        
#         combinations_category = st.selectbox("Combinations Category", 
#                                             options=enhanced_df['COMBINATIONS CATEGORY'].unique())
        
#         academic_level = st.selectbox("Academic Level*", 
#                                     options=enhanced_df['ACADEMIC LEVEL CATEGORY'].unique())
        
    
#     # Create derived features
#     sat_improvement = sat - prev_sat
#     relative_sat_change = sat_improvement / prev_sat if prev_sat > 0 else 0
#     prev_gpa_sat_interaction = prev_gpa * prev_sat
#     sat_retention = sat / prev_sat if prev_sat > 0 else 1.0
#     prev_gpa_squared = prev_gpa ** 2
#     sat_squared = sat ** 2
    
#     # Create input data for prediction
#     input_data = pd.DataFrame({
#         'prev_sat': [prev_sat],
#         'prev_gpa': [prev_gpa],
#         'sat': [sat],
#         'student_teacher_ratio': [student_teacher_ratio],
#         'SCHOOL OWNERSHIP': [school_ownership],
#         'SCHOOL CATEGORY': [school_category],
#         'SCHOOL TYPE': [school_type],
#         'COMBINATIONS CATEGORY': [combinations_category],
#         'ACADEMIC LEVEL CATEGORY': [academic_level],
#         'prev_gpa_sat_interaction': [prev_gpa_sat_interaction],
#         'sat_retention': [sat_retention],
#         'sat_improvement': [sat_improvement],
#         'relative_sat_change': [relative_sat_change],
#         'prev_gpa_squared': [prev_gpa_squared],
#         'sat_squared': [sat_squared]
#     })
    
#     # Model info
#     st.info("Using Linear Regression Model")
    
#     # Prediction button
#     if st.button("Generate GPA Prediction", use_container_width=True):
#         with st.spinner("Analyzing data and generating prediction..."):
#             import time
#             time.sleep(0.5)
            
#             # Make prediction
#             prediction = model.predict(input_data)[0]
            
#             # Create confidence interval (approximate)
#             model_rmse = 0.3  # Approximate RMSE for confidence interval
#             margin = 1.96 * model_rmse  # 95% confidence interval
#             lower_bound = max(0.0, prediction - margin)
#             upper_bound = min(4.5, prediction + margin)
            
            
#             st.markdown(f"""
#             <div class="card" style="background-color: #DBEAFE; text-align: center;">
#                 <h2 style="margin-bottom: 0.5rem;">Predicted GPA</h2>
#                 <div style="font-size: 3rem; font-weight: bold; color: #1E40AF;">{prediction:.2f}</div>
#                 <p>95% Confidence Interval: {lower_bound:.2f} - {upper_bound:.2f}</p>
#             </div>
#             """, unsafe_allow_html=True)
                
            
# # Batch Prediction Tab
# with tabs[1]:
    
#     # File uploader
#     uploaded_file = st.file_uploader("Upload CSV file with school data", type=["csv"])
    
#     # Options for batch processing
#     col1, col2 = st.columns(2)
#     with col1:
#         show_processing_details = st.checkbox("Show detailed processing reports", value=True)
#     with col2:
#         strict_validation = st.checkbox("Enable strict data validation", value=False)
    
#     # Sample data and template
#     col1, col2 = st.columns(2)
    
#     with col1:
#         with st.expander("View Required CSV Format"):
#             st.markdown("""
#             Your CSV file should include the following columns (case-insensitive):
#             - **prev_sat** - Previous SAT score
#             - **prev_gpa** - Previous GPA
#             - **sat** - Current SAT score  
#             - **student_teacher_ratio** - Student to teacher ratio
#             - **SCHOOL OWNERSHIP** - School ownership type
#             - **SCHOOL CATEGORY** - School category
#             - **SCHOOL TYPE** - Type of school
#             - **COMBINATIONS CATEGORY** - Combinations category
#             - **ACADEMIC LEVEL CATEGORY** - Academic level
            
#             **Note**: Missing values will be automatically handled!
#             """)
            
#             # Show sample CSV
#             sample_df = df.head(6)
#             st.dataframe(sample_df)
    
#     with col2:
#         st.download_button(
#             label="📥 Download Template CSV",
#             data=sample_df.to_csv(index=False),
#             file_name="gpa_prediction_template.csv",
#             mime="text/csv"
#         )
    
#     # Batch prediction functionality
#     if uploaded_file is not None:
#         try:
#             # Read the uploaded file
#             batch_data = pd.read_csv(uploaded_file)
#             st.success(f"✅ Successfully loaded {len(batch_data)} records from '{uploaded_file.name}'")
            
#             # Show initial data preview
#             with st.expander("📋 Preview Uploaded Data"):
#                 st.dataframe(batch_data.head())
#                 st.write(f"**Shape**: {batch_data.shape[0]} rows × {batch_data.shape[1]} columns")
            
#             # Check required columns (case insensitive)
#             required_cols = ['prev_sat', 'prev_gpa', 'sat', 'student_teacher_ratio', 
#                             'SCHOOL OWNERSHIP', 'SCHOOL CATEGORY', 'SCHOOL TYPE',
#                             'COMBINATIONS CATEGORY', 'ACADEMIC LEVEL CATEGORY']
            
#             # Normalize column names (handle case variations)
#             original_columns = batch_data.columns.tolist()
#             batch_data.columns = batch_data.columns.str.upper().str.strip()
            
#             # Create a mapping for numerical columns to handle case variations
#             numerical_column_mapping = {
#                 'PREV_SAT': 'PREV_SAT',
#                 'PREVIOUS_SAT': 'PREV_SAT', 
#                 'PREV GPA': 'PREV_GPA',
#                 'PREVIOUS_GPA': 'PREV_GPA',
#                 'PREVIOUS GPA': 'PREV_GPA',
#                 'SAT': 'SAT',
#                 'STUDENT_TEACHER_RATIO': 'STUDENT_TEACHER_RATIO',
#                 'STUDENT TEACHER RATIO': 'STUDENT_TEACHER_RATIO',
#                 'STUDENT-TEACHER RATIO': 'STUDENT_TEACHER_RATIO'
#             }
            
#             # Apply column name mapping
#             for old_name, new_name in numerical_column_mapping.items():
#                 if old_name in batch_data.columns and new_name not in batch_data.columns:
#                     batch_data[new_name] = batch_data[old_name]
            
#             batch_data_cols_upper = [col.upper() for col in batch_data.columns]
#             required_cols_numerical = ['PREV_SAT', 'PREV_GPA', 'SAT', 'STUDENT_TEACHER_RATIO']
#             required_cols_categorical = ['SCHOOL OWNERSHIP', 'SCHOOL CATEGORY', 'SCHOOL TYPE',
#                                         'COMBINATIONS CATEGORY', 'ACADEMIC LEVEL CATEGORY']
            
#             missing_numerical = [col for col in required_cols_numerical if col not in batch_data.columns]
#             missing_categorical = [col for col in required_cols_categorical if col not in batch_data.columns]
#             missing_cols = missing_numerical + missing_categorical
            
#             if missing_cols and strict_validation:
#                 st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
#                 st.info("💡 Tip: Disable 'strict validation' to proceed with partial data")
#             else:
#                 if missing_cols:
#                     st.warning(f"⚠️ Missing columns will be handled: {', '.join(missing_cols)}")
                
#                 if st.button("🚀 Run Batch Prediction", key="run_batch"):
#                     with st.spinner("Processing batch prediction..."):
#                         progress_bar = st.progress(0)
                        
#                         # Step 1: Handle missing values
#                         progress_bar.progress(25)
#                         processed_data, missing_report = handle_missing_values(batch_data, enhanced_df)
                        
#                         # Step 2: Standardize categorical values
#                         progress_bar.progress(50)
#                         processed_data, mapping_report = standardize_categorical_values(processed_data, enhanced_df)
                        
#                         # Step 3: Validate data ranges
#                         progress_bar.progress(75)
#                         processed_data, validation_report = validate_data_ranges(processed_data, enhanced_df)
                        
#                         # Step 4: Add derived features and make predictions
#                         try:
#                             processed_data = engineer_features(processed_data)
#                             predictions = model.predict(processed_data)
#                             processed_data['predicted_gpa'] = predictions
#                             progress_bar.progress(100)
                            
#                             # Display results
#                             st.markdown('<div class="sub-header">✅ Prediction Results</div>', unsafe_allow_html=True)
                            
#                             # Summary statistics
#                             col1, col2, col3, col4 = st.columns(4)
#                             with col1:
#                                 st.metric("Total Predictions", len(predictions))
#                             with col2:
#                                 st.metric("Average Predicted GPA", f"{np.mean(predictions):.2f}")
#                             with col3:
#                                 st.metric("Min GPA", f"{np.min(predictions):.2f}")
#                             with col4:
#                                 st.metric("Max GPA", f"{np.max(predictions):.2f}")
                            
#                             # Results table
#                             st.dataframe(processed_data, use_container_width=True)
                            
#                             # Download results
#                             st.download_button(
#                                 label="📥 Download Predictions CSV",
#                                 data=processed_data.to_csv(index=False),
#                                 file_name=f"gpa_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
#                                 mime="text/csv"
#                             )
                            
#                             # Visualization
#                             st.markdown('<div class="sub-header">📊 Prediction Analysis</div>', unsafe_allow_html=True)
                            
#                             col1, col2 = st.columns(2)
                            
#                             with col1:
#                                 # Distribution of predictions
#                                 fig1 = px.histogram(processed_data, x='predicted_gpa', nbins=20,
#                                                     title="Distribution of Predicted GPAs",
#                                                     color_discrete_sequence=['#1E40AF'])
#                                 fig1.update_layout(showlegend=False)
#                                 st.plotly_chart(fig1, use_container_width=True)
                            
#                             with col2:
#                                 # Box plot by categorical variable (if available)
#                                 selected_cat_feature = st.selectbox("Select Categorical Feature for Box Plot", 
#                                                                     options=['SCHOOL OWNERSHIP', 'SCHOOL CATEGORY', 
#                                                                              'SCHOOL TYPE', 'COMBINATIONS CATEGORY', 
#                                                                              'ACADEMIC LEVEL CATEGORY'], 
#                                                                     index=0)
                                
#                                 if selected_cat_feature in processed_data.columns:
#                                     fig2 = px.box(processed_data, x=selected_cat_feature, y='predicted_gpa',
#                                                     title=f"GPA Distribution by {selected_cat_feature}",
#                                                     color_discrete_sequence=['#1E40AF'])
#                                     fig2.update_xaxes(tickangle=45)
#                                     st.plotly_chart(fig2, use_container_width=True)
                            
#                             # Processing reports
#                             if show_processing_details:
#                                 st.markdown('<div class="sub-header">📋 Data Processing Report</div>', unsafe_allow_html=True)
                                
#                                 # Missing values report
#                                 if missing_report:
#                                     st.markdown("**Missing Values Handled:**")
#                                     for col, report in missing_report.items():
#                                         st.write(f"• {col}: {report}")
#                                 else:
#                                     st.success("✅ No missing values found!")
                                
#                                 # Categorical mapping report
#                                 if any(mapping_report.values()):
#                                     st.markdown("**Categorical Data Standardization:**")
#                                     for col, report in mapping_report.items():
#                                         if report['mapped'] or report['unmapped']:
#                                             st.write(f"**{col}:**")
#                                             for mapping in report['mapped'][:5]:  # Show first 5 mappings
#                                                 st.write(f"  • {mapping}")
#                                             if report['unmapped']:
#                                                 st.write(f"  • Unmapped values: {', '.join(report['unmapped'][:3])}")
                                
#                                 # Validation report
#                                 if validation_report:
#                                     st.markdown("**Data Validation:**")
#                                     for col, report in validation_report.items():
#                                         st.warning(f"• {col}: {report}")
#                                 else:
#                                     st.success("✅ All data within expected ranges!")
                            
#                         except Exception as e:
#                             st.error(f"❌ Error during prediction: {str(e)}")
#                             st.info("This might be due to data format issues. Please check your data and try again.")
                        
#                         finally:
#                             progress_bar.empty()
                        
#         except Exception as e:
#             st.error(f"❌ Error processing file: {str(e)}")
#             st.info("Please ensure your CSV file is properly formatted and try again.")

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
import shap
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="GPA Prediction Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load data
@st.cache_data
def load_data():
    """Load the sample data"""
    try:
        df = pd.read_csv('data/df.csv')
        return df
    except Exception as e:
        st.error(f"Could not load data: {e}")
        return None

# Load model
@st.cache_resource
def load_linear_regression_model():
    """Load the linear regression model"""
    try:
        model = joblib.load('models/linear_regression_pipeline.pkl')
        return model
    except Exception as e:
        st.error(f"Could not load model: {e}")
        return None

# Feature engineering function
def engineer_features(df):
    """Add derived features to the dataframe"""
    df = df.copy()
    
    # Convert column names to lowercase for consistency with the model
    column_mapping = {
        'PREV_SAT': 'prev_sat',
        'PREV_GPA': 'prev_gpa', 
        'SAT': 'sat',
        'STUDENT_TEACHER_RATIO': 'student_teacher_ratio'
    }
    
    # Rename columns to match expected format
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns:
            df[new_col] = df[old_col]
    
    # Ensure we have the required base columns
    required_cols = ['prev_sat', 'prev_gpa', 'sat', 'student_teacher_ratio']
    for col in required_cols:
        if col not in df.columns:
            # Set default values if columns are missing
            defaults = {'prev_sat': 500, 'prev_gpa': 3.0, 'sat': 500, 'student_teacher_ratio': 15}
            df[col] = defaults[col]
    
    # Create derived features safely
    df['sat_improvement'] = df['sat'] - df['prev_sat']
    df['relative_sat_change'] = df['sat_improvement'] / df['prev_sat'].replace(0, 1)
    df['prev_gpa_sat_interaction'] = df['prev_gpa'] * df['prev_sat']
    df['sat_retention'] = df['sat'] / df['prev_sat'].replace(0, 1)
    df['prev_gpa_squared'] = df['prev_gpa'] ** 2
    df['sat_squared'] = df['sat'] ** 2
    
    return df

def standardize_categorical_values(df, reference_df):
    """
    Standardize categorical values by converting to uppercase and mapping to reference values
    """
    df = df.copy()
    categorical_columns = ['SCHOOL OWNERSHIP', 'SCHOOL CATEGORY', 'SCHOOL TYPE', 
                          'COMBINATIONS CATEGORY', 'ACADEMIC LEVEL CATEGORY']
    
    # Get unique values from reference data for mapping
    reference_values = {}
    for col in categorical_columns:
        if col in reference_df.columns:
            reference_values[col] = set(reference_df[col].unique())
    
    mapping_report = {}
    
    for col in categorical_columns:
        if col in df.columns:
            original_values = df[col].unique()
            mapping_report[col] = {'original': list(original_values), 'mapped': [], 'unmapped': []}
            
            # Convert to uppercase for matching
            df[col] = df[col].astype(str).str.upper().str.strip()
            
            # Try to map values to reference values
            if col in reference_values:
                ref_upper = {val.upper(): val for val in reference_values[col]}
                
                def map_value(val):
                    val_upper = str(val).upper().strip()
                    if val_upper in ref_upper:
                        mapping_report[col]['mapped'].append(f"{val} -> {ref_upper[val_upper]}")
                        return ref_upper[val_upper]
                    else:
                        mapping_report[col]['unmapped'].append(val)
                        # Return the most similar value or the first reference value as fallback
                        return list(reference_values[col])[0]
                
                df[col] = df[col].apply(map_value)
    
    return df, mapping_report

def handle_missing_values(df, reference_df):
    """
    Handle missing values in the dataframe
    """
    df = df.copy()
    missing_report = {}
    
    # Define numerical and categorical columns (case insensitive)
    numerical_cols = ['PREV_SAT', 'PREV_GPA', 'SAT', 'STUDENT_TEACHER_RATIO']
    categorical_cols = ['SCHOOL OWNERSHIP', 'SCHOOL CATEGORY', 'SCHOOL TYPE', 
                       'COMBINATIONS CATEGORY', 'ACADEMIC LEVEL CATEGORY']
    
    # Ensure we have the required columns, create them if missing
    default_values = {
        'PREV_SAT': 500,
        'PREV_GPA': 3.0,
        'SAT': 500,
        'STUDENT_TEACHER_RATIO': 15
    }
    
    # Handle numerical missing values
    for col in numerical_cols:
        if col not in df.columns:
            # Create missing column with default value
            df[col] = default_values.get(col, 0)
            missing_report[col] = f"Created missing column with default value: {default_values.get(col, 0)}"
        else:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                # Convert column names to match reference_df
                ref_col = col.lower()
                if ref_col in reference_df.columns:
                    median_val = reference_df[ref_col].median()
                    df[col].fillna(median_val, inplace=True)
                    missing_report[col] = f"Filled {missing_count} missing values with median: {median_val:.2f}"
                else:
                    default_val = default_values.get(col, 0)
                    df[col].fillna(default_val, inplace=True)
                    missing_report[col] = f"Filled {missing_count} missing values with default: {default_val}"
    
    # Handle categorical missing values
    categorical_defaults = {
        'SCHOOL OWNERSHIP': 'PUBLIC',
        'SCHOOL CATEGORY': 'SECONDARY',
        'SCHOOL TYPE': 'MIXED',
        'COMBINATIONS CATEGORY': 'GENERAL',
        'ACADEMIC LEVEL CATEGORY': 'ORDINARY'
    }
    
    for col in categorical_cols:
        if col not in df.columns:
            # Create missing column with default value
            df[col] = categorical_defaults.get(col, 'Unknown')
            missing_report[col] = f"Created missing column with default value: {categorical_defaults.get(col, 'Unknown')}"
        else:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                if col in reference_df.columns:
                    mode_val = reference_df[col].mode()[0] if len(reference_df[col].mode()) > 0 else categorical_defaults.get(col, 'Unknown')
                    df[col].fillna(mode_val, inplace=True)
                    missing_report[col] = f"Filled {missing_count} missing values with mode: {mode_val}"
                else:
                    default_val = categorical_defaults.get(col, 'Unknown')
                    df[col].fillna(default_val, inplace=True)
                    missing_report[col] = f"Filled {missing_count} missing values with default: {default_val}"
    
    return df, missing_report

def validate_data_ranges(df, reference_df):
    """
    Validate that numerical values are within reasonable ranges
    """
    validation_report = {}
    
    # Define expected ranges based on reference data
    if not reference_df.empty:
        # Using .loc to avoid SettingWithCopyWarning if reference_df is a slice
        ranges = {
            'PREV_SAT': (reference_df['prev_sat'].min() * 0.5, reference_df['prev_sat'].max() * 1.5),
            'PREV_GPA': (0.0, 5.0),
            'SAT': (reference_df['sat'].min() * 0.5, reference_df['sat'].max() * 1.5),
            'STUDENT_TEACHER_RATIO': (1, 100)
        }
    else:
        ranges = {
            'PREV_SAT': (200, 800),
            'PREV_GPA': (0.0, 5.0),
            'SAT': (200, 800),
            'STUDENT_TEACHER_RATIO': (1, 100)
        }
    
    for col, (min_val, max_val) in ranges.items():
        if col in df.columns:
            out_of_range = df[(df[col] < min_val) | (df[col] > max_val)]
            if len(out_of_range) > 0:
                validation_report[col] = f"Warning: {len(out_of_range)} values out of expected range ({min_val:.1f}-{max_val:.1f})"
                # Clip values to valid range
                df[col] = df[col].clip(lower=min_val, upper=max_val)
    
    return df, validation_report

# Load data and model
df = load_data()
if df is not None:
    # Ensure 'STUDENT-TEACHER RATIO' is dropped consistently
    if 'STUDENT-TEACHER RATIO' in df.columns:
        df = df.drop('STUDENT-TEACHER RATIO', axis=1)
    df = df.dropna()

model = load_linear_regression_model()

if df is None or model is None:
    st.error("Failed to load required data or model. Please check file paths.")
    st.stop()

# Engineer features for the loaded data
# Ensure 'prev_sat' and other base columns are present before engineering
# This part is for the single prediction section, using the reference data's median
enhanced_df = engineer_features(df.copy()) # Use a copy to avoid modifying the original cached df

# Initialize session state variables if they don't exist
if 'processed_batch_data' not in st.session_state:
    st.session_state.processed_batch_data = None
if 'batch_prediction_run' not in st.session_state:
    st.session_state.batch_prediction_run = False


# Main content

st.header('🎓 School GPA Prediction Dashboard')
st.markdown("""
<div class="card">
    Enter School information below to predict the expected GPA. The more accurate the input data, 
    the more reliable the prediction will be. Required fields are marked with *
</div>
""", unsafe_allow_html=True)

# Create tabs for different prediction approaches
tabs = st.tabs(["📝 Standard Prediction", "📊 Batch Prediction"])

# Standard Prediction Tab
with tabs[0]:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="sub-header">Academic Information</div>', unsafe_allow_html=True)
        
        # Add tooltips to explain features
        st.markdown("""
        <div class="tooltip">Previous SAT 
            <span class="tooltiptext"> - Number of candidates who actually sat for the exams from previous academic year</span>
        </div>
        """, unsafe_allow_html=True)
        prev_sat = st.number_input("Previous SAT*", 
                            min_value=float(1), 
                            max_value=float(1000), 
                            value=float(enhanced_df['prev_sat'].median()))
        
        st.markdown("""
        <div class="tooltip">Previous GPA
            <span class="tooltiptext"> - School's GPA from previous academic year</span>
        </div>
        """, unsafe_allow_html=True)
        prev_gpa = st.number_input("Previous GPA*", 
                            min_value=1.0, 
                            max_value= 5.0, 
                            value=float(enhanced_df['prev_gpa'].median()))
        
        st.markdown("""
        <div class="tooltip">Current SAT
            <span class="tooltiptext"> - Number of candidates who actually sat for the exams</span>
        </div>
        """, unsafe_allow_html=True)
        sat = st.number_input("Current SAT*", 
                        min_value=float(enhanced_df['sat'].min()), 
                        max_value=float(enhanced_df['sat'].max()), 
                        value=float(enhanced_df['sat'].median()))
        student_teacher_ratio = st.number_input("Student-Teacher Ratio*", 
                                        min_value=float(enhanced_df['student_teacher_ratio'].min()), 
                                        max_value=float(enhanced_df['student_teacher_ratio'].max()), 
                                        value=float(enhanced_df['student_teacher_ratio'].median()))
        
    with col2:
        st.markdown('<div class="sub-header">School Information</div>', unsafe_allow_html=True)
        
        school_ownership = st.selectbox("School Ownership*", 
                                        options=enhanced_df['SCHOOL OWNERSHIP'].unique())
        
        school_category = st.selectbox("School Category*", 
                                        options=enhanced_df['SCHOOL CATEGORY'].unique())
        
        school_type = st.selectbox("School Type*", 
                                    options=enhanced_df['SCHOOL TYPE'].unique())
        
        combinations_category = st.selectbox("Combinations Category", 
                                            options=enhanced_df['COMBINATIONS CATEGORY'].unique())
        
        academic_level = st.selectbox("Academic Level*", 
                                    options=enhanced_df['ACADEMIC LEVEL CATEGORY'].unique())
        
    
    # Create derived features
    sat_improvement = sat - prev_sat
    relative_sat_change = sat_improvement / prev_sat if prev_sat > 0 else 0
    prev_gpa_sat_interaction = prev_gpa * prev_sat
    sat_retention = sat / prev_sat if prev_sat > 0 else 1.0
    prev_gpa_squared = prev_gpa ** 2
    sat_squared = sat ** 2
    
    # Create input data for prediction
    input_data = pd.DataFrame({
        'prev_sat': [prev_sat],
        'prev_gpa': [prev_gpa],
        'sat': [sat],
        'student_teacher_ratio': [student_teacher_ratio],
        'SCHOOL OWNERSHIP': [school_ownership],
        'SCHOOL CATEGORY': [school_category],
        'SCHOOL TYPE': [school_type],
        'COMBINATIONS CATEGORY': [combinations_category],
        'ACADEMIC LEVEL CATEGORY': [academic_level],
        'prev_gpa_sat_interaction': [prev_gpa_sat_interaction],
        'sat_retention': [sat_retention],
        'sat_improvement': [sat_improvement],
        'relative_sat_change': [relative_sat_change],
        'prev_gpa_squared': [prev_gpa_squared],
        'sat_squared': [sat_squared]
    })
    
    # Model info
    st.info("Using Linear Regression Model")
    
    # Prediction button
    if st.button("Generate GPA Prediction", use_container_width=True):
        with st.spinner("Analyzing data and generating prediction..."):
            import time
            time.sleep(0.5)
            
            # Make prediction
            prediction = model.predict(input_data)[0]
            
            # Create confidence interval (approximate)
            model_rmse = 0.3  # Approximate RMSE for confidence interval
            margin = 1.96 * model_rmse  # 95% confidence interval
            lower_bound = max(0.0, prediction - margin)
            upper_bound = min(4.5, prediction + margin)
            
            
            st.markdown(f"""
            <div class="card" style="background-color: #DBEAFE; text-align: center;">
                <h2 style="margin-bottom: 0.5rem;">Predicted GPA</h2>
                <div style="font-size: 3rem; font-weight: bold; color: #1E40AF;">{prediction:.2f}</div>
                <p>95% Confidence Interval: {lower_bound:.2f} - {upper_bound:.2f}</p>
            </div>
            """, unsafe_allow_html=True)
                
            
# Batch Prediction Tab
with tabs[1]:
    
    # File uploader
    uploaded_file = st.file_uploader("Upload CSV file with school data", type=["csv"])
    
    # Options for batch processing
    col1, col2 = st.columns(2)
    with col1:
        show_processing_details = st.checkbox("Show detailed processing reports", value=True)
    with col2:
        strict_validation = st.checkbox("Enable strict data validation", value=False)
    
    # Sample data and template
    col1, col2 = st.columns(2)
    
    with col1:
        with st.expander("View Required CSV Format"):
            st.markdown("""
            Your CSV file should include the following columns (case-insensitive):
            - **prev_sat** - Previous SAT score
            - **prev_gpa** - Previous GPA
            - **sat** - Current SAT score  
            - **student_teacher_ratio** - Student to teacher ratio
            - **SCHOOL OWNERSHIP** - School ownership type
            - **SCHOOL CATEGORY** - School category
            - **SCHOOL TYPE** - Type of school
            - **COMBINATIONS CATEGORY** - Combinations category
            - **ACADEMIC LEVEL CATEGORY** - Academic level
            
            **Note**: Missing values will be automatically handled!
            """)
            
            # Show sample CSV
            sample_df = df.head(6)
            st.dataframe(sample_df)
    
    with col2:
        st.download_button(
            label="📥 Download Template CSV",
            data=sample_df.to_csv(index=False),
            file_name="gpa_prediction_template.csv",
            mime="text/csv"
        )
    
    # Batch prediction functionality
    if uploaded_file is not None:
        try:
            # Read the uploaded file
            batch_data = pd.read_csv(uploaded_file)
            st.success(f"✅ Successfully loaded {len(batch_data)} records from '{uploaded_file.name}'")
            
            # Show initial data preview
            with st.expander("📋 Preview Uploaded Data"):
                st.dataframe(batch_data.head())
                st.write(f"**Shape**: {batch_data.shape[0]} rows × {batch_data.shape[1]} columns")
            
            # Check required columns (case insensitive)
            required_cols = ['prev_sat', 'prev_gpa', 'sat', 'student_teacher_ratio', 
                            'SCHOOL OWNERSHIP', 'SCHOOL CATEGORY', 'SCHOOL TYPE',
                            'COMBINATIONS CATEGORY', 'ACADEMIC LEVEL CATEGORY']
            
            # Normalize column names (handle case variations)
            original_columns = batch_data.columns.tolist()
            batch_data.columns = batch_data.columns.str.upper().str.strip()
            
            # Create a mapping for numerical columns to handle case variations
            numerical_column_mapping = {
                'PREV_SAT': 'PREV_SAT',
                'PREVIOUS_SAT': 'PREV_SAT', 
                'PREV GPA': 'PREV_GPA',
                'PREVIOUS_GPA': 'PREV_GPA',
                'PREVIOUS GPA': 'PREV_GPA',
                'SAT': 'SAT',
                'STUDENT_TEACHER_RATIO': 'STUDENT_TEACHER_RATIO',
                'STUDENT TEACHER RATIO': 'STUDENT_TEACHER_RATIO',
                'STUDENT-TEACHER RATIO': 'STUDENT_TEACHER_RATIO'
            }
            
            # Apply column name mapping
            for old_name, new_name in numerical_column_mapping.items():
                if old_name in batch_data.columns and new_name not in batch_data.columns:
                    batch_data[new_name] = batch_data[old_name]
            
            batch_data_cols_upper = [col.upper() for col in batch_data.columns]
            required_cols_numerical = ['PREV_SAT', 'PREV_GPA', 'SAT', 'STUDENT_TEACHER_RATIO']
            required_cols_categorical = ['SCHOOL OWNERSHIP', 'SCHOOL CATEGORY', 'SCHOOL TYPE',
                                        'COMBINATIONS CATEGORY', 'ACADEMIC LEVEL CATEGORY']
            
            missing_numerical = [col for col in required_cols_numerical if col not in batch_data.columns]
            missing_categorical = [col for col in required_cols_categorical if col not in batch_data.columns]
            missing_cols = missing_numerical + missing_categorical
            
            if missing_cols and strict_validation:
                st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
                st.info("💡 Tip: Disable 'strict validation' to proceed with partial data")
            else:
                if missing_cols:
                    st.warning(f"⚠️ Missing columns will be handled: {', '.join(missing_cols)}")
                
                if st.button("🚀 Run Batch Prediction", key="run_batch"):
                    with st.spinner("Processing batch prediction..."):
                        progress_bar = st.progress(0)
                        
                        # Step 1: Handle missing values
                        progress_bar.progress(25)
                        processed_data, missing_report = handle_missing_values(batch_data, enhanced_df)
                        
                        # Step 2: Standardize categorical values
                        progress_bar.progress(50)
                        processed_data, mapping_report = standardize_categorical_values(processed_data, enhanced_df)
                        
                        # Step 3: Validate data ranges
                        progress_bar.progress(75)
                        processed_data, validation_report = validate_data_ranges(processed_data, enhanced_df)
                        
                        # Step 4: Add derived features and make predictions
                        try:
                            processed_data = engineer_features(processed_data)
                            predictions = model.predict(processed_data)
                            processed_data['predicted_gpa'] = predictions
                            
                            # Store processed_data in session_state
                            st.session_state.processed_batch_data = processed_data
                            st.session_state.batch_prediction_run = True
                            st.session_state.missing_report = missing_report
                            st.session_state.mapping_report = mapping_report
                            st.session_state.validation_report = validation_report

                            progress_bar.progress(100)
                            
                            # Display results
                            st.markdown('<div class="sub-header">✅ Prediction Results</div>', unsafe_allow_html=True)
                            
                            # Summary statistics
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Total Predictions", len(predictions))
                            with col2:
                                st.metric("Average Predicted GPA", f"{np.mean(predictions):.2f}")
                            with col3:
                                st.metric("Min GPA", f"{np.min(predictions):.2f}")
                            with col4:
                                st.metric("Max GPA", f"{np.max(predictions):.2f}")
                            
                            # Results table
                            st.dataframe(processed_data, use_container_width=True)
                            
                            # Download results
                            st.download_button(
                                label="📥 Download Predictions CSV",
                                data=processed_data.to_csv(index=False),
                                file_name=f"gpa_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                            
                        except Exception as e:
                            st.error(f"❌ Error during prediction: {str(e)}")
                            st.info("This might be due to data format issues. Please check your data and try again.")
                        
                        finally:
                            progress_bar.empty()
                
                if st.session_state.batch_prediction_run and st.session_state.processed_batch_data is not None:
                    processed_data = st.session_state.processed_batch_data
                    missing_report = st.session_state.missing_report
                    mapping_report = st.session_state.mapping_report
                    validation_report = st.session_state.validation_report

                    st.markdown('<div class="sub-header">📊 Prediction Analysis</div>', unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Distribution of predictions
                        fig1 = px.histogram(processed_data, x='predicted_gpa', nbins=20,
                                            title="Distribution of Predicted GPAs",
                                            color_discrete_sequence=['#1E40AF'])
                        fig1.update_layout(showlegend=False)
                        st.plotly_chart(fig1, use_container_width=True)
                    
                    with col2:
                        # Box plot by categorical variable (if available)
                        # Ensure 'predicted_gpa' is in the dataframe before proceeding
                        if 'predicted_gpa' in processed_data.columns:
                            available_cat_features = [
                                'SCHOOL OWNERSHIP', 'SCHOOL CATEGORY', 'SCHOOL TYPE', 
                                'COMBINATIONS CATEGORY', 'ACADEMIC LEVEL CATEGORY'
                            ]
                            # Filter to only show columns that actually exist in processed_data
                            existing_cat_features = [col for col in available_cat_features if col in processed_data.columns]
                            
                            if existing_cat_features:
                                selected_cat_feature = st.selectbox("Select Categorical Feature for Box Plot", 
                                                                    options=existing_cat_features, 
                                                                    index=0, key="batch_boxplot_selector")
                                
                                fig2 = px.box(processed_data, x=selected_cat_feature, y='predicted_gpa',
                                                title=f"GPA Distribution by {selected_cat_feature}",
                                                color_discrete_sequence=['#1E40AF'])
                                fig2.update_xaxes(tickangle=45)
                                st.plotly_chart(fig2, use_container_width=True)
                            else:
                                st.warning("No categorical features available for box plot.")
                        else:
                            st.warning("Predicted GPA column not found for plotting.")


                    # Processing reports
                    if show_processing_details:
                        st.markdown('<div class="sub-header">📋 Data Processing Report</div>', unsafe_allow_html=True)
                        
                        # Missing values report
                        if missing_report:
                            st.markdown("**Missing Values Handled:**")
                            for col, report in missing_report.items():
                                st.write(f"• {col}: {report}")
                        else:
                            st.success("✅ No missing values found!")
                        
                        # Categorical mapping report
                        if any(mapping_report.values()):
                            st.markdown("**Categorical Data Standardization:**")
                            for col, report in mapping_report.items():
                                if report['unmapped']: #report['mapped'] or :
                                    st.write(f"**{col}:**")
                                    # for mapping in report['mapped'][:5]:  # Show first 5 mappings
                                    #     st.write(f"  • {mapping}")
                                    if report['unmapped']:
                                        st.write(f"  • Unmapped values: {', '.join(report['unmapped'][:3])}")
                        
                        # Validation report
                        if validation_report:
                            st.markdown("**Data Validation:**")
                            for col, report in validation_report.items():
                                st.warning(f"• {col}: {report}")
                        else:
                            st.success("✅ All data within expected ranges!")
                        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.info("Please ensure your CSV file is properly formatted and try again.")