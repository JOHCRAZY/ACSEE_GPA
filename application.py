import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from datetime import datetime
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="GPA Prediction Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/JOHCRAZY/ACSEE_GPA/blob/master/README.md',
        'Report a bug': 'https://github.com/JOHCRAZY/ACSEE_GPA/issues',
        'About': """
        ### GPA Prediction Dashboard
        This application predicts School GPA based on School information.
        **Version:** 1.0.0  
        **Last Updated:** June 2025   
        **Contact:** johcrazy.magiha@gmail.com    
        **GitHub:** [ACSEE_GPA](https://github.com/JOHCRAZY/ACSEE_GPA)        
        """
    }
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    .card {
        padding: 24px;
        border-radius: 16px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        border: 1px solid #e2e8f0;
        margin-bottom: 24px;
        transition: all 0.3s ease;
    }
    
    .card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.1);
    }
    
    .sub-header {
        font-size: 1.75rem;
        font-weight: 600;
        margin-bottom: 16px;
        color: #1e293b;
        background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 24px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #059669 0%, #047857 100%);
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
    }
    
    .stMetric {
        background: white;
        padding: 16px;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    .stSelectbox > div > div, .stTextInput > div > div {
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        transition: all 0.2s ease;
    }
    
    .stSelectbox > div > div:focus-within, .stTextInput > div > div:focus-within {
        border-color: #3b82f6;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
    }
    
    .css-1d391kg {
        background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
    }
    
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f5f9;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #94a3b8;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #64748b;
    }
    
    [data-testid="stSidebar"] button:contains("Settings") {display: none;}

/* Hide deploy button */
.stDeployButton {visibility: hidden;}

/* Hide Development section entirely */
.st-emotion-cache-6dpr85 .st-emotion-cache-17v3rtm:has(h2:contains("Development")) {
    display: none !important;
}

/* Alternative selector for Development section */
.st-emotion-cache-17v3rtm:has(input[name="runOnSave"]) {
    display: none !important;
}

/* Hide Wide mode toggle specifically */
.st-emotion-cache-17v3rtm:has(input[name="wideMode"]) {
    display: none !important;
}



/* Show only the theme dropdown */
.st-emotion-cache-17v3rtm:has(.row-widget.stSelectbox) {
    display: block !important;
}

.st-emotion-cache-17v3rtm:has(.row-widget.stSelectbox) h2 {
    display: none !important;
}



/* Hide run on save checkbox */
input[name="runOnSave"] {
    display: none !important;
}

/* Hide run on save label and description */
label:has(input[name="runOnSave"]) {
    display: none !important;
}

/* Hide wide mode checkbox */
input[name="wideMode"] {
    display: none !important;
}

/* Hide wide mode label and description */
label:has(input[name="wideMode"]) {
    display: none !important;
}



/* Keep theme dropdown container visible */
.row-widget.stSelectbox {
    display: block !important;
}

/* Hide Made with section entirely */
p:has(a[href*="streamlit.io"]) {
    display: none;
}

ul[role="option"]:has(span:contains("Clear cache")),
ul[role="option"]:has(span:contains("Developer options")) {
    display: none;
}
    @media (max-width: 768px) {
        .card {
            padding: 16px;
        }
        .sub-header {
            font-size: 1.5rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data(ttl=3600, max_entries=100)
def load_data():
    try:
        df = pd.read_csv('data/df_.csv')
        return df
    except Exception as e:
        st.error(f"Could not load data: {e}")
        return None

# Load model
@st.cache_resource(ttl=3600, max_entries=100)
def load_linear_regression_model():
    try:
        model = joblib.load('models/new/linear_regression_pipeline.pkl')
        return model
    except Exception as e:
        st.error(f"Could not load model: {e}")
        return None

# Feature engineering function
def engineer_features(df):
    df = df.copy()
    column_mapping = {
        'PREV_SAT': 'prev_sat',
        'PREV_GPA': 'prev_gpa',
        'SAT': 'sat',
        'STUDENT_TEACHER_RATIO': 'student_teacher_ratio'
    }
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns:
            df[new_col] = df[old_col]
    required_cols = ['prev_sat', 'prev_gpa', 'sat', 'student_teacher_ratio']
    for col in required_cols:
        if col not in df.columns:
            defaults = {'prev_sat': 500, 'prev_gpa': 3.0, 'sat': 500, 'student_teacher_ratio': 15}
            df[col] = defaults[col]
    df['sat_improvement'] = df['sat'] - df['prev_sat']
    df['relative_sat_change'] = df['sat_improvement'] / df['prev_sat'].replace(0, 1)
    df['prev_gpa_sat_interaction'] = df['prev_gpa'] * df['prev_sat']
    df['sat_retention'] = df['sat'] / df['prev_sat'].replace(0, 1)
    df['prev_gpa_squared'] = df['prev_gpa'] ** 2
    df['sat_squared'] = df['sat'] ** 2
    return df

def standardize_categorical_values(df, reference_df):
    df = df.copy()
    categorical_columns = ['SCHOOL OWNERSHIP', 'SCHOOL CATEGORY', 'SCHOOL TYPE',
                          'COMBINATIONS CATEGORY', 'ACADEMIC LEVEL CATEGORY', 'teacher_qualification']
    reference_values = {col: set(reference_df[col].unique()) for col in categorical_columns if col in reference_df.columns}
    mapping_report = {}
    for col in categorical_columns:
        if col in df.columns:
            original_values = df[col].unique()
            mapping_report[col] = {'original': list(original_values), 'mapped': [], 'unmapped': []}
            df[col] = df[col].astype(str).str.upper().str.strip()
            if col in reference_values:
                ref_upper = {val.upper(): val for val in reference_values[col]}
                def map_value(val):
                    val_upper = str(val).upper().strip()
                    if val_upper in ref_upper:
                        mapping_report[col]['mapped'].append(f"{val} -> {ref_upper[val_upper]}")
                        return ref_upper[val_upper]
                    else:
                        mapping_report[col]['unmapped'].append(val)
                        return list(reference_values[col])[0]
                df[col] = df[col].apply(map_value)
    return df, mapping_report

def handle_missing_values(df, reference_df):
    df = df.copy()
    missing_report = {}
    numerical_cols = ['PREV_SAT', 'PREV_GPA', 'SAT', 'STUDENT_TEACHER_RATIO', 'number_of_books', 'studying_hours']
    categorical_cols = ['SCHOOL OWNERSHIP', 'SCHOOL CATEGORY', 'SCHOOL TYPE',
                       'COMBINATIONS CATEGORY', 'ACADEMIC LEVEL CATEGORY', 'teacher_qualification']
    default_values = {
        'PREV_SAT': 400, 'PREV_GPA': 2.5, 'SAT': 300, 'STUDENT_TEACHER_RATIO': 15,
        'number_of_books': 500, 'studying_hours': 90
    }
    categorical_defaults = {
        'SCHOOL OWNERSHIP': 'GOVERNMENT', 'SCHOOL CATEGORY': 'BOYS AND GIRLS', 'SCHOOL TYPE': 'DAY AND BOARDING',
        'COMBINATIONS CATEGORY': 'MIXED', 'ACADEMIC LEVEL CATEGORY': 'ALEVEL ONLY', 'teacher_qualification': 'Intermediate'
    }
    for col in numerical_cols:
        if col not in df.columns:
            df[col] = default_values.get(col, 0)
            missing_report[col] = f"Created missing column with default value: {default_values.get(col, 0)}"
        else:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                ref_col = col.lower()
                if ref_col in reference_df.columns:
                    median_val = reference_df[ref_col].median()
                    df[col].fillna(median_val, inplace=True)
                    missing_report[col] = f"Filled {missing_count} missing values with median: {median_val:.2f}"
                else:
                    default_val = default_values.get(col, 0)
                    df[col].fillna(default_val, inplace=True)
                    missing_report[col] = f"Filled {missing_count} missing values with default: {default_val}"
    for col in categorical_cols:
        if col not in df.columns:
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
    validation_report = {}
    ranges = {
        'PREV_SAT': (1, 1500), 'PREV_GPA': (1.0, 5.0), 'SAT': (1, 1500),
        'STUDENT_TEACHER_RATIO': (0.1, 100), 'number_of_books': (0, 2000), 'studying_hours': (40, 168)
    }
    for col, (min_val, max_val) in ranges.items():
        if col in df.columns:
            out_of_range = df[(df[col] < min_val) | (df[col] > max_val)]
            if len(out_of_range) > 0:
                validation_report[col] = f"Warning: {len(out_of_range)} values out of expected range ({min_val:.1f}-{max_val:.1f})"
                df[col] = df[col].clip(lower=min_val, upper=max_val)
    return df, validation_report

# Load data and model
df = load_data()
if df is not None:
    if 'STUDENT-TEACHER RATIO' in df.columns:
        df = df.drop('STUDENT-TEACHER RATIO', axis=1)
    df = df.dropna()

model = load_linear_regression_model()

if df is None or model is None:
    st.error("Failed to load required data or model. Please check file paths.")
    st.stop()

enhanced_df = engineer_features(df.copy())

if 'processed_batch_data' not in st.session_state:
    st.session_state.processed_batch_data = None
if 'batch_prediction_run' not in st.session_state:
    st.session_state.batch_prediction_run = False

# Main content
st.markdown("""
    <div style='background: linear-gradient(90deg, #10b981 0%, #059669 100%); 
                padding: 30px; border-radius: 15px; margin: 20px 0; color: white;'>
        <div style='text-align: center;'>
            <h2 style='margin: 0; font-size: 2em;'>🎓 School GPA Prediction Dashboard</h2>
            <p style='margin: 5px 0; opacity: 0.9; font-size: 1.1em;'>
                Empowering ACSEE Schools with AI-driven Performance Insights
            </p>
        </div>
    </div>
""", unsafe_allow_html=True)

class GPAHistoryManager:
    def __init__(self, max_size_mb=10):
        self.max_size_mb = max_size_mb
        self.history_key = "gpa_prediction_history"
        self.init_session_state()

    def init_session_state(self):
        if self.history_key not in st.session_state:
            st.session_state[self.history_key] = []

    def add_prediction(self, input_data, prediction, confidence_interval):
        history_entry = {
            'timestamp': datetime.now().isoformat(),
            'prev_sat': input_data['prev_sat'].iloc[0],
            'prev_gpa': input_data['prev_gpa'].iloc[0],
            'sat': input_data['sat'].iloc[0],
            'student_teacher_ratio': input_data['student_teacher_ratio'].iloc[0],
            'school_ownership': input_data['SCHOOL OWNERSHIP'].iloc[0],
            'school_category': input_data['SCHOOL CATEGORY'].iloc[0],
            'school_type': input_data['SCHOOL TYPE'].iloc[0],
            'combinations_category': input_data['COMBINATIONS CATEGORY'].iloc[0],
            'academic_level': input_data['ACADEMIC LEVEL CATEGORY'].iloc[0],
            'prev_gpa_sat_interaction': input_data['prev_gpa_sat_interaction'].iloc[0],
            'sat_retention': input_data['sat_retention'].iloc[0],
            'sat_improvement': input_data['sat_improvement'].iloc[0],
            'relative_sat_change': input_data['relative_sat_change'].iloc[0],
            'studying_hours': input_data['studying_hours'].iloc[0],
            'teacher_qualification': input_data['teacher_qualification'].iloc[0],
            'number_of_books': input_data['number_of_books'].iloc[0],
            'prev_gpa_squared': input_data['prev_gpa_squared'].iloc[0],
            'sat_squared': input_data['sat_squared'].iloc[0],
            'predicted_gpa': round(prediction, 3),
            'confidence_lower': round(confidence_interval[0], 3),
            'confidence_upper': round(confidence_interval[1], 3),
        }
        st.session_state[self.history_key].append(history_entry)
        return True

    def get_history_dataframe(self):
        if not st.session_state[self.history_key]:
            return pd.DataFrame()
        df = pd.DataFrame(st.session_state[self.history_key])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp', ascending=False)
        return df

    def get_gpa_color(self, gpa_value):
        if gpa_value >= 3.5:
            return "#4CAF50"
        elif gpa_value >= 3.0:
            return "#2196F3"
        elif gpa_value >= 2.5:
            return "#FF9800"
        elif gpa_value >= 2.0:
            return "#FF5722"
        else:
            return "#F44336"

    def get_gpa_category(self, gpa_value):
        if gpa_value >= 3.5:
            return "Excellent"
        elif gpa_value >= 3.0:
            return "Good"
        elif gpa_value >= 2.5:
            return "Average"
        elif gpa_value >= 2.0:
            return "Below Average"
        else:
            return "Poor"

    def display_history(self, max_entries=20):
        df = self.get_history_dataframe()
        if df.empty:
            st.info("📝 No prediction history available yet. Make your first prediction!")
            return
        st.markdown("#### 🔍 Filter & Sort Options")
        filter_col1, filter_col2 = st.columns(2)
        with filter_col1:
            date_range = st.date_input(
                "Date Range",
                value=(df['timestamp'].min().date(), df['timestamp'].max().date()),
                min_value=df['timestamp'].min().date(),
                max_value=df['timestamp'].max().date()
            )
        with filter_col2:
            gpa_min, gpa_max = st.slider(
                "GPA Range",
                min_value=float(1), max_value=float(5),
                value=(float(1), float(5)), step=0.1
            )
        filtered_df = df.copy()
        if len(date_range) == 2:
            start_date, end_date = date_range
            filtered_df = filtered_df[
                (filtered_df['timestamp'].dt.date >= start_date) &
                (filtered_df['timestamp'].dt.date <= end_date)
            ]
        filtered_df = filtered_df[
            (filtered_df['predicted_gpa'] >= gpa_min) &
            (filtered_df['predicted_gpa'] <= gpa_max)
        ]
        display_df = filtered_df.head(max_entries)
        st.markdown(f"**Showing {len(display_df)} of {len(filtered_df)} filtered results**")
        clear = False
        if len(display_df) > 0:
            bulk_col1, bulk_col2, bulk_col3 = st.columns([2, 2, 2])
            with bulk_col1:
                if not st.session_state.get('show_clear_confirmation', False) or clear != True:
                    if st.button("🗑️ Clear All History"):
                        st.session_state['show_clear_confirmation'] = True
                        clear = True
                        st.rerun()
            with bulk_col3:
                if st.button("📥 Filtered (CSV)", help="Download filtered results as CSV"):
                    csv_data = self.export_to_csv(filtered_df)
                    st.download_button(
                        label="💾 Download CSV",
                        data=csv_data,
                        file_name=f"gpa_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            if st.session_state.get('show_clear_confirmation', False) or clear:
                st.markdown("---")
                st.warning("⚠️ **Confirm Deletion**")
                st.write("Are you sure you want to delete all prediction history? This action cannot be undone.")
                conf_col1, conf_col2, conf_col3 = st.columns([1, 1, 2])
                with conf_col1:
                    if st.button("✅ Yes, Delete All", type="secondary"):
                        st.session_state[self.history_key] = []
                        st.session_state['show_clear_confirmation'] = False
                        clear = False
                        st.success("✅ All history cleared!")
                        st.rerun()
                with conf_col2:
                    if st.button("❌ Cancel"):
                        st.session_state['show_clear_confirmation'] = False
                        clear = False
                        st.rerun()
        st.markdown("---")
        if len(display_df) > 0:
            st.markdown("#### 📋 Prediction History")
            display_table = display_df.copy()
            display_table['Date'] = display_table['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
            display_table['GPA Category'] = display_table['predicted_gpa'].apply(self.get_gpa_category)
            display_table['Confidence Range'] = display_table.apply(
                lambda row: f"{row['confidence_lower']:.2f} - {row['confidence_upper']:.2f}", axis=1
            )
            columns_to_show = [
                'Date', 'prev_sat', 'prev_gpa', 'sat', 'student_teacher_ratio', 'studying_hours', 'number_of_books',
                'school_type', 'school_category', 'school_ownership', 'combinations_category', 'teacher_qualification', 'academic_level', 'predicted_gpa', 'Confidence Range'
            ]
            display_table = display_table[columns_to_show]
            display_table.columns = [
                'Date & Time', 'Previous SAT', 'Previous GPA', 'SAT', 'Student-Teacher Ratio', 'Studying Hours', 'Number of Books',
                'School Type', 'School Category', 'School Ownership', 'Combinations Category', 'Teacher Qualification', 'Academic Level', 'Predicted GPA', 'Confidence Range'
            ]
            def color_gpa_row(row):
                gpa_value = float(row['Predicted GPA'])
                color = self.get_gpa_color(gpa_value)
                return [f'background-color: {color}; color: white; font-weight: bold' if col == 'Predicted GPA'
                       else f'background-color: {color}20' for col in row.index]
            styled_df = display_table.style.apply(color_gpa_row, axis=1)
            st.dataframe(styled_df, use_container_width=True, height=450)
            st.markdown("#### 🎨 GPA Color Legend")
            legend_cols = st.columns(5)
            legend_items = [
                ("Excellent (3.5+)", "#4CAF50"), ("Good (3.0-3.4)", "#2196F3"),
                ("Average (2.5-2.9)", "#FF9800"), ("Below Average (2.0-2.4)", "#FF5722"),
                ("Poor (<2.0)", "#F44336")
            ]
            for i, (category, color) in enumerate(legend_items):
                with legend_cols[i]:
                    st.markdown(f"""
                        <div style='background-color: {color}; color: white; padding: 5px; 
                                   border-radius: 5px; text-align: center; font-size: 0.8em;'>
                            {category}
                        </div>
                    """, unsafe_allow_html=True)

    def export_to_csv(self, df):
        if df.empty:
            return ""
        export_df = df.copy()
        export_df['timestamp'] = export_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        column_order = [
            'timestamp', 'prev_sat', 'prev_gpa', 'sat', 'student_teacher_ratio', 'studying_hours', 'number_of_books',
            'school_ownership', 'school_category', 'school_type',
            'combinations_category', 'teacher_qualification', 'academic_level', 'predicted_gpa', 'confidence_lower', 'confidence_upper'
        ]
        available_columns = [col for col in column_order if col in export_df.columns]
        export_df = export_df[available_columns]
        column_mapping = {
            'timestamp': 'Date & Time', 'prev_sat': 'Previous SAT', 'prev_gpa': 'Previous GPA', 'sat': 'SAT',
            'student_teacher_ratio': 'Student-Teacher Ratio', 'studying_hours': 'Studying Hours', 'number_of_books': 'Number of Books',
            'school_ownership': 'School Ownership', 'school_category': 'School Category', 'school_type': 'School Type',
            'combinations_category': 'Combinations Category', 'teacher_qualification': 'Teacher Qualification',
            'academic_level': 'Academic Level', 'predicted_gpa': 'Predicted GPA', 'confidence_lower': 'Confidence Lower',
            'confidence_upper': 'Confidence Upper'
        }
        export_df = export_df.rename(columns=column_mapping)
        return export_df.to_csv(index=False)

def initialize_history_manager():
    if 'history_manager' not in st.session_state:
        st.session_state.history_manager = GPAHistoryManager()
    return st.session_state.history_manager

def add_prediction_to_history(input_data, prediction, confidence_interval):
    history_manager = initialize_history_manager()
    return history_manager.add_prediction(input_data, prediction, confidence_interval)

def display_history_tab():
    history_manager = initialize_history_manager()
    history_manager.display_history(max_entries=10)

def make_prediction_with_history(model, input_data):
    prediction = model.predict(input_data)[0]
    prediction = max(0.0, min(4.95, prediction))  # Clip prediction to 0.0–5.0
    model_rmse = 0.1174
    margin = 1.96 * model_rmse  # Margin of error ≈ 0.23
    lower_bound = max(0.0, prediction - margin)
    upper_bound = min(5.0, prediction + margin)  # Update cap to 5.0
    confidence_interval = (lower_bound, upper_bound)
    prediction = float(prediction)
    confidence_interval = (float(confidence_interval[0]), float(confidence_interval[1]))
    add_prediction_to_history(input_data, prediction, confidence_interval)
    return prediction, confidence_interval

# Tabs
tabs = st.tabs(["📝 Standard Prediction", "📊 Batch Prediction", "Prediction History"])

# Standard Prediction Tab
with tabs[0]:
    st.markdown("""
        <div class="card">
            <center>Enter School information below to predict the expected GPA. The more accurate the input data,
            the more reliable the prediction will be. Required fields are marked with *</center>
        </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        prev_sat = st.number_input(
            "Previous SAT*", min_value=1, max_value=1000, value=200, step=10,
            help="Number of candidates who sat for exams in the previous academic year."
        )
        prev_gpa = st.number_input(
            "Previous GPA*", min_value=1.0, max_value=5.0, value=float(enhanced_df['prev_gpa'].median()), step=0.1,
            help="School's GPA from the previous academic year."
        )
        sat = st.number_input(
            "Current SAT*", min_value=1, max_value=1000, value=300,
            help="Number of candidates who sat for exams this year."
        )
        student_teacher_ratio = st.number_input(
            "Student-Teacher Ratio*", min_value=0.01, max_value=100.0, value=float(enhanced_df['student_teacher_ratio'].median()),
            help="Ratio of students to teachers."
        )
        studying_hours = st.number_input(
            "Students Studying Hours (Per Week)*", min_value=40.0, max_value=168.0, value=enhanced_df['studying_hours'].median(),
            help="Hours per week."
        )
        number_of_books = st.number_input(
            "Number of Books Available*", min_value=0, max_value=2000, value=int(enhanced_df.number_of_books.median()),
            help="Number of textbooks or resources."
        )
    with col2:
        school_ownership = st.selectbox(
            "School Ownership*", options=enhanced_df['SCHOOL OWNERSHIP'].unique(),
            help="Type of school ownership (e.g., Government, Private)."
        )
        school_category = st.selectbox(
            "School Category*", options=enhanced_df['SCHOOL CATEGORY'].unique(),
            help="Category of the school (e.g., Boys, Girls, Mixed)."
        )
        school_type = st.selectbox(
            "School Type*", options=enhanced_df['SCHOOL TYPE'].unique(),
            help="Type of school accommodation (e.g., Day, Boarding)."
        )
        combinations_category = st.selectbox(
            "Combinations Category*", options=enhanced_df['COMBINATIONS CATEGORY'].unique(),
            help="Category of subject combinations offered."
        )
        academic_level = st.selectbox(
            "Academic Level*", options=enhanced_df['ACADEMIC LEVEL CATEGORY'].unique(),
            help="Academic level of the school (e.g., A-Level)."
        )
        descriptions = f""" 
        _**Basic:**_  Represents lower qualifications (e.g., minimal training or experience).               
        _**Intermediate:**_  Represents moderate qualifications (e.g., standard teaching credentials).            
        _**Advanced:**_  Represents high qualifications (e.g., advanced degrees or extensive experience).
        """
        options = ['Basic', 'Intermediate', 'Advanced']
        teacher_qualification = st.selectbox(
            "Teacher Qualification*", options=options,
            help=descriptions
        )

    inputs_valid = all([prev_sat > 0, 0 <= prev_gpa <= 5.0, sat > 0, student_teacher_ratio > 0])
    if not inputs_valid:
        st.warning("Please ensure all numerical inputs are within valid ranges.")

    sat_improvement = sat - prev_sat
    relative_sat_change = sat_improvement / prev_sat if prev_sat > 0 else 0
    prev_gpa_sat_interaction = prev_gpa * prev_sat
    sat_retention = sat / prev_sat if prev_sat > 0 else 1.0
    prev_gpa_squared = prev_gpa ** 2
    sat_squared = sat ** 2

    input_data = pd.DataFrame({
        'prev_sat': [prev_sat], 'prev_gpa': [prev_gpa], 'sat': [sat], 'student_teacher_ratio': [student_teacher_ratio],
        'SCHOOL OWNERSHIP': [school_ownership], 'SCHOOL CATEGORY': [school_category], 'SCHOOL TYPE': [school_type],
        'COMBINATIONS CATEGORY': [combinations_category], 'ACADEMIC LEVEL CATEGORY': [academic_level],
        'prev_gpa_sat_interaction': [prev_gpa_sat_interaction], 'sat_retention': [sat_retention],
        'sat_improvement': [sat_improvement], 'relative_sat_change': [relative_sat_change],
        'prev_gpa_squared': [prev_gpa_squared], 'sat_squared': [sat_squared],
        'teacher_qualification': [teacher_qualification], 'studying_hours': [studying_hours],
        'number_of_books': [number_of_books]
    })
    
    
    if st.button("Generate GPA Prediction", use_container_width=True):
        with st.spinner("Analyzing data and generating prediction..."):
            prediction, confidence_interval = make_prediction_with_history(model, input_data)
            st.markdown(f"""
                <div class="card">
                    <h2 style='text-align: center; color: #1e293b;'>Predicted GPA</h2>
                    <div style='text-align: center; font-size: 3rem; font-weight: bold; color: #10b981;'>
                        {prediction:.2f}
                    </div>
                    <p style='text-align: center; color: #64748b;'>
                        95% Confidence Interval: {min(confidence_interval):.2f} - {max(confidence_interval):.2f}
                    </p>
                </div>
            """, unsafe_allow_html=True)

            # GPA Overview with Colored Boxes
            st.markdown('<div class="sub-header">GPA Overview</div>', unsafe_allow_html=True)
            difference = prediction - prev_gpa
            if difference > 0.1:
                st.markdown(f"""
                    <div style='background: #e6ffed; padding: 15px; border-radius: 8px; border-left: 5px solid #22c55e;'>
                        <h4 style='color: #15803d; margin: 0;'>Improvement 📈</h4>
                        <p style='margin: 5px 0;'>Your predicted GPA is {difference:.2f} points higher than your previous GPA of {prev_gpa:.2f}.</p>
                        <p style='margin: 5px 0; color: #15803d;'><strong>Tip:</strong> Keep up the good work!</p>
                    </div>
                """, unsafe_allow_html=True)
            elif difference < -0.1:
                st.markdown(f"""
                    <div style='background: #fee2e2; padding: 15px; border-radius: 8px; border-left: 5px solid #ef4444;'>
                        <h4 style='color: #dc2626; margin: 0;'>Decline 📉</h4>
                        <p style='margin: 5px 0;'>Your predicted GPA is {abs(difference):.2f} points lower than your previous GPA of {prev_gpa:.2f}.</p>
                        <p style='margin: 5px 0; color: #dc2626;'><strong>Tip:</strong> Check the recommendations below.</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div style='background: #e0f2fe; padding: 15px; border-radius: 8px; border-left: 5px solid #3b82f6;'>
                        <h4 style='color: #1e40af; margin: 0;'>Stable 🔄</h4>
                        <p style='margin: 5px 0;'>Your predicted GPA is close to your previous GPA of {prev_gpa:.2f}.</p>
                        <p style='margin: 5px 0; color: #1e40af;'><strong>Tip:</strong> Small changes could boost your GPA.</p>
                    </div>
                """, unsafe_allow_html=True)

            # Key Factors with Colored Indicators
            # st.markdown('<div class="sub-header">Key Factors</div>', unsafe_allow_html=True)
            reference_data = load_data()
            medians = reference_data[['student_teacher_ratio', 'number_of_books', 'studying_hours']].median().to_dict()

            
            qualification = input_data['teacher_qualification'].iloc[0]
        
            # Recommendations
            st.markdown('<div class="sub-header">Recommendations</div>', unsafe_allow_html=True)
            recommendations = []
            for feat in ['student_teacher_ratio', 'number_of_books', 'studying_hours']:
                input_val = input_data[feat].iloc[0]
                median_val = medians[feat]
                feat_display = feat.replace('_', ' ').title()
                if feat == 'student_teacher_ratio' and input_val > median_val:
                    recommendations.append(f"Consider reducing {feat_display} for better results.")
                elif feat in ['number_of_books', 'studying_hours'] and input_val < median_val:
                    recommendations.append(f"Increase {feat_display}.")
            if qualification == 'Basic':
                recommendations.append("Encourage teacher training to enhance qualifications.")
            if not recommendations:
                recommendations.append("Maintain current efforts!")

            for rec in recommendations:
                st.markdown(f"<span style='color: #1e40af;'>• {rec}</span>", unsafe_allow_html=True)

    
# Batch Prediction Tab
with tabs[1]:
    uploaded_file = st.file_uploader("Upload CSV file with school data", type=["csv"])
    col1, col2 = st.columns([4, 1])
    with col1:
        show_processing_details = st.checkbox("Show detailed processing reports", value=True)
    with col2:
        strict_validation = st.checkbox("Enable strict data validation", value=False)
    col1, col2 = st.columns([4, 1])
    with col1:
        with st.expander("View Required CSV Format"):
            st.markdown("""
            Your CSV file should include the following columns (case-insensitive):
            - **prev_sat** - Number of candidates in previous examination.
            - **prev_gpa** - PSchool's GPA from the previous academic year.
            - **sat** - Number of candidates sat for examination.
            - **student_teacher_ratio** - Ratio of students to teachers.
            - **studying_hours** - Average hours students study per week.
            - **number_of_books** - Number of textbooks or resources available.
            - **teacher_qualification** - Qualification level of teachers (e.g., Basic, Intermediate, Advanced).
            - **SCHOOL OWNERSHIP** - Type of school ownership (e.g., Government, Private).
            - **SCHOOL CATEGORY** - Category of the school (e.g., Boys, Girls, Mixed).
            - **SCHOOL TYPE** - Type of school accommodation (e.g., Day, Boarding).
            - **COMBINATIONS CATEGORY** - Category of subject combinations offered.
            - **ACADEMIC LEVEL CATEGORY** - Academic level of the school (e.g., A-Level).
            
            **Note**: Missing values will be automatically handled!
            """)
            sample_df = df.head(10)
            st.dataframe(sample_df)
    with col2:
        st.download_button(
            label="📥 Download Template CSV",
            data=sample_df.to_csv(index=False),
            file_name="gpa_prediction_template.csv",
            mime="text/csv"
        )
    if uploaded_file is not None:
        try:
            batch_data = pd.read_csv(uploaded_file)
            st.success(f"✅ Successfully loaded {len(batch_data)} records from '{uploaded_file.name}'")
            with st.expander("📋 Preview Uploaded Data"):
                st.dataframe(batch_data.head(10))
                st.write(f"**Shape**: {batch_data.shape[0]} rows × {batch_data.shape[1]} columns")
            required_cols = ['prev_sat', 'prev_gpa', 'sat', 'student_teacher_ratio', 'studying_hours', 'number_of_books',
                            'SCHOOL OWNERSHIP', 'SCHOOL CATEGORY', 'SCHOOL TYPE',
                            'COMBINATIONS CATEGORY', 'ACADEMIC LEVEL CATEGORY', 'teacher_qualification']
            original_columns = batch_data.columns.tolist()
            batch_data.columns = batch_data.columns.str.upper().str.strip()
            numerical_column_mapping = {
                'PREV_SAT': 'PREV_SAT', 'PREVIOUS_SAT': 'PREV_SAT', 'PREV GPA': 'PREV_GPA',
                'PREVIOUS_GPA': 'PREV_GPA', 'PREVIOUS GPA': 'PREV_GPA', 'SAT': 'SAT','STUDYING_HOURS' : 'studying_hours', 'NUMBER_OF_BOOKS': 'number_of_books',
                'STUDENT_TEACHER_RATIO': 'STUDENT_TEACHER_RATIO', 'STUDENT TEACHER RATIO': 'STUDENT_TEACHER_RATIO',
                'STUDENT-TEACHER RATIO': 'STUDENT_TEACHER_RATIO', 'TEACHER_QUALIFICATION': 'teacher_qualification',
            }
            for old_name, new_name in numerical_column_mapping.items():
                if old_name in batch_data.columns and new_name not in batch_data.columns:
                    batch_data[new_name] = batch_data[old_name]
            batch_data_cols_upper = [col.upper() for col in batch_data.columns]
            required_cols_numerical = ['PREV_SAT', 'PREV_GPA', 'SAT', 'STUDENT_TEACHER_RATIO', 'STUDYING_HOURS', 'NUMBER_OF_BOOKS']
            required_cols_categorical = ['SCHOOL OWNERSHIP', 'SCHOOL CATEGORY', 'SCHOOL TYPE', 'TEACHER_QUALIFICATION',
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
                        processed_data, missing_report = handle_missing_values(batch_data, enhanced_df)
                        progress_bar.progress(25)
                        processed_data, mapping_report = standardize_categorical_values(processed_data, enhanced_df)
                        progress_bar.progress(50)
                        processed_data, validation_report = validate_data_ranges(processed_data, enhanced_df)
                        progress_bar.progress(75)
                        try:
                            processed_data = engineer_features(processed_data)
                            predictions = model.predict(processed_data)
                            processed_data['predicted_gpa'] = predictions
                            st.session_state.processed_batch_data = processed_data
                            st.session_state.batch_prediction_run = True
                            st.session_state.missing_report = missing_report
                            st.session_state.mapping_report = mapping_report
                            st.session_state.validation_report = validation_report
                            progress_bar.progress(100)
                            st.markdown('<div class="sub-header">Prediction Results</div>', unsafe_allow_html=True)
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Total Predictions", len(predictions))
                            with col2:
                                st.metric("Average Predicted GPA", f"{np.mean(predictions):.2f}")
                            with col3:
                                st.metric("Min GPA", f"{np.min(predictions):.2f}")
                            with col4:
                                st.metric("Max GPA", f"{np.max(predictions):.2f}")
                            st.dataframe(processed_data, use_container_width=True)
                            st.download_button(
                                label="📥 Download Predictions CSV",
                                data=processed_data.to_csv(index=False),
                                file_name=f"gpa_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                        except Exception as e:
                            st.error(f"❌ Error during prediction: {str(e)} {processed_data.columns}")
                            st.info("This might be due to data format issues. Please check your data and try again.")
                        finally:
                            progress_bar.empty()
                if st.session_state.batch_prediction_run and st.session_state.processed_batch_data is not None:
                    processed_data = st.session_state.processed_batch_data
                    missing_report = st.session_state.missing_report
                    mapping_report = st.session_state.mapping_report
                    validation_report = st.session_state.validation_report
                    st.markdown('<div class="sub-header">Prediction Analysis</div>', unsafe_allow_html=True)
                    col1, col2 = st.columns(2)
                    with col1:
                        fig1 = px.histogram(processed_data, x='predicted_gpa', nbins=20,
                                            title="Distribution of Predicted GPAs",
                                            color_discrete_sequence=['#1E40AF'])
                        fig1.update_layout(showlegend=False)
                        st.plotly_chart(fig1, use_container_width=True)
                    with col2:
                        if 'predicted_gpa' in processed_data.columns:
                            available_cat_features = [
                                'SCHOOL OWNERSHIP', 'SCHOOL CATEGORY', 'SCHOOL TYPE',
                                'COMBINATIONS CATEGORY', 'ACADEMIC LEVEL CATEGORY', 'teacher_qualification'
                            ]
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
                    if show_processing_details:
                        st.markdown('<div class="sub-header">Data Processing Report</div>', unsafe_allow_html=True)
                        if missing_report:
                            st.markdown("**Missing Values Handled:**")
                            for col, report in missing_report.items():
                                st.write(f"• {col}: {report}")
                        else:
                            st.success("✅ No missing values found!")
                        if any(mapping_report.values()):
                            st.markdown("**Categorical Data Standardization:**")
                            for col, report in mapping_report.items():
                                if report['unmapped']:
                                    st.write(f"**{col}:**")
                                    if report['unmapped']:
                                        st.write(f"  • Unmapped values: {', '.join(report['unmapped'][:3])}")
                        if validation_report:
                            st.markdown("**Data Validation:**")
                            for col, report in validation_report.items():
                                st.warning(f"• {col}: {report}")
                        else:
                            st.success("✅ All data within expected ranges!")
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.info("Please ensure your CSV file is properly formatted and try again.")

@st.dialog("👥 Contributors")
def display_contributors_modal():
    contributors_data = [
        {"name": "Joseph Magiha", "role": "Member", "email": "johcrazy.magiha@gmail.com", "contributions": ["UI/UX", "Deployment"], "github": "JOHCRAZY"},
        {"name": "Irene Bushiri", "role": "Member", "email": "bushiriirene213@gmail.com", "contributions": ["Data Analysis", "Model Validation"], "github": None},
        {"name": "Mwanabay Kasim", "role": "Member", "email": "mwanabaysindi@gmail.com", "contributions": ["Project Documentation", "Testing"], "github": None},
        {"name": "Ibrahim Kitupe", "role": "Member", "email": None, "contributions": ["Data Acquisition", "Model Development"], "github": None}
    ]
    st.markdown("""
        <div style='text-align: center; margin-bottom: 20px;'>
            <h2 style='color: #1f77b4;'>🌟 Meet Our Amazing Team</h2>
            <p style='color: #666; font-style: italic;'>The brilliant minds behind the ACSEE GPA Prediction Dashboard</p>
        </div>
    """, unsafe_allow_html=True)
    for i, contributor in enumerate(contributors_data):
        with st.expander(f"👤 **{contributor['name']}** - Project {contributor['role']}", expanded=i==0):
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown(f"""
                    <div style='width: 80px; height: 80px; border-radius: 50%; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                                display: flex; align-items: center; justify-content: center; color: white; font-size: 30px; font-weight: bold; margin: 10px auto;'>
                        {contributor['name'][0]}
                    </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"**Role:** {contributor['role']}")
                if contributor['email']:
                    st.markdown(f"📧 {contributor['email']}")
                if contributor['github']:
                    st.markdown(f"🔗 [GitHub Profile](https://github.com/{contributor['github']})")
                st.markdown("**Contributions:**")
                contributions_html = " ".join([
                    f"<span style='background-color: #e1f5fe; color: #01579b; padding: 2px 8px; border-radius: 12px; font-size: 0.8em; margin: 2px;'>{contrib}</span>"
                    for contrib in contributor['contributions']
                ])
                st.markdown(contributions_html, unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.markdown("[⭐ Star Repository](https://github.com/JOHCRAZY/ACSEE_GPA/stargazers)")
    with col2:
        st.markdown("[🔗 Report Issue](https://github.com/JOHCRAZY/ACSEE_GPA/issues)")
    with col3:
        st.markdown("[📚 View Docs](https://github.com/JOHCRAZY/ACSEE_GPA/blob/master/README.md)")

def display_footer():
    st.markdown("---")
    link_cols = st.columns([2, 2, 2])
    with link_cols[0]:
        st.markdown("""
            **📞 Contact & Project Information**
            - 📧 **Email:** johcrazy.magiha@gmail.com
            - 🏫 **Institution:** [Eastern Africa Statistical Training Centre](https://www.eastc.ac.tz)
            - 💡 **Purpose:** Grade Point Average Prediction for Tanzania [ACSEE](https://www.necta.go.tz/pages/acsee) Schools
            - 🛠️ **Built with:** [Python](https://www.python.org/) • [Streamlit](https://streamlit.io/)
        """)
    with link_cols[1]:
        current_time = datetime.now()
        st.markdown(f"""
            <div style='text-align: center; border-radius: 8px; font-size: 0.9em;'>
                <strong>GPA Prediction Dashboard v1.0.0</strong><br>
                Built with ❤️ using Streamlit & Python<br>
                © 2025 | Last Build: {current_time.strftime("%B %Y")}<br>
                <em>EASTC 🇹🇿</em>
            </div>
        """, unsafe_allow_html=True)
        st.markdown("---")
        if st.button("👥 Contributors", help="Meet our amazing contributors", use_container_width=True, type="secondary"):
            display_contributors_modal()
    with link_cols[2]:
        st.markdown("""
            **🔗 Quick Access Links**
            - 🐙 [GitHub Repository](https://github.com/JOHCRAZY/ACSEE_GPA)
            - 📖 [Documentation](https://github.com/JOHCRAZY/ACSEE_GPA/blob/master/README.md)
            - 🐛 [Report Issues](https://github.com/JOHCRAZY/ACSEE_GPA/issues)
            - ⭐ [Star the Project](https://github.com/JOHCRAZY/ACSEE_GPA/stargazers)
        """)

with tabs[2]:
    display_history_tab()

display_footer()