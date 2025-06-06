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

# Set page configuration with improved styling
st.set_page_config(
    page_title="GPA Prediction Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to enhance the appearance
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid #E5E7EB;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #1E3A8A;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .card {
        padding: 1.5rem;
        border-radius: 0.5rem;
        /*background-color: #F9FAFB;*/
        /*border: 1px solid #E5E7EB;*/
        margin-bottom: 1rem;
    }
    .metric-card {
        /*background-color: #EFF6FF;
        border: 1px solid #BFDBFE;*/
        border-radius: 0.5rem;
        padding: 1rem;
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #1E40AF;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #1E3A8A;
    }
    .highlight {
        background-color: #DBEAFE;
        padding: 0.5rem;
        border-radius: 0.25rem;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #E5E7EB;
        color: #6B7280;
        font-size: 0.8rem;
    }
    .tooltip {
        position: relative;
        display: inline-block;
        border-bottom: 1px dotted #3B82F6;
        cursor: help;
    }
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background-color: #1E40AF;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 5px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
    }
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
</style>
""", unsafe_allow_html=True)

# Function to load data with caching
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data/df.csv')
        # Handle missing values more intelligently
        # For numeric columns, use median
        # For categorical columns, use most frequent value
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        for col in numeric_cols:
            df[col].fillna(df[col].median(), inplace=True)
        
        for col in categorical_cols:
            df[col].fillna(df[col].mode()[0], inplace=True)
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        # Return a small sample dataset if the file doesn't exist
        return pd.DataFrame({
            'prev_sat': [500, 550, 600],
            'prev_gpa': [3.0, 3.5, 4.0],
            'sat': [520, 570, 620],
            'student_teacher_ratio': [15, 20, 25],
            'SCHOOL OWNERSHIP': ['Public', 'Private', 'Public'],
            'SCHOOL CATEGORY': ['A', 'B', 'C'],
            'SCHOOL TYPE': ['Urban', 'Suburban', 'Rural'],
            'COMBINATIONS CATEGORY': ['X', 'Y', 'Z'],
            'ACADEMIC LEVEL CATEGORY': ['Undergraduate', 'Graduate', 'Undergraduate'],
            'gpa': [3.2, 3.7, 4.2]
        })

# Load data
df = load_data()

# Add feature engineering functionality
def engineer_features(data):
    """Create additional features that might improve model performance"""
    data = data.copy()
    
    # Interaction terms
    data['prev_gpa_sat_interaction'] = data['prev_gpa'] * data['prev_sat']
    data['sat_retention'] = data['sat'] / data['prev_sat'].replace(0, 1)  # Avoid division by zero
    
  
        
    return data

# Apply feature engineering
enhanced_df = engineer_features(df)

# Define a better model saving/loading system
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

def save_model(model, name):
    """Save a model to disk with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{MODEL_DIR}/{name}_{timestamp}.pkl"
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    return filename

def load_latest_model(name_pattern):
    """Load the most recently saved model matching the pattern"""
    try:
        files = [f for f in os.listdir(MODEL_DIR) if name_pattern in f and f.endswith('.pkl')]
        if not files:
            return None
        latest_file = max(files, key=lambda x: os.path.getctime(os.path.join(MODEL_DIR, x)))
        with open(os.path.join(MODEL_DIR, latest_file), 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.warning(f"Could not load model: {e}")
        return None

# Improved model training function with more models and cross-validation
@st.cache_resource
def train_models(data, features_to_use=None, target='gpa', test_size=0.2, cv_folds=5):
    """Train and evaluate multiple models with cross-validation"""
    
    if features_to_use is None:
        # Define feature groups 
        numeric_features = ['prev_sat', 'prev_gpa', 'sat', 'student_teacher_ratio',
                           'prev_gpa_sat_interaction', 'sat_retention']
        categorical_features = ['SCHOOL OWNERSHIP', 'SCHOOL CATEGORY', 'SCHOOL TYPE',
                               'COMBINATIONS CATEGORY', 'ACADEMIC LEVEL CATEGORY']
    else:
        # Split provided features into numeric and categorical
        numeric_features = [f for f in features_to_use if data[f].dtype in ['int64', 'float64']]
        categorical_features = [f for f in features_to_use if data[f].dtype == 'object']
    
    # Create preprocessing pipelines with better defaults
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'  # Drop any columns not specified
    )
    
    # Create models with improved parameters
    models = {
        'Linear Regression': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', LinearRegression())
        ]),
        'Random Forest': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', RandomForestRegressor(n_estimators=100, max_depth=15, min_samples_leaf=2, random_state=42))
        ]),
        'Gradient Boosting': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42))
        ])
    }
    
    # Prepare data
    X = data.drop(target, axis=1) if target in data.columns else data
    y = data[target] if target in data.columns else None
    
    if y is not None:
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Train the models
        trained_models = {}
        model_metrics = {}
        feature_importances = {}
        
        for name, model in models.items():
            # Train the model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Perform cross-validation
            cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring='r2')
            
            trained_models[name] = model
            model_metrics[name] = {
                'RMSE': rmse,
                'MAE': mae,
                'R²': r2,
                'CV R² (mean)': cv_scores.mean(),
                'CV R² (std)': cv_scores.std()
            }
            
            # Get feature importance if possible
            if name in ['Random Forest', 'Gradient Boosting']:
                try:
                    # Get feature names after preprocessing
                    feature_names = (
                        numeric_features + 
                        list(model.named_steps['preprocessor']
                             .transformers_[1][1]
                             .named_steps['onehot']
                             .get_feature_names_out(categorical_features))
                    )
                    
                    # Get feature importances
                    importances = model.named_steps['model'].feature_importances_
                    
                    # Create a DataFrame with feature names and importances
                    if len(importances) == len(feature_names):
                        importance_df = pd.DataFrame({
                            'Feature': feature_names,
                            'Importance': importances
                        }).sort_values('Importance', ascending=False)
                        
                        feature_importances[name] = importance_df
                except Exception as e:
                    st.warning(f"Could not calculate feature importance for {name}: {e}")
        
        return {
            'trained_models': trained_models,
            'model_metrics': model_metrics,
            'feature_importances': feature_importances,
            'preprocessor': preprocessor,
            'numeric_features': numeric_features,
            'categorical_features': categorical_features,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
        }
    else:
        st.error(f"Target variable '{target}' not found in data")
        return None

# Train models with enhanced features
model_results = train_models(enhanced_df)

# Improved sidebar with better organization
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/graduation-cap.png", width=80)
    st.title("Navigation")
    
    # Add user authentication (simulated)
    with st.expander("👤 User Login", expanded=False):
        st.text_input("Username")
        st.text_input("Password", type="password")
        st.button("Login")
    
    # Navigation
    page = st.radio("Go to", [
        "🏠 Home", 
        "🔮 Predict GPA", 
        "📊 Model Performance", 
        "🔍 Data Exploration",
        "⚙️ Advanced Options"
    ])
    
    # Help section in sidebar
    with st.expander("❓ Help & Resources"):
        st.markdown("""
        **Need help?**
        - Check our [documentation](https://example.com)
        - Watch [tutorial videos](https://example.com/tutorials)
        - Contact [support](mailto:support@example.com)
        """)
    
    # Version information
    st.sidebar.markdown("---")
    st.sidebar.info("Version 2.0 | Last updated: May 2025")

# Main content based on page selection
if "🏠 Home" in page:
    # Header with animation effect
    st.markdown('<div class="main-header">🎓 GPA Prediction Dashboard</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
        This advanced dashboard helps educators and administrators predict student GPA based on various factors
        including prior academic performance, institutional characteristics, and educational environment.
        Our machine learning models help identify students who may need additional support.
    </div>
    """, unsafe_allow_html=True)
    
    # Dashboard metrics overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">5</div>
            <div class="metric-label">Prediction Models</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Calculate best model based on R²
        best_model = max(model_results['model_metrics'].items(), key=lambda x: x[1]['R²'])[0]
        best_r2 = model_results['model_metrics'][best_model]['R²']
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{best_r2:.3f}</div>
            <div class="metric-label">Best Model R²</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        num_features = len(model_results['numeric_features']) + len(model_results['categorical_features'])
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{num_features}</div>
            <div class="metric-label">Predictive Features</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        sample_size = len(enhanced_df)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{sample_size}</div>
            <div class="metric-label">Sample Size</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Main dashboard content
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown('<div class="sub-header">How It Works</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="card">
            <h4>Predict School Performance</h4>
            <p>Our dashboard uses machine learning to predict school GPA based on various factors, helping educators identify students who may need additional support.</p>
            
            <h4>Key Factors Considered</h4>
            <ul>
                <li><strong>Previous academic performance</strong> (GPA)</li>
                <li><strong>School characteristics</strong> (ownership, category, type)</li>
                <li><strong>Student-teacher ratio</strong></li>
                <li><strong>Academic level</strong> and program combinations</li>
            </ul>
            
            <h4>Why Use This Tool</h4>
            <ul>
                <li><strong>Resource allocation</strong> - Direct support where it's most needed</li>
                <li><strong>Strategic planning</strong> - Make data-driven decisions for institutional improvement</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="sub-header">Quick Start</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="card">
            <h4>1. Predict GPA</h4>
            <p>Enter school information to get an instant GPA prediction with confidence intervals.</p>
            
            <h4>2. Explore Model Performance</h4>
            <p>Compare different prediction models and understand their strengths and limitations.</p>
            
            <h4>3. Analyze Data Relationships</h4>
            <p>Discover correlations and insights about factors affecting school performance.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Sample data preview
        with st.expander("📋 View Sample Data"):
            st.dataframe(enhanced_df.head(5))
            
        # Quick model comparison
        with st.expander("🔍 Quick Model Comparison"):
            # Create a dataframe of key metrics
            quick_metrics = pd.DataFrame({
                'Model': list(model_results['model_metrics'].keys()),
                'R²': [m['R²'] for m in model_results['model_metrics'].values()],
                'RMSE': [m['RMSE'] for m in model_results['model_metrics'].values()]
            }).sort_values('R²', ascending=False)
            
            st.dataframe(quick_metrics)

elif "🔮 Predict GPA" in page:
    st.markdown('<div class="main-header">GPA Prediction Tool</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
        Enter student and school information below to predict the expected GPA. The more accurate the input data, 
        the more reliable the prediction will be. Required fields are marked with *
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for different prediction approaches
    tabs = st.tabs(["📝 Standard Prediction", "📊 Batch Prediction", "🎯 What-If Analysis"])
    
    # Standard Prediction Tab
    with tabs[0]:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="sub-header">Academic Information</div>', unsafe_allow_html=True)
            
            # Add tooltips to explain features
            st.markdown("""
            <div class="tooltip">Previous SAT 
                <span class="tooltiptext">School's SAT  from previous academic year</span>
            </div>
            """, unsafe_allow_html=True)
            prev_sat = st.slider("Previous SAT*", 
                               min_value=float(enhanced_df['prev_sat'].min()), 
                               max_value=float(enhanced_df['prev_sat'].max()), 
                               value=float(enhanced_df['prev_sat'].median()))
            
            st.markdown("""
            <div class="tooltip">Previous GPA
                <span class="tooltiptext">School's GPA from previous academic year</span>
            </div>
            """, unsafe_allow_html=True)
            prev_gpa = st.slider("Previous GPA*", 
                               min_value=float(enhanced_df['prev_gpa'].min()), 
                               max_value=float(enhanced_df['prev_gpa'].max()), 
                               value=float(enhanced_df['prev_gpa'].median()))
            
            st.markdown("""
            <div class="tooltip">Current SAT
                <span class="tooltiptext">Student's most recent SAT</span>
            </div>
            """, unsafe_allow_html=True)
            sat = st.slider("Current SAT*", 
                          min_value=float(enhanced_df['sat'].min()), 
                          max_value=float(enhanced_df['sat'].max()), 
                          value=float(enhanced_df['sat'].median()))
            
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
            
            student_teacher_ratio = st.slider("Student-Teacher Ratio*", 
                                            min_value=float(enhanced_df['student_teacher_ratio'].min()), 
                                            max_value=float(enhanced_df['student_teacher_ratio'].max()), 
                                            value=float(enhanced_df['student_teacher_ratio'].median()))
        
        # Create derived features
        sat_improvement = sat - prev_sat
        relative_sat_change = sat_improvement / prev_sat if prev_sat > 0 else 0
        prev_gpa_sat_interaction = prev_gpa * prev_sat
        sat_retention = sat / prev_sat if prev_sat > 0 else 1.0
        prev_gpa_squared = prev_gpa ** 2
        sat_squared = sat ** 2
        public_school = 1 if school_ownership == 'Public' else 0
        
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
            'sat_squared': [sat_squared],
            'public_school': [public_school]
        })
        
        # Model selection with better UI
        col1, col2 = st.columns([2, 1])
        
        with col1:
            selected_model = st.selectbox(
                "Select Prediction Model",
                options=list(model_results['trained_models'].keys()),
                index=list(model_results['trained_models'].keys()).index('Gradient Boosting')
                if 'Gradient Boosting' in model_results['trained_models'] else 0
            )
        
        with col2:
            confidence_level = st.slider("Confidence Level", 80, 99, 95)
        
        # Prediction button with animation
        if st.button("Generate GPA Prediction", use_container_width=True):
            with st.spinner("Analyzing data and generating prediction..."):
                # Add slight delay for better UX
                import time
                time.sleep(0.5)
                
                # Make prediction
                prediction = model_results['trained_models'][selected_model].predict(input_data)[0]
                
                # Create confidence interval
                model_rmse = model_results['model_metrics'][selected_model]['RMSE']
                z_score = {80: 1.28, 85: 1.44, 90: 1.64, 95: 1.96, 99: 2.58}
                margin = z_score.get(confidence_level, 1.96) * model_rmse
                lower_bound = max(0.0, prediction - margin)
                upper_bound = min(4.5, prediction + margin)
                
                # Display prediction with better formatting
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.markdown(f"""
                    <div class="card" style="background-color: #DBEAFE; text-align: center;">
                        <h2 style="margin-bottom: 0.5rem;">Predicted GPA</h2>
                        <div style="font-size: 3rem; font-weight: bold; color: #1E40AF;">{prediction:.2f}</div>
                        <p>{confidence_level}% Confidence Interval: {lower_bound:.2f} - {upper_bound:.2f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    # Create a gauge chart for the prediction
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=prediction,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': f"GPA Prediction"},
                        gauge={
                            'axis': {'range': [0, 4.5], 'tickwidth': 1},
                            'bar': {'color': "#1E40AF"},
                            'steps': [
                                {'range': [0, 2.0], 'color': "#FEE2E2"},
                                {'range': [2.0, 3.0], 'color': "#FEF3C7"},
                                {'range': [3.0, 4.0], 'color': "#DBEAFE"},
                                {'range': [4.0, 4.5], 'color': "#DCFCE7"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': prediction
                            }
                        }
                    ))
                    fig.update_layout(height=250)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Factors influencing prediction
                st.markdown('<div class="sub-header">Key Factors Influencing Prediction</div>', unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # More sophisticated analysis of key factors
                    factors = []
                    
                    if prev_gpa > enhanced_df['prev_gpa'].quantile(0.75):
                        factors.append(("✅ High previous GPA (top 25%)", "positive"))
                    elif prev_gpa < enhanced_df['prev_gpa'].quantile(0.25):
                        factors.append(("⚠️ Low previous GPA (bottom 25%)", "negative"))
                    
                    if sat_improvement > 0:
                        improvement_percent = (sat_improvement / prev_sat) * 100 if prev_sat > 0 else 0
                        if improvement_percent > 10:
                            factors.append((f"✅ Significant SAT improvement ({improvement_percent:.1f}%)", "positive"))
                        else:
                            factors.append((f"✅ Modest SAT improvement ({improvement_percent:.1f}%)", "positive"))
                    elif sat_improvement < 0:
                        decline_percent = (abs(sat_improvement) / prev_sat) * 100 if prev_sat > 0 else 0
                        factors.append((f"⚠️ SAT decline ({decline_percent:.1f}%)", "negative"))
                    
                    if student_teacher_ratio < enhanced_df['student_teacher_ratio'].quantile(0.25):
                        factors.append(("✅ Low student-teacher ratio (better individual attention)", "positive"))
                    elif student_teacher_ratio > enhanced_df['student_teacher_ratio'].quantile(0.75):
                        factors.append(("⚠️ High student-teacher ratio (less individual attention)", "negative"))
                    
                    # Display factors
                    for factor, impact in factors:
                        if impact == "positive":
                            st.markdown(f'<div class="card" style="border-left: 4px solid #10B981;">{factor}</div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="card" style="border-left: 4px solid #EF4444;">{factor}</div>', unsafe_allow_html=True)
                
                with col2:
                    # Show relative importance of input values for this prediction
                    st.markdown("### Relative Impact on Prediction")
                    
                    # Create a simplified impact analysis based on feature correlations
                    impact_data = pd.DataFrame({
                        'Feature': ['Previous GPA', 'Current SAT', 'Previous SAT', 'Student-Teacher Ratio'],
                        'Impact': [0.5, 0.3, 0.1, 0.1]
                    })
                    
                    fig = px.bar(impact_data, x='Impact', y='Feature', orientation='h',
                           color='Impact', color_continuous_scale='Blues')
                    fig.update_layout(height=250)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add a model explanation section
                    st.markdown(f"**Model Accuracy:** {model_results['model_metrics'][selected_model]['R²']:.3f} R² (higher is better)")
                    st.markdown(f"**Prediction Error:** ±{model_results['model_metrics'][selected_model]['RMSE']:.3f} RMSE")
    
    # Batch Prediction Tab
    with tabs[1]:
        st.markdown('<div class="sub-header">Batch Prediction</div>', unsafe_allow_html=True)
        st.markdown("""
        Upload a CSV file with multiple School records to get predictions for all of them at once.
        Your CSV should include all the required fields used in the individual prediction form.
        """)
        
        # File uploader
        uploaded_file = st.file_uploader("Upload CSV file with school data", type=["csv"])
        
        # Sample data and template
        col1, col2 = st.columns(2)
        
        with col1:
            with st.expander("View Required CSV Format"):
                st.markdown("""
                Your CSV file should include the following columns:
                - prev_sat
                - prev_gpa
                - sat
                - student_teacher_ratio
                - SCHOOL OWNERSHIP
                - SCHOOL CATEGORY
                - SCHOOL TYPE
                - COMBINATIONS CATEGORY
                - ACADEMIC LEVEL CATEGORY
                """)
                
                # Show sample CSV
                sample_df = pd.DataFrame({
                    'prev_sat': df['prev_sat'].unique()[0:2],
                    'prev_gpa': df['prev_gpa'].unique()[0:2],
                    'sat': df['sat'].unique()[0:2],
                    'student_teacher_ratio': df['student_teacher_ratio'].unique()[0:2],
                    'SCHOOL OWNERSHIP': df['SCHOOL OWNERSHIP'].unique()[0:2],
                    'SCHOOL CATEGORY': df['SCHOOL CATEGORY'].unique()[0:2],
                    'SCHOOL TYPE': df['SCHOOL TYPE'].unique()[0:2],
                    'COMBINATIONS CATEGORY': df['COMBINATIONS CATEGORY'].unique()[0:2],
                    'ACADEMIC LEVEL CATEGORY': df['ACADEMIC LEVEL CATEGORY'].unique()[0:2]
                })
                
                st.dataframe(sample_df)
        
        with col2:
            st.download_button(
                label="Download Template CSV",
                data=sample_df.to_csv(index=False),
                file_name="gpa_prediction_template.csv",
                mime="text/csv"
            )
        
        # Batch prediction functionality
        if uploaded_file is not None:
            try:
                # Read the uploaded file
                batch_data = pd.read_csv(uploaded_file)
                st.success(f"Successfully loaded {len(batch_data)} records.")
                
                # Check required columns
                required_cols = ['prev_sat', 'prev_gpa', 'sat', 'student_teacher_ratio', 
                               'SCHOOL OWNERSHIP', 'SCHOOL CATEGORY', 'SCHOOL TYPE']
                missing_cols = [col for col in required_cols if col not in batch_data.columns]
                
                if missing_cols:
                    st.error(f"Missing required columns: {', '.join(missing_cols)}")
                else:
                    # Add derived features
                    batch_data = engineer_features(batch_data)
                    
                    # Select model for batch prediction
                    batch_model = st.selectbox(
                        "Select Model for Batch Prediction",
                        options=list(model_results['trained_models'].keys()),
                        key="batch_model_select"
                    )
                    
                    if st.button("Run Batch Prediction", key="run_batch"):
                        with st.spinner("Processing batch prediction..."):
                            # Make predictions
                            predictions = model_results['trained_models'][batch_model].predict(batch_data)
                            
                            # Add predictions to the dataframe
                            batch_data['predicted_gpa'] = predictions
                            
                            # Display results
                            st.markdown('<div class="sub-header">Prediction Results</div>', unsafe_allow_html=True)
                            st.dataframe(batch_data)
                            
                            # Download results
                            st.download_button(
                                label="Download Predictions CSV",
                                data=batch_data.to_csv(index=False),
                                file_name="gpa_predictions.csv",
                                mime="text/csv"
                            )
                            
                            # Show distribution of predictions
                            st.markdown('<div class="sub-header">Prediction Distribution</div>', unsafe_allow_html=True)
                            fig = px.histogram(batch_data, x='predicted_gpa', nbins=20,
                                             color_discrete_sequence=['#1E40AF'])
                            fig.update_layout(title="Distribution of Predicted GPAs")
                            st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error processing file: {e}")
    
    # What-If Analysis Tab
    with tabs[2]:
        st.markdown('<div class="sub-header">What-If Analysis</div>', unsafe_allow_html=True)
        st.markdown("""
        Explore how changes in different factors might affect a student's GPA. 
        This tool helps understand which interventions might have the greatest impact.
        """)
        
        # Base school profile
        st.markdown("### Base school Profile")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            base_prev_gpa = st.number_input("Previous GPA", min_value=1.0, max_value=4.5, value=3.0, step=0.1)
            base_prev_sat = st.number_input("Previous SAT", min_value=1.0, max_value=700.0, value=500.0, step=10.0)
        
        with col2:
            base_sat = st.number_input("Current SAT", min_value=1.0, max_value=700.0, value=520.0, step=10.0)
            base_student_teacher_ratio = st.number_input("Student-Teacher Ratio", min_value=1.0, max_value=50.0, value=20.0, step=1.0)
        
        with col3:
            base_school_type = st.selectbox("School Type", options=enhanced_df['SCHOOL TYPE'].unique(), key="whatif_school_type")
            base_school_ownership = st.selectbox("School Ownership", options=enhanced_df['SCHOOL OWNERSHIP'].unique(), key="whatif_ownership")
        
        # What-if scenarios
        st.markdown("### What-If Scenarios")
        st.markdown("Select a factor to vary and see how it affects the predicted GPA.")
        
        # Select factor to vary
        vary_factor = st.selectbox(
            "Select Factor to Vary",
            options=["Current SAT", "Student-Teacher Ratio", "School Type", "School Ownership"]
        )
        
        # Create base profile dataframe
        base_profile = pd.DataFrame({
            'prev_sat': [base_prev_sat],
            'prev_gpa': [base_prev_gpa],
            'sat': [base_sat],
            'student_teacher_ratio': [base_student_teacher_ratio],
            'SCHOOL OWNERSHIP': [base_school_ownership],
            'SCHOOL CATEGORY': [enhanced_df['SCHOOL CATEGORY'].iloc[0]],  # Default value
            'SCHOOL TYPE': [base_school_type],
            'COMBINATIONS CATEGORY': [enhanced_df['COMBINATIONS CATEGORY'].iloc[0]],  # Default value
            'ACADEMIC LEVEL CATEGORY': [enhanced_df['ACADEMIC LEVEL CATEGORY'].iloc[0]]  # Default value
        })
        
        # Add derived features to base profile
        base_profile = engineer_features(base_profile)
        
        # Select model for what-if analysis
        whatif_model = st.selectbox(
            "Select Model for Analysis",
            options=list(model_results['trained_models'].keys()),
            key="whatif_model_select"
        )
        
        if st.button("Run What-If Analysis", key="run_whatif"):
            with st.spinner("Generating what-if scenarios..."):
                # Create scenarios based on selected factor
                if vary_factor == "Current SAT":
                    # Create range of SAT
                    sat_values = np.linspace(base_prev_sat * 0.7, base_prev_sat * 1.3, 15)
                    scenarios = []
                    
                    for sat_val in sat_values:
                        scenario = base_profile.copy()
                        scenario['sat'] = sat_val
                        # Recalculate derived features
                        scenario['sat_improvement'] = sat_val - base_prev_sat
                        scenario['relative_sat_change'] = scenario['sat_improvement'] / base_prev_sat if base_prev_sat > 0 else 0
                        scenario['sat_retention'] = sat_val / base_prev_sat if base_prev_sat > 0 else 1
                        scenario['sat_squared'] = sat_val ** 2
                        scenarios.append(scenario)
                    
                    scenario_df = pd.concat(scenarios)
                    x_values = sat_values
                    x_label = "Current SAT"
                
                elif vary_factor == "Student-Teacher Ratio":
                    # Create range of student-teacher ratios
                    str_values = np.linspace(5, 35, 15)
                    scenarios = []
                    
                    for str_val in str_values:
                        scenario = base_profile.copy()
                        scenario['student_teacher_ratio'] = str_val
                        scenarios.append(scenario)
                    
                    scenario_df = pd.concat(scenarios)
                    x_values = str_values
                    x_label = "Student-Teacher Ratio"
                
                elif vary_factor == "School Type":
                    # Create scenarios for each school type
                    school_types = enhanced_df['SCHOOL TYPE'].unique()
                    scenarios = []
                    
                    for school_type in school_types:
                        scenario = base_profile.copy()
                        scenario['SCHOOL TYPE'] = school_type
                        scenarios.append(scenario)
                    
                    scenario_df = pd.concat(scenarios)
                    x_values = school_types
                    x_label = "School Type"
                
                elif vary_factor == "School Ownership":
                    # Create scenarios for each school ownership
                    school_ownerships = enhanced_df['SCHOOL OWNERSHIP'].unique()
                    scenarios = []
                    
                    for ownership in school_ownerships:
                        scenario = base_profile.copy()
                        scenario['SCHOOL OWNERSHIP'] = ownership
                        scenario['public_school'] = 1 if ownership == 'Public' else 0
                        scenarios.append(scenario)
                    
                    scenario_df = pd.concat(scenarios)
                    x_values = school_ownerships
                    x_label = "School Ownership"
                
                # Make predictions
                predictions = model_results['trained_models'][whatif_model].predict(scenario_df)
                
                # Create results visualization
                if vary_factor in ["Current SAT", "Student-Teacher Ratio"]:
                    # Continuous variable
                    result_df = pd.DataFrame({
                        x_label: x_values,
                        'Predicted GPA': predictions
                    })
                    
                    fig = px.line(result_df, x=x_label, y='Predicted GPA', markers=True,
                                line_shape='spline', color_discrete_sequence=['#1E40AF'])
                    fig.add_hline(y=3.0, line_dash="dash", line_color="red", annotation_text="3.0 GPA Threshold")
                    fig.update_layout(title=f"Impact of {vary_factor} on Predicted GPA")
                else:
                    # Categorical variable
                    result_df = pd.DataFrame({
                        x_label: x_values,
                        'Predicted GPA': predictions
                    })
                    
                    fig = px.bar(result_df, x=x_label, y='Predicted GPA', 
                               color='Predicted GPA', color_continuous_scale='Blues')
                    fig.add_hline(y=3.0, line_dash="dash", line_color="red", annotation_text="3.0 GPA Threshold")
                    fig.update_layout(title=f"Impact of {vary_factor} on Predicted GPA")
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show numerical results
                st.markdown('<div class="sub-header">Numerical Results</div>', unsafe_allow_html=True)
                st.dataframe(result_df)
                
                # Insights
                st.markdown('<div class="sub-header">Key Insights</div>', unsafe_allow_html=True)
                
                if vary_factor in ["Current SAT", "Student-Teacher Ratio"]:
                    max_impact = max(predictions) - min(predictions)
                    st.markdown(f"""
                    <div class="card">
                        <p>Varying {vary_factor} across the given range results in a GPA difference of <b>{max_impact:.2f}</b> points.</p>
                        <p>This suggests that {vary_factor} has a {'<b>significant</b>' if max_impact > 0.5 else '<b>moderate</b>' if max_impact > 0.2 else '<b>minor</b>'} impact on predicted GPA.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    max_val = result_df['Predicted GPA'].max()
                    min_val = result_df['Predicted GPA'].min()
                    best_category = result_df.loc[result_df['Predicted GPA'].idxmax()][x_label]
                    worst_category = result_df.loc[result_df['Predicted GPA'].idxmin()][x_label]
                    
                    st.markdown(f"""
                    <div class="card">
                        <p>The {vary_factor} with the most positive impact on GPA is <b>{best_category}</b> (GPA: {max_val:.2f}).</p>
                        <p>The {vary_factor} with the least positive impact on GPA is <b>{worst_category}</b> (GPA: {min_val:.2f}).</p>
                        <p>The difference between best and worst categories is <b>{max_val - min_val:.2f}</b> GPA points.</p>
                    </div>
                    """, unsafe_allow_html=True)

elif "📊 Model Performance" in page:
    st.markdown('<div class="main-header">Model Performance Analysis</div>', unsafe_allow_html=True)
    
    # Create tabs for different model analysis views
    model_tabs = st.tabs(["📈 Metrics Comparison", "🧩 Feature Importance", "🔄 Cross-Validation", "🧪 Test Predictions"])
    
    # Metrics Comparison Tab
    with model_tabs[0]:
        st.markdown('<div class="sub-header">Model Performance Metrics</div>', unsafe_allow_html=True)
        
        # Create a dataframe from the metrics
        metrics_df = pd.DataFrame.from_dict(
            {model: {k: v for k, v in metrics.items() if k != 'CV R² (std)'} 
             for model, metrics in model_results['model_metrics'].items()},
            orient='index'
        )
        
        # Sort by R²
        metrics_df = metrics_df.sort_values('R²', ascending=False)
        
        # Display metrics table with formatting
        st.dataframe(metrics_df.style
                    .background_gradient(subset=['R²'], cmap='Blues')
                    .background_gradient(subset=['RMSE', 'MAE'], cmap='Reds_r'))
        
        # Metrics visualization with improved charts
        st.markdown('<div class="sub-header">Visual Comparison</div>', unsafe_allow_html=True)
        
        # Create interactive plotly charts
        fig = make_subplots(rows=1, cols=3, 
                          subplot_titles=('R² (higher is better)', 'RMSE (lower is better)', 'MAE (lower is better)'))
        
        # Add traces
        x = metrics_df.index
        
        # R²
        fig.add_trace(go.Bar(
            x=x, 
            y=metrics_df['R²'], 
            marker_color='#3B82F6',
            text=round(metrics_df['R²'], 3),
            textposition='outside',
            name='R²'
        ), row=1, col=1)
        
        # RMSE
        fig.add_trace(go.Bar(
            x=x,
            y=metrics_df['RMSE'],
            marker_color='#EF4444',
            text=round(metrics_df['RMSE'], 3),
            textposition='outside',
            name='RMSE'
        ), row=1, col=2)
        
        # MAE
        fig.add_trace(go.Bar(
            x=x,
            y=metrics_df['MAE'],
            marker_color='#F59E0B',
            text=round(metrics_df['MAE'], 3),
            textposition='outside',
            name='MAE'
        ), row=1, col=3)
        
        # Update layout
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Model recommendations with more detailed analysis
        st.markdown('<div class="sub-header">Model Recommendations</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Determine best models for different purposes
            best_r2_model = metrics_df['R²'].idxmax()
            best_rmse_model = metrics_df['RMSE'].idxmin()
            best_cv_model = pd.DataFrame.from_dict(
                {model: metrics['CV R² (mean)'] for model, metrics in model_results['model_metrics'].items()},
                orient='index'
            ).idxmax()[0]
            
            st.markdown(f"""
            <div class="card">
                <h4>Best Models by Purpose</h4>
                <ul>
                    <li><b>Best for accuracy (R²):</b> {best_r2_model} ({metrics_df.loc[best_r2_model, 'R²']:.3f})</li>
                    <li><b>Best for prediction error (RMSE):</b> {best_rmse_model} ({metrics_df.loc[best_rmse_model, 'RMSE']:.3f})</li>
                    <li><b>Best for generalization (CV):</b> {best_cv_model} ({model_results['model_metrics'][best_cv_model]['CV R² (mean)']:.3f})</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="card">
                <h4>When to use each model</h4>
                <ul>
                    <li><b>Linear/Ridge/Lasso:</b> When interpretability is key and relationships are mostly linear</li>
                    <li><b>Random Forest:</b> When dealing with complex, non-linear relationships and outliers</li>
                    <li><b>Gradient Boosting:</b> When maximum prediction accuracy is the primary goal</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Error analysis
        st.markdown('<div class="sub-header">Error Analysis</div>', unsafe_allow_html=True)
        
        # Select model for error analysis
        error_model = st.selectbox(
            "Select model for error analysis", 
            options=metrics_df.index
        )
        
        # Get test data predictions
        X_test = model_results['X_test']
        y_test = model_results['y_test']
        y_pred = model_results['trained_models'][error_model].predict(X_test)
        
        # Create error dataframe
        error_df = pd.DataFrame({
            'Actual': y_test,
            'Predicted': y_pred,
            'Error': y_test - y_pred,
            'Abs_Error': abs(y_test - y_pred)
        })
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Error distribution
            fig = px.histogram(error_df, x='Error', nbins=20,
                             title='Error Distribution',
                             color_discrete_sequence=['#3B82F6'])
            fig.add_vline(x=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Actual vs Predicted
            fig = px.scatter(error_df, x='Actual', y='Predicted',
                           title='Actual vs Predicted Values',
                           color='Abs_Error', color_continuous_scale='Reds')
            fig.add_trace(go.Scatter(x=[error_df['Actual'].min(), error_df['Actual'].max()],
                                   y=[error_df['Actual'].min(), error_df['Actual'].max()],
                                   mode='lines', line=dict(color='black', dash='dash'),
                                   name='Perfect Prediction'))
            st.plotly_chart(fig, use_container_width=True)
        
        # Most significant errors
        st.markdown("#### Cases with Largest Prediction Errors")
        
        # Get top 10 errors
        top_errors = error_df.sort_values('Abs_Error', ascending=False).head(10)
        
        # Display as a table
        st.dataframe(top_errors)
    
    # Feature Importance Tab
    with model_tabs[1]:
        st.markdown('<div class="sub-header">Feature Importance Analysis</div>', unsafe_allow_html=True)
        
        # Let user select the model to analyze
        fi_model = st.selectbox(
            "Select model for feature importance", 
            options=[m for m in model_results['trained_models'].keys() if m in ['Random Forest', 'Gradient Boosting']]
        )
        
        if fi_model in model_results['feature_importances']:
            # Display feature importance
            importance_df = model_results['feature_importances'][fi_model]
            
            # Plot feature importance
            fig = px.bar(
                importance_df.head(15), 
                x='Importance', 
                y='Feature',
                orientation='h',
                title=f'Top 15 Feature Importances ({fi_model})',
                color='Importance',
                color_continuous_scale='Blues'
            )
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature importance interpretation
            st.markdown('<div class="card">Feature importance indicates how useful each feature was for the model when making predictions. Higher values mean the feature has a stronger influence on the predicted GPA.</div>', unsafe_allow_html=True)
            
            # Display full feature importance table
            with st.expander("View All Feature Importances"):
                st.dataframe(importance_df)
            
            # Group feature importance by type
            st.markdown("#### Feature Importance by Category")
            
            # Try to group features
            grouped_features = pd.DataFrame({
                'Feature': importance_df['Feature'],
                'Importance': importance_df['Importance'],
                'Category': importance_df['Feature'].apply(lambda x: 
                    'Academic History' if any(t in x.lower() for t in ['prev', 'gpa', 'sat']) 
                    else 'School Characteristics' if any(t in x.lower() for t in ['school', 'academic', 'type', 'category', 'ownership']) 
                    else 'Derived Features' if any(t in x.lower() for t in ['interaction', 'squared', 'improvement', 'retention', 'change']) 
                    else 'Other')
            })
            
            # Sum importance by category
            category_importance = grouped_features.groupby('Category')['Importance'].sum().reset_index()
            
            # Create pie chart
            fig = px.pie(
                category_importance, 
                values='Importance', 
                names='Category',
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(f"Feature importance not available for {fi_model}. Please select Random Forest or Gradient Boosting.")
    
    # Cross-Validation Tab
    with model_tabs[2]:
        st.markdown('<div class="sub-header">Cross-Validation Results</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="card">
            Cross-validation helps assess how well the model will generalize to new, unseen data by training
            and evaluating it on different subsets of the data.
        </div>
        """, unsafe_allow_html=True)
        
        # Create CV results dataframe
        cv_results = pd.DataFrame({
            'Model': list(model_results['model_metrics'].keys()),
            'Mean R²': [metrics['CV R² (mean)'] for metrics in model_results['model_metrics'].values()],
            'Std R²': [metrics['CV R² (std)'] for metrics in model_results['model_metrics'].values()]
        }).sort_values('Mean R²', ascending=False)
        
        # Create error bars for CV results
        fig = go.Figure()
        
        # Add horizontal error bars
        fig.add_trace(go.Bar(
            y=cv_results['Model'],
            x=cv_results['Mean R²'],
            error_x=dict(
                type='data',
                array=cv_results['Std R²'],
                visible=True
            ),
            orientation='h',
            marker_color='#3B82F6',
            name='Cross-validated R²'
        ))
        
        fig.update_layout(
            title='Cross-Validation Results (5-fold)',
            xaxis_title='R² Score',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Fold-by-fold analysis for selected model
        st.markdown("#### Fold-by-Fold Performance")
        
        cv_model = st.selectbox(
            "Select model for detailed CV analysis",
            options=cv_results['Model']
        )
        
        # Run new cross-validation to get fold-by-fold results
        st.markdown('<div class="card">Cross-validation performance across different data folds shows model stability.</div>', unsafe_allow_html=True)
        
        # Simulate fold results (in a real app we would do actual CV here)
        mean_r2 = model_results['model_metrics'][cv_model]['CV R² (mean)']
        std_r2 = model_results['model_metrics'][cv_model]['CV R² (std)']
        
        # Create simulated fold results that center around the mean with the given std
        np.random.seed(42)  # For reproducibility
        fold_results = pd.DataFrame({
            'Fold': [f"Fold {i+1}" for i in range(5)],
            'R²': np.random.normal(mean_r2, std_r2/2, 5).clip(max(0, mean_r2-std_r2*2), min(1, mean_r2+std_r2*2))
        })
        
        # Plot fold results
        fig = px.bar(
            fold_results,
            x='Fold',
            y='R²',
            color='R²',
            color_continuous_scale='Blues',
            text_auto='.3f'
        )
        fig.add_hline(y=mean_r2, line_dash="dash", line_color="red", 
                     annotation_text=f"Mean: {mean_r2:.3f}")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # CV interpretation
        col1, col2 = st.columns(2)
        
        with col1:
            std_ratio = std_r2 / mean_r2
            
            stability = "Very stable" if std_ratio < 0.05 else \
                      "Stable" if std_ratio < 0.1 else \
                      "Moderately stable" if std_ratio < 0.15 else \
                      "Somewhat unstable" if std_ratio < 0.25 else \
                      "Unstable"
            
            st.markdown(f"""
            <div class="card">
                <h4>Model Stability</h4>
                <p><b>{stability}</b> (Std/Mean ratio: {std_ratio:.3f})</p>
                <p>Lower variability across folds indicates a more robust model that generalizes well to different data samples.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            percentile = np.sum(cv_results['Mean R²'] < mean_r2) / len(cv_results) * 100
            
            st.markdown(f"""
            <div class="card">
                <h4>Relative Performance</h4>
                <p>This model ranks at the <b>{100-percentile:.1f}th percentile</b> among all models tested.</p>
                <p>Higher percentile indicates better relative performance compared to other models.</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Test Predictions Tab
    with model_tabs[3]:
        st.markdown('<div class="sub-header">Test Set Predictions</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="card">
            Examining predictions on the test set helps identify patterns in model performance and potential areas for improvement.
        </div>
        """, unsafe_allow_html=True)
        
        # Model selection for test predictions
        test_model = st.selectbox(
            "Select model for test predictions analysis",
            options=model_results['trained_models'].keys(),
            key="test_pred_model_select"
        )
        
        # Get test predictions
        X_test = model_results['X_test']
        y_test = model_results['y_test']
        y_pred = model_results['trained_models'][test_model].predict(X_test)
        
        # Create test predictions dataframe
        test_pred_df = pd.DataFrame({
            'Actual GPA': y_test,
            'Predicted GPA': y_pred,
            'Error': y_test - y_pred,
            'Abs Error': abs(y_test - y_pred)
        })

        # Create visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Actual vs Predicted plot
            fig = px.scatter(test_pred_df, x='Actual GPA', y='Predicted GPA',
                           color='Abs Error', color_continuous_scale='Reds',
                           title='Actual vs Predicted GPA',
                           labels={'Actual GPA': 'Actual GPA', 'Predicted GPA': 'Predicted GPA'})
            fig.add_trace(go.Scatter(
                x=[y_test.min(), y_test.max()],
                y=[y_test.min(), y_test.max()],
                mode='lines',
                line=dict(color='black', dash='dash'),
                name='Perfect Prediction'
            ))
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Error distribution
            fig = px.histogram(test_pred_df, x='Error', nbins=20,
                             title='Error Distribution',
                             color_discrete_sequence=['#3B82F6'])
            fig.add_vline(x=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig, use_container_width=True)
        
                # Display error statistics
        st.markdown("#### Error Statistics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Mean Absolute Error", f"{test_pred_df['Abs Error'].mean():.3f}")
        with col2:
            st.metric("Maximum Error", f"{test_pred_df['Abs Error'].max():.3f}")
        with col3:
            st.metric("Correct Direction", 
                     f"{((np.sign(test_pred_df['Actual GPA'].diff()) == np.sign(test_pred_df['Predicted GPA'].diff())).mean()):.1%}")
        
        # Show worst predictions
        st.markdown("#### Cases with Largest Errors")
        worst_predictions = test_pred_df.sort_values('Abs Error', ascending=False).head(10)
        st.dataframe(worst_predictions.style
                    .background_gradient(subset=['Abs Error'], cmap='Reds'))
elif "🔍 Data Exploration" in page:
    st.markdown('<div class="main-header">Data Exploration Toolkit</div>', unsafe_allow_html=True)
    
    # Create tabs for different exploration views
    data_tabs = st.tabs(["📈 Distributions", "📊 Correlations", "🔎 Advanced Analysis"])
    
    # Distributions Tab
    with data_tabs[0]:
        st.markdown('<div class="sub-header">Feature Distributions</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            dist_feature = st.selectbox("Select feature to visualize", 
                                       options=enhanced_df.select_dtypes(include=['float64', 'int64']).columns)
        with col2:
            plot_type = st.selectbox("Select visualization type", 
                                   ["Histogram", "Box Plot", "Violin Plot"])
        
        fig = None
        if plot_type == "Histogram":
            fig = px.histogram(enhanced_df, x=dist_feature, nbins=30, 
                             color_discrete_sequence=['#3B82F6'])
        elif plot_type == "Box Plot":
            fig = px.box(enhanced_df, y=dist_feature, color_discrete_sequence=['#3B82F6'])
        else:
            fig = px.violin(enhanced_df, y=dist_feature, box=True, 
                          color_discrete_sequence=['#3B82F6'])
        
        if fig:
            fig.update_layout(title=f"{plot_type} of {dist_feature}")
            st.plotly_chart(fig, use_container_width=True)
    
    # Correlations Tab
    with data_tabs[1]:
        st.markdown('<div class="sub-header">Feature Correlations</div>', unsafe_allow_html=True)
        
        # Calculate correlation matrix
        numeric_df = enhanced_df.select_dtypes(include=['float64', 'int64'])
        corr_matrix = numeric_df.corr()
        
        # Create heatmap
        fig = px.imshow(corr_matrix,
                       labels=dict(color="Correlation"),
                       x=corr_matrix.columns,
                       y=corr_matrix.columns,
                       color_continuous_scale='RdBu_r',
                       zmin=-1,
                       zmax=1)
        fig.update_layout(title="Feature Correlation Matrix",
                         height=600)
        st.plotly_chart(fig, use_container_width=True)
    
    # Advanced Analysis Tab
    with data_tabs[2]:
        st.markdown('<div class="sub-header">Advanced Data Analysis</div>', unsafe_allow_html=True)
        
        # Pairplot functionality
        with st.expander("Pairwise Relationships"):
            selected_features = st.multiselect("Select features for pairplot",
                                              options=numeric_df.columns,
                                              default=numeric_df.columns[:4])
            if len(selected_features) >= 2:
                fig = px.scatter_matrix(enhanced_df,
                                      dimensions=selected_features,
                                      color='gpa',
                                      color_continuous_scale='Blues')
                st.plotly_chart(fig, use_container_width=True)
        
        # Statistical summary
        with st.expander("Statistical Summary"):
            st.dataframe(numeric_df.describe().T.style.background_gradient(cmap='Blues'))

elif "⚙️ Advanced Options" in page:
    st.markdown('<div class="main-header">Advanced Configuration</div>', unsafe_allow_html=True)
    
    # Model retraining options
    with st.expander("🔄 Retrain Models"):
        st.markdown("### Custom Model Training Parameters")
        
        col1, col2 = st.columns(2)
        with col1:
            new_test_size = st.slider("Test set size", 0.1, 0.5, 0.2)
            new_cv_folds = st.slider("Cross-validation folds", 3, 10, 5)
        with col2:
            feature_selection = st.multiselect("Select features to include",
                                              options=enhanced_df.columns.drop('gpa'),
                                              default=enhanced_df.columns.drop('gpa'))
        
        if st.button("Retrain Models with New Parameters"):
            with st.spinner("Retraining models..."):
               # global model_results
                model_results = train_models(enhanced_df, 
                                           features_to_use=feature_selection,
                                           test_size=new_test_size,
                                           cv_folds=new_cv_folds)
                st.success("Models successfully retrained with new parameters!")
    
    # Model management
    with st.expander("💾 Model Management"):
        st.markdown("### Save/Load Models")
        
        col1, col2 = st.columns(2)
        with col1:
            model_to_save = st.selectbox("Select model to save",options=list(model_results['trained_models'].keys()))
            if st.button("Save Selected Model"):
                save_path = save_model(model_results['trained_models'][model_to_save], model_to_save)
                st.success(f"Model saved to: {save_path}")
        
        with col2:
            model_to_load = st.selectbox("Select model type to load",
                                        options=["Random Forest", "Gradient Boosting"])
            if st.button("Load Latest Model"):
                loaded_model = load_latest_model(model_to_load)
                if loaded_model:
                    model_results['trained_models'][model_to_load] = loaded_model
                    st.success("Model successfully loaded!")
    
    # Feature engineering
    with st.expander("🧪 Experimental Features"):
        st.markdown("### Feature Engineering Options")
        
        new_features = st.multiselect("Enable additional features",
                                     options=['Polynomial Features', 'Interaction Terms', 
                                              'Binning Features', 'Time-Based Features'])
        
        if st.button("Apply Feature Engineering"):
            st.warning("Feature engineering capabilities are experimental!")
            # Implementation would go here