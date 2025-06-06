import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Set page configuration
st.set_page_config(
    page_title="GPA Prediction Dashboard",
    page_icon="🎓",
    layout="wide"
)

df = pd.read_csv('data/df.csv').dropna(axis=0)

# Title and description
st.title("🎓 GPA Prediction Dashboard")
st.markdown("""
This dashboard predicts student GPA based on various factors including previous academic performance,
school characteristics, and student-teacher ratio.
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Predict GPA", "Model Performance", "Data Exploration"])

# Load or train models
@st.cache_resource
def load_or_train_models():
    # Define feature groups
    numeric_features = ['prev_sat', 'prev_gpa', 'sat', 'student_teacher_ratio',
                        'prev_gpa_sat_interaction', 'sat_retention']
    categorical_features = ['SCHOOL OWNERSHIP', 'SCHOOL CATEGORY', 'SCHOOL TYPE',
                            'COMBINATIONS CATEGORY', 'ACADEMIC LEVEL CATEGORY']
    
    # Create preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Create models
    models = {
        'Linear Regression': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', LinearRegression())
        ]),
        'Random Forest': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', RandomForestRegressor(n_estimators=100, random_state=42))
        ]),
        'Gradient Boosting': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', GradientBoostingRegressor(n_estimators=100, random_state=42))
        ])
    }
        
    
    dummy_data = df[['prev_sat','prev_gpa','sat','student_teacher_ratio','SCHOOL OWNERSHIP',
                    'SCHOOL CATEGORY', 'SCHOOL TYPE','COMBINATIONS CATEGORY', 'ACADEMIC LEVEL CATEGORY','gpa']]
    # Create interaction terms
    dummy_data['prev_gpa_sat_interaction'] = dummy_data['prev_gpa'] * dummy_data['prev_sat']
    dummy_data['sat_retention'] = dummy_data['sat'] / dummy_data['prev_sat']
    
    # Split the data
    X = dummy_data.drop('gpa', axis=1)
    y = dummy_data['gpa']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the models
    trained_models = {}
    model_metrics = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        trained_models[name] = model
        model_metrics[name] = {
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2
        }
    
    return trained_models, model_metrics, dummy_data, preprocessor, numeric_features, categorical_features

# Load or train models
models, metrics, sample_data, preprocessor, numeric_features, categorical_features = load_or_train_models()

# Home page
if page == "Home":
    st.header("Welcome to the GPA Prediction Dashboard")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("About This Dashboard")
        st.write("""
        This dashboard helps educators and administrators predict student GPA based on:
        
        - Previous academic performance (SAT scores, GPA)
        - School characteristics (ownership, category, type)
        - Student-teacher ratio
        - Academic level and combinations category
        
        Use the sidebar to navigate between different features of the dashboard.
        """)
        
        st.subheader("How It Works")
        st.write("""
        1. **Predict GPA**: Input student and school information to get a GPA prediction
        2. **Model Performance**: Compare the performance of different prediction models
        3. **Data Exploration**: Explore relationships between different factors and GPA
        """)
    
    with col2:
        st.subheader("Sample Data")
        st.dataframe(sample_data.head(5))
        
        st.subheader("Models Used")
        for model_name in models.keys():
            st.write(f"- {model_name}")

# Predict GPA page
elif page == "Predict GPA":
    st.header("Predict Student GPA")
    st.write("Enter student and school information to predict the expected GPA.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Academic Information")
        prev_sat = st.number_input("Previous Number of student sat for exam", min_value=1.0, max_value=676.0, value=300.0, step=1.0)
        prev_gpa = st.number_input("Previous GPA", min_value=1.29, max_value=4.57, value=3.0, step=0.01)
        sat = st.number_input("Number of student sat for exam", min_value=1.0, max_value=645.0, value=320.0, step=1.0)
        student_teacher_ratio = st.number_input("Student-Teacher Ratio", min_value=0.5, max_value=30.0, value=15.0, step=0.1)
        
    with col2:
        st.subheader("School Information")
        school_ownership = st.selectbox("School Ownership", df['SCHOOL OWNERSHIP'].unique())
        school_category = st.selectbox("School Category", df['SCHOOL CATEGORY'].unique())
        school_type = st.selectbox("School Type", df['SCHOOL TYPE'].unique())
        combinations_category = st.selectbox("Combinations Category", df['COMBINATIONS CATEGORY'].unique())
        academic_level = st.selectbox("Academic Level", df['ACADEMIC LEVEL CATEGORY'].unique())
        
    
    # Calculate derived features
    prev_gpa_sat_interaction = prev_gpa * prev_sat
    sat_retention = sat / prev_sat if prev_sat > 0 else 1.0
    
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
        'sat_retention': [sat_retention]
    })
    
    # Model selection
    selected_model = 'Linear Regression' #st.selectbox("Select Prediction Model", list(models.keys()))
    
    if st.button("Predict GPA"):
        # Make prediction
        prediction = models[selected_model].predict(input_data)[0]
        
        # Display prediction
        st.success(f"### Predicted GPA: {prediction:.2f}")
        
        # Show prediction confidence range (simplified approach)
        model_rmse = metrics[selected_model]['RMSE']
        st.info(f"Prediction confidence range: {max(1.0, prediction - model_rmse):.2f} to {min(4.5, prediction + model_rmse):.2f}")


# Model Performance page
elif page == "Model Performance":
    st.header("Model Performance Comparison")
    
    # Create a dataframe from the metrics
    metrics_df = pd.DataFrame.from_dict(metrics, orient='index')
    
    # Display metrics table
    st.subheader("Model Metrics")
    st.dataframe(metrics_df.style.highlight_min(axis=0, color='lightgreen'))
    
    # Plot metrics comparison
    st.subheader("Metrics Comparison")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # RMSE
    sns.barplot(x=metrics_df.index, y='RMSE', data=metrics_df, ax=axes[0])
    axes[0].set_title('RMSE (lower is better)')
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45)
    
    # MAE
    sns.barplot(x=metrics_df.index, y='MAE', data=metrics_df, ax=axes[1])
    axes[1].set_title('MAE (lower is better)')
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45)
    
    # R²
    sns.barplot(x=metrics_df.index, y='R²', data=metrics_df, ax=axes[2])
    axes[2].set_title('R² (higher is better)')
    axes[2].set_xticklabels(axes[2].get_xticklabels(), rotation=45)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Model recommendations
    st.subheader("Model Recommendations")
    
    best_r2_model = metrics_df['R²'].idxmax()
    best_rmse_model = metrics_df['RMSE'].idxmin()
    
    st.write(f"- **Best model for accuracy (R²)**: {best_r2_model}")
    st.write(f"- **Best model for prediction error (RMSE)**: {best_rmse_model}")
    
    st.write("""
    **When to use each model:**
    - **Linear Regression**: Good for understanding direct relationships between variables
    - **Random Forest**: Strong general-purpose model with good handling of non-linear relationships
    - **Gradient Boosting**: Often provides the best predictions but may be more complex
    """)

# Data Exploration page
elif page == "Data Exploration":
    st.header("Data Exploration")
    
    # Display sample data statistics
    st.subheader("Data Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Numeric Features Summary")
        st.dataframe(sample_data[numeric_features].describe())
    
    with col2:
        st.write("Categorical Features Distribution")
        selected_cat_feature = st.selectbox("Select categorical feature", categorical_features)
        
        # Plot distribution of selected categorical feature
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.countplot(y=selected_cat_feature, data=sample_data, ax=ax)
        st.pyplot(fig)
    
    # Correlation analysis
    st.subheader("Correlation Between Features and GPA")
    
    # Calculate correlations
    corr = sample_data[numeric_features + ['gpa']].corr()
    
    # Plot correlation with GPA
    fig, ax = plt.subplots(figsize=(10, 6))
    corr_with_gpa = corr['gpa'].drop('gpa').sort_values(ascending=False)
    sns.barplot(x=corr_with_gpa.values, y=corr_with_gpa.index, ax=ax)
    ax.set_title('Correlation with GPA')
    ax.set_xlabel('Correlation Coefficient')
    st.pyplot(fig)
    
    # Feature relationships
    st.subheader("Feature Relationships")
    
    # Feature selection for scatter plot
    x_feature = st.selectbox("Select X-axis feature", numeric_features, index=1)  # Default to prev_gpa
    
    # Create scatter plot
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = sns.scatterplot(x=x_feature, y='gpa', data=sample_data, alpha=0.6, ax=ax)
    ax.set_title(f'{x_feature} vs GPA')
    
    # Add trend line
    sns.regplot(x=x_feature, y='gpa', data=sample_data, scatter=False, ax=ax, color='red')
    
    st.pyplot(fig)
    
    # Feature importance (from Random Forest model)
    st.subheader("Feature Importance")
    
    # Get the feature names after preprocessing
    # This is a simplified approach - in a real app, you'd need to extract the feature names properly
    try:
        # Try to extract feature importance from Random Forest model
        rf_model = models['Random Forest'].named_steps['model']
        feature_importances = rf_model.feature_importances_
        
        # Create a DataFrame with feature importances
        # Note: This is simplified and may not correctly match feature names
        importance_df = pd.DataFrame({
            'Feature': numeric_features + [f'Encoded_{cat}' for cat in categorical_features],
            'Importance': feature_importances[:len(numeric_features) + len(categorical_features)]
        }).sort_values('Importance', ascending=False)
        
        # Plot feature importance
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax)
        ax.set_title('Feature Importance (Random Forest)')
        st.pyplot(fig)
    except:
        st.write("Feature importance visualization is not available")

# Add footer
st.sidebar.markdown("---")
st.sidebar.info("""
**About**  
This dashboard was created to help predict student GPA based on various factors.
""")