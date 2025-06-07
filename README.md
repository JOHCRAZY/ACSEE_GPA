# ACSEE GPA Prediction Dashboard

A professional Streamlit web application designed to predict Tanzania's Advanced Certificate of Secondary Education Examination (ACSEE) school GPA based on academic and institutional data. The dashboard supports both single and batch predictions using a pre-trained Linear Regression model with comprehensive data preprocessing and interactive visualizations.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Single Prediction](#single-prediction)
  - [Batch Prediction](#batch-prediction)
- [Model Performance](#model-performance)
- [File Structure](#file-structure)
- [Dependencies](#dependencies)
- [Data Requirements](#data-requirements)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

The ACSEE GPA Prediction Dashboard is a web-based tool built with Streamlit and Python, leveraging machine learning to predict Tanzania secondary school GPA based on previous academic performance, current examination data, and institutional characteristics. The application provides an intuitive interface for educators and administrators to assess school performance predictions with detailed data processing reports.

## ✨ Features

### Core Functionality
- **Single Prediction**: Input individual school data to predict GPA with detailed results display
- **Batch Prediction**: Upload CSV files to predict GPAs for multiple schools with comprehensive analysis

### Data Processing
- **Robust Data Preprocessing**: Handles missing values, validates numerical ranges, and standardizes categorical data
- **Comprehensive Validation**: Validates examination scores, student-teacher ratios, and categorical field consistency
- **Detailed Processing Reports**: Provides transparent data processing summaries including missing value handling and validation results

### Visualizations & Analytics
- **GPA Distribution Analysis**: Interactive histogram showing predicted GPA distributions
- **Categorical Analysis**: Box plots comparing GPA distributions across different school categories
- **File Preview**: Visual preview of uploaded data before processing
- **Summary Statistics**: Detailed statistical summaries of predictions and input data

### User Experience
- **Professional Interface**: Clean, modern design optimized for educational use
- **Comprehensive Error Handling**: Clear error messages and validation feedback
- **Export Functionality**: Download prediction results as CSV files
- **Interactive Data Tables**: Sortable and searchable result displays

## 🚀 Installation

### Prerequisites
- Python 3.7 or higher
- Git (for cloning the repository)

### Step-by-Step Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/JOHCRAZY/ACSEE_GPA.git
   cd ACSEE_GPA
   ```

2. **Set Up a Virtual Environment** (recommended)
   ```bash
   # Create virtual environment
   python -m venv venv
   
   # Activate virtual environment
   # On macOS/Linux:
   source venv/bin/activate
   
   # On Windows:
   venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify Required Files**
   - Ensure dataset exists: `data/df.csv`
   - Ensure model exists: `models/linear_regression_pipeline.pkl`

5. **Run the Application**
   ```bash
   streamlit run application.py
   ```

The application will open in your default web browser at `http://localhost:8501`.

## 📖 Usage

### Single Prediction

1. Navigate to the **"Single Prediction"** tab
2. Enter the required academic and institutional information:
   - **Previous SAT**: Number of candidates in previous examination
   - **Previous GPA**: School's previous GPA performance
   - **Current SAT**: Number of candidates in current examination
   - **School Details**: Select ownership type, category, type, combinations, and academic level
   - **Student-Teacher Ratio**: Current ratio (if available)
3. Click **"Generate GPA Prediction"** to view results
4. Review the predicted GPA with confidence metrics

### Batch Prediction

1. Navigate to the **"Batch Prediction"** tab
2. Upload a CSV file containing school data (see [Data Requirements](#data-requirements))
3. Configure processing options:
   - Enable strict validation for enhanced quality checks
   - Enable detailed reports for comprehensive analysis
4. Click **"Run Batch Prediction"**
5. Review results:
   - Interactive prediction table
   - GPA distribution histogram
   - Categorical analysis box plots
   - Download processed results

## 📊 Model Performance

The application uses a **Linear Regression** model selected for optimal performance on ACSEE data:

| Metric | Value | Description |
|--------|-------|-------------|
| **Test RMSE** | 0.2750 | Root Mean Square Error on test set |
| **Test MAE** | 0.1939 | Mean Absolute Error on test set |
| **Test R²** | 0.6739 | Coefficient of determination (67.39% variance explained) |
| **CV RMSE** | 0.2990 | Cross-validation RMSE |

### Model Comparison
The Linear Regression model was selected over Random Forest (R²: 0.4323) and Gradient Boosting (R²: 0.4387) based on superior performance metrics and model interpretability for educational predictions.

## 📁 File Structure

```
ACSEE_GPA/
├── application.py                                    # Main Streamlit application
├── data/
│   └── df.csv                                        # ACSEE dataset (required)
├── models/
│   ├── linear_regression_pipeline.pkl                # Primary model (required)
│   ├── random_forest_pipeline.pkl                    # Alternative model
│   └── gradient_boosting_pipeline.pkl                # Alternative model
├── temp_data/                                        # Temporary processing files
├── requirements.txt                                  # Python dependencies
├── README.md                                         # This documentation
└── LICENSE                                           # MIT License
```

**Total Project Size**: ~20MB

## 📦 Dependencies

Core dependencies for the ACSEE GPA Prediction Dashboard:

| Package | Version | Purpose |
|---------|---------|---------|
| streamlit | 1.37.1 | Web application framework |
| pandas | 2.2.2 | Data manipulation and analysis |
| numpy | 1.26.4 | Numerical computing |
| joblib | 1.4.2 | Model serialization and loading |
| scikit-learn | 1.5.1 | Machine learning algorithms |
| plotly | 5.24.1 | Interactive visualizations |

### Installation
```bash
pip install streamlit==1.37.1 pandas==2.2.2 numpy==1.26.4 joblib==1.4.2 scikit-learn==1.5.1 plotly==5.24.1
```

For complete environment setup:
```bash
pip install -r requirements.txt
```

## 📊 Data Requirements

### Input Data Schema

The CSV file for batch predictions must include these columns (case-insensitive):

| Column Name | Data Type | Valid Range/Values | Description |
|-------------|-----------|-------------------|-------------|
| `prev_sat` | Numeric | - | Number of candidates in previous examination |
| `prev_gpa` | Numeric | 1 - 5 | Previous school GPA |
| `sat` | Numeric |  -  | Number of candidates in current examination |
| `SCHOOL OWNERSHIP` | Categorical | PRIVATE, GOVERNMENT | School ownership type |
| `SCHOOL CATEGORY` | Categorical | BOYS AND GIRLS, GIRLS ONLY, BOYS ONLY | Student gender composition |
| `SCHOOL TYPE` | Categorical | BOARDING, DAY AND BOARDING, DAY | Accommodation type |
| `COMBINATIONS CATEGORY` | Categorical | MIXED, ARTS, SCIENCE | Subject combinations offered |
| `ACADEMIC LEVEL CATEGORY` | Categorical | COMBINED OA, ALEVEL ONLY | Academic level focus |
| `STUDENTS` | Numeric |  -  | Total number of students |
| `TEACHERS` | Numeric |  -  | Total number of teachers |
| `STUDENT-TEACHER RATIO` |  Numeric | - | Student-to-teacher ratio |

### Data Distribution Statistics

**Target Variable (GPA)**:
- Range: 1.0 - 5.0
- Mean: 2.51 ± 0.45
- Distribution: Near-normal with slight right skew

**Key Predictors**:
- **Previous SAT**: 1-676 candidates (mean: 112)
- **Current SAT**: 1-645 candidates (mean: 117)
- **Student-Teacher Ratio**: 0.03-41.86 (mean: 16.03)

### Sample Data Format

```csv
prev_sat,prev_gpa,sat,SCHOOL OWNERSHIP,SCHOOL CATEGORY,SCHOOL TYPE,COMBINATIONS CATEGORY,ACADEMIC LEVEL CATEGORY
112,2.64,117,PRIVATE,BOYS AND GIRLS,BOARDING,MIXED,COMBINED OA
85,2.35,92,GOVERNMENT,GIRLS ONLY,DAY AND BOARDING,ARTS,ALEVEL ONLY
```

### Data Quality Notes

- **Missing Values**: Automatically handled by preprocessing pipeline
- **Categorical Standardization**: Text values are case-insensitive and automatically standardized
- **Validation**: Comprehensive range validation with detailed error reporting
- **Template Download**: Available within the application interface

## 🤝 Contributing

We welcome contributions to improve the ACSEE GPA Prediction Dashboard! 

### Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally
   ```bash
   git clone https://github.com/your-username/ACSEE_GPA.git
   cd ACSEE_GPA
   ```
3. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

### Development Guidelines

- Follow **PEP 8** Python style guidelines
- Include comprehensive tests for new functionality
- Update documentation for any API changes
- Ensure model compatibility with existing pipeline
- Test with sample ACSEE data before submitting

### Submitting Changes

1. **Commit with descriptive messages**
   ```bash
   git commit -m "Add: Brief description of your enhancement"
   ```
2. **Push to your branch**
   ```bash
   git push origin feature/your-feature-name
   ```
3. **Create a pull request** with detailed description

### Areas for Contribution

- Model performance improvements
- Additional visualization features
- Enhanced data validation
- Mobile responsiveness
- Internationalization (Swahili language support)

## 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for complete details.

### MIT License Summary

- ✅ Commercial use permitted
- ✅ Modification allowed
- ✅ Distribution allowed
- ✅ Private use allowed
- ❌ No warranty provided
- ❌ No liability assumed

---

## 🛠️ Technical Details

### System Requirements
- **Memory**: Minimum 4GB RAM recommended for optimal performance
- **Storage**: ~25MB for complete application and dependencies
- **Browser**: Modern web browser with JavaScript enabled
- **Python**: Version 3.9+ required

### Tanzania ACSEE Context
- **Target Audience**: Tanzania secondary education administrators and educators
- **Examination System**: Advanced Certificate of Secondary Education Examination
- **GPA Scale**: 1.0-5.0 scale based on subject performance
- **Data Source**: Historical ACSEE school performance data

### Performance Characteristics
- **Prediction Speed**: <1 second for single predictions
- **Batch Processing**: ~100 schools per second
- **Model Size**: 6.2MB (complete pipeline)
- **Memory Usage**: ~50MB during operation

### Troubleshooting

**Common Issues:**

| Issue | Solution |
|-------|----------|
| Model file not found | Ensure `linear_regression_pipeline.pkl` exists in `models/` directory |
| Data format errors | Check CSV column names match requirements (case-insensitive) |
| Memory errors | Reduce batch size or increase available RAM |
| Port conflicts | Streamlit will automatically find available port |
| Package version conflicts | Use virtual environment and install exact versions |

**Data Validation Errors:**
- Ensure SAT values are positive integers
- Verify GPA values are within 1.0-5.0 range
- Check categorical values match expected options
- Validate student-teacher ratios are positive

---

## 📞 Support & Contact

For questions, issues, or feature requests:

- **GitHub Issues**: [Create an issue](https://github.com/JOHCRAZY/ACSEE_GPA/issues)
- **Repository**: https://github.com/JOHCRAZY/ACSEE_GPA
- **Documentation**: Available in repository README and code comments

---

**Built for Tanzania's Educational Excellence using Streamlit and Machine Learning**

*Supporting data-driven decisions in ACSEE school performance assessment*