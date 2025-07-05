# Water Pump Functionality Prediction System - Complete Documentation

## Table of Contents
1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Directory Structure](#directory-structure)
4. [File Descriptions](#file-descriptions)
5. [Setup Instructions](#setup-instructions)
6. [Usage Guide](#usage-guide)
7. [API Documentation](#api-documentation)
8. [Development Guide](#development-guide)

## Project Overview

This is a comprehensive machine learning system designed to predict water pump functionality in Tanzania. The system classifies water pumps into three categories:
- **Functional**: Working properly
- **Functional needs repair**: Working but requires maintenance  
- **Non-functional**: Not working

The project uses real data from the Tanzania Ministry of Water and includes a complete ML pipeline from data ingestion to model deployment with an interactive web interface.

## System Architecture

### Technology Stack
- **Frontend**: Streamlit (Multi-page web application)
- **Backend**: Python with scikit-learn, XGBoost, LightGBM, CatBoost
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly, Matplotlib, Seaborn, Folium
- **Deployment**: Replit with workflow automation

### Core Components
1. **Data Ingestion**: Automated data loading and preprocessing
2. **Feature Engineering**: Advanced data transformation pipeline
3. **Model Training**: Multiple ML algorithms with hyperparameter tuning
4. **Model Evaluation**: Comprehensive performance metrics
5. **Prediction Service**: Real-time prediction capabilities
6. **Geospatial Analysis**: Interactive mapping and location-based insights

## Directory Structure

```
water-pump-prediction/
├── 📁 src/                          # Core source code
│   ├── 📁 components/               # ML pipeline components
│   │   ├── __init__.py
│   │   ├── data_ingestion.py        # Data loading and splitting
│   │   ├── data_transformation.py   # Data preprocessing pipeline
│   │   ├── model_trainer.py         # Model training and evaluation
│   │   ├── file_handler.py          # File processing utilities
│   │   ├── advanced_eda.py          # Advanced exploratory data analysis
│   │   ├── training_guard.py        # Training safety mechanisms
│   │   └── training_report.py       # Training report generation
│   ├── 📁 pipeline/                 # End-to-end pipelines
│   │   ├── __init__.py
│   │   ├── train_pipeline.py        # Complete training pipeline
│   │   └── predict_pipeline.py      # Prediction pipeline
│   ├── 📁 utils/                    # Utility functions
│   │   ├── __init__.py
│   │   └── utils.py                 # Common utility functions
│   ├── exception.py                 # Custom exception handling
│   ├── logger.py                    # Logging configuration
│   └── __init__.py
├── 📁 pages/                        # Streamlit pages
│   ├── 01_🔍_EDA.py                # Exploratory Data Analysis
│   ├── 02_⚙️_Preprocessing.py       # Data preprocessing interface
│   ├── 03_🤖_Model_Training.py      # Model training interface
│   ├── 04_📊_Model_Evaluation.py    # Model evaluation interface
│   ├── 05_🎯_Predictions.py         # Prediction interface
│   └── 06_🗺️_Geospatial_Analysis.py # Geographic analysis
├── 📁 artifacts/                    # Generated files and models
│   ├── model.pkl                    # Trained model
│   ├── preprocessor.pkl             # Data preprocessing pipeline
│   ├── label_encoder.pkl            # Label encoding mappings
│   ├── merged_train_data.csv        # Processed training data
│   ├── test_data.csv                # Test dataset
│   └── training_report_*.txt        # Training reports
├── 📁 logs/                         # Application logs
├── 📁 config/                       # Configuration files
│   └── config.yaml                  # Application configuration
├── 📁 sample_data/                  # Sample datasets
├── 📁 uploaded_data/                # User uploaded files
├── 📁 attached_assets/              # Project assets
├── app.py                           # Main Streamlit application
├── replit.md                        # Project documentation
└── requirements files               # Dependencies
```

## File Descriptions

### 🎯 Main Application Files

#### `app.py`
**Purpose**: Main Streamlit application entry point
- Provides homepage with project overview
- Handles file uploads (single or multiple files)
- Loads Tanzania water pump dataset
- Navigation hub for all application features
- Session state management

#### `app_safe.py` & `app_simple.py`
**Purpose**: Alternative versions of the main application
- Simplified interfaces for testing
- Reduced functionality for debugging
- Backup versions with different configurations

### 🔧 Core ML Pipeline (`src/components/`)

#### `data_ingestion.py`
**Purpose**: Data loading and initial processing
- Loads raw data from various sources
- Handles train/test splits
- Manages data quality checks
- Supports multiple file formats

#### `data_transformation.py`
**Purpose**: Data preprocessing pipeline
- Missing value imputation
- Feature scaling and normalization
- Categorical encoding (OneHot, Label)
- Feature engineering transformations
- Pipeline serialization

#### `model_trainer.py`
**Purpose**: Model training and evaluation
- Supports 9 ML algorithms:
  - Random Forest
  - XGBoost
  - LightGBM
  - CatBoost
  - SVM
  - Extra Trees
  - AdaBoost
  - K-Nearest Neighbors
  - Neural Networks
- Hyperparameter tuning with GridSearchCV
- Cross-validation
- Model comparison and selection

#### `file_handler.py`
**Purpose**: Advanced file processing
- Automatic file type detection
- Merging multiple data files
- Data validation and cleaning
- Format standardization

#### `advanced_eda.py`
**Purpose**: Comprehensive exploratory data analysis
- Statistical analysis
- Data quality assessment
- Correlation analysis
- Feature importance analysis
- Visualization generation

#### `training_guard.py`
**Purpose**: Training safety and monitoring
- Prevents overfitting
- Memory usage monitoring
- Training time limits
- Performance thresholds

#### `training_report.py`
**Purpose**: Detailed training documentation
- Model performance metrics
- Training history
- Feature importance rankings
- Recommendations for improvement

### 🔄 Pipeline Components (`src/pipeline/`)

#### `train_pipeline.py`
**Purpose**: Complete training workflow
- Orchestrates entire training process
- Handles data ingestion → transformation → training
- Manages model persistence
- Generates training reports

#### `predict_pipeline.py`
**Purpose**: Prediction workflow
- Loads trained models
- Processes new data
- Generates predictions
- Handles batch processing

### 🛠️ Utility Files

#### `src/utils.py`
**Purpose**: Common utility functions
- Model serialization/deserialization
- Performance metric calculations
- Feature importance extraction
- Data validation helpers

#### `src/exception.py`
**Purpose**: Custom error handling
- Detailed error messages
- File and line number tracking
- Graceful error recovery
- Logging integration

#### `src/logger.py`
**Purpose**: Logging configuration
- Timestamped log files
- Console and file output
- Structured log formats
- Debug information

### 📊 Streamlit Pages

#### `01_🔍_EDA.py`
**Purpose**: Interactive data exploration
- Dataset overview and statistics
- Missing value analysis
- Correlation matrices
- Distribution plots
- Feature relationship analysis

#### `02_⚙️_Preprocessing.py`
**Purpose**: Data preprocessing interface
- Interactive data cleaning
- Feature selection
- Transformation options
- Preview processed data

#### `03_🤖_Model_Training.py`
**Purpose**: Model training interface
- Algorithm selection
- Hyperparameter configuration
- Training progress monitoring
- Performance visualization

#### `04_📊_Model_Evaluation.py`
**Purpose**: Model evaluation dashboard
- Performance metrics
- Confusion matrices
- ROC curves
- Model comparison
- Cross-validation results

#### `05_🎯_Predictions.py`
**Purpose**: Prediction interface
- Single pump prediction
- Batch prediction from CSV
- Confidence scores
- Prediction explanations

#### `06_🗺️_Geospatial_Analysis.py`
**Purpose**: Geographic analysis
- Interactive maps
- Pump location visualization
- Regional performance analysis
- Water quality mapping

### 📋 Utility Scripts

#### `prediction_examples.py`
**Purpose**: Example usage scripts
- Demonstrates prediction API
- Shows different input formats
- Provides integration examples
- API usage patterns

#### `csv_processor.py`
**Purpose**: CSV file processing
- Batch CSV processing
- Automated predictions
- Output formatting
- Summary reports

#### `flask_api.py`
**Purpose**: REST API implementation
- HTTP endpoints for predictions
- JSON input/output
- API documentation
- Health checks

#### `simple_prediction_api.py`
**Purpose**: Simplified API wrapper
- Easy-to-use prediction interface
- Minimal setup required
- Quick integration

### 📈 Analysis Scripts

#### `run_comprehensive_eda.py`
**Purpose**: Automated EDA generation
- Complete dataset analysis
- Report generation
- Visualization creation
- Statistical summaries

#### `demo_training_report.py`
**Purpose**: Training report examples
- Sample training outputs
- Report formatting
- Performance visualization

#### `test_model_training.py` & `test_full_training.py`
**Purpose**: Training validation
- Model training tests
- Performance benchmarks
- Integration testing

### 🗂️ Configuration Files

#### `config/config.yaml`
**Purpose**: Application configuration
- Model parameters
- File paths
- Logging settings
- Default values

#### `pyproject.toml`
**Purpose**: Python project configuration
- Dependencies
- Package metadata
- Build settings

## Setup Instructions

### Prerequisites
- Python 3.11 or higher
- 4GB+ RAM (recommended for large datasets)
- Internet connection for package installation

### Local Installation

1. **Clone or download the project**
   ```bash
   git clone [repository-url]
   cd water-pump-prediction
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Create required directories**
   ```bash
   mkdir -p artifacts logs uploaded_data
   ```

5. **Run the application**
   ```bash
   streamlit run app.py --server.port 5000 --server.address 0.0.0.0
   ```

### Replit Setup

1. **Import project to Replit**
   - Upload all files to Replit
   - Ensure directory structure is maintained

2. **Install dependencies**
   - Dependencies are automatically installed from `pyproject.toml`

3. **Configure workflow**
   - Workflow is pre-configured to run on port 5000
   - Starts automatically when you run the project

## Usage Guide

### 1. Data Upload
- **Option 1**: Upload 3 separate CSV files (training features, labels, test features)
- **Option 2**: Upload single combined CSV file
- **Option 3**: Use built-in Tanzania dataset

### 2. Data Exploration
- Navigate to EDA page
- View dataset statistics
- Analyze missing values
- Explore feature relationships

### 3. Data Preprocessing
- Select preprocessing options
- Handle missing values
- Apply feature transformations
- Preview processed data

### 4. Model Training
- Choose ML algorithms
- Configure hyperparameters
- Start training process
- Monitor progress

### 5. Model Evaluation
- Review performance metrics
- Compare different models
- Analyze predictions
- Generate reports

### 6. Making Predictions
- Single pump prediction
- Batch processing
- Download results
- View confidence scores

### 7. Geographic Analysis
- View pump locations on map
- Regional performance analysis
- Water quality insights
- Infrastructure analysis

## API Documentation

### REST API Endpoints

```python
# Health check
GET /health

# Single prediction
POST /predict
{
    "pump_data": {
        "latitude": -5.0,
        "longitude": 35.0,
        "population": 1000,
        "water_quality": "good"
    }
}

# Batch prediction
POST /predict/batch
{
    "pumps": [
        {"latitude": -5.0, "longitude": 35.0, ...},
        {"latitude": -6.0, "longitude": 36.0, ...}
    ]
}

# Model information
GET /model/info
```

### Python API Usage

```python
from prediction_examples import WaterPumpPredictor

# Initialize predictor
predictor = WaterPumpPredictor()

# Single prediction
result = predictor.predict_single({
    'latitude': -5.0,
    'longitude': 35.0,
    'population': 1000,
    'water_quality': 'good'
})

# Batch prediction
results = predictor.predict_batch([pump1_data, pump2_data])
```

## Development Guide

### Adding New Features

1. **Create new component**
   ```python
   # src/components/new_feature.py
   class NewFeature:
       def __init__(self):
           pass
       
       def process(self, data):
           # Implementation
           return processed_data
   ```

2. **Add to pipeline**
   ```python
   # src/pipeline/train_pipeline.py
   from src.components.new_feature import NewFeature
   
   # Integrate into training workflow
   ```

3. **Create Streamlit page**
   ```python
   # pages/07_🆕_New_Feature.py
   import streamlit as st
   
   def main():
       st.title("New Feature")
       # Implementation
   ```

### Testing

1. **Unit tests**
   ```bash
   python -m pytest tests/
   ```

2. **Integration tests**
   ```bash
   python test_model_training.py
   ```

3. **Manual testing**
   ```bash
   streamlit run app.py
   ```

### Debugging

1. **Check logs**
   ```bash
   tail -f logs/latest.log
   ```

2. **Enable debug mode**
   ```python
   # In your script
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

3. **Use Streamlit debug info**
   ```python
   st.write("Debug info:", st.session_state)
   ```

### Performance Optimization

1. **Data loading optimization**
   - Use `pd.read_csv(nrows=1000)` for large files
   - Implement data sampling
   - Cache processed data

2. **Model training optimization**
   - Use cross-validation strategically
   - Implement early stopping
   - Optimize hyperparameter ranges

3. **Memory management**
   - Clear unused variables
   - Use generators for large datasets
   - Monitor memory usage

## Troubleshooting

### Common Issues

1. **Import errors**
   - Check Python path
   - Verify all dependencies installed
   - Check file permissions

2. **Memory issues**
   - Reduce dataset size
   - Increase available RAM
   - Use data sampling

3. **Performance issues**
   - Use fewer model algorithms
   - Reduce hyperparameter search space
   - Implement caching

### Support

For technical support:
1. Check logs in `logs/` directory
2. Review error messages in Streamlit interface
3. Consult this documentation
4. Check the replit.md file for recent changes

---

**Last Updated**: July 5, 2025
**Version**: 1.0
**Author**: Water Pump Prediction System Team