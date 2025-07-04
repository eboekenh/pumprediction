# Water Pump Functionality Prediction System

## Overview

This is a machine learning system for predicting water pump functionality in Tanzania. The application uses Streamlit for the web interface and implements a complete ML pipeline from data ingestion to model deployment. The system classifies water pumps into three categories: Functional, Functional needs repair, and Non-functional.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit multi-page application
- **UI Components**: Interactive forms, charts, maps, and data visualizations
- **Visualization Libraries**: Plotly, Matplotlib, Seaborn, Folium for geospatial analysis
- **Navigation**: Sidebar-based page navigation with 6 main sections

### Backend Architecture
- **ML Pipeline**: Modular pipeline with separate components for data ingestion, transformation, and model training
- **Data Processing**: Sklearn-based preprocessing with StandardScaler, LabelEncoder, and OneHotEncoder
- **Model Training**: Supports Random Forest and XGBoost with hyperparameter tuning
- **Prediction Service**: Real-time prediction pipeline with trained model serving

### Core Components Structure
```
src/
├── components/          # ML pipeline components
│   ├── data_ingestion.py
│   ├── data_transformation.py
│   └── model_trainer.py
├── pipeline/           # End-to-end pipelines
│   ├── train_pipeline.py
│   └── predict_pipeline.py
├── exception.py        # Custom exception handling
├── logger.py          # Logging configuration
└── utils.py           # Utility functions
```

## Key Components

### 1. Data Ingestion (`DataIngestion`)
- **Purpose**: Load and split dataset into train/test sets
- **Features**: Supports custom data paths or default sample data
- **Output**: Saves train.csv and test.csv to artifacts directory

### 2. Data Transformation (`DataTransformation`)
- **Purpose**: Preprocess data for ML models
- **Features**: Automatic column type detection, missing value imputation, scaling, encoding
- **Pipeline**: Uses sklearn ColumnTransformer for different preprocessing steps

### 3. Model Training (`ModelTrainer`)
- **Purpose**: Train and evaluate ML models with hyperparameter tuning
- **Models**: Random Forest, XGBoost
- **Features**: Grid search CV, cross-validation, model comparison

### 4. Prediction Pipeline (`PredictPipeline`)
- **Purpose**: Make predictions on new data
- **Features**: Load trained model and preprocessor, transform input data, return predictions

### 5. Streamlit Pages
- **EDA**: Exploratory data analysis with interactive visualizations
- **Preprocessing**: Data quality assessment and transformation
- **Model Training**: Model selection and training interface
- **Model Evaluation**: Performance metrics and model comparison
- **Predictions**: Single and batch prediction interface
- **Geospatial Analysis**: Interactive maps and location-based insights

## Data Flow

1. **Data Ingestion**: Raw data → Train/Test split → CSV files in artifacts/
2. **Data Transformation**: Raw data → Preprocessing pipeline → Transformed arrays
3. **Model Training**: Transformed data → Hyperparameter tuning → Best model selection
4. **Model Evaluation**: Test data → Predictions → Performance metrics
5. **Prediction**: New data → Preprocessing → Model inference → Results

## External Dependencies

### Core ML Libraries
- **scikit-learn**: Core ML algorithms and preprocessing
- **xgboost**: Gradient boosting framework
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations

### Visualization Libraries
- **streamlit**: Web application framework
- **plotly**: Interactive visualizations
- **matplotlib**: Static plotting
- **seaborn**: Statistical visualizations
- **folium**: Interactive maps
- **streamlit-folium**: Streamlit-Folium integration

### Utility Libraries
- **pickle**: Model serialization
- **os, sys**: System operations
- **datetime**: Timestamp generation
- **dataclasses**: Configuration classes

## Deployment Strategy

### Artifact Management
- **Model Persistence**: Trained models saved as pickle files in artifacts/
- **Preprocessor Storage**: Sklearn pipelines saved for consistent preprocessing
- **Configuration**: Dataclass-based configuration for each component

### Session Management
- **Streamlit State**: Uses st.session_state for data persistence across pages
- **File Paths**: Centralized artifact storage in artifacts/ directory

### Error Handling
- **Custom Exceptions**: Detailed error messages with file and line information
- **Logging**: Comprehensive logging with timestamps and structured format
- **Graceful Degradation**: Fallback mechanisms for missing data/models

### Scalability Considerations
- **Modular Design**: Separate components for easy maintenance and updates
- **Pipeline Architecture**: Extensible pipeline system for adding new models
- **Configuration Management**: Centralized configuration for easy parameter updates

## Recent Changes
- July 04, 2025: Enhanced data ingestion system to handle separate training features, labels, and test files
- July 04, 2025: Added FileHandler component for automatic file identification and merging
- July 04, 2025: Fixed EDA correlation crashes by replacing complex calculations with safe Pearson correlation and recommendations
- July 04, 2025: Implemented realistic data processing for Tanzania water pump dataset (59,400 training, 14,850 test samples)
- July 04, 2025: Updated model training with more conservative hyperparameters to prevent overfitting
- July 04, 2025: Integrated real dataset with proper class distribution (54.3% functional, 38.4% non-functional, 7.3% needs repair)
- July 04, 2025: Replaced unrealistic sample data with authentic Tanzania water pump competition data
- July 04, 2025: Added realistic train-test splits and validation procedures for large-scale ML pipeline

## Changelog
- July 04, 2025. Initial setup

## User Preferences

Preferred communication style: Simple, everyday language.