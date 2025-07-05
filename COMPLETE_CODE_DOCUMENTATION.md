# Complete Code Documentation with Line-by-Line Comments

This document contains all the core scripts with detailed explanations for each line of code.

## Table of Contents

1. [Data Ingestion](#data-ingestion)
2. [Data Transformation](#data-transformation)
3. [Model Training](#model-training)
4. [Exception Handling](#exception-handling)
5. [Utility Functions](#utility-functions)
6. [Training Pipeline](#training-pipeline)
7. [Prediction Pipeline](#prediction-pipeline)
8. [Streamlit Pages](#streamlit-pages)

---

## Data Ingestion

### `src/components/data_ingestion.py`

```python
import os                           # For file and directory operations
import sys                          # For system-specific parameters and functions
import pandas as pd                 # For data manipulation and analysis
import numpy as np                  # For numerical operations
from sklearn.model_selection import train_test_split  # For splitting data into train/test sets
from dataclasses import dataclass   # For creating data classes
from src.exception import CustomException  # Custom exception handling
from src.logger import logging      # Logging functionality
from typing import Tuple            # For type hints

@dataclass
class DataIngestionConfig:
    """Configuration class for data ingestion paths and settings"""
    # Define default paths for train, test, and raw data files
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "raw_data.csv")

class DataIngestion:
    """Main class for handling data ingestion and processing"""
    
    def __init__(self):
        """Initialize the data ingestion component with configuration"""
        self.ingestion_config = DataIngestionConfig()
    
    def _merge_train_data(self, train_features_path: str, train_labels_path: str) -> pd.DataFrame:
        """
        Private method to merge training features and labels
        
        Args:
            train_features_path: Path to CSV file containing training features
            train_labels_path: Path to CSV file containing training labels
            
        Returns:
            DataFrame with merged features and labels
        """
        try:
            # Read training features CSV file into DataFrame
            train_features = pd.read_csv(train_features_path)
            # Read training labels CSV file into DataFrame
            train_labels = pd.read_csv(train_labels_path)
            
            # Log the shapes of loaded data for debugging
            logging.info(f"Training features shape: {train_features.shape}")
            logging.info(f"Training labels shape: {train_labels.shape}")
            
            # Merge features and labels on the common 'id' column using inner join
            merged_df = pd.merge(train_features, train_labels, on='id', how='inner')
            
            # Log the shape of merged data
            logging.info(f"Merged training data shape: {merged_df.shape}")
            
            return merged_df
            
        except Exception as e:
            # Log any errors that occur during merging
            logging.error(f"Error merging train data: {str(e)}")
            # Raise custom exception with system details
            raise CustomException(e, sys)
    
    def _identify_file_types(self, file_paths: list) -> dict:
        """
        Private method to automatically identify file types (features, labels, test)
        
        Args:
            file_paths: List of file paths to analyze
            
        Returns:
            Dictionary mapping file types to their paths
        """
        file_info = {}  # Initialize empty dictionary for file information
        
        # Iterate through each file path
        for path in file_paths:
            # Check if file exists
            if os.path.exists(path):
                # Read only first 5 rows to check file structure (for efficiency)
                df = pd.read_csv(path, nrows=5)
                
                # Identify labels file: has target column and few columns (typically 2-3)
                if 'status_group' in df.columns and len(df.columns) <= 3:
                    file_info['labels'] = path
                    logging.info(f"Identified labels file: {path}")
                
                # Identify features file: many columns, no target column
                elif 'status_group' not in df.columns and len(df.columns) > 10:
                    if 'features' not in file_info:
                        # First features file is training features
                        file_info['features'] = path
                        logging.info(f"Identified features file: {path}")
                    else:
                        # Second features file is test features
                        file_info['test_features'] = path
                        logging.info(f"Identified test features file: {path}")
        
        return file_info
    
    def initiate_data_ingestion(self, data_path=None, train_features_path=None, 
                               train_labels_path=None, test_features_path=None) -> Tuple[str, str]:
        """
        Main method to initiate the data ingestion process
        
        Args:
            data_path: Path to single combined dataset file (optional)
            train_features_path: Path to training features file (optional)
            train_labels_path: Path to training labels file (optional)
            test_features_path: Path to test features file (optional)
            
        Returns:
            Tuple containing paths to train and test data files
        """
        try:
            # Log the start of data ingestion process
            logging.info("Entered data ingestion method")
            
            # Create artifacts directory if it doesn't exist
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            
            # Handle case where separate training features and labels are provided
            if train_features_path and train_labels_path:
                logging.info("Loading separate training features and labels files")
                # Merge training features and labels
                train_df = self._merge_train_data(train_features_path, train_labels_path)
                
                # Load test features if provided
                if test_features_path and os.path.exists(test_features_path):
                    test_df = pd.read_csv(test_features_path)
                    logging.info(f"Test features loaded, shape: {test_df.shape}")
                else:
                    # If no test features provided, split training data
                    train_df, test_df = train_test_split(
                        train_df,                    # Data to split
                        test_size=0.2,              # 20% for testing
                        random_state=42,            # For reproducible results
                        stratify=train_df['status_group']  # Maintain class distribution
                    )
                    
            # Handle case where single combined file is provided
            elif data_path and os.path.exists(data_path):
                logging.info(f"Loading data from single file: {data_path}")
                df = pd.read_csv(data_path)
                # Split data into train and test sets
                train_df, test_df = train_test_split(
                    df,                                        # Data to split
                    test_size=0.2,                            # 20% for testing
                    random_state=42,                          # For reproducible results
                    stratify=df['status_group'] if 'status_group' in df.columns else None
                )
            else:
                # Handle case where no data paths provided - look for processed data
                train_data_path = "artifacts/merged_train_data.csv"
                test_data_path = "artifacts/test_data.csv"
                
                # Check if processed Tanzania data exists
                if os.path.exists(train_data_path) and os.path.exists(test_data_path):
                    train_df = pd.read_csv(train_data_path)
                    test_df = pd.read_csv(test_data_path)
                    logging.info(f"Real Tanzania data loaded from processed files")
                    
                    # Split full training dataset for validation
                    train_df, test_df = train_test_split(
                        train_df,                             # Data to split
                        test_size=0.2,                       # 20% for validation
                        random_state=42,                     # For reproducible results
                        stratify=train_df['status_group'] if 'status_group' in train_df.columns else None
                    )
                else:
                    # Fallback to sample data if no real data found
                    sample_path = "sample_data/sample_water_pumps.csv"
                    if os.path.exists(sample_path):
                        df = pd.read_csv(sample_path)
                        logging.info(f"Fallback to sample data from: {sample_path}")
                        train_df, test_df = train_test_split(
                            df,                              # Data to split
                            test_size=0.25,                 # 25% for testing
                            random_state=42,                # For reproducible results
                            stratify=df['status_group'] if 'status_group' in df.columns else None
                        )
                    else:
                        # No data found - raise error
                        raise FileNotFoundError("No data file found. Please upload and process data first.")
            
            # Log final dataset shapes
            logging.info(f"Final train dataset shape: {train_df.shape}")
            logging.info(f"Final test dataset shape: {test_df.shape}")
            
            # Save raw merged data to artifacts directory
            train_df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Raw data saved")
            
            # Save train and test sets to separate files
            train_df.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_df.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            
            logging.info("Data ingestion completed successfully")
            
            # Return paths to saved train and test files
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
                
        except Exception as e:
            # Log any errors that occur during data ingestion
            logging.error(f"Error in data ingestion: {str(e)}")
            # Raise custom exception with system details
            raise CustomException(e, sys)
    
    def get_data_info(self, data_path: str) -> dict:
        """
        Get comprehensive information about the dataset
        
        Args:
            data_path: Path to the dataset file
            
        Returns:
            Dictionary containing dataset information
        """
        try:
            # Load dataset
            df = pd.read_csv(data_path)
            
            # Create comprehensive information dictionary
            info = {
                'shape': df.shape,                           # Number of rows and columns
                'columns': df.columns.tolist(),             # List of column names
                'dtypes': df.dtypes.to_dict(),              # Data types of each column
                'missing_values': df.isnull().sum().to_dict(), # Missing values per column
                'memory_usage': df.memory_usage().sum(),    # Total memory usage
                'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
                'categorical_columns': df.select_dtypes(include=['object']).columns.tolist()
            }
            
            # Add target column information if it exists
            if 'status_group' in df.columns:
                info['target_distribution'] = df['status_group'].value_counts().to_dict()
            
            return info
            
        except Exception as e:
            # Log any errors that occur while getting data info
            logging.error(f"Error getting data info: {str(e)}")
            # Raise custom exception with system details
            raise CustomException(e, sys)
```

---

## Data Transformation

### `src/components/data_transformation.py`

```python
import sys                          # For system-specific parameters and functions
import os                           # For file and directory operations
import pandas as pd                 # For data manipulation and analysis
import numpy as np                  # For numerical operations
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer  # For applying different transformations to different columns
from sklearn.pipeline import Pipeline           # For creating processing pipelines
from sklearn.impute import SimpleImputer       # For handling missing values
from dataclasses import dataclass              # For creating data classes
import pickle                                  # For serializing objects
from src.exception import CustomException       # Custom exception handling
from src.logger import logging                 # Logging functionality
from src.utils import save_object              # Utility function for saving objects

@dataclass
class DataTransformationConfig:
    """Configuration class for data transformation settings"""
    # Define path where preprocessor object will be saved
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    """Main class for handling data transformation and preprocessing"""
    
    def __init__(self):
        """Initialize the data transformation component with configuration"""
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformer_object(self, df: pd.DataFrame):
        """
        Create a comprehensive preprocessing pipeline
        
        Args:
            df: Input DataFrame to analyze for creating appropriate transformations
            
        Returns:
            sklearn ColumnTransformer object with preprocessing pipeline
        """
        try:
            # Identify numerical columns (int, float types)
            numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            # Identify categorical columns (object/string types)
            categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
            
            # Handle boolean columns that may be stored as object type
            boolean_columns = []
            remaining_categorical = []
            
            # Check each categorical column for boolean values
            for col in categorical_columns:
                unique_vals = df[col].dropna().unique()  # Get unique values, excluding NaN
                # Check if column contains only boolean values (True/False)
                if len(unique_vals) <= 2 and all(isinstance(x, bool) for x in unique_vals):
                    boolean_columns.append(col)
                else:
                    remaining_categorical.append(col)
            
            # Update categorical columns to exclude boolean columns
            categorical_columns = remaining_categorical
            
            # Remove target column from all lists if present
            if 'status_group' in numerical_columns:
                numerical_columns.remove('status_group')
            if 'status_group' in categorical_columns:
                categorical_columns.remove('status_group')
            if 'status_group' in boolean_columns:
                boolean_columns.remove('status_group')
            
            # Remove ID columns that shouldn't be used for modeling
            id_columns = ['id']
            numerical_columns = [col for col in numerical_columns if col not in id_columns]
            categorical_columns = [col for col in categorical_columns if col not in id_columns]
            boolean_columns = [col for col in boolean_columns if col not in id_columns]
            
            # Log the identified column types for debugging
            logging.info(f"Numerical columns: {numerical_columns}")
            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Boolean columns: {boolean_columns}")
            
            # Create numerical processing pipeline
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),  # Fill missing values with median
                    ("scaler", StandardScaler())                   # Standardize numerical features
                ]
            )
            
            # Create categorical processing pipeline with cardinality limit
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),  # Fill missing with "missing"
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, max_categories=20))  # One-hot encode
                ]
            )
            
            # Combine boolean columns with categorical for processing
            all_categorical_columns = categorical_columns + boolean_columns
            
            # Create main preprocessor that combines all pipelines
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", num_pipeline, numerical_columns),        # Apply numerical pipeline
                    ("cat", cat_pipeline, all_categorical_columns)   # Apply categorical pipeline
                ]
            )
            
            return preprocessor
            
        except Exception as e:
            # Log any errors that occur during transformer creation
            logging.error(f"Error creating data transformer: {str(e)}")
            # Raise custom exception with system details
            raise CustomException(e, sys)
    
    def initiate_data_transformation(self, train_path: str, test_path: str):
        """
        Main method to initiate data transformation process
        
        Args:
            train_path: Path to training data CSV file
            test_path: Path to test data CSV file
            
        Returns:
            Tuple containing transformed arrays and preprocessor file path
        """
        try:
            # Load training and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            # Log successful data loading
            logging.info("Data loaded for transformation")
            logging.info(f"Training data shape: {train_df.shape}")
            logging.info(f"Test data shape: {test_df.shape}")
            
            # Convert boolean columns to strings to avoid encoding issues
            for col in train_df.select_dtypes(include=['object']).columns:
                unique_vals = train_df[col].dropna().unique()
                # Check if column contains boolean values
                if len(unique_vals) <= 2 and all(isinstance(x, bool) for x in unique_vals):
                    train_df[col] = train_df[col].astype(str)  # Convert to string
                    test_df[col] = test_df[col].astype(str)    # Convert to string
            
            # Create preprocessor object
            preprocessor_obj = self.get_data_transformer_object(train_df)
            
            # Define target column name
            target_column = "status_group"
            
            # Check if target column exists in training data
            if target_column in train_df.columns:
                # Separate features and target for training data
                input_feature_train_df = train_df.drop(columns=[target_column], axis=1)
                target_feature_train_df = train_df[target_column]
                
                # Separate features and target for test data
                input_feature_test_df = test_df.drop(columns=[target_column], axis=1)
                target_feature_test_df = test_df[target_column]
                
                # Log transformation start
                logging.info("Starting feature transformation...")
                logging.info(f"Processing {len(input_feature_train_df.columns)} features")
                
                # Handle large datasets by using sampling for fitting
                max_samples = 10000
                if len(input_feature_train_df) > max_samples:
                    logging.info(f"Large dataset detected ({len(input_feature_train_df)} samples), using sample of {max_samples} for preprocessing")
                    # Sample data for fitting the preprocessor
                    train_sample = input_feature_train_df.sample(n=max_samples, random_state=42)
                    # Fit preprocessor on sample
                    input_feature_train_arr = preprocessor_obj.fit_transform(train_sample)
                    # Transform test data
                    input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)
                    
                    # Transform full training set
                    input_feature_train_arr = preprocessor_obj.transform(input_feature_train_df)
                else:
                    # For smaller datasets, fit and transform normally
                    input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
                    input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)
                
                logging.info("Feature transformation completed")
                
                # Encode target labels to numerical values
                label_encoder = LabelEncoder()
                target_feature_train_encoded = label_encoder.fit_transform(target_feature_train_df)
                target_feature_test_encoded = label_encoder.transform(target_feature_test_df)
                
                # Save label encoder for future use
                label_encoder_path = os.path.join("artifacts", "label_encoder.pkl")
                save_object(file_path=label_encoder_path, obj=label_encoder)
                
                # Combine features and target into single arrays
                train_arr = np.c_[input_feature_train_arr, target_feature_train_encoded]
                test_arr = np.c_[input_feature_test_arr, target_feature_test_encoded]
                
                logging.info("Data transformation completed successfully")
                
            else:
                # If no target column exists, just transform features
                input_feature_train_arr = preprocessor_obj.fit_transform(train_df)
                input_feature_test_arr = preprocessor_obj.transform(test_df)
                
                train_arr = input_feature_train_arr
                test_arr = input_feature_test_arr
                
                logging.info("Data transformation completed (no target column)")
            
            # Save preprocessor object for future use
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )
            
            # Return transformed arrays and preprocessor path
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
            
        except Exception as e:
            # Log any errors that occur during transformation
            logging.error(f"Error in data transformation: {str(e)}")
            # Raise custom exception with system details
            raise CustomException(e, sys)
    
    def get_feature_names(self, preprocessor_obj, original_df):
        """
        Get feature names after transformation (for interpretability)
        
        Args:
            preprocessor_obj: Fitted preprocessor object
            original_df: Original DataFrame before transformation
            
        Returns:
            List of feature names after transformation
        """
        try:
            # Get original column types
            numerical_columns = original_df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_columns = original_df.select_dtypes(include=['object']).columns.tolist()
            
            # Remove target and ID columns
            if 'status_group' in numerical_columns:
                numerical_columns.remove('status_group')
            if 'status_group' in categorical_columns:
                categorical_columns.remove('status_group')
            
            id_columns = ['id']
            numerical_columns = [col for col in numerical_columns if col not in id_columns]
            categorical_columns = [col for col in categorical_columns if col not in id_columns]
            
            # Build feature names list
            feature_names = []
            
            # Add numerical feature names (unchanged)
            feature_names.extend(numerical_columns)
            
            # Add categorical feature names (one-hot encoded)
            if categorical_columns:
                cat_transformer = preprocessor_obj.named_transformers_['cat']
                if hasattr(cat_transformer, 'named_steps'):
                    onehot_encoder = cat_transformer.named_steps['onehot']
                    # Get feature names from one-hot encoder
                    cat_feature_names = onehot_encoder.get_feature_names_out(categorical_columns)
                    feature_names.extend(cat_feature_names)
            
            return feature_names
            
        except Exception as e:
            # Log any errors that occur while getting feature names
            logging.error(f"Error getting feature names: {str(e)}")
            return None
```

---

## Model Training

### `src/components/model_trainer.py`

```python
import os                           # For file and directory operations
import sys                          # For system-specific parameters and functions
import pandas as pd                 # For data manipulation and analysis
import numpy as np                  # For numerical operations
import time                         # For timing operations
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.svm import SVC                    # Support Vector Machine
from sklearn.neural_network import MLPClassifier  # Neural Network
from sklearn.neighbors import KNeighborsClassifier  # K-Nearest Neighbors
from sklearn.ensemble import VotingClassifier      # Ensemble voting classifier
from sklearn.model_selection import GridSearchCV, cross_val_score  # Model selection tools
from sklearn.metrics import classification_report, accuracy_score, f1_score  # Evaluation metrics
from xgboost import XGBClassifier              # XGBoost classifier
# from lightgbm import LGBMClassifier          # Disabled due to system library requirements
# from catboost import CatBoostClassifier      # Disabled due to system library requirements
from dataclasses import dataclass             # For creating data classes
import pickle                                 # For serializing objects
from src.exception import CustomException      # Custom exception handling
from src.logger import logging                # Logging functionality
from src.utils import save_object, evaluate_models  # Utility functions
from src.components.training_report import TrainingReportGenerator  # Training report generator

@dataclass
class ModelTrainerConfig:
    """Configuration class for model training settings"""
    # Define paths for saving trained models and reports
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")
    model_report_file_path: str = os.path.join("artifacts", "model_report.pkl")
    training_report_file_path: str = os.path.join("artifacts", "training_report.txt")

class ModelTrainer:
    """Main class for training and evaluating machine learning models"""
    
    def __init__(self):
        """Initialize the model trainer with configuration"""
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self, train_array, test_array, selected_models=None):
        """
        Main method to initiate model training with multiple algorithms
        
        Args:
            train_array: Training data array (features + target)
            test_array: Test data array (features + target)
            selected_models: List of model names to train (if None, trains all models)
            
        Returns:
            Best model accuracy score
        """
        try:
            # Log the start of model training
            logging.info("Starting model training")
            
            # Initialize training report generator
            report_generator = TrainingReportGenerator()
            
            # Create dataset information for reporting
            dataset_info = {
                'training_samples': len(train_array),           # Number of training samples
                'test_samples': len(test_array),               # Number of test samples
                'total_features': train_array.shape[1] - 1,   # Number of features (excluding target)
                'target_classes': len(np.unique(train_array[:, -1]))  # Number of target classes
            }
            
            # Start training session in report generator
            report_generator.start_training_session(dataset_info)
            
            # Split arrays into features and target
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],    # All columns except last (features)
                train_array[:, -1],     # Last column (target)
                test_array[:, :-1],     # All columns except last (features)
                test_array[:, -1]       # Last column (target)
            )
            
            # Define models with optimized parameters for large datasets
            models = {
                "Random Forest": RandomForestClassifier(
                    n_estimators=50,        # Number of trees (reduced for speed)
                    max_depth=10,           # Maximum depth of trees (reduced for speed)
                    min_samples_split=20,   # Minimum samples required to split (increased for speed)
                    min_samples_leaf=10,    # Minimum samples in leaf node (increased for speed)
                    random_state=42,        # For reproducible results
                    n_jobs=-1               # Use all available cores
                ),
                "XGBoost": XGBClassifier(
                    n_estimators=50,        # Number of boosting rounds (reduced for speed)
                    max_depth=4,            # Maximum depth of trees (reduced for speed)
                    learning_rate=0.2,      # Learning rate (increased for faster convergence)
                    subsample=0.8,          # Subsample ratio of training instances
                    colsample_bytree=0.8,   # Subsample ratio of columns
                    random_state=42,        # For reproducible results
                    eval_metric='mlogloss', # Evaluation metric
                    n_jobs=-1               # Use all available cores
                ),
                "Support Vector Machine": SVC(
                    kernel='rbf',           # Radial basis function kernel
                    C=1.0,                  # Regularization parameter
                    gamma='scale',          # Kernel coefficient
                    random_state=42,        # For reproducible results
                    probability=True        # Enable probability estimates
                ),
                "Extra Trees": ExtraTreesClassifier(
                    n_estimators=50,        # Number of trees (reduced for speed)
                    max_depth=10,           # Maximum depth of trees
                    min_samples_split=20,   # Minimum samples required to split
                    min_samples_leaf=10,    # Minimum samples in leaf node
                    random_state=42,        # For reproducible results
                    n_jobs=-1               # Use all available cores
                ),
                "AdaBoost": AdaBoostClassifier(
                    n_estimators=50,        # Number of boosting rounds
                    learning_rate=1.0,      # Learning rate
                    random_state=42         # For reproducible results
                ),
                "K-Nearest Neighbors": KNeighborsClassifier(
                    n_neighbors=5,          # Number of neighbors
                    weights='distance',     # Weight function
                    n_jobs=-1               # Use all available cores
                ),
                "Neural Network": MLPClassifier(
                    hidden_layer_sizes=(100, 50),  # Hidden layer sizes
                    max_iter=100,           # Maximum iterations (reduced for speed)
                    random_state=42,        # For reproducible results
                    early_stopping=True,    # Stop when validation score stops improving
                    validation_fraction=0.1 # Fraction of data for validation
                )
            }
            
            # Parameters for hyperparameter tuning (empty for speed - using defaults)
            params = {
                "Random Forest": {},        # Use default parameters
                "XGBoost": {},             # Use default parameters
                "Support Vector Machine": {},  # Use default parameters
                "Extra Trees": {},         # Use default parameters
                "AdaBoost": {},            # Use default parameters
                "K-Nearest Neighbors": {},  # Use default parameters
                "Neural Network": {}       # Use default parameters
            }
            
            # Filter models based on user selection
            if selected_models:
                models = {name: model for name, model in models.items() if name in selected_models}
                params = {name: param for name, param in params.items() if name in selected_models}
            
            # Train and evaluate each model
            model_report = {}
            
            for model_name, model in models.items():
                logging.info(f"Training {model_name} with default parameters")
                
                # Time the training process
                start_time = time.time()
                model.fit(X_train, y_train)        # Train the model
                training_time = time.time() - start_time
                
                # Evaluate model on test set
                test_score = model.score(X_test, y_test)
                logging.info(f"{model_name} - Default params used")
                logging.info(f"{model_name} - Test score: {test_score}")
                
                # Store model score
                model_report[model_name] = test_score
                
                # Extract hyperparameters for reporting
                hyperparams = {
                    param: getattr(model, param)
                    for param in ['n_estimators', 'max_depth', 'random_state']
                    if hasattr(model, param)
                }
                
                # Add model results to training report
                report_generator.add_model_result(
                    model_name=model_name,
                    model=model,
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    training_time=training_time,
                    hyperparams=hyperparams
                )
            
            # Find best model based on test scores
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            # Get the best model object
            best_model = models[best_model_name]
            
            # Log best model information
            logging.info(f"Best model: {best_model_name}")
            logging.info(f"Best model score: {best_model_score}")
            
            # Check if best model performance is acceptable
            if best_model_score < 0.6:
                raise CustomException("No best model found with acceptable performance", sys)
            
            # Get parameters for best model
            best_params = params[best_model_name]
            
            # Since we're using empty params (no grid search), use the model directly
            if not best_params:
                logging.info(f"Using {best_model_name} with default optimized parameters")
                best_model_final = best_model
                # Train the final model
                best_model_final.fit(X_train, y_train)
                # Get the actual parameters used by the model
                best_params_used = best_model_final.get_params()
            else:
                # If parameters are provided, perform grid search
                grid_search = GridSearchCV(
                    estimator=best_model,       # Model to tune
                    param_grid=best_params,     # Parameters to search
                    cv=3,                       # 3-fold cross validation
                    scoring='accuracy',         # Scoring metric
                    n_jobs=-1                   # Use all available cores
                )
                
                # Fit grid search
                grid_search.fit(X_train, y_train)
                best_model_final = grid_search.best_estimator_
                best_params_used = grid_search.best_params_
            
            # Make predictions on test set
            y_pred = best_model_final.predict(X_test)
            
            # Calculate final metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            # Log final performance
            logging.info(f"Final model accuracy: {accuracy}")
            logging.info(f"Final model F1 score: {f1}")
            
            # Save trained model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model_final
            )
            
            # Create detailed report
            detailed_report = {
                'model_name': best_model_name,
                'best_params': best_params_used,
                'best_score': best_model_score,
                'accuracy': accuracy,
                'f1_score': f1,
                'model_report': model_report,
                'classification_report': classification_report(y_test, y_pred)
            }
            
            # Save model report
            save_object(
                file_path=self.model_trainer_config.model_report_file_path,
                obj=detailed_report
            )
            
            # Finalize training report
            report_generator.set_best_model(best_model_name, best_model_score)
            report_generator.finish_training_session()
            
            # Save training report
            training_report_path = report_generator.save_report()
            if training_report_path:
                logging.info(f"Training report saved to {training_report_path}")
            
            return accuracy
            
        except Exception as e:
            # Log any errors that occur during training
            logging.error(f"Error in model training: {str(e)}")
            # Raise custom exception with system details
            raise CustomException(str(e), sys)
    
    def get_model_comparison(self, X_train, y_train, X_test, y_test):
        """
        Compare multiple models without extensive hyperparameter tuning
        
        Args:
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target
            
        Returns:
            Dictionary of model performances
        """
        try:
            # Define models for comparison
            models = {
                "Random Forest": RandomForestClassifier(random_state=42),
                "XGBoost": XGBClassifier(random_state=42, eval_metric='mlogloss')
            }
            
            results = {}
            
            # Train and evaluate each model
            for name, model in models.items():
                # Train model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='weighted')
                
                # Perform cross-validation
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
                
                # Store results
                results[name] = {
                    'accuracy': accuracy,
                    'f1_score': f1,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'classification_report': classification_report(y_test, y_pred)
                }
            
            return results
            
        except Exception as e:
            # Log any errors that occur during model comparison
            logging.error(f"Error in model comparison: {str(e)}")
            # Raise custom exception with system details
            raise CustomException(str(e), sys)
```

---

## Exception Handling

### `src/exception.py`

```python
import sys                          # For system-specific parameters and functions
from src.logger import logging      # Logging functionality

def error_message_detail(error, error_detail: sys):
    """
    Create detailed error message with file location and line number
    
    Args:
        error: The error object that was raised
        error_detail: System error details containing traceback information
        
    Returns:
        Formatted error message string with file and line information
    """
    # Extract traceback information from error details
    _, _, exc_tb = error_detail.exc_info()
    # Get the filename where the error occurred
    file_name = exc_tb.tb_frame.f_code.co_filename
    # Get the line number where the error occurred
    line_number = exc_tb.tb_lineno
    
    # Format comprehensive error message
    error_message = f"Error occurred in python script [{file_name}] line number [{line_number}] error message [{str(error)}]"
    
    return error_message

class CustomException(Exception):
    """
    Custom exception class that provides detailed error information
    including file location and line number for better debugging
    """
    
    def __init__(self, error_message, error_detail: sys):
        """
        Initialize custom exception with detailed error information
        
        Args:
            error_message: The original error message
            error_detail: System error details for extracting traceback
        """
        # Call parent class constructor with error message
        super().__init__(error_message)
        # Create detailed error message with file and line information
        self.error_message = error_message_detail(error_message, error_detail=error_detail)
    
    def __str__(self):
        """
        Return string representation of the exception
        
        Returns:
            Detailed error message with file and line information
        """
        return self.error_message
```

---

## Utility Functions

### `src/utils.py`

```python
import os                           # For file and directory operations
import sys                          # For system-specific parameters and functions
import pandas as pd                 # For data manipulation and analysis
import numpy as np                  # For numerical operations
import pickle                       # For serializing and deserializing objects
from sklearn.metrics import accuracy_score, f1_score  # Evaluation metrics
from sklearn.model_selection import GridSearchCV      # Grid search for hyperparameter tuning
from src.exception import CustomException              # Custom exception handling
from src.logger import logging                        # Logging functionality

def save_object(file_path, obj):
    """
    Save any Python object to a file using pickle serialization
    
    Args:
        file_path: Path where the object should be saved
        obj: Python object to be saved
    """
    try:
        # Get directory path from file path
        dir_path = os.path.dirname(file_path)
        # Create directory if it doesn't exist
        os.makedirs(dir_path, exist_ok=True)
        
        # Open file in binary write mode and save object
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
            
    except Exception as e:
        # Log any errors that occur during saving
        logging.error(f"Error saving object: {str(e)}")
        # Raise custom exception with system details
        raise CustomException(e, sys)

def load_object(file_path):
    """
    Load a Python object from a file using pickle deserialization
    
    Args:
        file_path: Path to the file containing the pickled object
        
    Returns:
        Deserialized Python object
    """
    try:
        # Open file in binary read mode and load object
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
            
    except Exception as e:
        # Log any errors that occur during loading
        logging.error(f"Error loading object: {str(e)}")
        # Raise custom exception with system details
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    """
    Evaluate multiple machine learning models with optional hyperparameter tuning
    
    Args:
        X_train: Training features
        y_train: Training target values
        X_test: Test features
        y_test: Test target values
        models: Dictionary of model objects {name: model}
        param: Dictionary of hyperparameters {name: params}
        
    Returns:
        Dictionary of model performance scores {name: score}
    """
    try:
        report = {}  # Initialize empty report dictionary
        
        # Iterate through each model
        for model_name, model in models.items():
            # Get hyperparameters for current model
            params = param.get(model_name, {})
            
            # Check if hyperparameters are provided
            if not params:
                # Train with default parameters if no hyperparameters provided
                logging.info(f"Training {model_name} with default parameters")
                
                # Fit model on training data
                model.fit(X_train, y_train)
                
                # Make predictions on test data
                y_pred = model.predict(X_test)
                
                # Calculate accuracy score
                test_score = accuracy_score(y_test, y_pred)
                
                # Store result in report
                report[model_name] = test_score
                
                # Log results
                logging.info(f"{model_name} - Default params used")
                logging.info(f"{model_name} - Test score: {test_score}")
            else:
                # Perform grid search with hyperparameter tuning
                gs = GridSearchCV(
                    estimator=model,        # Model to tune
                    param_grid=params,      # Parameters to search
                    cv=3,                   # 3-fold cross validation
                    scoring='accuracy',     # Scoring metric
                    n_jobs=-1,              # Use all available cores
                    verbose=1               # Show progress
                )
                
                # Fit grid search on training data
                gs.fit(X_train, y_train)
                
                # Get best model from grid search
                best_model = gs.best_estimator_
                
                # Make predictions on test data
                y_pred = best_model.predict(X_test)
                
                # Calculate accuracy score
                test_score = accuracy_score(y_test, y_pred)
                
                # Store result in report
                report[model_name] = test_score
                
                # Log results
                logging.info(f"{model_name} - Best params: {gs.best_params_}")
                logging.info(f"{model_name} - Test score: {test_score}")
        
        return report
        
    except Exception as e:
        # Log any errors that occur during evaluation
        logging.error(f"Error evaluating models: {str(e)}")
        # Raise custom exception with system details
        raise CustomException(e, sys)

def get_feature_importance(model, feature_names):
    """
    Extract feature importance from a trained model
    
    Args:
        model: Trained machine learning model
        feature_names: List of feature names
        
    Returns:
        DataFrame with features sorted by importance (or None if not available)
    """
    try:
        # Check if model has feature importance attribute
        if hasattr(model, 'feature_importances_'):
            # Create DataFrame with feature names and importance scores
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)  # Sort by importance
            
            return importance_df
        else:
            # Model doesn't support feature importance
            return None
            
    except Exception as e:
        # Log any errors that occur while getting feature importance
        logging.error(f"Error getting feature importance: {str(e)}")
        return None

def calculate_metrics(y_true, y_pred):
    """
    Calculate comprehensive classification metrics
    
    Args:
        y_true: True target values
        y_pred: Predicted target values
        
    Returns:
        Dictionary containing various evaluation metrics
    """
    try:
        # Import additional metrics
        from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
        from sklearn.metrics import confusion_matrix, classification_report
        
        # Calculate all metrics
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),                      # Overall accuracy
            'precision': precision_score(y_true, y_pred, average='weighted'), # Weighted precision
            'recall': recall_score(y_true, y_pred, average='weighted'),       # Weighted recall
            'f1_score': f1_score(y_true, y_pred, average='weighted'),         # Weighted F1 score
            'confusion_matrix': confusion_matrix(y_true, y_pred),             # Confusion matrix
            'classification_report': classification_report(y_true, y_pred)     # Detailed report
        }
        
        return metrics
        
    except Exception as e:
        # Log any errors that occur during metric calculation
        logging.error(f"Error calculating metrics: {str(e)}")
        # Raise custom exception with system details
        raise CustomException(e, sys)
```

---

## Training Pipeline

### `src/pipeline/train_pipeline.py`

```python
import sys                          # For system-specific parameters and functions
import os                           # For file and directory operations
from src.components.data_ingestion import DataIngestion      # Data loading component
from src.components.data_transformation import DataTransformation  # Data preprocessing component
from src.components.model_trainer import ModelTrainer       # Model training component
from src.exception import CustomException                    # Custom exception handling
from src.logger import logging                              # Logging functionality

class TrainPipeline:
    """
    Complete training pipeline that orchestrates the entire ML workflow
    from data ingestion to model training
    """
    
    def __init__(self):
        """Initialize pipeline with all required components"""
        self.data_ingestion = DataIngestion()           # Component for loading data
        self.data_transformation = DataTransformation() # Component for preprocessing data
        self.model_trainer = ModelTrainer()             # Component for training models
    
    def run_pipeline(self, data_path: str = None, selected_models: list = None):
        """
        Execute the complete training pipeline
        
        Args:
            data_path: Path to the dataset file (optional)
            selected_models: List of model names to train (optional, trains all if None)
            
        Returns:
            Model performance score (accuracy) of the best model
        """
        try:
            # Log the start of the pipeline
            logging.info("Starting training pipeline")
            
            # Step 1: Data ingestion - load and split data
            train_data_path, test_data_path = self.data_ingestion.initiate_data_ingestion(data_path)
            
            # Step 2: Data transformation - preprocess the data
            train_arr, test_arr, _ = self.data_transformation.initiate_data_transformation(
                train_data_path, test_data_path
            )
            
            # Step 3: Model training - train and evaluate models
            model_score = self.model_trainer.initiate_model_trainer(train_arr, test_arr, selected_models)
            
            # Log successful completion
            logging.info("Training pipeline completed successfully")
            return model_score
            
        except Exception as e:
            # Log any errors that occur during pipeline execution
            logging.error(f"Error in training pipeline: {str(e)}")
            # Raise custom exception with system details
            raise CustomException(e, sys)
    
    def get_pipeline_info(self):
        """
        Get information about the pipeline components
        
        Returns:
            Dictionary with descriptions of each pipeline stage
        """
        return {
            'data_ingestion': 'Loads and splits data into train/test sets',
            'data_transformation': 'Preprocesses data with scaling and encoding',
            'model_trainer': 'Trains and evaluates ML models with hyperparameter tuning'
        }
```

---

## Prediction Pipeline

### `src/pipeline/predict_pipeline.py`

```python
import sys                          # For system-specific parameters and functions
import os                           # For file and directory operations
import pandas as pd                 # For data manipulation and analysis
import numpy as np                  # For numerical operations
from src.exception import CustomException  # Custom exception handling
from src.logger import logging      # Logging functionality
from src.utils import load_object   # Utility function for loading saved objects

class PredictPipeline:
    """Pipeline for making predictions using trained models"""
    
    def __init__(self):
        """Initialize prediction pipeline"""
        pass
    
    def predict(self, features):
        """
        Make predictions using the trained model
        
        Args:
            features: Input features as DataFrame or array
            
        Returns:
            Array of predictions
        """
        try:
            # Define paths to saved model artifacts
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            
            # Load trained model from file
            model = load_object(file_path=model_path)
            # Load preprocessor from file
            preprocessor = load_object(file_path=preprocessor_path)
            
            # Transform input features using saved preprocessor
            data_scaled = preprocessor.transform(features)
            
            # Make predictions using trained model
            predictions = model.predict(data_scaled)
            
            # Load label encoder to convert numerical predictions back to original labels
            label_encoder_path = os.path.join("artifacts", "label_encoder.pkl")
            if os.path.exists(label_encoder_path):
                label_encoder = load_object(file_path=label_encoder_path)
                # Convert numerical predictions back to original string labels
                predictions = label_encoder.inverse_transform(predictions)
            
            return predictions
            
        except Exception as e:
            # Log any errors that occur during prediction
            logging.error(f"Error in prediction pipeline: {str(e)}")
            # Raise custom exception with system details
            raise CustomException(e, sys)
    
    def predict_proba(self, features):
        """
        Get prediction probabilities for each class
        
        Args:
            features: Input features as DataFrame or array
            
        Returns:
            Array of prediction probabilities
        """
        try:
            # Define paths to saved model artifacts
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            
            # Load trained model from file
            model = load_object(file_path=model_path)
            # Load preprocessor from file
            preprocessor = load_object(file_path=preprocessor_path)
            
            # Transform input features using saved preprocessor
            data_scaled = preprocessor.transform(features)
            
            # Get prediction probabilities using trained model
            probabilities = model.predict_proba(data_scaled)
            
            return probabilities
            
        except Exception as e:
            # Log any errors that occur while getting probabilities
            logging.error(f"Error getting prediction probabilities: {str(e)}")
            # Raise custom exception with system details
            raise CustomException(e, sys)

class CustomData:
    """Custom data class for creating prediction input with water pump features"""
    
    def __init__(self, **kwargs):
        """
        Initialize custom data object with water pump features
        
        Args:
            **kwargs: Feature values passed as keyword arguments
        """
        # Geographic features - location information
        self.longitude = kwargs.get('longitude', 0.0)           # GPS longitude coordinate
        self.latitude = kwargs.get('latitude', 0.0)             # GPS latitude coordinate
        self.region = kwargs.get('region', 'unknown')           # Administrative region
        self.district_code = kwargs.get('district_code', 0)     # District code number
        self.lga = kwargs.get('lga', 'unknown')                 # Local Government Area
        self.ward = kwargs.get('ward', 'unknown')               # Ward name
        
        # Technical features - pump specifications
        self.pump_type = kwargs.get('pump_type', 'unknown')                         # Type of water pump
        self.extraction_type = kwargs.get('extraction_type', 'unknown')             # Water extraction method
        self.extraction_type_group = kwargs.get('extraction_type_group', 'unknown') # Extraction type category
        self.extraction_type_class = kwargs.get('extraction_type_class', 'unknown') # Extraction type class
        self.management = kwargs.get('management', 'unknown')                       # Management type
        self.management_group = kwargs.get('management_group', 'unknown')           # Management category
        self.payment = kwargs.get('payment', 'unknown')                             # Payment method
        self.payment_type = kwargs.get('payment_type', 'unknown')                   # Payment type
        
        # Water features - quality and quantity information
        self.water_quality = kwargs.get('water_quality', 'unknown')         # Water quality level
        self.quality_group = kwargs.get('quality_group', 'unknown')         # Quality category
        self.quantity = kwargs.get('quantity', 'unknown')                   # Water quantity available
        self.quantity_group = kwargs.get('quantity_group', 'unknown')       # Quantity category
        self.source = kwargs.get('source', 'unknown')                       # Water source
        self.source_type = kwargs.get('source_type', 'unknown')             # Source type
        self.source_class = kwargs.get('source_class', 'unknown')           # Source classification
        self.waterpoint_type = kwargs.get('waterpoint_type', 'unknown')     # Type of water point
        self.waterpoint_type_group = kwargs.get('waterpoint_type_group', 'unknown') # Water point category
        
        # Installation features - construction and funding information
        self.installer = kwargs.get('installer', 'unknown')         # Who installed the pump
        self.funder = kwargs.get('funder', 'unknown')               # Who funded the pump
        self.construction_year = kwargs.get('construction_year', 0)  # Year of construction
        self.population = kwargs.get('population', 0)               # Population served
        self.gps_height = kwargs.get('gps_height', 0)               # GPS elevation
        
        # Additional features - other relevant information
        self.num_private = kwargs.get('num_private', 0)                         # Number of private connections
        self.basin = kwargs.get('basin', 'unknown')                             # Water basin
        self.subvillage = kwargs.get('subvillage', 'unknown')                   # Sub-village name
        self.region_code = kwargs.get('region_code', 0)                         # Region code number
        self.lga_code = kwargs.get('lga_code', 0)                               # LGA code number
        self.ward_code = kwargs.get('ward_code', 0)                             # Ward code number
        self.public_meeting = kwargs.get('public_meeting', 'unknown')           # Public meeting held
        self.scheme_management = kwargs.get('scheme_management', 'unknown')     # Scheme management type
        self.scheme_name = kwargs.get('scheme_name', 'unknown')                 # Scheme name
        self.permit = kwargs.get('permit', 'unknown')                           # Permit status
        self.recorded_by = kwargs.get('recorded_by', 'unknown')                 # Who recorded the data
    
    def get_data_as_data_frame(self):
        """
        Convert custom data object to pandas DataFrame for model input
        
        Returns:
            DataFrame with single row containing all features
        """
        try:
            # Create dictionary with all feature values in list format (for DataFrame creation)
            custom_data_input_dict = {
                'longitude': [self.longitude],
                'latitude': [self.latitude],
                'region': [self.region],
                'district_code': [self.district_code],
                'lga': [self.lga],
                'ward': [self.ward],
                'pump_type': [self.pump_type],
                'extraction_type': [self.extraction_type],
                'extraction_type_group': [self.extraction_type_group],
                'extraction_type_class': [self.extraction_type_class],
                'management': [self.management],
                'management_group': [self.management_group],
                'payment': [self.payment],
                'payment_type': [self.payment_type],
                'water_quality': [self.water_quality],
                'quality_group': [self.quality_group],
                'quantity': [self.quantity],
                'quantity_group': [self.quantity_group],
                'source': [self.source],
                'source_type': [self.source_type],
                'source_class': [self.source_class],
                'waterpoint_type': [self.waterpoint_type],
                'waterpoint_type_group': [self.waterpoint_type_group],
                'installer': [self.installer],
                'funder': [self.funder],
                'construction_year': [self.construction_year],
                'population': [self.population],
                'gps_height': [self.gps_height],
                'num_private': [self.num_private],
                'basin': [self.basin],
                'subvillage': [self.subvillage],
                'region_code': [self.region_code],
                'lga_code': [self.lga_code],
                'ward_code': [self.ward_code],
                'public_meeting': [self.public_meeting],
                'scheme_management': [self.scheme_management],
                'scheme_name': [self.scheme_name],
                'permit': [self.permit],
                'recorded_by': [self.recorded_by]
            }
            
            # Create and return DataFrame from dictionary
            return pd.DataFrame(custom_data_input_dict)
            
        except Exception as e:
            # Log any errors that occur during DataFrame creation
            logging.error(f"Error creating DataFrame: {str(e)}")
            # Raise custom exception with system details
            raise CustomException(e, sys)
```

---

## Streamlit Pages

### Main Application - `app.py`

```python
import streamlit as st              # Streamlit framework for web applications
import pandas as pd                 # For data manipulation and analysis
import numpy as np                  # For numerical operations
from src.logger import logging      # Logging functionality
from src.exception import CustomException  # Custom exception handling
import sys                          # For system-specific parameters and functions
import os                           # For file and directory operations

# Configure Streamlit page settings
st.set_page_config(
    page_title="Water Pump Functionality Prediction",  # Browser tab title
    page_icon="💧",                                     # Browser tab icon
    layout="wide",                                      # Use full width of browser
    initial_sidebar_state="expanded"                    # Start with sidebar open
)

def main():
    """Main application function that creates the homepage"""
    try:
        # Create main title and description
        st.title("💧 Water Pump Functionality Prediction")
        st.markdown("""
        ## Tanzania Water Pump Status Prediction System
        
        This application predicts the functionality status of water pumps in Tanzania using machine learning.
        The system can classify pumps into three categories:
        - **Functional**: Working properly
        - **Functional needs repair**: Working but requires maintenance
        - **Non-functional**: Not working
        
        ### Features:
        - **Exploratory Data Analysis**: Comprehensive data exploration and visualization
        - **Data Preprocessing**: Automated data cleaning and feature engineering
        - **Model Training**: Train and compare multiple ML models with hyperparameter tuning
        - **Model Evaluation**: Detailed performance metrics and comparisons
        - **Predictions**: Make predictions on new data
        - **Geospatial Analysis**: Interactive maps and location-based insights
        
        ### Navigation:
        Use the sidebar to navigate between different sections of the application.
        """)
        
        # Create sidebar for navigation
        st.sidebar.title("Navigation")
        st.sidebar.markdown("Select a page from the dropdown or use the pages above.")
        
        # Create dataset information section
        st.header("📊 Dataset Information")
        col1, col2, col3 = st.columns(3)  # Create three columns
        
        # Display dataset metrics in columns
        with col1:
            st.metric("Target Classes", "3")                               # Number of target classes
            st.caption("Functional, Needs Repair, Non-functional")        # Description
        
        with col2:
            st.metric("Features", "40+")                                   # Number of features
            st.caption("Geographic, Technical, Management data")          # Description
        
        with col3:
            st.metric("Data Source", "DrivenData")                        # Data source
            st.caption("Tanzania Ministry of Water")                      # Description
        
        # Create quick start guide section
        st.header("🚀 Quick Start Guide")
        st.markdown("""
        1. **Upload Data**: Go to the EDA page to upload your dataset
        2. **Explore**: Use the EDA tools to understand your data
        3. **Preprocess**: Clean and prepare your data for modeling
        4. **Train**: Build and train machine learning models
        5. **Evaluate**: Compare model performance
        6. **Predict**: Make predictions on new data
        7. **Analyze**: Explore geospatial patterns
        """)
        
        # Create data upload section
        st.header("📁 Data Upload")
        
        # Multiple file upload option
        st.subheader("Option 1: Upload Multiple Files (Recommended)")
        st.markdown("Upload separate files for training features, training labels, and test features:")
        
        # File uploader widget for multiple files
        uploaded_files = st.file_uploader(
            "Upload training features, training labels, and test features CSV files",
            type=['csv'],                    # Accept only CSV files
            accept_multiple_files=True,      # Allow multiple file selection
            help="Upload 3 CSV files: training features (X), training labels (y), and test features"
        )
        
        # Process uploaded files if exactly 3 files are uploaded
        if uploaded_files and len(uploaded_files) == 3:
            if st.button("Process Uploaded Files", type="primary"):  # Primary button (highlighted)
                try:
                    # Save uploaded files temporarily
                    temp_paths = []
                    for i, uploaded_file in enumerate(uploaded_files):
                        # Create temporary file path
                        temp_path = f"temp_upload_{i}_{uploaded_file.name}"
                        # Write uploaded file content to temporary file
                        with open(temp_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        temp_paths.append(temp_path)
                    
                    # Import and initialize file handler
                    from src.components.file_handler import FileHandler
                    file_handler = FileHandler()
                    
                    # Process files with loading spinner
                    with st.spinner("Processing uploaded files..."):
                        results = file_handler.process_uploaded_files(temp_paths)
                    
                    # Check if processing was successful
                    if results['success']:
                        st.success("Files processed successfully!")
                        
                        # Store results in session state for other pages to access
                        st.session_state['file_processing_results'] = results
                        st.session_state['data_uploaded'] = True
                        st.session_state['train_data_path'] = results['saved_files']['merged_train']
                        if 'test' in results['saved_files']:
                            st.session_state['test_data_path'] = results['saved_files']['test']
                        
                        # Display processing summary
                        st.subheader("Processing Summary")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            # Display number of training rows
                            st.metric("Training Data", f"{results['summary']['train_data_info']['shape'][0]:,} rows")
                        
                        with col2:
                            # Calculate and display total number of features
                            num_features = len(results['summary']['train_data_info']['numerical_columns']) + len(results['summary']['train_data_info']['categorical_columns'])
                            st.metric("Features", f"{num_features}")
                        
                        with col3:
                            # Display test data information if available
                            if 'test_data_info' in results['summary']:
                                st.metric("Test Data", f"{results['summary']['test_data_info']['shape'][0]:,} rows")
                            else:
                                st.metric("Test Data", "Not available")
                        
                        # Show identified file types
                        st.subheader("Identified Files")
                        for file_type, path in results['identified_files'].items():
                            # Display file type and filename
                            st.write(f"**{file_type.replace('_', ' ').title()}**: {os.path.basename(path)}")
                        
                        # Show target distribution if available
                        if 'target_distribution' in results['summary']['train_data_info']:
                            st.subheader("Target Distribution")
                            target_dist = results['summary']['train_data_info']['target_distribution']
                            for status, count in target_dist.items():
                                # Calculate and display percentage
                                percentage = count/sum(target_dist.values())*100
                                st.write(f"**{status}**: {count:,} ({percentage:.1f}%)")
                    
                    else:
                        # Display error if processing failed
                        st.error(f"Error processing files: {results['error']}")
                    
                    # Clean up temporary files
                    for temp_path in temp_paths:
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                            
                except Exception as e:
                    # Handle any errors during file processing
                    st.error(f"Error processing files: {str(e)}")
        
        # Show warning if wrong number of files uploaded
        elif uploaded_files and len(uploaded_files) != 3:
            st.warning(f"Please upload exactly 3 CSV files. You uploaded {len(uploaded_files)} files.")
        
        st.markdown("---")  # Add horizontal line separator
        
        # Single file upload option
        st.subheader("Option 2: Upload Single Combined File")
        uploaded_file = st.file_uploader(
            "Upload a single CSV file with both features and target variable",
            type=['csv'],                    # Accept only CSV files
            key="single_file",              # Unique key for this widget
            help="Upload a CSV file containing water pump data with features and target variable"
        )
        
        # Process single uploaded file
        if uploaded_file is not None:
            try:
                # Create artifacts directory if it doesn't exist
                os.makedirs("artifacts", exist_ok=True)
                file_path = os.path.join("artifacts", "uploaded_data.csv")
                
                # Read uploaded file into DataFrame
                df = pd.read_csv(uploaded_file)
                # Save DataFrame to CSV file
                df.to_csv(file_path, index=False)
                
                # Display success message with file information
                st.success(f"✅ File uploaded successfully! {df.shape[0]} rows, {df.shape[1]} columns")
                
                # Create two columns for displaying information
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Dataset Preview")
                    st.dataframe(df.head())  # Show first 5 rows
                
                with col2:
                    st.subheader("Dataset Info")
                    st.write(f"**Shape:** {df.shape}")                                    # Dimensions
                    st.write(f"**Memory Usage:** {df.memory_usage().sum() / 1024:.2f} KB") # Memory usage
                    st.write(f"**Missing Values:** {df.isnull().sum().sum()}")            # Total missing values
                
                # Store file information in session state
                st.session_state.data_path = file_path
                st.session_state.data_uploaded = True
                
            except Exception as e:
                # Handle errors during file reading
                st.error(f"Error reading file: {str(e)}")
                logging.error(f"Error reading uploaded file: {str(e)}")
        
        # Real Tanzania dataset option
        st.header("🎯 Tanzania Water Pump Dataset")
        if st.button("Load Real Tanzania Dataset", type="primary"):
            try:
                # Define paths to processed data files
                train_data_path = "artifacts/merged_train_data.csv"
                test_data_path = "artifacts/test_data.csv"
                
                # Check if processed data exists
                if os.path.exists(train_data_path):
                    # Store data paths in session state
                    st.session_state['data_uploaded'] = True
                    st.session_state['train_data_path'] = train_data_path
                    st.session_state['test_data_path'] = test_data_path
                    
                    # Load sample data for display (only first 1000 rows for performance)
                    df = pd.read_csv(train_data_path, nrows=1000)
                    df_full_info = pd.read_csv(train_data_path, nrows=0)  # Get column info only
                    
                    # Count total rows without loading full dataset
                    with open(train_data_path, 'r') as f:
                        row_count = sum(1 for line in f) - 1  # Subtract header row
                    
                    st.success("✅ Tanzania water pump dataset loaded successfully!")
                    
                    # Display dataset summary metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Training Samples", f"{row_count:,}")
                    with col2:
                        st.metric("Features", f"{len(df_full_info.columns)-1}")  # Exclude target column
                    with col3:
                        # Count test data rows if test file exists
                        if os.path.exists(test_data_path):
                            with open(test_data_path, 'r') as f:
                                test_row_count = sum(1 for line in f) - 1
                            st.metric("Test Samples", f"{test_row_count:,}")
                    
                    # Show target distribution using sample data
                    if 'status_group' in df.columns:
                        st.subheader("Target Distribution (Sample)")
                        target_counts = df['status_group'].value_counts()
                        for status, count in target_counts.items():
                            # Calculate percentage from sample
                            percentage = (count / len(df)) * 100
                            st.write(f"**{status}**: {count:,} ({percentage:.1f}% of sample)")
                    
                    st.rerun()  # Refresh the page to update display
                else:
                    # Show warning if data not found
                    st.warning("Real dataset not found. Please process the uploaded files first using the upload option above.")
            except Exception as e:
                # Handle errors during dataset loading
                st.error(f"Error loading real dataset: {str(e)}")
        
        # Create footer section
        st.markdown("---")  # Horizontal line
        st.markdown("**Built with Streamlit** | **Data Source:** DrivenData Competition")
        
    except Exception as e:
        # Handle any unexpected errors in the main function
        st.error(f"An error occurred: {str(e)}")
        logging.error(f"Error in main app: {str(e)}")

# Run main function if script is executed directly
if __name__ == "__main__":
    main()
```

---

## Streamlit EDA Page

### `pages/01_🔍_EDA.py`

```python
import streamlit as st              # Streamlit framework for web applications
import pandas as pd                 # For data manipulation and analysis
import numpy as np                  # For numerical operations
import matplotlib.pyplot as plt     # For creating static plots
import seaborn as sns               # For statistical visualizations
import plotly.express as px         # For interactive visualizations
import plotly.graph_objects as go   # For advanced interactive plots
from plotly.subplots import make_subplots  # For subplot creation
import folium                       # For creating interactive maps
from streamlit_folium import st_folium      # Streamlit integration for folium
from src.components.advanced_eda import AdvancedEDA  # Advanced EDA component
from src.logger import logging      # Logging functionality
from src.exception import CustomException  # Custom exception handling
import sys                          # For system-specific parameters and functions
import os                           # For file and directory operations

# Configure Streamlit page settings for EDA
st.set_page_config(page_title="EDA", page_icon="🔍", layout="wide")

def load_full_dataset():
    """Load the complete merged training dataset"""
    try:
        # Check if processed data file exists
        if os.path.exists("artifacts/merged_train_data.csv"):
            df = pd.read_csv("artifacts/merged_train_data.csv")  # Load full dataset
            # Display success message with dataset dimensions
            st.success(f"Loaded complete dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
            return df
        else:
            # Show warning if no processed data found
            st.warning("No processed data found. Please upload and process data on the main page first.")
            return None
    except Exception as e:
        # Handle errors during data loading
        st.error(f"Error loading dataset: {str(e)}")
        return None

def display_comprehensive_overview(df):
    """Display comprehensive dataset overview with key metrics"""
    st.header("📊 Complete Dataset Overview")
    
    # Create four columns for key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Display total number of records with comma formatting
        st.metric("Total Records", f"{df.shape[0]:,}")
    
    with col2:
        # Display total number of features
        st.metric("Total Features", df.shape[1])
    
    with col3:
        # Calculate and display total missing values
        st.metric("Missing Values", f"{df.isnull().sum().sum():,}")
    
    with col4:
        # Calculate memory usage in MB and display
        st.metric("Memory Usage", f"{df.memory_usage().sum() / 1024 / 1024:.1f} MB")
    
    # Display target variable distribution if present
    if 'status_group' in df.columns:
        st.subheader("🎯 Target Variable Distribution")
        target_counts = df['status_group'].value_counts()        # Count each class
        target_pct = df['status_group'].value_counts(normalize=True) * 100  # Calculate percentages
        
        # Create two columns for visualization and details
        col1, col2 = st.columns(2)
        
        with col1:
            # Create interactive pie chart for target distribution
            fig = px.pie(values=target_counts.values, names=target_counts.index, 
                        title="Water Pump Status Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Detailed Counts")
            # Display count and percentage for each target class
            for status, count in target_counts.items():
                pct = target_pct[status]
                st.write(f"**{status}**: {count:,} ({pct:.1f}%)")

def display_missing_values_analysis(df):
    """Display comprehensive missing values analysis"""
    st.header("🔍 Missing Values Analysis")
    
    # Calculate missing values statistics
    missing_data = df.isnull().sum()                        # Count missing values per column
    missing_percentage = (missing_data / len(df)) * 100     # Calculate missing percentages
    
    # Create DataFrame with missing values information
    missing_df = pd.DataFrame({
        'Column': missing_data.index,
        'Missing Count': missing_data.values,
        'Missing Percentage': missing_percentage.values
    })
    # Filter to only columns with missing values and sort by count
    missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
    
    if len(missing_df) > 0:
        # Create two columns for visualization and table
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Missing Values by Column")
            # Create horizontal bar chart of missing percentages
            fig = px.bar(missing_df, x='Missing Percentage', y='Column', 
                        title="Missing Values Percentage by Column",
                        orientation='h')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Missing Values Table")
            # Display detailed table of missing values
            st.dataframe(missing_df, use_container_width=True)
        
        # Categorize columns by missing percentage thresholds
        high_missing = missing_df[missing_df['Missing Percentage'] > 50]['Column'].tolist()
        medium_missing = missing_df[(missing_df['Missing Percentage'] > 10) & 
                                  (missing_df['Missing Percentage'] <= 50)]['Column'].tolist()
        low_missing = missing_df[(missing_df['Missing Percentage'] > 0) & 
                               (missing_df['Missing Percentage'] <= 10)]['Column'].tolist()
        
        st.subheader("Missing Values Categories")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**High Missing (>50%)**")
            for col in high_missing:
                st.write(f"- {col}")
        
        with col2:
            st.write("**Medium Missing (10-50%)**")
            for col in medium_missing:
                st.write(f"- {col}")
        
        with col3:
            st.write("**Low Missing (<10%)**")
            for col in low_missing:
                st.write(f"- {col}")
    else:
        # Display success message if no missing values found
        st.success("No missing values found in the dataset!")

def display_data_types_analysis(df):
    """Display comprehensive data types analysis"""
    st.header("🔢 Data Types Analysis")
    
    # Identify numerical and categorical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Remove ID and target columns from feature lists
    if 'id' in numerical_cols:
        numerical_cols.remove('id')
    if 'id' in categorical_cols:
        categorical_cols.remove('id')
    if 'status_group' in numerical_cols:
        numerical_cols.remove('status_group')
    if 'status_group' in categorical_cols:
        categorical_cols.remove('status_group')
    
    # Display feature types in two columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"Numerical Features ({len(numerical_cols)})")
        for col in numerical_cols:
            st.write(f"- {col}")
    
    with col2:
        st.subheader(f"Categorical Features ({len(categorical_cols)})")
        for col in categorical_cols:
            st.write(f"- {col}")
    
    # Display statistical summary for numerical columns
    if numerical_cols:
        st.subheader("📈 Numerical Features Statistics")
        st.dataframe(df[numerical_cols].describe(), use_container_width=True)
    
    # Display unique value counts for categorical columns
    if categorical_cols:
        st.subheader("📊 Categorical Features Unique Values")
        cat_unique = pd.DataFrame({
            'Column': categorical_cols,
            'Unique Values': [df[col].nunique() for col in categorical_cols],
            'Most Frequent': [str(df[col].mode().iloc[0]) if len(df[col].mode()) > 0 else 'N/A' for col in categorical_cols]
        })
        st.dataframe(cat_unique, use_container_width=True)
```

---

## Streamlit Predictions Page

### `pages/05_🎯_Predictions.py`

```python
import streamlit as st              # Streamlit framework for web applications
import pandas as pd                 # For data manipulation and analysis
import numpy as np                  # For numerical operations
import matplotlib.pyplot as plt     # For creating static plots
import seaborn as sns               # For statistical visualizations
import plotly.express as px         # For interactive visualizations
import plotly.graph_objects as go   # For advanced interactive plots
from src.pipeline.predict_pipeline import PredictPipeline, CustomData  # Prediction pipeline
from src.utils import load_object   # Utility function for loading saved objects
from src.logger import logging      # Logging functionality
from src.exception import CustomException  # Custom exception handling
import os                           # For file and directory operations
import sys                          # For system-specific parameters and functions

# Configure Streamlit page settings for predictions
st.set_page_config(page_title="Predictions", page_icon="🎯", layout="wide")

def check_model_availability():
    """Check if trained model and preprocessor are available"""
    model_path = "artifacts/model.pkl"                  # Path to trained model
    preprocessor_path = "artifacts/preprocessor.pkl"    # Path to preprocessor
    
    # Check if both files exist
    model_available = os.path.exists(model_path)
    preprocessor_available = os.path.exists(preprocessor_path)
    
    # Return True only if both components are available
    return model_available and preprocessor_available

def load_sample_data_for_prediction():
    """Load sample data for prediction examples"""
    try:
        sample_path = "sample_data/sample_water_pumps.csv"
        if os.path.exists(sample_path):
            df = pd.read_csv(sample_path)          # Load sample data
            # Remove target column if present (for prediction input)
            if 'status_group' in df.columns:
                df = df.drop('status_group', axis=1)
            return df
        else:
            return None
    except Exception as e:
        st.error(f"Error loading sample data: {str(e)}")
        return None

def display_single_prediction_form():
    """Display interactive form for single water pump prediction"""
    st.subheader("🎯 Single Prediction")
    
    # Create form for user input
    with st.form("single_prediction_form"):
        st.write("Enter water pump information:")
        
        # Create three columns for organized input layout
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Geographic information inputs
            st.write("**Geographic Information**")
            longitude = st.number_input("Longitude", value=0.0, format="%.6f")      # GPS longitude
            latitude = st.number_input("Latitude", value=0.0, format="%.6f")       # GPS latitude
            region = st.selectbox("Region", [                                       # Administrative region
                "Dodoma", "Arusha", "Kilimanjaro", "Tanga", "Morogoro",
                "Pwani", "Lindi", "Mtwara", "Ruvuma", "Iringa", "Mbeya",
                "Singida", "Tabora", "Rukwa", "Kigoma", "Shinyanga",
                "Kagera", "Mwanza", "Mara", "Manyara", "Dar es Salaam"
            ])
            gps_height = st.number_input("GPS Height", value=0)                     # Elevation above sea level
            
        with col2:
            # Technical specifications inputs
            st.write("**Technical Specifications**")
            extraction_type = st.selectbox("Extraction Type", [                    # Water extraction method
                "gravity", "handpump", "submersible", "motorpump", "rope pump",
                "wind-powered", "other"
            ])
            pump_type = st.selectbox("Pump Type", [                                # Type of pump mechanism
                "afridev", "india mark ii", "india mark iii", "ksb", "other"
            ])
            waterpoint_type = st.selectbox("Waterpoint Type", [                    # Type of water access point
                "communal standpipe", "hand pump", "improved spring",
                "cattle trough", "dam", "other"
            ])
            
        with col3:
            # Management and payment information inputs
            st.write("**Management & Payment**")
            management = st.selectbox("Management", [                              # Who manages the pump
                "vwc", "wug", "water authority", "school", "parastatal",
                "private operator", "other"
            ])
            payment = st.selectbox("Payment", [                                    # Payment method for water
                "pay per bucket", "pay monthly", "pay annually", "never pay", "other"
            ])
            water_quality = st.selectbox("Water Quality", [                        # Quality of water
                "soft", "salty", "milky", "colored", "fluoride", "unknown"
            ])
            quantity = st.selectbox("Quantity", [                                  # Amount of water available
                "enough", "insufficient", "dry", "seasonal", "unknown"
            ])
        
        # Additional information section
        st.write("**Additional Information**")
        col4, col5, col6 = st.columns(3)
        
        with col4:
            population = st.number_input("Population Served", value=0, min_value=0)  # People served by pump
            construction_year = st.number_input("Construction Year", value=2000, min_value=1900, max_value=2025)
            
        with col5:
            installer = st.text_input("Installer", value="unknown")                 # Who installed the pump
            funder = st.text_input("Funder", value="unknown")                      # Who funded the pump
            
        with col6:
            source = st.selectbox("Water Source", [                                # Source of water
                "spring", "shallow well", "machine dbh", "hand dtw", "other"
            ])
            basin = st.selectbox("Basin", [                                        # Water basin
                "Lake Victoria", "Pangani", "Internal", "Ruvuma / Southern Coast", 
                "Rufiji", "Lake Tanganyika", "Lake Nyasa", "Wami / Ruvu", "Lake Rukwa"
            ])
        
        # Submit button for form
        submitted = st.form_submit_button("🔮 Predict Pump Status", type="primary")
        
        # Process prediction when form is submitted
        if submitted:
            try:
                # Create custom data object with form inputs
                custom_data = CustomData(
                    longitude=longitude,
                    latitude=latitude,
                    region=region,
                    gps_height=gps_height,
                    extraction_type=extraction_type,
                    pump_type=pump_type,
                    waterpoint_type=waterpoint_type,
                    management=management,
                    payment=payment,
                    water_quality=water_quality,
                    quantity=quantity,
                    population=population,
                    construction_year=construction_year,
                    installer=installer,
                    funder=funder,
                    source=source,
                    basin=basin
                )
                
                # Convert to DataFrame for prediction
                pred_df = custom_data.get_data_as_data_frame()
                
                # Initialize prediction pipeline and make prediction
                predict_pipeline = PredictPipeline()
                results = predict_pipeline.predict(pred_df)
                
                # Display prediction results
                st.success("🎯 Prediction Complete!")
                
                # Create columns for result display
                col1, col2 = st.columns(2)
                
                with col1:
                    # Display main prediction result
                    st.subheader("Prediction Result")
                    prediction = results[0]
                    
                    # Style result based on prediction
                    if prediction == "functional":
                        st.success(f"**Status**: {prediction.title()}")
                        st.info("✅ This water pump is predicted to be working properly.")
                    elif prediction == "functional needs repair":
                        st.warning(f"**Status**: {prediction.title()}")
                        st.info("⚠️ This water pump is working but needs maintenance.")
                    else:
                        st.error(f"**Status**: {prediction.title()}")
                        st.info("❌ This water pump is predicted to be non-functional.")
                
                with col2:
                    # Display prediction confidence if available
                    st.subheader("Prediction Details")
                    st.write(f"**Predicted Class**: {prediction}")
                    st.write("**Model Used**: Best trained model from artifacts")
                    
                    # Try to get prediction probabilities
                    try:
                        probabilities = predict_pipeline.predict_proba(pred_df)
                        if probabilities is not None:
                            st.write("**Confidence Scores**:")
                            # Display probability for each class
                            prob_dict = {
                                "Functional": probabilities[0][0],
                                "Needs Repair": probabilities[0][1],
                                "Non-functional": probabilities[0][2]
                            }
                            for status, prob in prob_dict.items():
                                st.write(f"- {status}: {prob:.2%}")
                    except:
                        pass  # Skip if probabilities not available
                        
            except Exception as e:
                # Handle prediction errors
                st.error(f"Error making prediction: {str(e)}")
                logging.error(f"Error in single prediction: {str(e)}")

def display_batch_prediction():
    """Display interface for batch prediction from CSV file"""
    st.subheader("📊 Batch Prediction")
    st.write("Upload a CSV file to predict multiple water pumps at once.")
    
    # File uploader for batch prediction
    uploaded_file = st.file_uploader(
        "Choose CSV file for batch prediction",
        type=['csv'],
        help="Upload a CSV file containing water pump features (without target column)"
    )
    
    if uploaded_file is not None:
        try:
            # Read uploaded CSV file
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ File uploaded: {df.shape[0]} rows, {df.shape[1]} columns")
            
            # Display preview of uploaded data
            st.subheader("Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Button to start batch prediction
            if st.button("🔮 Predict All Pumps", type="primary"):
                with st.spinner("Making predictions..."):
                    try:
                        # Remove target column if present
                        if 'status_group' in df.columns:
                            df_features = df.drop('status_group', axis=1)
                        else:
                            df_features = df.copy()
                        
                        # Initialize prediction pipeline
                        predict_pipeline = PredictPipeline()
                        
                        # Make batch predictions
                        predictions = predict_pipeline.predict(df_features)
                        
                        # Add predictions to original DataFrame
                        df_with_predictions = df.copy()
                        df_with_predictions['predicted_status'] = predictions
                        
                        # Display results summary
                        st.success("🎯 Batch Prediction Complete!")
                        
                        # Create summary statistics
                        prediction_counts = pd.Series(predictions).value_counts()
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("Prediction Summary")
                            for status, count in prediction_counts.items():
                                percentage = (count / len(predictions)) * 100
                                st.write(f"**{status.title()}**: {count} ({percentage:.1f}%)")
                        
                        with col2:
                            # Create pie chart of predictions
                            fig = px.pie(values=prediction_counts.values, 
                                       names=prediction_counts.index,
                                       title="Predicted Status Distribution")
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Display results table
                        st.subheader("Detailed Results")
                        st.dataframe(df_with_predictions, use_container_width=True)
                        
                        # Download button for results
                        csv = df_with_predictions.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Predictions as CSV",
                            data=csv,
                            file_name="water_pump_predictions.csv",
                            mime="text/csv"
                        )
                        
                    except Exception as e:
                        st.error(f"Error during batch prediction: {str(e)}")
                        logging.error(f"Error in batch prediction: {str(e)}")
                        
        except Exception as e:
            st.error(f"Error reading uploaded file: {str(e)}")

def display_prediction_examples():
    """Display prediction examples using sample data"""
    st.subheader("💡 Prediction Examples")
    
    # Load sample data for examples
    sample_df = load_sample_data_for_prediction()
    
    if sample_df is not None:
        st.write("Here are some example predictions using sample data:")
        
        # Select random samples for examples
        if len(sample_df) > 5:
            examples = sample_df.sample(n=5, random_state=42)
        else:
            examples = sample_df.copy()
        
        # Make predictions for examples
        try:
            predict_pipeline = PredictPipeline()
            example_predictions = predict_pipeline.predict(examples)
            
            # Display examples with predictions
            for i, (idx, row) in enumerate(examples.iterrows()):
                with st.expander(f"Example {i+1}: {example_predictions[i].title()}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Key Features:**")
                        st.write(f"- Region: {row.get('region', 'N/A')}")
                        st.write(f"- Extraction Type: {row.get('extraction_type', 'N/A')}")
                        st.write(f"- Water Quality: {row.get('water_quality', 'N/A')}")
                        st.write(f"- Management: {row.get('management', 'N/A')}")
                    
                    with col2:
                        st.write("**Prediction:**")
                        prediction = example_predictions[i]
                        if prediction == "functional":
                            st.success(f"✅ {prediction.title()}")
                        elif prediction == "functional needs repair":
                            st.warning(f"⚠️ {prediction.title()}")
                        else:
                            st.error(f"❌ {prediction.title()}")
                            
        except Exception as e:
            st.error(f"Error generating examples: {str(e)}")
    else:
        st.info("No sample data available for examples.")

def display_model_info():
    """Display information about the trained model"""
    st.subheader("🤖 Model Information")
    
    try:
        # Load model report if available
        report_path = "artifacts/model_report.pkl"
        if os.path.exists(report_path):
            report = load_object(report_path)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Model Details:**")
                st.write(f"- Best Model: {report.get('model_name', 'N/A')}")
                st.write(f"- Accuracy: {report.get('accuracy', 'N/A'):.3f}")
                st.write(f"- F1 Score: {report.get('f1_score', 'N/A'):.3f}")
            
            with col2:
                st.write("**Training Information:**")
                st.write("- Features: Geographic, Technical, Management data")
                st.write("- Target Classes: 3 (Functional, Needs Repair, Non-functional)")
                st.write("- Data Source: Tanzania Ministry of Water")
        else:
            st.info("Model report not available. Train a model first.")
            
    except Exception as e:
        st.error(f"Error loading model information: {str(e)}")

def main():
    """Main function for predictions page"""
    st.title("🎯 Water Pump Predictions")
    st.markdown("""
    Make predictions about water pump functionality using the trained machine learning model.
    You can predict for a single pump or upload a CSV file for batch predictions.
    """)
    
    # Check if model is available
    if not check_model_availability():
        st.error("🚫 **Model not available!**")
        st.markdown("""
        No trained model found. Please:
        1. Go to the **Model Training** page
        2. Train a model with your data
        3. Return here to make predictions
        """)
        return
    
    st.success("✅ **Model loaded successfully!** Ready for predictions.")
    
    # Create tabs for different prediction modes
    tab1, tab2, tab3, tab4 = st.tabs(["Single Prediction", "Batch Prediction", "Examples", "Model Info"])
    
    with tab1:
        display_single_prediction_form()
    
    with tab2:
        display_batch_prediction()
    
    with tab3:
        display_prediction_examples()
    
    with tab4:
        display_model_info()

# Run main function if script is executed directly
if __name__ == "__main__":
    main()
```

---

## Prediction Examples Utility

### `prediction_examples.py`

```python
#!/usr/bin/env python3
"""
Practical examples for using the trained water pump model
This script demonstrates various ways to use the prediction system
"""

import pandas as pd                 # For data manipulation and analysis
import numpy as np                  # For numerical operations
from src.utils import load_object   # Utility function for loading saved objects
import json                         # For JSON operations

class WaterPumpPredictor:
    """Water pump functionality predictor with easy-to-use interface"""
    
    def __init__(self):
        """Initialize predictor with trained model components"""
        self.model = None               # Placeholder for trained model
        self.preprocessor = None        # Placeholder for data preprocessor
        self.label_encoder = None       # Placeholder for label encoder
        self.load_components()          # Load all components on initialization
    
    def load_components(self):
        """Load all model components from saved files"""
        try:
            # Load trained model from pickle file
            self.model = load_object("artifacts/model.pkl")
            # Load preprocessor (scaler and encoder) from pickle file
            self.preprocessor = load_object("artifacts/preprocessor.pkl")
            # Load label encoder for converting predictions back to text
            self.label_encoder = load_object("artifacts/label_encoder.pkl")
            print("✅ All model components loaded successfully")
        except Exception as e:
            print(f"❌ Error loading components: {str(e)}")
    
    def predict_single(self, pump_data):
        """
        Predict functionality for a single water pump
        
        Args:
            pump_data: Dictionary containing pump features
            
        Returns:
            Dictionary with prediction, confidence, and probabilities
        """
        try:
            # Convert input dictionary to DataFrame (required for preprocessing)
            df = pd.DataFrame([pump_data])
            
            # Remove ID column if present (not needed for prediction)
            if 'id' in df.columns:
                df = df.drop('id', axis=1)
            
            # Transform features using trained preprocessor
            features_transformed = self.preprocessor.transform(df)
            
            # Make prediction using trained model
            prediction = self.model.predict(features_transformed)[0]     # Get first (only) prediction
            probabilities = self.model.predict_proba(features_transformed)[0]  # Get prediction probabilities
            
            # Convert numerical prediction back to text label
            predicted_label = self.label_encoder.inverse_transform([prediction])[0]
            confidence = max(probabilities)  # Highest probability is confidence score
            
            # Create dictionary of probabilities for all classes
            class_probs = {}
            for i, class_name in enumerate(self.label_encoder.classes_):
                class_probs[class_name] = probabilities[i]
            
            # Return comprehensive prediction results
            return {
                'prediction': predicted_label,      # Predicted class label
                'confidence': confidence,           # Confidence score (0-1)
                'probabilities': class_probs        # Probability for each class
            }
            
        except Exception as e:
            # Return error information if prediction fails
            return {'error': str(e)}
    
    def predict_batch(self, pumps_data):
        """
        Predict functionality for multiple water pumps
        
        Args:
            pumps_data: List of dictionaries or DataFrame containing pump data
            
        Returns:
            List of prediction results for each pump
        """
        try:
            # Convert input to DataFrame if it's a list
            if isinstance(pumps_data, list):
                df = pd.DataFrame(pumps_data)
            else:
                df = pumps_data.copy()
            
            # Remove ID column if present
            if 'id' in df.columns:
                df = df.drop('id', axis=1)
            
            # Transform all features using trained preprocessor
            features_transformed = self.preprocessor.transform(df)
            
            # Make predictions for all pumps
            predictions = self.model.predict(features_transformed)
            probabilities = self.model.predict_proba(features_transformed)
            
            # Convert numerical predictions back to text labels
            predicted_labels = self.label_encoder.inverse_transform(predictions)
            
            # Create list of results for each pump
            results = []
            for i in range(len(predicted_labels)):
                # Get confidence (highest probability)
                confidence = max(probabilities[i])
                
                # Create probability dictionary for this prediction
                class_probs = {}
                for j, class_name in enumerate(self.label_encoder.classes_):
                    class_probs[class_name] = probabilities[i][j]
                
                # Add result for this pump
                results.append({
                    'prediction': predicted_labels[i],
                    'confidence': confidence,
                    'probabilities': class_probs
                })
            
            return results
            
        except Exception as e:
            # Return error information if batch prediction fails
            return [{'error': str(e)}]
    
    def get_feature_importance(self, top_n=10):
        """Get top N most important features from the trained model"""
        try:
            # Check if model has feature importance attribute
            if hasattr(self.model, 'feature_importances_'):
                # Get feature names from preprocessor (if available)
                try:
                    # This is a simplified approach - in practice, getting feature names
                    # from a fitted preprocessor can be complex
                    feature_names = [f"feature_{i}" for i in range(len(self.model.feature_importances_))]
                except:
                    feature_names = [f"feature_{i}" for i in range(len(self.model.feature_importances_))]
                
                # Create DataFrame with feature importance
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': self.model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                # Return top N features
                return importance_df.head(top_n)
            else:
                return None
                
        except Exception as e:
            print(f"Error getting feature importance: {str(e)}")
            return None

def example_single_prediction():
    """Example: Predict functionality for a single water pump"""
    print("\n🔍 Example 1: Single Water Pump Prediction")
    print("=" * 50)
    
    # Create predictor instance
    predictor = WaterPumpPredictor()
    
    # Example pump data (typical values for a water pump in Tanzania)
    pump_data = {
        'longitude': 34.5,                      # GPS longitude
        'latitude': -6.2,                       # GPS latitude
        'region': 'Kilimanjaro',                # Administrative region
        'extraction_type': 'gravity',           # How water is extracted
        'management': 'vwc',                    # Who manages the pump
        'payment': 'pay monthly',               # Payment method
        'water_quality': 'soft',                # Quality of water
        'quantity': 'enough',                   # Amount of water
        'source': 'spring',                     # Water source
        'waterpoint_type': 'communal standpipe', # Type of access point
        'construction_year': 2010,              # When pump was built
        'population': 250                       # People served
    }
    
    # Make prediction
    result = predictor.predict_single(pump_data)
    
    # Display results
    if 'error' not in result:
        print(f"✅ Prediction: {result['prediction']}")
        print(f"🎯 Confidence: {result['confidence']:.2%}")
        print("\n📊 Class Probabilities:")
        for class_name, prob in result['probabilities'].items():
            print(f"   {class_name}: {prob:.2%}")
    else:
        print(f"❌ Error: {result['error']}")

def example_batch_prediction():
    """Example: Predict for multiple pumps from CSV file"""
    print("\n🔍 Example 2: Batch Prediction from CSV")
    print("=" * 50)
    
    # Create predictor instance
    predictor = WaterPumpPredictor()
    
    # Create sample data for multiple pumps
    pumps_data = [
        {
            'longitude': 34.5, 'latitude': -6.2, 'region': 'Kilimanjaro',
            'extraction_type': 'gravity', 'management': 'vwc', 'water_quality': 'soft',
            'quantity': 'enough', 'construction_year': 2010, 'population': 250
        },
        {
            'longitude': 35.1, 'latitude': -7.1, 'region': 'Morogoro', 
            'extraction_type': 'handpump', 'management': 'water authority', 'water_quality': 'salty',
            'quantity': 'insufficient', 'construction_year': 1995, 'population': 150
        },
        {
            'longitude': 33.8, 'latitude': -5.5, 'region': 'Arusha',
            'extraction_type': 'submersible', 'management': 'private operator', 'water_quality': 'soft',
            'quantity': 'enough', 'construction_year': 2015, 'population': 400
        }
    ]
    
    # Make batch predictions
    results = predictor.predict_batch(pumps_data)
    
    # Display results
    print(f"📊 Processed {len(results)} water pumps:")
    for i, result in enumerate(results):
        if 'error' not in result:
            print(f"\nPump {i+1}:")
            print(f"  Prediction: {result['prediction']}")
            print(f"  Confidence: {result['confidence']:.2%}")
        else:
            print(f"\nPump {i+1}: Error - {result['error']}")

def example_feature_importance():
    """Example: Display most important features for predictions"""
    print("\n🔍 Example 3: Feature Importance Analysis")
    print("=" * 50)
    
    # Create predictor instance
    predictor = WaterPumpPredictor()
    
    # Get feature importance
    importance = predictor.get_feature_importance(top_n=10)
    
    if importance is not None:
        print("📈 Top 10 Most Important Features:")
        for idx, row in importance.iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
    else:
        print("❌ Feature importance not available for this model type")

def example_api_style_prediction():
    """Example: API-style prediction function for integration"""
    print("\n🔍 Example 4: API-Style Prediction Function")
    print("=" * 50)
    
    def predict_pump_status(pump_features):
        """
        API-style function to predict pump status
        
        Args:
            pump_features: Dictionary with pump characteristics
            
        Returns:
            JSON-style response with prediction and recommendation
        """
        try:
            # Initialize predictor
            predictor = WaterPumpPredictor()
            
            # Make prediction
            result = predictor.predict_single(pump_features)
            
            if 'error' not in result:
                # Get recommendation based on prediction
                recommendation = get_recommendation(result['prediction'])
                
                # Return API-style response
                return {
                    'status': 'success',
                    'prediction': {
                        'pump_status': result['prediction'],
                        'confidence': round(result['confidence'], 3),
                        'recommendation': recommendation
                    },
                    'probabilities': {k: round(v, 3) for k, v in result['probabilities'].items()}
                }
            else:
                return {
                    'status': 'error',
                    'message': result['error']
                }
                
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def get_recommendation(prediction):
        """Get maintenance recommendation based on prediction"""
        recommendations = {
            'functional': 'Pump is working well. Continue regular maintenance.',
            'functional needs repair': 'Schedule maintenance visit soon to prevent breakdown.',
            'non functional': 'Immediate repair or replacement required.'
        }
        return recommendations.get(prediction, 'No recommendation available.')
    
    # Example usage of API-style function
    sample_pump = {
        'longitude': 34.7, 'latitude': -6.8, 'region': 'Tanga',
        'extraction_type': 'handpump', 'management': 'vwc',
        'water_quality': 'soft', 'quantity': 'enough',
        'construction_year': 2005, 'population': 180
    }
    
    # Make API call
    api_response = predict_pump_status(sample_pump)
    
    # Display API response
    print("📡 API Response:")
    print(json.dumps(api_response, indent=2))

def main():
    """Run all prediction examples"""
    print("🚰 Water Pump Prediction Examples")
    print("=" * 60)
    print("This script demonstrates how to use the trained model for predictions")
    
    # Run all examples
    example_single_prediction()
    example_batch_prediction()
    example_feature_importance()
    example_api_style_prediction()
    
    print("\n✅ All examples completed!")
    print("\n💡 Tips for using the predictor:")
    print("- Ensure all required features are provided")
    print("- Use consistent feature names and value formats")
    print("- Check confidence scores to assess prediction reliability")
    print("- Higher confidence (>0.8) indicates more reliable predictions")

# Run examples if script is executed directly
if __name__ == "__main__":
    main()
```

This completes the comprehensive code documentation with detailed line-by-line comments for all major components including:

✓ **Core ML Components**: Data ingestion, transformation, model training with detailed explanations
✓ **Exception Handling**: Custom exception system with file/line tracking
✓ **Utility Functions**: Model saving/loading, evaluation, metrics calculation
✓ **Pipeline Components**: Training and prediction pipelines
✓ **Streamlit Pages**: Main app, EDA page, predictions page with user interfaces
✓ **Utility Scripts**: Prediction examples and API usage patterns

Each script is thoroughly documented with comments explaining:
- What each line does
- Why it's needed
- How it fits into the overall system
- Expected inputs and outputs
- Error handling strategies

The documentation provides everything needed to understand, modify, and extend the water pump prediction system.
