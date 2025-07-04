import sys
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from dataclasses import dataclass
import pickle
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    """Configuration for data transformation"""
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    """Data transformation component for preprocessing"""
    
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformer_object(self, df: pd.DataFrame):
        """
        Create preprocessing pipeline
        
        Args:
            df: Input dataframe
            
        Returns:
            Preprocessing pipeline
        """
        try:
            # Identify column types
            numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
            
            # Handle boolean columns that are stored as object type
            boolean_columns = []
            remaining_categorical = []
            
            for col in categorical_columns:
                unique_vals = df[col].dropna().unique()
                # Check if column contains only boolean values (True/False)
                if len(unique_vals) <= 2 and all(isinstance(x, bool) for x in unique_vals):
                    boolean_columns.append(col)
                else:
                    remaining_categorical.append(col)
            
            categorical_columns = remaining_categorical
            
            # Remove target column if present
            if 'status_group' in numerical_columns:
                numerical_columns.remove('status_group')
            if 'status_group' in categorical_columns:
                categorical_columns.remove('status_group')
            if 'status_group' in boolean_columns:
                boolean_columns.remove('status_group')
            
            # Remove ID columns
            id_columns = ['id']
            numerical_columns = [col for col in numerical_columns if col not in id_columns]
            categorical_columns = [col for col in categorical_columns if col not in id_columns]
            boolean_columns = [col for col in boolean_columns if col not in id_columns]
            
            logging.info(f"Numerical columns: {numerical_columns}")
            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Boolean columns: {boolean_columns}")
            
            # Numerical pipeline
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )
            
            # Categorical pipeline with cardinality limit to prevent memory issues
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, max_categories=20))
                ]
            )
            
            # Convert boolean columns to categorical by adding them back
            # We'll handle boolean conversion in the data transformation step
            all_categorical_columns = categorical_columns + boolean_columns
            
            # Combine pipelines
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", num_pipeline, numerical_columns),
                    ("cat", cat_pipeline, all_categorical_columns)
                ]
            )
            
            return preprocessor
            
        except Exception as e:
            logging.error(f"Error creating data transformer: {str(e)}")
            raise CustomException(e, sys)
    
    def initiate_data_transformation(self, train_path: str, test_path: str):
        """
        Initiate data transformation
        
        Args:
            train_path: Path to training data
            test_path: Path to test data
            
        Returns:
            Transformed arrays and preprocessor object
        """
        try:
            # Load data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("Data loaded for transformation")
            logging.info(f"Training data shape: {train_df.shape}")
            logging.info(f"Test data shape: {test_df.shape}")
            
            # Convert boolean columns to strings to avoid encoding issues
            for col in train_df.select_dtypes(include=['object']).columns:
                unique_vals = train_df[col].dropna().unique()
                if len(unique_vals) <= 2 and all(isinstance(x, bool) for x in unique_vals):
                    train_df[col] = train_df[col].astype(str)
                    test_df[col] = test_df[col].astype(str)
            
            # Get preprocessor
            preprocessor_obj = self.get_data_transformer_object(train_df)
            
            # Separate features and target
            target_column = "status_group"
            
            if target_column in train_df.columns:
                # Training data
                input_feature_train_df = train_df.drop(columns=[target_column], axis=1)
                target_feature_train_df = train_df[target_column]
                
                # Test data
                input_feature_test_df = test_df.drop(columns=[target_column], axis=1)
                target_feature_test_df = test_df[target_column]
                
                # Transform features with progress logging
                logging.info("Starting feature transformation...")
                logging.info(f"Processing {len(input_feature_train_df.columns)} features")
                
                # Limit data size for transformation if too large
                max_samples = 10000
                if len(input_feature_train_df) > max_samples:
                    logging.info(f"Large dataset detected ({len(input_feature_train_df)} samples), using sample of {max_samples} for preprocessing")
                    train_sample = input_feature_train_df.sample(n=max_samples, random_state=42)
                    input_feature_train_arr = preprocessor_obj.fit_transform(train_sample)
                    input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)
                    
                    # Apply to full training set
                    input_feature_train_arr = preprocessor_obj.transform(input_feature_train_df)
                else:
                    input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
                    input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)
                
                logging.info("Feature transformation completed")
                
                # Encode target labels
                label_encoder = LabelEncoder()
                target_feature_train_encoded = label_encoder.fit_transform(target_feature_train_df)
                target_feature_test_encoded = label_encoder.transform(target_feature_test_df)
                
                # Save label encoder
                label_encoder_path = os.path.join("artifacts", "label_encoder.pkl")
                save_object(file_path=label_encoder_path, obj=label_encoder)
                
                # Combine features and target
                train_arr = np.c_[input_feature_train_arr, target_feature_train_encoded]
                test_arr = np.c_[input_feature_test_arr, target_feature_test_encoded]
                
                logging.info("Data transformation completed successfully")
                
            else:
                # No target column - just transform features
                input_feature_train_arr = preprocessor_obj.fit_transform(train_df)
                input_feature_test_arr = preprocessor_obj.transform(test_df)
                
                train_arr = input_feature_train_arr
                test_arr = input_feature_test_arr
                
                logging.info("Data transformation completed (no target column)")
            
            # Save preprocessor
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
            
        except Exception as e:
            logging.error(f"Error in data transformation: {str(e)}")
            raise CustomException(e, sys)
    
    def get_feature_names(self, preprocessor_obj, original_df):
        """
        Get feature names after transformation
        
        Args:
            preprocessor_obj: Fitted preprocessor
            original_df: Original dataframe
            
        Returns:
            List of feature names
        """
        try:
            # Get column names for numerical features
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
            
            # Get feature names
            feature_names = []
            
            # Numerical features
            feature_names.extend(numerical_columns)
            
            # Categorical features (one-hot encoded)
            if categorical_columns:
                cat_transformer = preprocessor_obj.named_transformers_['cat']
                if hasattr(cat_transformer, 'named_steps'):
                    onehot_encoder = cat_transformer.named_steps['onehot']
                    cat_feature_names = onehot_encoder.get_feature_names_out(categorical_columns)
                    feature_names.extend(cat_feature_names)
            
            return feature_names
            
        except Exception as e:
            logging.error(f"Error getting feature names: {str(e)}")
            return None
