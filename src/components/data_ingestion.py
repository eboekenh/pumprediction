import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from typing import Tuple

@dataclass
class DataIngestionConfig:
    """Configuration for data ingestion"""
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "raw_data.csv")

class DataIngestion:
    """Data ingestion component for loading and splitting data"""
    
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def _merge_train_data(self, train_features_path: str, train_labels_path: str) -> pd.DataFrame:
        """
        Merge training features and labels based on common id column
        
        Args:
            train_features_path: Path to training features file
            train_labels_path: Path to training labels file
            
        Returns:
            Merged DataFrame with features and labels
        """
        try:
            # Load training features and labels
            train_features = pd.read_csv(train_features_path)
            train_labels = pd.read_csv(train_labels_path)
            
            logging.info(f"Training features shape: {train_features.shape}")
            logging.info(f"Training labels shape: {train_labels.shape}")
            
            # Merge on 'id' column
            merged_df = pd.merge(train_features, train_labels, on='id', how='inner')
            
            logging.info(f"Merged training data shape: {merged_df.shape}")
            
            return merged_df
            
        except Exception as e:
            logging.error(f"Error merging train data: {str(e)}")
            raise CustomException(e, sys)
    
    def _identify_file_types(self, file_paths: list) -> dict:
        """
        Identify which file contains features, labels, and test data
        
        Args:
            file_paths: List of file paths to analyze
            
        Returns:
            Dictionary with file types identified
        """
        file_info = {}
        
        for path in file_paths:
            if os.path.exists(path):
                df = pd.read_csv(path, nrows=5)  # Read first 5 rows to check structure
                
                # Check if it's labels file (has target column)
                if 'status_group' in df.columns and len(df.columns) <= 3:
                    file_info['labels'] = path
                    logging.info(f"Identified labels file: {path}")
                
                # Check if it's features file (many columns, no target)
                elif 'status_group' not in df.columns and len(df.columns) > 10:
                    if 'features' not in file_info:
                        file_info['features'] = path
                        logging.info(f"Identified features file: {path}")
                    else:
                        file_info['test_features'] = path
                        logging.info(f"Identified test features file: {path}")
        
        return file_info
    
    def initiate_data_ingestion(self, data_path=None, train_features_path=None, 
                               train_labels_path=None, test_features_path=None) -> Tuple[str, str]:
        """
        Initiate data ingestion process
        
        Args:
            data_path: Path to the dataset (for single file)
            train_features_path: Path to training features file
            train_labels_path: Path to training labels file  
            test_features_path: Path to test features file
            
        Returns:
            Tuple of train and test data paths
        """
        try:
            logging.info("Entered data ingestion method")
            
            # Create artifacts directory
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            
            # Handle separate training features and labels files
            if train_features_path and train_labels_path:
                logging.info("Loading separate training features and labels files")
                train_df = self._merge_train_data(train_features_path, train_labels_path)
                
                # Load test features if provided
                if test_features_path and os.path.exists(test_features_path):
                    test_df = pd.read_csv(test_features_path)
                    logging.info(f"Test features loaded, shape: {test_df.shape}")
                else:
                    # Split train data if no separate test provided
                    train_df, test_df = train_test_split(
                        train_df, 
                        test_size=0.2, 
                        random_state=42,
                        stratify=train_df['status_group']
                    )
                    
            elif data_path and os.path.exists(data_path):
                logging.info(f"Loading data from single file: {data_path}")
                df = pd.read_csv(data_path)
                # Split data into train and test
                train_df, test_df = train_test_split(
                    df, 
                    test_size=0.2, 
                    random_state=42,
                    stratify=df['status_group'] if 'status_group' in df.columns else None
                )
            else:
                # Load sample data if no path provided
                sample_path = "sample_data/sample_water_pumps.csv"
                if os.path.exists(sample_path):
                    df = pd.read_csv(sample_path)
                    logging.info(f"Sample data loaded from: {sample_path}")
                    train_df, test_df = train_test_split(
                        df, 
                        test_size=0.25,  # 25% for test set (more realistic)
                        random_state=42,
                        stratify=df['status_group'] if 'status_group' in df.columns else None
                    )
                else:
                    raise FileNotFoundError("No data file found. Please upload data first.")
            
            logging.info(f"Final train dataset shape: {train_df.shape}")
            logging.info(f"Final test dataset shape: {test_df.shape}")
            
            # Save raw merged data
            train_df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Raw data saved")
            
            # Save train and test sets
            train_df.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_df.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            
            logging.info("Data ingestion completed successfully")
            
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
                
        except Exception as e:
            logging.error(f"Error in data ingestion: {str(e)}")
            raise CustomException(e, sys)
    
    def get_data_info(self, data_path: str) -> dict:
        """
        Get information about the dataset
        
        Args:
            data_path: Path to the dataset
            
        Returns:
            Dictionary containing dataset information
        """
        try:
            df = pd.read_csv(data_path)
            
            info = {
                'shape': df.shape,
                'columns': df.columns.tolist(),
                'dtypes': df.dtypes.to_dict(),
                'missing_values': df.isnull().sum().to_dict(),
                'memory_usage': df.memory_usage().sum(),
                'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
                'categorical_columns': df.select_dtypes(include=['object']).columns.tolist()
            }
            
            # Add target column info if exists
            if 'status_group' in df.columns:
                info['target_distribution'] = df['status_group'].value_counts().to_dict()
            
            return info
            
        except Exception as e:
            logging.error(f"Error getting data info: {str(e)}")
            raise CustomException(e, sys)
