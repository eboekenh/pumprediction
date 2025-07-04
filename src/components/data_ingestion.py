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
    
    def initiate_data_ingestion(self, data_path: str = None) -> Tuple[str, str]:
        """
        Initiate data ingestion process
        
        Args:
            data_path: Path to the dataset
            
        Returns:
            Tuple of train and test data paths
        """
        try:
            logging.info("Entered data ingestion method")
            
            # Use provided path or default sample data
            if data_path and os.path.exists(data_path):
                df = pd.read_csv(data_path)
                logging.info(f"Data loaded from: {data_path}")
            else:
                # Load sample data if no path provided
                sample_path = "sample_data/sample_water_pumps.csv"
                if os.path.exists(sample_path):
                    df = pd.read_csv(sample_path)
                    logging.info(f"Sample data loaded from: {sample_path}")
                else:
                    raise FileNotFoundError("No data file found. Please upload data first.")
            
            logging.info(f"Dataset shape: {df.shape}")
            
            # Create artifacts directory
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            
            # Save raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Raw data saved")
            
            # Split data if target column exists
            if 'status_group' in df.columns:
                # Split data
                train_set, test_set = train_test_split(
                    df, 
                    test_size=0.2, 
                    random_state=42,
                    stratify=df['status_group']
                )
                
                # Save train and test sets
                train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
                test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
                
                logging.info(f"Train set shape: {train_set.shape}")
                logging.info(f"Test set shape: {test_set.shape}")
                logging.info("Data ingestion completed successfully")
                
                return (
                    self.ingestion_config.train_data_path,
                    self.ingestion_config.test_data_path
                )
            else:
                # If no target column, just save as raw data
                df.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
                logging.info("Data saved without splitting (no target column found)")
                
                return (
                    self.ingestion_config.train_data_path,
                    self.ingestion_config.train_data_path
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
