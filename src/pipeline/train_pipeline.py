import sys
import os
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException
from src.logger import logging

class TrainPipeline:
    """Complete training pipeline"""
    
    def __init__(self):
        self.data_ingestion = DataIngestion()
        self.data_transformation = DataTransformation()
        self.model_trainer = ModelTrainer()
    
    def run_pipeline(self, data_path: str = None, selected_models: list = None):
        """
        Run the complete training pipeline
        
        Args:
            data_path: Path to the dataset
            selected_models: List of model names to train
            
        Returns:
            Model performance score
        """
        try:
            logging.info("Starting training pipeline")
            
            # Data ingestion
            train_data_path, test_data_path = self.data_ingestion.initiate_data_ingestion(data_path)
            
            # Data transformation
            train_arr, test_arr, _ = self.data_transformation.initiate_data_transformation(
                train_data_path, test_data_path
            )
            
            # Model training
            model_score = self.model_trainer.initiate_model_trainer(train_arr, test_arr, selected_models)
            
            logging.info("Training pipeline completed successfully")
            return model_score
            
        except Exception as e:
            logging.error(f"Error in training pipeline: {str(e)}")
            raise CustomException(e, sys)
    
    def get_pipeline_info(self):
        """
        Get information about the pipeline components
        
        Returns:
            Dictionary with pipeline information
        """
        return {
            'data_ingestion': 'Loads and splits data into train/test sets',
            'data_transformation': 'Preprocesses data with scaling and encoding',
            'model_trainer': 'Trains and evaluates ML models with hyperparameter tuning'
        }
