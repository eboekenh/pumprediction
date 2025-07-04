import sys
import os
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

class PredictPipeline:
    """Pipeline for making predictions"""
    
    def __init__(self):
        pass
    
    def predict(self, features):
        """
        Make predictions using trained model
        
        Args:
            features: Input features
            
        Returns:
            Predictions array
        """
        try:
            # Load model and preprocessor
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            
            # Transform features
            data_scaled = preprocessor.transform(features)
            
            # Make predictions
            predictions = model.predict(data_scaled)
            
            # Load label encoder to get original labels
            label_encoder_path = os.path.join("artifacts", "label_encoder.pkl")
            if os.path.exists(label_encoder_path):
                label_encoder = load_object(file_path=label_encoder_path)
                predictions = label_encoder.inverse_transform(predictions)
            
            return predictions
            
        except Exception as e:
            logging.error(f"Error in prediction pipeline: {str(e)}")
            raise CustomException(e, sys)
    
    def predict_proba(self, features):
        """
        Get prediction probabilities
        
        Args:
            features: Input features
            
        Returns:
            Prediction probabilities
        """
        try:
            # Load model and preprocessor
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            
            # Transform features
            data_scaled = preprocessor.transform(features)
            
            # Get probabilities
            probabilities = model.predict_proba(data_scaled)
            
            return probabilities
            
        except Exception as e:
            logging.error(f"Error getting prediction probabilities: {str(e)}")
            raise CustomException(e, sys)

class CustomData:
    """Custom data class for creating prediction input"""
    
    def __init__(self, **kwargs):
        """
        Initialize custom data with water pump features
        
        Args:
            **kwargs: Feature values
        """
        # Geographic features
        self.longitude = kwargs.get('longitude', 0.0)
        self.latitude = kwargs.get('latitude', 0.0)
        self.region = kwargs.get('region', 'unknown')
        self.district_code = kwargs.get('district_code', 0)
        self.lga = kwargs.get('lga', 'unknown')
        self.ward = kwargs.get('ward', 'unknown')
        
        # Technical features
        self.pump_type = kwargs.get('pump_type', 'unknown')
        self.extraction_type = kwargs.get('extraction_type', 'unknown')
        self.extraction_type_group = kwargs.get('extraction_type_group', 'unknown')
        self.extraction_type_class = kwargs.get('extraction_type_class', 'unknown')
        self.management = kwargs.get('management', 'unknown')
        self.management_group = kwargs.get('management_group', 'unknown')
        self.payment = kwargs.get('payment', 'unknown')
        self.payment_type = kwargs.get('payment_type', 'unknown')
        
        # Water features
        self.water_quality = kwargs.get('water_quality', 'unknown')
        self.quality_group = kwargs.get('quality_group', 'unknown')
        self.quantity = kwargs.get('quantity', 'unknown')
        self.quantity_group = kwargs.get('quantity_group', 'unknown')
        self.source = kwargs.get('source', 'unknown')
        self.source_type = kwargs.get('source_type', 'unknown')
        self.source_class = kwargs.get('source_class', 'unknown')
        self.waterpoint_type = kwargs.get('waterpoint_type', 'unknown')
        self.waterpoint_type_group = kwargs.get('waterpoint_type_group', 'unknown')
        
        # Installation features
        self.installer = kwargs.get('installer', 'unknown')
        self.funder = kwargs.get('funder', 'unknown')
        self.construction_year = kwargs.get('construction_year', 0)
        self.population = kwargs.get('population', 0)
        self.gps_height = kwargs.get('gps_height', 0)
        
        # Additional features
        self.num_private = kwargs.get('num_private', 0)
        self.basin = kwargs.get('basin', 'unknown')
        self.subvillage = kwargs.get('subvillage', 'unknown')
        self.region_code = kwargs.get('region_code', 0)
        self.lga_code = kwargs.get('lga_code', 0)
        self.ward_code = kwargs.get('ward_code', 0)
        self.public_meeting = kwargs.get('public_meeting', 'unknown')
        self.scheme_management = kwargs.get('scheme_management', 'unknown')
        self.scheme_name = kwargs.get('scheme_name', 'unknown')
        self.permit = kwargs.get('permit', 'unknown')
        self.recorded_by = kwargs.get('recorded_by', 'unknown')
    
    def get_data_as_data_frame(self):
        """
        Convert custom data to pandas DataFrame
        
        Returns:
            DataFrame with single row
        """
        try:
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
            
            return pd.DataFrame(custom_data_input_dict)
            
        except Exception as e:
            logging.error(f"Error creating DataFrame: {str(e)}")
            raise CustomException(e, sys)
