#!/usr/bin/env python3
"""
Simple API example for water pump predictions
"""

import pandas as pd
import numpy as np
from src.utils import load_object
import json

class WaterPumpAPI:
    """Simple API for water pump predictions"""
    
    def __init__(self):
        self.model = load_object("artifacts/model.pkl")
        self.preprocessor = load_object("artifacts/preprocessor.pkl")
        self.label_encoder = load_object("artifacts/label_encoder.pkl")
        print("✅ Model components loaded")
    
    def predict(self, pump_data):
        """Make prediction for a single pump"""
        try:
            # Convert to DataFrame
            df = pd.DataFrame([pump_data])
            
            # Remove id if present
            if 'id' in df.columns:
                df = df.drop('id', axis=1)
            
            # Transform and predict
            features_transformed = self.preprocessor.transform(df)
            prediction = self.model.predict(features_transformed)[0]
            probabilities = self.model.predict_proba(features_transformed)[0]
            
            # Convert numpy types to Python types for JSON serialization
            predicted_label = self.label_encoder.inverse_transform([prediction])[0]
            confidence = float(max(probabilities))
            
            # Build response
            result = {
                'status': 'success',
                'prediction': predicted_label,
                'confidence': round(confidence, 3),
                'probabilities': {
                    str(class_name): round(float(prob), 3) 
                    for class_name, prob in zip(self.label_encoder.classes_, probabilities)
                }
            }
            
            return result
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}

def demo_predictions():
    """Demo different types of pumps"""
    
    api = WaterPumpAPI()
    
    # Test cases with different scenarios
    test_cases = [
        {
            'name': 'Well-maintained pump',
            'data': {
                'amount_tsh': 6000.0,
                'date_recorded': '2013-03-06',
                'funder': 'World Bank',
                'gps_height': 1390,
                'installer': 'DWE',
                'longitude': 34.938093,
                'latitude': -9.856323,
                'wpt_name': 'Good Pump',
                'num_private': 0,
                'basin': 'Pangani',
                'subvillage': 'Village1',
                'region': 'Kilimanjaro',
                'region_code': 9,
                'district_code': 4,
                'lga': 'Hai',
                'ward': 'Ward1',
                'population': 500,
                'public_meeting': True,
                'recorded_by': 'Surveyor',
                'scheme_management': 'VWC',
                'scheme_name': 'Scheme1',
                'permit': True,
                'construction_year': 2010,
                'extraction_type': 'gravity',
                'extraction_type_group': 'gravity',
                'extraction_type_class': 'gravity',
                'management': 'vwc',
                'management_group': 'user-group',
                'payment': 'pay annually',
                'payment_type': 'annually',
                'water_quality': 'soft',
                'quality_group': 'good',
                'quantity': 'enough',
                'quantity_group': 'enough',
                'source': 'spring',
                'source_type': 'spring',
                'source_class': 'groundwater',
                'waterpoint_type': 'communal standpipe',
                'waterpoint_type_group': 'communal standpipe'
            }
        },
        {
            'name': 'Problematic pump',
            'data': {
                'amount_tsh': 0.0,
                'date_recorded': '2011-12-06',
                'funder': 'Government Of Tanzania',
                'gps_height': 1400,
                'installer': 'DWE',
                'longitude': 34.973782,
                'latitude': -9.856323,
                'wpt_name': 'Problem Pump',
                'num_private': 0,
                'basin': 'Lake Victoria',
                'subvillage': 'Village2',
                'region': 'Kagera',
                'region_code': 4,
                'district_code': 2,
                'lga': 'Bukoba Rural',
                'ward': 'Ward2',
                'population': 100,
                'public_meeting': False,
                'recorded_by': 'Surveyor',
                'scheme_management': 'None',
                'scheme_name': '',
                'permit': False,
                'construction_year': 1990,
                'extraction_type': 'other',
                'extraction_type_group': 'other',
                'extraction_type_class': 'other',
                'management': 'unknown',
                'management_group': 'unknown',
                'payment': 'never pay',
                'payment_type': 'never pay',
                'water_quality': 'unknown',
                'quality_group': 'unknown',
                'quantity': 'dry',
                'quantity_group': 'dry',
                'source': 'other',
                'source_type': 'other',
                'source_class': 'unknown',
                'waterpoint_type': 'other',
                'waterpoint_type_group': 'other'
            }
        }
    ]
    
    print("🎯 Water Pump Prediction Demo")
    print("=" * 50)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. {test_case['name']}")
        print("-" * 30)
        
        result = api.predict(test_case['data'])
        
        if result['status'] == 'success':
            print(f"Prediction: {result['prediction']}")
            print(f"Confidence: {result['confidence']}")
            print("Probabilities:")
            for class_name, prob in result['probabilities'].items():
                print(f"  {class_name}: {prob}")
                
            # Add recommendation
            if result['prediction'] == 'functional':
                print("Recommendation: Continue regular maintenance")
            elif result['prediction'] == 'functional needs repair':
                print("Recommendation: Schedule repair within 30 days")
            else:
                print("Recommendation: Immediate repair or replacement required")
        else:
            print(f"Error: {result['message']}")

def batch_prediction_example():
    """Example of batch prediction from CSV file"""
    
    print("\n\n📊 Batch Prediction Example")
    print("=" * 50)
    
    api = WaterPumpAPI()
    
    try:
        # Load sample data
        df = pd.read_csv("artifacts/merged_train_data.csv", nrows=10)
        features_df = df.drop(['status_group'], axis=1, errors='ignore')
        
        print(f"Processing {len(features_df)} pumps...")
        
        # Make predictions
        results = []
        for i, row in features_df.iterrows():
            pump_data = row.to_dict()
            result = api.predict(pump_data)
            results.append(result)
        
        # Summary
        predictions = [r['prediction'] for r in results if r['status'] == 'success']
        if predictions:
            functional = predictions.count('functional')
            needs_repair = predictions.count('functional needs repair')
            non_functional = predictions.count('non functional')
            
            print(f"\nResults Summary:")
            print(f"Functional: {functional}")
            print(f"Needs repair: {needs_repair}")
            print(f"Non-functional: {non_functional}")
            
            # Show first few results
            print(f"\nFirst 5 predictions:")
            for i, result in enumerate(results[:5]):
                if result['status'] == 'success':
                    print(f"  Pump {i+1}: {result['prediction']} (confidence: {result['confidence']})")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    demo_predictions()
    batch_prediction_example()
    
    print("\n\n✅ Demo completed!")
    print("\nNext steps:")
    print("• Use this API class in your web applications")
    print("• Create REST endpoints with Flask or FastAPI")
    print("• Process CSV files in batch")
    print("• Integrate with monitoring systems")