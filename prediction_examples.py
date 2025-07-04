#!/usr/bin/env python3
"""
Practical examples for using the trained water pump model
"""

import pandas as pd
import numpy as np
from src.utils import load_object
import json

class WaterPumpPredictor:
    """Water pump functionality predictor"""
    
    def __init__(self):
        """Initialize predictor with trained model components"""
        self.model = None
        self.preprocessor = None
        self.label_encoder = None
        self.load_components()
    
    def load_components(self):
        """Load all model components"""
        try:
            self.model = load_object("artifacts/model.pkl")
            self.preprocessor = load_object("artifacts/preprocessor.pkl")
            self.label_encoder = load_object("artifacts/label_encoder.pkl")
            print("✅ All model components loaded successfully")
        except Exception as e:
            print(f"❌ Error loading components: {str(e)}")
    
    def predict_single(self, pump_data):
        """
        Predict functionality for a single water pump
        
        Args:
            pump_data: Dictionary with pump features
            
        Returns:
            Dictionary with prediction and confidence
        """
        try:
            # Convert to DataFrame
            df = pd.DataFrame([pump_data])
            
            # Remove id if present
            if 'id' in df.columns:
                df = df.drop('id', axis=1)
            
            # Transform features
            features_transformed = self.preprocessor.transform(df)
            
            # Make prediction
            prediction = self.model.predict(features_transformed)[0]
            probabilities = self.model.predict_proba(features_transformed)[0]
            
            # Decode prediction
            predicted_label = self.label_encoder.inverse_transform([prediction])[0]
            confidence = max(probabilities)
            
            # Get probabilities for all classes
            class_probs = {}
            for i, class_name in enumerate(self.label_encoder.classes_):
                class_probs[class_name] = probabilities[i]
            
            return {
                'prediction': predicted_label,
                'confidence': confidence,
                'probabilities': class_probs
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def predict_batch(self, pumps_data):
        """
        Predict functionality for multiple water pumps
        
        Args:
            pumps_data: List of dictionaries or DataFrame
            
        Returns:
            List of prediction results
        """
        try:
            # Convert to DataFrame if needed
            if isinstance(pumps_data, list):
                df = pd.DataFrame(pumps_data)
            else:
                df = pumps_data.copy()
            
            # Remove id if present
            if 'id' in df.columns:
                df = df.drop('id', axis=1)
            
            # Transform features
            features_transformed = self.preprocessor.transform(df)
            
            # Make predictions
            predictions = self.model.predict(features_transformed)
            probabilities = self.model.predict_proba(features_transformed)
            
            # Decode predictions
            predicted_labels = self.label_encoder.inverse_transform(predictions)
            
            results = []
            for i, (pred, probs) in enumerate(zip(predicted_labels, probabilities)):
                class_probs = {}
                for j, class_name in enumerate(self.label_encoder.classes_):
                    class_probs[class_name] = probs[j]
                
                results.append({
                    'prediction': pred,
                    'confidence': max(probs),
                    'probabilities': class_probs
                })
            
            return results
            
        except Exception as e:
            return [{'error': str(e)}]
    
    def get_feature_importance(self, top_n=10):
        """Get top N most important features"""
        try:
            if hasattr(self.model, 'feature_importances_'):
                # Get feature names from preprocessor
                feature_names = []
                if hasattr(self.preprocessor, 'get_feature_names_out'):
                    feature_names = self.preprocessor.get_feature_names_out()
                else:
                    feature_names = [f"feature_{i}" for i in range(len(self.model.feature_importances_))]
                
                # Create importance DataFrame
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': self.model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                return importance_df.head(top_n).to_dict('records')
            else:
                return {'error': 'Model does not have feature importances'}
                
        except Exception as e:
            return {'error': str(e)}

def example_single_prediction():
    """Example: Predict for a single water pump"""
    print("\n🎯 Example 1: Single Pump Prediction")
    print("-" * 40)
    
    predictor = WaterPumpPredictor()
    
    # Example pump data
    pump_data = {
        'amount_tsh': 6000.0,
        'date_recorded': '2011-03-14',
        'funder': 'World Bank',
        'gps_height': 1390,
        'installer': 'DWE',
        'longitude': 34.938093,
        'latitude': -9.856323,
        'wpt_name': 'Zahanati',
        'num_private': 0,
        'basin': 'Pangani',
        'subvillage': 'Shuleni',
        'region': 'Kilimanjaro',
        'region_code': 9,
        'district_code': 4,
        'lga': 'Hai',
        'ward': 'Shira',
        'population': 280,
        'public_meeting': True,
        'recorded_by': 'GeoData Consultants Ltd',
        'scheme_management': 'VWC',
        'scheme_name': 'Roman',
        'permit': False,
        'construction_year': 2011,
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
    
    result = predictor.predict_single(pump_data)
    
    if 'error' not in result:
        print(f"🎯 Prediction: {result['prediction']}")
        print(f"📊 Confidence: {result['confidence']:.3f}")
        print("\n📈 All probabilities:")
        for class_name, prob in result['probabilities'].items():
            print(f"   {class_name}: {prob:.3f}")
    else:
        print(f"❌ Error: {result['error']}")

def example_batch_prediction():
    """Example: Predict for multiple pumps from CSV"""
    print("\n🎯 Example 2: Batch Prediction from CSV")
    print("-" * 40)
    
    predictor = WaterPumpPredictor()
    
    try:
        # Load sample data
        df = pd.read_csv("artifacts/merged_train_data.csv", nrows=5)
        
        # Remove target column for prediction
        features_df = df.drop(['status_group'], axis=1, errors='ignore')
        
        print(f"📊 Loaded {len(features_df)} samples for prediction")
        
        # Make batch predictions
        results = predictor.predict_batch(features_df)
        
        # Display results
        for i, result in enumerate(results):
            if 'error' not in result:
                print(f"\nSample {i+1}:")
                print(f"  Prediction: {result['prediction']}")
                print(f"  Confidence: {result['confidence']:.3f}")
            else:
                print(f"Sample {i+1}: Error - {result['error']}")
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")

def example_feature_importance():
    """Example: Show most important features"""
    print("\n🎯 Example 3: Feature Importance Analysis")
    print("-" * 40)
    
    predictor = WaterPumpPredictor()
    
    importance = predictor.get_feature_importance(top_n=15)
    
    if isinstance(importance, list):
        print("📈 Top 15 Most Important Features:")
        for i, feat in enumerate(importance, 1):
            print(f"{i:2d}. {feat['feature']}: {feat['importance']:.4f}")
    else:
        print(f"❌ Error: {importance.get('error', 'Unknown error')}")

def example_api_style_prediction():
    """Example: API-style prediction function"""
    print("\n🎯 Example 4: API-Style Prediction Function")
    print("-" * 40)
    
    def predict_pump_status(pump_features):
        """
        API-style function to predict pump status
        
        Args:
            pump_features: Dictionary with pump characteristics
            
        Returns:
            JSON-style response
        """
        predictor = WaterPumpPredictor()
        result = predictor.predict_single(pump_features)
        
        if 'error' not in result:
            return {
                'status': 'success',
                'data': {
                    'pump_status': result['prediction'],
                    'confidence_score': round(result['confidence'], 3),
                    'risk_assessment': 'high' if result['prediction'] == 'non functional' else 
                                    'medium' if result['prediction'] == 'functional needs repair' else 'low',
                    'recommendation': get_recommendation(result['prediction'])
                }
            }
        else:
            return {
                'status': 'error',
                'message': result['error']
            }
    
    def get_recommendation(prediction):
        """Get maintenance recommendation based on prediction"""
        recommendations = {
            'functional': 'Regular maintenance scheduled',
            'functional needs repair': 'Schedule repair within 30 days',
            'non functional': 'Immediate repair or replacement required'
        }
        return recommendations.get(prediction, 'Unknown status')
    
    # Test the API function
    test_pump = {
        'amount_tsh': 0.0,
        'date_recorded': '2013-03-06',
        'funder': 'Government Of Tanzania',
        'gps_height': 1399,
        'installer': 'DWE',
        'longitude': 34.973782,
        'latitude': -9.856323,
        'wpt_name': 'Zahanati',
        'num_private': 0,
        'basin': 'Pangani',
        'subvillage': 'Shuleni',
        'region': 'Kilimanjaro',
        'region_code': 9,
        'district_code': 4,
        'lga': 'Hai',
        'ward': 'Shira',
        'population': 280,
        'public_meeting': True,
        'recorded_by': 'GeoData Consultants Ltd',
        'scheme_management': 'VWC',
        'scheme_name': 'Roman',
        'permit': False,
        'construction_year': 1999,
        'extraction_type': 'gravity',
        'extraction_type_group': 'gravity',
        'extraction_type_class': 'gravity',
        'management': 'vwc',
        'management_group': 'user-group',
        'payment': 'never pay',
        'payment_type': 'never pay',
        'water_quality': 'soft',
        'quality_group': 'good',
        'quantity': 'dry',
        'quantity_group': 'dry',
        'source': 'spring',
        'source_type': 'spring',
        'source_class': 'groundwater',
        'waterpoint_type': 'communal standpipe',
        'waterpoint_type_group': 'communal standpipe'
    }
    
    response = predict_pump_status(test_pump)
    print("📡 API Response:")
    print(json.dumps(response, indent=2))

def main():
    """Run all prediction examples"""
    print("🚀 Water Pump Prediction Examples")
    print("=" * 50)
    
    # Run examples
    example_single_prediction()
    example_batch_prediction()
    example_feature_importance()
    example_api_style_prediction()
    
    print("\n✅ All examples completed!")
    print("\nYou can now use these patterns to:")
    print("• Integrate predictions into web apps")
    print("• Create API endpoints")
    print("• Batch process pump data")
    print("• Analyze feature importance")

if __name__ == "__main__":
    main()