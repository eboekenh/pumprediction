#!/usr/bin/env python3
"""
Flask API for water pump predictions
"""

from flask import Flask, request, jsonify
import pandas as pd
from src.utils import load_object
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

class ModelService:
    """Service to load and use the trained model"""
    
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.label_encoder = None
        self.load_model()
    
    def load_model(self):
        """Load the trained model components"""
        try:
            self.model = load_object("artifacts/model.pkl")
            self.preprocessor = load_object("artifacts/preprocessor.pkl")
            self.label_encoder = load_object("artifacts/label_encoder.pkl")
            logger.info("Model components loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise e
    
    def predict(self, pump_data):
        """Make prediction for pump data"""
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
            
            # Convert to serializable types
            predicted_label = self.label_encoder.inverse_transform([prediction])[0]
            confidence = float(max(probabilities))
            
            return {
                'prediction': predicted_label,
                'confidence': round(confidence, 3),
                'probabilities': {
                    str(class_name): round(float(prob), 3) 
                    for class_name, prob in zip(self.label_encoder.classes_, probabilities)
                }
            }
            
        except Exception as e:
            raise Exception(f"Prediction error: {str(e)}")

# Initialize model service
try:
    model_service = ModelService()
except Exception as e:
    logger.error(f"Failed to initialize model service: {str(e)}")
    model_service = None

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    if model_service and model_service.model:
        return jsonify({'status': 'healthy', 'model_loaded': True})
    else:
        return jsonify({'status': 'unhealthy', 'model_loaded': False}), 500

@app.route('/predict', methods=['POST'])
def predict_pump():
    """Predict water pump functionality"""
    try:
        if not model_service:
            return jsonify({'error': 'Model service not available'}), 500
        
        # Get data from request
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Make prediction
        result = model_service.predict(data)
        
        return jsonify({
            'status': 'success',
            'data': result
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """Predict for multiple pumps"""
    try:
        if not model_service:
            return jsonify({'error': 'Model service not available'}), 500
        
        # Get data from request
        data = request.get_json()
        
        if not data or 'pumps' not in data:
            return jsonify({'error': 'No pump data provided'}), 400
        
        pumps = data['pumps']
        if not isinstance(pumps, list):
            return jsonify({'error': 'Pumps data must be a list'}), 400
        
        # Make predictions
        results = []
        for i, pump_data in enumerate(pumps):
            try:
                result = model_service.predict(pump_data)
                result['pump_id'] = i
                results.append(result)
            except Exception as e:
                results.append({'pump_id': i, 'error': str(e)})
        
        return jsonify({
            'status': 'success',
            'data': results,
            'summary': {
                'total_pumps': len(pumps),
                'successful_predictions': len([r for r in results if 'error' not in r])
            }
        })
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/model/info', methods=['GET'])
def model_info():
    """Get model information"""
    try:
        if not model_service:
            return jsonify({'error': 'Model service not available'}), 500
        
        info = {
            'model_type': model_service.model.__class__.__name__ if model_service.model else None,
            'classes': model_service.label_encoder.classes_.tolist() if model_service.label_encoder else None,
            'features_count': len(model_service.model.feature_importances_) if hasattr(model_service.model, 'feature_importances_') else None
        }
        
        return jsonify({
            'status': 'success',
            'data': info
        })
        
    except Exception as e:
        logger.error(f"Model info error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/', methods=['GET'])
def home():
    """API documentation"""
    docs = {
        'name': 'Water Pump Prediction API',
        'version': '1.0.0',
        'endpoints': {
            'GET /health': 'Check API health status',
            'POST /predict': 'Predict single pump functionality',
            'POST /predict/batch': 'Predict multiple pumps functionality',
            'GET /model/info': 'Get model information'
        },
        'example_request': {
            'amount_tsh': 6000.0,
            'funder': 'World Bank',
            'gps_height': 1390,
            'longitude': 34.938093,
            'latitude': -9.856323,
            'population': 280,
            'construction_year': 2010,
            'extraction_type': 'gravity',
            'management': 'vwc',
            'payment': 'pay annually',
            'water_quality': 'soft',
            'quantity': 'enough',
            'source': 'spring',
            'waterpoint_type': 'communal standpipe'
        }
    }
    
    return jsonify(docs)

if __name__ == '__main__':
    print("🚀 Starting Water Pump Prediction API...")
    print("📋 Available endpoints:")
    print("  GET  /         - API documentation")
    print("  GET  /health   - Health check")
    print("  POST /predict  - Single prediction")
    print("  POST /predict/batch - Batch predictions")
    print("  GET  /model/info - Model information")
    print("\n🔗 API will run on http://0.0.0.0:5001")
    
    app.run(host='0.0.0.0', port=5001, debug=True)