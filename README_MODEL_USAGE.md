# Water Pump Model Usage Guide

Your trained XGBoost model is now ready to use! Here's a comprehensive guide on how to open and use the `model.pkl` file in different ways.

## Model Details
- **Model Type**: XGBoost Classifier
- **Accuracy**: 76.2%
- **Classes**: functional, functional needs repair, non functional
- **Features**: 341 transformed features

## Files Created
- `artifacts/model.pkl` - Trained XGBoost model
- `artifacts/preprocessor.pkl` - Data preprocessing pipeline
- `artifacts/label_encoder.pkl` - Target label encoder

## Usage Methods

### 1. Basic Python Usage

```python
import pandas as pd
from src.utils import load_object

# Load model components
model = load_object("artifacts/model.pkl")
preprocessor = load_object("artifacts/preprocessor.pkl")
label_encoder = load_object("artifacts/label_encoder.pkl")

# Prepare your data
pump_data = {
    'amount_tsh': 6000.0,
    'funder': 'World Bank',
    'gps_height': 1390,
    'construction_year': 2010,
    'extraction_type': 'gravity',
    'management': 'vwc',
    'payment': 'pay annually',
    'water_quality': 'soft',
    'quantity': 'enough',
    'source': 'spring',
    'waterpoint_type': 'communal standpipe'
    # ... add all required features
}

# Make prediction
df = pd.DataFrame([pump_data])
features_transformed = preprocessor.transform(df)
prediction = model.predict(features_transformed)[0]
predicted_label = label_encoder.inverse_transform([prediction])[0]

print(f"Prediction: {predicted_label}")
```

### 2. Using the API Class

```python
from simple_prediction_api import WaterPumpAPI

# Initialize API
api = WaterPumpAPI()

# Single prediction
result = api.predict(pump_data)
print(f"Status: {result['prediction']}")
print(f"Confidence: {result['confidence']}")
```

### 3. CSV File Processing

```python
from csv_processor import CSVPumpProcessor

# Initialize processor
processor = CSVPumpProcessor()

# Process CSV file
result_df = processor.process_csv_file('input.csv', 'output_with_predictions.csv')

# Get summary report
report = processor.create_summary_report(result_df)
print(f"Pumps needing immediate attention: {report['recommendations']['immediate_attention']}")
```

### 4. Web API Integration (Flask)

Install Flask first:
```bash
pip install flask
```

Then run the API:
```bash
python flask_api.py
```

API endpoints:
- `GET /health` - Check API status
- `POST /predict` - Single prediction
- `POST /predict/batch` - Batch predictions
- `GET /model/info` - Model information

Example API request:
```bash
curl -X POST http://localhost:5001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "amount_tsh": 6000.0,
    "funder": "World Bank",
    "gps_height": 1390,
    "construction_year": 2010,
    "extraction_type": "gravity",
    "management": "vwc"
  }'
```

### 5. Streamlit Web Interface

The trained model is already integrated into the Streamlit app. Go to the **Predictions** page to:
- Make single predictions with a form
- Upload CSV files for batch processing
- View prediction results with confidence scores

## Example Outputs

### Single Prediction
```json
{
  "prediction": "functional",
  "confidence": 0.834,
  "probabilities": {
    "functional": 0.834,
    "functional needs repair": 0.045,
    "non functional": 0.121
  }
}
```

### CSV Processing Output
The processed CSV includes:
- All original columns
- `predicted_status` - The prediction
- `prediction_confidence` - Confidence score
- `prob_functional` - Probability for functional
- `prob_functional_needs_repair` - Probability for needs repair
- `prob_non_functional` - Probability for non-functional

### Batch Summary
```
Total pumps processed: 100
Prediction breakdown:
  functional: 72
  non functional: 26
  functional needs repair: 2

Recommendations:
  Immediate attention needed: 26
  Schedule repairs: 2
  Continue regular maintenance: 72
```

## Required Features

Your input data must include these features:
- `amount_tsh` - Amount of water available
- `date_recorded` - Date when recorded
- `funder` - Organization that funded the well
- `gps_height` - Altitude of the well
- `installer` - Organization that installed the well
- `longitude`, `latitude` - GPS coordinates
- `population` - Population around the well
- `construction_year` - Year the well was constructed
- `extraction_type` - How water is extracted
- `management` - Who manages the well
- `payment` - Payment system
- `water_quality` - Quality of the water
- `quantity` - Amount of water
- `source` - Source of the water
- `waterpoint_type` - Type of water point

## Integration Examples

### Web Application
```python
# In your web app
from simple_prediction_api import WaterPumpAPI

api = WaterPumpAPI()

@app.route('/check_pump', methods=['POST'])
def check_pump():
    pump_data = request.json
    result = api.predict(pump_data)
    return jsonify(result)
```

### Monitoring System
```python
# Regular monitoring
processor = CSVPumpProcessor()

# Process daily reports
daily_report = processor.process_csv_file('daily_pumps.csv')
summary = processor.create_summary_report(daily_report)

# Alert if many pumps need attention
if summary['recommendations']['immediate_attention'] > 10:
    send_alert("High number of pumps need immediate attention")
```

### Mobile App Backend
```python
# Mobile API endpoint
def mobile_predict(pump_location, pump_features):
    api = WaterPumpAPI()
    
    # Add location data
    pump_features.update({
        'longitude': pump_location['lng'],
        'latitude': pump_location['lat']
    })
    
    result = api.predict(pump_features)
    
    return {
        'status': result['prediction'],
        'confidence': result['confidence'],
        'recommendation': get_maintenance_recommendation(result['prediction'])
    }
```

## Testing Your Integration

Use the provided examples to test:

1. **Run the basic example**:
   ```bash
   python load_model_example.py
   ```

2. **Test the API**:
   ```bash
   python simple_prediction_api.py
   ```

3. **Process a CSV file**:
   ```bash
   python csv_processor.py
   ```

4. **Start the Flask API**:
   ```bash
   python flask_api.py
   ```

## Troubleshooting

### Common Issues:
1. **Missing features**: Ensure all required features are present
2. **Data types**: Match the data types from training data
3. **Feature names**: Use exact feature names from the original dataset
4. **Preprocessing**: Always use the provided preprocessor

### Error Handling:
```python
try:
    result = api.predict(pump_data)
    if result.get('status') == 'error':
        print(f"Prediction error: {result['message']}")
except Exception as e:
    print(f"System error: {str(e)}")
```

## Performance Tips

1. **Batch Processing**: Use batch predictions for multiple pumps
2. **Caching**: Cache the loaded model components
3. **Validation**: Validate input data before prediction
4. **Monitoring**: Log prediction requests and results

Your model is production-ready and can be integrated into any application using these methods!