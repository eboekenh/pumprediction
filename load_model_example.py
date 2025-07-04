#!/usr/bin/env python3
"""
Example script showing how to load and use the trained model.pkl file
"""

import pickle
import pandas as pd
import numpy as np
from src.utils import load_object

def load_and_inspect_model():
    """Load and inspect the trained model"""
    
    print("🔍 Loading trained model...")
    
    try:
        # Method 1: Using the utility function
        model = load_object("artifacts/model.pkl")
        print(f"✅ Model loaded successfully: {type(model)}")
        
        # Method 2: Direct pickle loading (alternative)
        # with open("artifacts/model.pkl", "rb") as f:
        #     model = pickle.load(f)
        
        # Inspect model properties
        print(f"📊 Model type: {model.__class__.__name__}")
        
        # If it's a scikit-learn model, show some properties
        if hasattr(model, 'feature_importances_'):
            print(f"📈 Number of features: {len(model.feature_importances_)}")
            print(f"🎯 Model classes: {model.classes_}")
        
        # Show model parameters
        if hasattr(model, 'get_params'):
            params = model.get_params()
            print(f"⚙️ Model parameters: {len(params)} parameters")
            for key, value in list(params.items())[:5]:  # Show first 5 params
                print(f"   {key}: {value}")
            if len(params) > 5:
                print(f"   ... and {len(params) - 5} more parameters")
        
        return model
        
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return None

def load_preprocessor():
    """Load the preprocessor"""
    
    print("\n🔧 Loading preprocessor...")
    
    try:
        preprocessor = load_object("artifacts/preprocessor.pkl")
        print(f"✅ Preprocessor loaded successfully: {type(preprocessor)}")
        
        # Show preprocessor info
        if hasattr(preprocessor, 'named_transformers_'):
            transformers = preprocessor.named_transformers_
            print(f"🔄 Number of transformers: {len(transformers)}")
            for name, transformer in transformers.items():
                print(f"   {name}: {type(transformer)}")
        
        return preprocessor
        
    except Exception as e:
        print(f"❌ Error loading preprocessor: {str(e)}")
        return None

def load_label_encoder():
    """Load the label encoder"""
    
    print("\n🏷️ Loading label encoder...")
    
    try:
        label_encoder = load_object("artifacts/label_encoder.pkl")
        print(f"✅ Label encoder loaded successfully: {type(label_encoder)}")
        
        # Show class mappings
        if hasattr(label_encoder, 'classes_'):
            print(f"🎯 Classes: {label_encoder.classes_}")
        
        return label_encoder
        
    except Exception as e:
        print(f"❌ Error loading label encoder: {str(e)}")
        return None

def make_sample_prediction():
    """Make a sample prediction using the loaded model"""
    
    print("\n🎯 Making sample prediction...")
    
    try:
        # Load all components
        model = load_object("artifacts/model.pkl")
        preprocessor = load_object("artifacts/preprocessor.pkl")
        label_encoder = load_object("artifacts/label_encoder.pkl")
        
        # Load sample data
        if not pd.io.common.file_exists("artifacts/merged_train_data.csv"):
            print("❌ No sample data found")
            return
        
        # Load a few samples for prediction
        df = pd.read_csv("artifacts/merged_train_data.csv", nrows=5)
        
        # Remove target column and id
        features = df.drop(['status_group', 'id'], axis=1, errors='ignore')
        
        print(f"📊 Sample features shape: {features.shape}")
        print(f"📋 Sample features: {list(features.columns)[:5]}...")
        
        # Transform features
        features_transformed = preprocessor.transform(features)
        print(f"🔄 Transformed features shape: {features_transformed.shape}")
        
        # Make predictions
        predictions = model.predict(features_transformed)
        probabilities = model.predict_proba(features_transformed)
        
        # Decode predictions
        predicted_labels = label_encoder.inverse_transform(predictions)
        
        print(f"🎯 Predictions: {predicted_labels}")
        print(f"📊 Prediction probabilities shape: {probabilities.shape}")
        
        # Show detailed results
        for i, (pred, prob) in enumerate(zip(predicted_labels, probabilities)):
            print(f"   Sample {i+1}: {pred} (confidence: {max(prob):.3f})")
        
    except Exception as e:
        print(f"❌ Error making prediction: {str(e)}")

def main():
    """Main function to demonstrate model loading and usage"""
    
    print("🚀 Model Loading and Inspection Example")
    print("=" * 50)
    
    # Load and inspect model
    model = load_and_inspect_model()
    
    # Load preprocessor
    preprocessor = load_preprocessor()
    
    # Load label encoder
    label_encoder = load_label_encoder()
    
    # Make sample prediction
    if model and preprocessor and label_encoder:
        make_sample_prediction()
    
    print("\n✅ Model inspection completed!")

if __name__ == "__main__":
    main()