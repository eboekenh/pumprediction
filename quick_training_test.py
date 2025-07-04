#!/usr/bin/env python3

import sys
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import numpy as np

sys.path.append('src')
from src.components.data_transformation import DataTransformation

def quick_training_test():
    """Quick training test with small subset"""
    
    try:
        # Load data
        data_path = "artifacts/merged_train_data.csv"
        if not os.path.exists(data_path):
            print("❌ Data not found")
            return False
        
        df = pd.read_csv(data_path)
        print(f"✅ Loaded data: {df.shape}")
        
        # Use only a small subset for quick testing (1000 samples)
        df_sample = df.sample(n=1000, random_state=42)
        print(f"✅ Using sample: {df_sample.shape}")
        
        # Convert boolean columns to strings
        for col in df_sample.select_dtypes(include=['object']).columns:
            unique_vals = df_sample[col].dropna().unique()
            if len(unique_vals) <= 2 and all(isinstance(x, bool) for x in unique_vals):
                df_sample[col] = df_sample[col].astype(str)
                print(f"✅ Converted boolean column: {col}")
        
        # Split target and features
        if 'status_group' not in df_sample.columns:
            print("❌ Target column not found")
            return False
        
        X = df_sample.drop(['status_group', 'id'], axis=1, errors='ignore')
        y = df_sample['status_group']
        
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"✅ Split data: Train={X_train.shape}, Test={X_test.shape}")
        
        # Initialize transformation
        data_transformation = DataTransformation()
        
        # Get preprocessor
        preprocessor = data_transformation.get_data_transformer_object(X_train)
        
        # Transform data
        X_train_transformed = preprocessor.fit_transform(X_train)
        X_test_transformed = preprocessor.transform(X_test)
        
        print(f"✅ Transformed data: Train={X_train_transformed.shape}, Test={X_test_transformed.shape}")
        
        # Encode target
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train)
        y_test_encoded = label_encoder.transform(y_test)
        
        # Train simple model
        model = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
        model.fit(X_train_transformed, y_train_encoded)
        
        # Predict and evaluate
        y_pred = model.predict(X_test_transformed)
        accuracy = accuracy_score(y_test_encoded, y_pred)
        
        print(f"✅ Model trained successfully! Accuracy: {accuracy:.4f}")
        
        # Save artifacts for testing
        import pickle
        os.makedirs("artifacts", exist_ok=True)
        
        with open("artifacts/model.pkl", "wb") as f:
            pickle.dump(model, f)
        
        with open("artifacts/preprocessor.pkl", "wb") as f:
            pickle.dump(preprocessor, f)
            
        model_report = {
            'model_name': 'RandomForest',
            'accuracy': accuracy,
            'best_params': model.get_params(),
            'model_report': {'RandomForest': accuracy}
        }
        
        with open("artifacts/model_report.pkl", "wb") as f:
            pickle.dump(model_report, f)
        
        print("✅ Artifacts saved successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

if __name__ == "__main__":
    success = quick_training_test()
    if success:
        print("\n🎉 Quick training test successful!")
    else:
        print("\n❌ Quick training test failed!")