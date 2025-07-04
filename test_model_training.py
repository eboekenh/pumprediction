#!/usr/bin/env python3

import sys
import os
sys.path.append('src')

from src.pipeline.train_pipeline import TrainPipeline
from src.logger import logging

def test_model_training():
    """Test model training with Tanzania dataset"""
    
    try:
        # Check if data exists
        data_path = "artifacts/merged_train_data.csv"
        if not os.path.exists(data_path):
            print(f"❌ Data not found at {data_path}")
            return
        
        print(f"✅ Found data at {data_path}")
        
        # Initialize pipeline
        train_pipeline = TrainPipeline()
        
        # Run training
        print("🚀 Starting model training...")
        model_score = train_pipeline.run_pipeline(data_path)
        
        print(f"✅ Model training completed! Score: {model_score:.4f}")
        
        # Check if artifacts were created
        artifacts_created = []
        for artifact in ["artifacts/model.pkl", "artifacts/preprocessor.pkl", "artifacts/model_report.pkl"]:
            if os.path.exists(artifact):
                artifacts_created.append(artifact)
                print(f"✅ Created: {artifact}")
            else:
                print(f"❌ Missing: {artifact}")
        
        return len(artifacts_created) == 3
        
    except Exception as e:
        print(f"❌ Error during training: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_model_training()
    if success:
        print("\n🎉 Model training test passed!")
    else:
        print("\n❌ Model training test failed!")