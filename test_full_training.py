#!/usr/bin/env python3

import sys
import os
import pandas as pd
import time
sys.path.append('src')

from src.pipeline.train_pipeline import TrainPipeline
from src.logger import logging

def test_full_training():
    """Test model training with full Tanzania dataset"""
    
    try:
        # Check if data exists
        data_path = "artifacts/merged_train_data.csv"
        if not os.path.exists(data_path):
            print(f"❌ Data not found at {data_path}")
            return False
        
        df = pd.read_csv(data_path)
        print(f"✅ Found full dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
        
        # Initialize pipeline
        train_pipeline = TrainPipeline()
        
        # Record start time
        start_time = time.time()
        
        # Run training
        print("🚀 Starting full dataset training...")
        model_score = train_pipeline.run_pipeline(data_path)
        
        # Record end time
        end_time = time.time()
        training_time = end_time - start_time
        
        print(f"✅ Training completed in {training_time:.1f} seconds!")
        print(f"✅ Best model score: {model_score:.4f}")
        
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
    success = test_full_training()
    if success:
        print("\n🎉 Full dataset training test passed!")
    else:
        print("\n❌ Full dataset training test failed!")