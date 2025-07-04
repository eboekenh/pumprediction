#!/usr/bin/env python3

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append('src')

from src.components.file_handler import FileHandler
from src.logger import logging

def process_real_tanzania_data():
    """Process the real Tanzania water pump dataset"""
    
    try:
        # File paths
        train_features_path = "uploaded_data/training_features.csv"
        train_labels_path = "uploaded_data/training_labels.csv"
        test_features_path = "uploaded_data/test_features.csv"
        
        # Check if files exist
        for path in [train_features_path, train_labels_path, test_features_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"File not found: {path}")
        
        # Initialize FileHandler
        file_handler = FileHandler()
        
        # Process the files
        print("Processing Tanzania water pump dataset...")
        results = file_handler.process_uploaded_files([
            train_features_path,
            train_labels_path, 
            test_features_path
        ])
        
        if results['success']:
            print("✅ Dataset processed successfully!")
            
            # Display summary
            summary = results['summary']
            print(f"\n📊 Dataset Summary:")
            print(f"Training samples: {summary['train_data_info']['shape'][0]:,}")
            print(f"Test samples: {summary['test_data_info']['shape'][0]:,}")
            print(f"Total features: {len(summary['train_data_info']['numerical_columns']) + len(summary['train_data_info']['categorical_columns'])}")
            print(f"Numerical features: {len(summary['train_data_info']['numerical_columns'])}")
            print(f"Categorical features: {len(summary['train_data_info']['categorical_columns'])}")
            
            # Target distribution
            if 'target_distribution' in summary['train_data_info']:
                print(f"\n🎯 Target Distribution:")
                target_dist = summary['train_data_info']['target_distribution']
                total = sum(target_dist.values())
                for status, count in target_dist.items():
                    percentage = (count / total) * 100
                    print(f"  {status}: {count:,} ({percentage:.1f}%)")
            
            # Missing values info
            print(f"\n❓ Data Quality:")
            print(f"Columns with missing values: {summary['train_data_info']['missing_values_count']}")
            
            # Files saved
            print(f"\n💾 Files saved:")
            for file_type, path in results['saved_files'].items():
                print(f"  {file_type}: {path}")
            
            return results
            
        else:
            print(f"❌ Error processing dataset: {results['error']}")
            return None
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return None

if __name__ == "__main__":
    results = process_real_tanzania_data()