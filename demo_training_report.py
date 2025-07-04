#!/usr/bin/env python3
"""
Demo script to show what the training report looks like
"""

from src.components.training_report import TrainingReportGenerator
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import pandas as pd

def create_demo_report():
    """Create a demo training report with sample data"""
    
    # Load some sample data for demo
    try:
        df = pd.read_csv("artifacts/merged_train_data.csv", nrows=1000)
        
        # Prepare data
        X = df.drop(['status_group', 'id'], axis=1, errors='ignore')
        y = df['status_group']
        
        # Simple preprocessing - just use numeric columns for demo
        X_numeric = X.select_dtypes(include=[np.number])
        
        # Fill missing values
        X_numeric = X_numeric.fillna(X_numeric.mean())
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_numeric, y, test_size=0.2, random_state=42
        )
        
        print("Creating demo training report...")
        
        # Initialize report generator
        report_generator = TrainingReportGenerator()
        
        # Dataset info
        dataset_info = {
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'total_features': X_train.shape[1],
            'target_classes': len(np.unique(y))
        }
        
        report_generator.start_training_session(dataset_info)
        
        # Train models and track performance
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=10, random_state=42),
            'XGBoost': XGBClassifier(n_estimators=10, random_state=42, eval_metric='mlogloss')
        }
        
        for model_name, model in models.items():
            print(f"Training {model_name}...")
            
            import time
            start_time = time.time()
            model.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            # Add results to report
            hyperparams = {
                'n_estimators': getattr(model, 'n_estimators', None),
                'max_depth': getattr(model, 'max_depth', None),
                'random_state': getattr(model, 'random_state', None)
            }
            
            report_generator.add_model_result(
                model_name=model_name,
                model=model,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                training_time=training_time,
                hyperparams=hyperparams
            )
        
        # Finalize report
        report_generator.set_best_model('XGBoost', 0.756)
        report_generator.finish_training_session()
        
        # Generate and save report
        report_path = report_generator.save_report("artifacts/demo_training_report.txt")
        
        if report_path:
            print(f"Demo report saved to: {report_path}")
            
            # Display preview
            with open(report_path, 'r') as f:
                content = f.read()
            
            print("\n" + "="*60)
            print("PREVIEW OF TRAINING REPORT:")
            print("="*60)
            
            lines = content.split('\n')
            for line in lines[:50]:  # First 50 lines
                print(line)
            
            if len(lines) > 50:
                print("... (report continues)")
            
        return report_path
        
    except Exception as e:
        print(f"Error creating demo report: {str(e)}")
        return None

if __name__ == "__main__":
    create_demo_report()
    
    print("\nThe training report includes:")
    print("• Model comparison table with accuracy, F1, precision, recall, and training time")
    print("• Detailed results for each algorithm")
    print("• Hyperparameter settings used")
    print("• Per-class performance metrics")
    print("• Confusion matrices")
    print("• Training duration and recommendations")
    print("• Best model selection summary")
    print("\nThis report is automatically generated when you train models in the Streamlit app!")