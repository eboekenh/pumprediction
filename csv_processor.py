#!/usr/bin/env python3
"""
CSV file processor for water pump predictions
"""

import pandas as pd
import numpy as np
from src.utils import load_object
import csv
import os

class CSVPumpProcessor:
    """Process CSV files for water pump predictions"""
    
    def __init__(self):
        """Initialize with trained model components"""
        self.model = load_object("artifacts/model.pkl")
        self.preprocessor = load_object("artifacts/preprocessor.pkl")
        self.label_encoder = load_object("artifacts/label_encoder.pkl")
        print("Model components loaded successfully")
    
    def process_csv_file(self, input_file, output_file=None):
        """
        Process a CSV file and add predictions
        
        Args:
            input_file: Path to input CSV file
            output_file: Path to output CSV file (optional)
        """
        try:
            # Read CSV file
            df = pd.read_csv(input_file)
            print(f"Loaded {len(df)} records from {input_file}")
            
            # Make predictions
            predictions_data = self._predict_dataframe(df)
            
            # Add predictions to original data
            result_df = df.copy()
            result_df['predicted_status'] = predictions_data['predictions']
            result_df['prediction_confidence'] = predictions_data['confidences']
            
            # Add probability columns
            for class_name in self.label_encoder.classes_:
                result_df[f'prob_{class_name.replace(" ", "_")}'] = predictions_data['probabilities'][class_name]
            
            # Save results
            if output_file:
                result_df.to_csv(output_file, index=False)
                print(f"Results saved to {output_file}")
            
            return result_df
            
        except Exception as e:
            print(f"Error processing CSV: {str(e)}")
            return None
    
    def _predict_dataframe(self, df):
        """Make predictions for a DataFrame"""
        # Prepare features
        features_df = df.copy()
        
        # Remove known non-feature columns
        columns_to_remove = ['id', 'status_group', 'predicted_status', 'prediction_confidence']
        for col in columns_to_remove:
            if col in features_df.columns:
                features_df = features_df.drop(col, axis=1)
        
        # Transform features
        features_transformed = self.preprocessor.transform(features_df)
        
        # Make predictions
        predictions = self.model.predict(features_transformed)
        probabilities = self.model.predict_proba(features_transformed)
        
        # Convert to readable format
        predicted_labels = self.label_encoder.inverse_transform(predictions)
        confidences = np.max(probabilities, axis=1)
        
        # Organize probabilities by class
        prob_dict = {}
        for i, class_name in enumerate(self.label_encoder.classes_):
            prob_dict[class_name] = probabilities[:, i]
        
        return {
            'predictions': predicted_labels,
            'confidences': confidences,
            'probabilities': prob_dict
        }
    
    def create_summary_report(self, df_with_predictions):
        """Create a summary report of predictions"""
        if df_with_predictions is None:
            return None
        
        # Count predictions
        pred_counts = df_with_predictions['predicted_status'].value_counts()
        
        # Calculate confidence statistics
        avg_confidence = df_with_predictions['prediction_confidence'].mean()
        min_confidence = df_with_predictions['prediction_confidence'].min()
        max_confidence = df_with_predictions['prediction_confidence'].max()
        
        # High confidence predictions (>0.8)
        high_conf = df_with_predictions[df_with_predictions['prediction_confidence'] > 0.8]
        
        report = {
            'total_pumps': len(df_with_predictions),
            'predictions': pred_counts.to_dict(),
            'confidence_stats': {
                'average': round(avg_confidence, 3),
                'minimum': round(min_confidence, 3),
                'maximum': round(max_confidence, 3),
                'high_confidence_count': len(high_conf)
            },
            'recommendations': {
                'immediate_attention': len(df_with_predictions[
                    df_with_predictions['predicted_status'] == 'non functional'
                ]),
                'needs_repair': len(df_with_predictions[
                    df_with_predictions['predicted_status'] == 'functional needs repair'
                ]),
                'maintenance_ok': len(df_with_predictions[
                    df_with_predictions['predicted_status'] == 'functional'
                ])
            }
        }
        
        return report

def demo_csv_processing():
    """Demo CSV processing with sample data"""
    
    print("CSV Processing Demo")
    print("=" * 50)
    
    processor = CSVPumpProcessor()
    
    # Process sample data
    input_file = "artifacts/merged_train_data.csv"
    output_file = "artifacts/predictions_output.csv"
    
    # Process first 100 records as demo
    df_sample = pd.read_csv(input_file, nrows=100)
    
    # Remove target column for prediction
    if 'status_group' in df_sample.columns:
        df_sample = df_sample.drop('status_group', axis=1)
    
    # Save as temporary input file
    temp_input = "artifacts/temp_input.csv"
    df_sample.to_csv(temp_input, index=False)
    
    # Process the file
    result_df = processor.process_csv_file(temp_input, output_file)
    
    if result_df is not None:
        # Create summary report
        report = processor.create_summary_report(result_df)
        
        print("\nSummary Report:")
        print(f"Total pumps processed: {report['total_pumps']}")
        print("\nPrediction breakdown:")
        for status, count in report['predictions'].items():
            print(f"  {status}: {count}")
        
        print(f"\nConfidence statistics:")
        print(f"  Average: {report['confidence_stats']['average']}")
        print(f"  Range: {report['confidence_stats']['minimum']} - {report['confidence_stats']['maximum']}")
        print(f"  High confidence (>0.8): {report['confidence_stats']['high_confidence_count']}")
        
        print(f"\nRecommendations:")
        print(f"  Immediate attention needed: {report['recommendations']['immediate_attention']}")
        print(f"  Schedule repairs: {report['recommendations']['needs_repair']}")
        print(f"  Continue regular maintenance: {report['recommendations']['maintenance_ok']}")
        
        # Show sample predictions
        print(f"\nSample predictions:")
        sample_data = result_df[['predicted_status', 'prediction_confidence']].head()
        for i, row in sample_data.iterrows():
            print(f"  Pump {i+1}: {row['predicted_status']} (confidence: {row['prediction_confidence']:.3f})")
    
    # Clean up temp file
    if os.path.exists(temp_input):
        os.remove(temp_input)

if __name__ == "__main__":
    demo_csv_processing()
    
    print("\n" + "=" * 50)
    print("Usage Examples:")
    print("\n1. Process a CSV file:")
    print("   processor = CSVPumpProcessor()")
    print("   result = processor.process_csv_file('input.csv', 'output.csv')")
    print("\n2. Get summary report:")
    print("   report = processor.create_summary_report(result)")
    print("\n3. Process without saving:")
    print("   result = processor.process_csv_file('input.csv')")
    print("\nOutput CSV will include:")
    print("  - All original columns")
    print("  - predicted_status")
    print("  - prediction_confidence") 
    print("  - prob_functional, prob_functional_needs_repair, prob_non_functional")