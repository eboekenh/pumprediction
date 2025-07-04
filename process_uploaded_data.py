#!/usr/bin/env python3

import os
import sys
from src.components.file_handler import FileHandler
from src.components.advanced_eda import AdvancedEDA

def process_attached_files():
    """Process the three attached CSV files"""
    
    # Paths to the uploaded files
    file_paths = [
        "attached_assets/4910797b-ee55-40a7-8668-10efd5c1b960_1751639642324.csv",  # Training features
        "attached_assets/0bf8bc6e-30d0-4c50-956a-603fc693d966_1751639642325.csv",  # Training labels
        "attached_assets/702ddfc5-68cd-4d1d-a0de-f5f566f76d91_1751639642325.csv"   # Test features
    ]
    
    print("Processing Tanzania Water Pump Dataset...")
    print("=" * 50)
    
    # Initialize file handler
    file_handler = FileHandler()
    
    try:
        # Process the files
        results = file_handler.process_uploaded_files(file_paths)
        
        if results['success']:
            print("✅ Files processed successfully!")
            print("\nIdentified Files:")
            for file_type, path in results['identified_files'].items():
                print(f"  - {file_type.replace('_', ' ').title()}: {os.path.basename(path)}")
            
            print("\nTraining Data Summary:")
            train_info = results['summary']['train_data_info']
            print(f"  - Shape: {train_info['shape']}")
            print(f"  - Target distribution: {train_info['target_distribution']}")
            print(f"  - Numerical columns: {len(train_info['numerical_columns'])}")
            print(f"  - Categorical columns: {len(train_info['categorical_columns'])}")
            
            if 'test_data_info' in results['summary']:
                test_info = results['summary']['test_data_info']
                print(f"\nTest Data Summary:")
                print(f"  - Shape: {test_info['shape']}")
                print(f"  - Numerical columns: {len(test_info['numerical_columns'])}")
                print(f"  - Categorical columns: {len(test_info['categorical_columns'])}")
            
            # Validation results
            validation = results['validation_results']
            print(f"\nData Validation:")
            print(f"  - Column consistency: {'✅' if validation['column_consistency']['consistent'] else '❌'}")
            print(f"  - Data type consistency: {'✅' if validation['dtype_consistency']['consistent'] else '❌'}")
            print(f"  - Range consistency: {'✅' if validation['range_consistency']['consistent'] else '❌'}")
            
            # Now run advanced EDA on the merged training data
            print("\n" + "=" * 50)
            print("Running Advanced EDA Analysis...")
            
            # Load the merged training data
            import pandas as pd
            merged_train_path = results['saved_files']['merged_train']
            df = pd.read_csv(merged_train_path)
            
            # Initialize advanced EDA
            eda = AdvancedEDA()
            
            # Generate comprehensive report
            eda_report = eda.generate_comprehensive_report(df)
            
            print(f"\nAdvanced EDA Results:")
            print(f"  - Dataset shape: {eda_report['dataset_info']['shape']}")
            print(f"  - Missing values: {eda_report['dataset_info']['missing_values_total']} ({eda_report['dataset_info']['missing_percentage']:.1f}%)")
            print(f"  - Duplicate rows: {eda_report['dataset_info']['duplicate_rows']}")
            
            # Print preprocessing recommendations
            print(f"\nPreprocessing Recommendations ({len(eda_report['preprocessing_recommendations'])} items):")
            for i, rec in enumerate(eda_report['preprocessing_recommendations'][:5], 1):  # Show top 5
                print(f"  {i}. [{rec['priority']}] {rec['category']}: {rec['recommendation'][:80]}...")
            
            # Print feature engineering suggestions
            print(f"\nFeature Engineering Suggestions ({len(eda_report['feature_engineering_suggestions'])} items):")
            for i, sug in enumerate(eda_report['feature_engineering_suggestions'][:3], 1):  # Show top 3
                print(f"  {i}. {sug['category']}: {sug['suggestion'][:80]}...")
            
            # Save the EDA report as markdown
            report_file = eda.save_report_to_markdown(eda_report, "comprehensive_eda_report.md")
            print(f"\n📄 Comprehensive EDA report saved to: {report_file}")
            
            print(f"\n✅ Processing complete! Files saved to artifacts directory.")
            print(f"Ready for model training and analysis in the Streamlit app.")
            
        else:
            print(f"❌ Error processing files: {results['error']}")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    process_attached_files()