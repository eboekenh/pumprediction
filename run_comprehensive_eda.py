#!/usr/bin/env python3

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

# Add src to path
sys.path.append('src')

from src.components.advanced_eda import AdvancedEDA
from src.components.file_handler import FileHandler

def run_comprehensive_eda():
    """Run comprehensive EDA on the full Tanzania water pump dataset"""
    
    print("=" * 80)
    print("COMPREHENSIVE EDA ANALYSIS - TANZANIA WATER PUMP DATASET")
    print("=" * 80)
    print(f"Analysis started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Load the merged training data
    data_path = "artifacts/merged_train_data.csv"
    
    if not os.path.exists(data_path):
        print("❌ Error: No processed data found.")
        print("Please run the data processing script first or upload data through the Streamlit app.")
        return
    
    print(f"📊 Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"✅ Dataset loaded successfully: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print()
    
    # Initialize Advanced EDA
    print("🔬 Initializing Advanced EDA Analysis...")
    eda = AdvancedEDA()
    print()
    
    # 1. BASIC DATASET INFORMATION
    print("1. BASIC DATASET INFORMATION")
    print("-" * 40)
    print(f"Dataset Shape: {df.shape}")
    print(f"Total Missing Values: {df.isnull().sum().sum():,}")
    print(f"Missing Percentage: {(df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100:.2f}%")
    print(f"Duplicate Rows: {df.duplicated().sum():,}")
    print(f"Memory Usage: {df.memory_usage().sum() / 1024 / 1024:.2f} MB")
    print()
    
    # 2. TARGET VARIABLE ANALYSIS
    print("2. TARGET VARIABLE ANALYSIS")
    print("-" * 40)
    if 'status_group' in df.columns:
        target_dist = df['status_group'].value_counts()
        target_pct = df['status_group'].value_counts(normalize=True) * 100
        
        print("Target Distribution:")
        for status, count in target_dist.items():
            pct = target_pct[status]
            print(f"  - {status}: {count:,} ({pct:.1f}%)")
        
        # Check class imbalance
        max_class_pct = target_pct.max()
        min_class_pct = target_pct.min()
        imbalance_ratio = max_class_pct / min_class_pct
        
        print(f"\nClass Imbalance Analysis:")
        print(f"  - Majority class: {target_pct.idxmax()} ({max_class_pct:.1f}%)")
        print(f"  - Minority class: {target_pct.idxmin()} ({min_class_pct:.1f}%)")
        print(f"  - Imbalance ratio: {imbalance_ratio:.1f}:1")
        
        if imbalance_ratio > 3:
            print("  ⚠️  Significant class imbalance detected - consider balancing techniques")
        else:
            print("  ✅ Class distribution is relatively balanced")
    print()
    
    # 3. DATA TYPES ANALYSIS
    print("3. DATA TYPES ANALYSIS")
    print("-" * 40)
    data_types = eda.analyze_data_types(df)
    
    print(f"Numerical Features ({len(data_types['numerical'])}): {data_types['numerical']}")
    print(f"Categorical Features ({len(data_types['categorical'])}): {data_types['categorical']}")
    print()
    
    # Display statistics for numerical features
    if data_types['numerical']:
        print("Numerical Features Statistics:")
        num_stats = df[data_types['numerical']].describe()
        print(num_stats.round(2))
        print()
    
    # Display categorical features info
    if data_types['categorical']:
        print("Categorical Features Information:")
        for col in data_types['categorical']:
            unique_count = df[col].nunique()
            most_frequent = df[col].mode().iloc[0] if len(df[col].mode()) > 0 else 'N/A'
            missing_count = df[col].isnull().sum()
            print(f"  - {col}: {unique_count} unique values, most frequent: {most_frequent}, missing: {missing_count}")
        print()
    
    # 4. MISSING VALUES ANALYSIS
    print("4. MISSING VALUES ANALYSIS")
    print("-" * 40)
    missing_data = df.isnull().sum()
    missing_percentage = (missing_data / len(df)) * 100
    
    cols_with_missing = missing_data[missing_data > 0]
    
    if len(cols_with_missing) > 0:
        print("Columns with missing values:")
        for col, count in cols_with_missing.items():
            pct = missing_percentage[col]
            print(f"  - {col}: {count:,} ({pct:.1f}%)")
        
        # Categorize by missing percentage
        high_missing = missing_data[missing_percentage > 50].index.tolist()
        medium_missing = missing_data[(missing_percentage > 10) & (missing_percentage <= 50)].index.tolist()
        low_missing = missing_data[(missing_percentage > 0) & (missing_percentage <= 10)].index.tolist()
        
        if high_missing:
            print(f"\n  High Missing (>50%): {high_missing}")
        if medium_missing:
            print(f"  Medium Missing (10-50%): {medium_missing}")
        if low_missing:
            print(f"  Low Missing (<10%): {low_missing}")
    else:
        print("✅ No missing values found in the dataset!")
    print()
    
    # 5. CORRELATION ANALYSIS
    print("5. COMPREHENSIVE CORRELATION ANALYSIS")
    print("-" * 40)
    print("Calculating correlations (this may take a few minutes for the full dataset)...")
    
    correlation_results = eda.calculate_correlation_matrix(df)
    
    # Pearson correlations (numerical features)
    if 'pearson' in correlation_results:
        print("\n📊 Numerical Features Correlation (Pearson):")
        pearson_corr = correlation_results['pearson']
        
        # Find highly correlated pairs
        high_corr_pairs = []
        for i in range(len(pearson_corr.columns)):
            for j in range(i+1, len(pearson_corr.columns)):
                corr_val = pearson_corr.iloc[i, j]
                if abs(corr_val) > 0.7:  # Threshold for high correlation
                    high_corr_pairs.append((pearson_corr.columns[i], pearson_corr.columns[j], corr_val))
        
        if high_corr_pairs:
            print("  High correlations (|r| > 0.7):")
            for col1, col2, corr in high_corr_pairs:
                print(f"    - {col1} ↔ {col2}: {corr:.3f}")
        else:
            print("  ✅ No highly correlated numerical features found")
    
    # Spearman correlations
    if 'spearman' in correlation_results:
        print("\n📊 Spearman Correlation (Rank-based):")
        spearman_corr = correlation_results['spearman']
        print("  Calculated successfully for non-linear relationships")
    
    # Categorical correlations
    if 'cramers_v' in correlation_results:
        print("\n📊 Categorical Features Correlation (Cramér's V):")
        cramers_matrix = correlation_results['cramers_v']
        print("  Calculated successfully for categorical associations")
    
    # Feature-target relationships
    if 'mutual_info_numerical' in correlation_results:
        print("\n🎯 Feature-Target Relationships (Mutual Information):")
        
        # Top numerical features
        mi_num = correlation_results['mutual_info_numerical'].sort_values(ascending=False)
        print("  Top numerical features by importance:")
        for i, (feature, importance) in enumerate(mi_num.head(10).items(), 1):
            print(f"    {i:2d}. {feature}: {importance:.4f}")
    
    if 'mutual_info_categorical' in correlation_results:
        # Top categorical features
        mi_cat = correlation_results['mutual_info_categorical'].sort_values(ascending=False)
        print("\n  Top categorical features by importance:")
        for i, (feature, importance) in enumerate(mi_cat.head(10).items(), 1):
            print(f"    {i:2d}. {feature}: {importance:.4f}")
    
    print()
    
    # 6. OUTLIER ANALYSIS
    print("6. OUTLIER ANALYSIS")
    print("-" * 40)
    print("Detecting outliers using multiple methods...")
    
    outlier_results = eda.detect_outliers(df)
    
    print("\nOutlier Summary by Column:")
    for col, methods in outlier_results.items():
        if col != 'multivariate':
            print(f"  {col}:")
            for method, stats in methods.items():
                print(f"    - {method.upper()}: {stats['count']:,} outliers ({stats['percentage']:.1f}%)")
    
    if 'multivariate' in outlier_results:
        multi_stats = outlier_results['multivariate']
        print(f"\nMultivariate Outliers ({multi_stats['method']}):")
        print(f"  - Outliers: {multi_stats['count']:,} ({multi_stats['percentage']:.1f}%)")
    
    print()
    
    # 7. PREPROCESSING RECOMMENDATIONS
    print("7. PREPROCESSING RECOMMENDATIONS")
    print("-" * 40)
    print("Generating preprocessing recommendations...")
    
    recommendations = eda.generate_preprocessing_recommendations(df)
    
    # Group by priority
    priorities = ['High', 'Medium', 'Low']
    for priority in priorities:
        priority_recs = [rec for rec in recommendations if rec['priority'] == priority]
        if priority_recs:
            print(f"\n{priority} Priority Recommendations:")
            for i, rec in enumerate(priority_recs, 1):
                print(f"  {i}. [{rec['category']}] {rec['recommendation']}")
                if 'suggested_method' in rec:
                    print(f"     Method: {rec['suggested_method']}")
    print()
    
    # 8. FEATURE ENGINEERING SUGGESTIONS
    print("8. FEATURE ENGINEERING SUGGESTIONS")
    print("-" * 40)
    print("Generating feature engineering suggestions...")
    
    suggestions = eda.generate_feature_engineering_suggestions(df)
    
    for i, suggestion in enumerate(suggestions, 1):
        print(f"{i}. {suggestion['category']}: {suggestion['suggestion']}")
        if 'new_features' in suggestion:
            print(f"   New features: {', '.join(suggestion['new_features'])}")
    print()
    
    # 9. GENERATE COMPREHENSIVE REPORT
    print("9. GENERATING COMPREHENSIVE REPORT")
    print("-" * 40)
    print("Creating detailed report...")
    
    report = eda.generate_comprehensive_report(df)
    
    # Save report to markdown
    report_filename = eda.save_report_to_markdown(report, "comprehensive_eda_report.md")
    print(f"✅ Comprehensive report saved as: {report_filename}")
    
    # 10. SUMMARY AND CONCLUSIONS
    print("\n" + "=" * 80)
    print("ANALYSIS SUMMARY AND CONCLUSIONS")
    print("=" * 80)
    
    print("\n🔍 KEY FINDINGS:")
    print(f"• Dataset contains {df.shape[0]:,} water pump records with {df.shape[1]} features")
    print(f"• Target variable shows {'significant' if imbalance_ratio > 3 else 'moderate'} class imbalance")
    print(f"• Found {len(cols_with_missing)} columns with missing values")
    print(f"• Identified {len([r for r in recommendations if r['priority'] == 'High'])} high-priority preprocessing needs")
    
    print("\n📋 RECOMMENDED NEXT STEPS:")
    print("1. Address high-priority preprocessing recommendations")
    print("2. Handle class imbalance if necessary (SMOTE, class weights)")
    print("3. Implement feature engineering suggestions")
    print("4. Consider feature selection based on mutual information scores")
    print("5. Apply appropriate scaling and encoding transformations")
    print("6. Proceed with model training using Random Forest and XGBoost")
    
    print(f"\n✅ Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("💡 Detailed results are available in the Streamlit EDA interface")
    print("📄 Full report saved as 'comprehensive_eda_report.md'")
    print("=" * 80)

if __name__ == "__main__":
    run_comprehensive_eda()