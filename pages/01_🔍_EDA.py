import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
from src.components.advanced_eda import AdvancedEDA
from src.logger import logging
from src.exception import CustomException
import sys
import os

st.set_page_config(page_title="EDA", page_icon="🔍", layout="wide")

def load_full_dataset():
    """Load the complete merged training dataset"""
    try:
        # Check if processed data exists
        if os.path.exists("artifacts/merged_train_data.csv"):
            df = pd.read_csv("artifacts/merged_train_data.csv")
            st.success(f"Loaded complete dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
            return df
        else:
            st.warning("No processed data found. Please upload and process data on the main page first.")
            return None
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None

def display_comprehensive_overview(df):
    """Display comprehensive dataset overview"""
    st.header("📊 Complete Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", f"{df.shape[0]:,}")
    
    with col2:
        st.metric("Total Features", df.shape[1])
    
    with col3:
        st.metric("Missing Values", f"{df.isnull().sum().sum():,}")
    
    with col4:
        st.metric("Memory Usage", f"{df.memory_usage().sum() / 1024 / 1024:.1f} MB")
    
    # Target distribution
    if 'status_group' in df.columns:
        st.subheader("🎯 Target Variable Distribution")
        target_counts = df['status_group'].value_counts()
        target_pct = df['status_group'].value_counts(normalize=True) * 100
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(values=target_counts.values, names=target_counts.index, 
                        title="Water Pump Status Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Detailed Counts")
            for status, count in target_counts.items():
                pct = target_pct[status]
                st.write(f"**{status}**: {count:,} ({pct:.1f}%)")

def display_missing_values_analysis(df):
    """Display comprehensive missing values analysis"""
    st.header("🔍 Missing Values Analysis")
    
    missing_data = df.isnull().sum()
    missing_percentage = (missing_data / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'Column': missing_data.index,
        'Missing Count': missing_data.values,
        'Missing Percentage': missing_percentage.values
    })
    missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
    
    if len(missing_df) > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Missing Values by Column")
            fig = px.bar(missing_df, x='Missing Percentage', y='Column', 
                        title="Missing Values Percentage by Column",
                        orientation='h')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Missing Values Table")
            st.dataframe(missing_df, use_container_width=True)
        
        # Categorize columns by missing percentage
        high_missing = missing_df[missing_df['Missing Percentage'] > 50]['Column'].tolist()
        medium_missing = missing_df[(missing_df['Missing Percentage'] > 10) & 
                                  (missing_df['Missing Percentage'] <= 50)]['Column'].tolist()
        low_missing = missing_df[(missing_df['Missing Percentage'] > 0) & 
                               (missing_df['Missing Percentage'] <= 10)]['Column'].tolist()
        
        st.subheader("Missing Values Categories")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**High Missing (>50%)**")
            for col in high_missing:
                st.write(f"- {col}")
        
        with col2:
            st.write("**Medium Missing (10-50%)**")
            for col in medium_missing:
                st.write(f"- {col}")
        
        with col3:
            st.write("**Low Missing (<10%)**")
            for col in low_missing:
                st.write(f"- {col}")
    else:
        st.success("No missing values found in the dataset!")

def display_data_types_analysis(df):
    """Display data types analysis"""
    st.header("🔢 Data Types Analysis")
    
    # Analyze data types
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Remove ID and target from features
    if 'id' in numerical_cols:
        numerical_cols.remove('id')
    if 'id' in categorical_cols:
        categorical_cols.remove('id')
    if 'status_group' in numerical_cols:
        numerical_cols.remove('status_group')
    if 'status_group' in categorical_cols:
        categorical_cols.remove('status_group')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"Numerical Features ({len(numerical_cols)})")
        for col in numerical_cols:
            st.write(f"- {col}")
    
    with col2:
        st.subheader(f"Categorical Features ({len(categorical_cols)})")
        for col in categorical_cols:
            st.write(f"- {col}")
    
    # Display basic statistics for numerical columns
    if numerical_cols:
        st.subheader("📈 Numerical Features Statistics")
        st.dataframe(df[numerical_cols].describe(), use_container_width=True)
    
    # Display unique values count for categorical columns
    if categorical_cols:
        st.subheader("📊 Categorical Features Unique Values")
        cat_unique = pd.DataFrame({
            'Column': categorical_cols,
            'Unique Values': [df[col].nunique() for col in categorical_cols],
            'Most Frequent': [df[col].mode().iloc[0] if len(df[col].mode()) > 0 else 'N/A' for col in categorical_cols]
        })
        st.dataframe(cat_unique, use_container_width=True)

def display_comprehensive_correlation_analysis(df):
    """Display comprehensive correlation analysis"""
    st.header("🔗 Comprehensive Correlation Analysis")
    
    # Initialize advanced EDA
    eda = AdvancedEDA()
    
    # Analyze data types
    data_types = eda.analyze_data_types(df)
    
    # Calculate correlation matrix
    correlation_results = eda.calculate_correlation_matrix(df)
    
    # Display numerical correlations
    if 'pearson' in correlation_results:
        st.subheader("📊 Numerical Features Correlation (Pearson)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Correlation heatmap
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(correlation_results['pearson'], annot=True, cmap='coolwarm', 
                       center=0, square=True, ax=ax, fmt='.2f')
            plt.title('Pearson Correlation Matrix')
            st.pyplot(fig)
        
        with col2:
            # Spearman correlation
            if 'spearman' in correlation_results:
                st.write("**Spearman Correlation (Rank-based)**")
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(correlation_results['spearman'], annot=True, cmap='coolwarm', 
                           center=0, square=True, ax=ax, fmt='.2f')
                plt.title('Spearman Correlation Matrix')
                st.pyplot(fig)
    
    # Display categorical correlations
    if 'cramers_v' in correlation_results:
        st.subheader("📊 Categorical Features Correlation (Cramér's V)")
        
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(correlation_results['cramers_v'], annot=True, cmap='viridis', 
                   square=True, ax=ax, fmt='.2f')
        plt.title("Cramér's V Correlation Matrix")
        st.pyplot(fig)
    
    # Display feature-target relationships
    if 'mutual_info_numerical' in correlation_results:
        st.subheader("🎯 Feature-Target Relationships (Mutual Information)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Numerical Features**")
            mi_num = correlation_results['mutual_info_numerical'].sort_values(ascending=False)
            fig = px.bar(x=mi_num.values, y=mi_num.index, orientation='h',
                        title="Mutual Information: Numerical Features vs Target")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'mutual_info_categorical' in correlation_results:
                st.write("**Categorical Features**")
                mi_cat = correlation_results['mutual_info_categorical'].sort_values(ascending=False)
                fig = px.bar(x=mi_cat.values, y=mi_cat.index, orientation='h',
                            title="Mutual Information: Categorical Features vs Target")
                st.plotly_chart(fig, use_container_width=True)

def display_outlier_analysis(df):
    """Display comprehensive outlier analysis"""
    st.header("🚨 Outlier Analysis")
    
    # Initialize advanced EDA
    eda = AdvancedEDA()
    eda.analyze_data_types(df)
    
    # Detect outliers
    outlier_results = eda.detect_outliers(df)
    
    # Display outlier summary
    st.subheader("📊 Outlier Summary")
    
    outlier_summary = []
    for col, methods in outlier_results.items():
        if col != 'multivariate':
            for method, stats in methods.items():
                outlier_summary.append({
                    'Column': col,
                    'Method': method.upper(),
                    'Outliers Count': stats['count'],
                    'Percentage': f"{stats['percentage']:.2f}%"
                })
    
    if outlier_summary:
        outlier_df = pd.DataFrame(outlier_summary)
        st.dataframe(outlier_df, use_container_width=True)
        
        # Show multivariate outliers if available
        if 'multivariate' in outlier_results:
            st.subheader("🔍 Multivariate Outliers")
            multi_stats = outlier_results['multivariate']
            st.write(f"**Method**: {multi_stats['method']}")
            st.write(f"**Outliers**: {multi_stats['count']:,} ({multi_stats['percentage']:.2f}%)")
    
    # Display outlier visualization for top numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'id' in numerical_cols:
        numerical_cols.remove('id')
    if 'status_group' in numerical_cols:
        numerical_cols.remove('status_group')
    
    if numerical_cols:
        st.subheader("📈 Outlier Visualization")
        
        # Select top 6 numerical columns for visualization
        top_cols = numerical_cols[:6]
        
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=top_cols,
            specs=[[{"type": "xy"} for _ in range(3)] for _ in range(2)]
        )
        
        for i, col in enumerate(top_cols):
            row = i // 3 + 1
            col_idx = i % 3 + 1
            
            fig.add_trace(
                go.Box(y=df[col], name=col, showlegend=False),
                row=row, col=col_idx
            )
        
        fig.update_layout(height=600, title_text="Box Plots for Outlier Detection")
        st.plotly_chart(fig, use_container_width=True)

def display_preprocessing_recommendations(df):
    """Display comprehensive preprocessing recommendations"""
    st.header("⚙️ Preprocessing Recommendations")
    
    # Initialize advanced EDA
    eda = AdvancedEDA()
    eda.analyze_data_types(df)
    eda.calculate_correlation_matrix(df)
    eda.detect_outliers(df)
    
    # Generate recommendations
    recommendations = eda.generate_preprocessing_recommendations(df)
    
    # Display recommendations by priority
    priorities = ['High', 'Medium', 'Low']
    
    for priority in priorities:
        priority_recs = [rec for rec in recommendations if rec['priority'] == priority]
        
        if priority_recs:
            st.subheader(f"🔴 {priority} Priority Recommendations" if priority == 'High' 
                        else f"🟡 {priority} Priority Recommendations" if priority == 'Medium'
                        else f"🟢 {priority} Priority Recommendations")
            
            for i, rec in enumerate(priority_recs, 1):
                with st.expander(f"{i}. {rec['category']}: {rec['recommendation'][:50]}..."):
                    st.write(f"**Category**: {rec['category']}")
                    st.write(f"**Recommendation**: {rec['recommendation']}")
                    
                    if 'columns' in rec:
                        st.write(f"**Affected Columns**: {rec['columns']}")
                    
                    if 'suggested_method' in rec:
                        st.write(f"**Suggested Method**: {rec['suggested_method']}")
                    
                    if 'pairs' in rec:
                        st.write("**Highly Correlated Pairs**:")
                        for pair in rec['pairs']:
                            st.write(f"- {pair[0]} ↔ {pair[1]} (r = {pair[2]:.3f})")

def display_feature_engineering_suggestions(df):
    """Display feature engineering suggestions"""
    st.header("🔧 Feature Engineering Suggestions")
    
    # Initialize advanced EDA
    eda = AdvancedEDA()
    suggestions = eda.generate_feature_engineering_suggestions(df)
    
    for i, suggestion in enumerate(suggestions, 1):
        with st.expander(f"{i}. {suggestion['category']}: {suggestion['suggestion'][:50]}..."):
            st.write(f"**Category**: {suggestion['category']}")
            st.write(f"**Suggestion**: {suggestion['suggestion']}")
            
            if 'columns' in suggestion:
                st.write(f"**Source Columns**: {suggestion['columns']}")
            
            if 'new_features' in suggestion:
                st.write(f"**New Features**: {suggestion['new_features']}")

def generate_comprehensive_report(df):
    """Generate and display comprehensive EDA report"""
    st.header("📄 Comprehensive EDA Report")
    
    # Initialize advanced EDA
    eda = AdvancedEDA()
    
    # Generate comprehensive report
    with st.spinner("Generating comprehensive EDA report..."):
        report = eda.generate_comprehensive_report(df)
    
    # Display report summary
    st.subheader("📊 Report Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Dataset Shape", f"{report['dataset_info']['shape'][0]:,} × {report['dataset_info']['shape'][1]}")
        st.metric("Missing Values", f"{report['dataset_info']['missing_values_total']:,}")
    
    with col2:
        st.metric("Missing Percentage", f"{report['dataset_info']['missing_percentage']:.1f}%")
        st.metric("Duplicate Rows", f"{report['dataset_info']['duplicate_rows']:,}")
    
    with col3:
        st.metric("Numerical Features", len(report['data_types']['numerical']))
        st.metric("Categorical Features", len(report['data_types']['categorical']))
    
    # Save report to markdown
    if st.button("💾 Save Report as Markdown", type="primary"):
        with st.spinner("Saving report..."):
            filename = eda.save_report_to_markdown(report, "comprehensive_eda_report.md")
            st.success(f"Report saved as: {filename}")
            
            # Display download link
            with open(filename, 'r') as f:
                st.download_button(
                    label="📥 Download EDA Report",
                    data=f.read(),
                    file_name=filename,
                    mime="text/markdown"
                )

def main():
    """Main EDA page function"""
    st.title("🔍 Comprehensive Exploratory Data Analysis")
    st.markdown("---")
    
    # Load complete dataset
    df = load_full_dataset()
    
    if df is not None:
        # Display comprehensive overview
        display_comprehensive_overview(df)
        st.markdown("---")
        
        # Create tabs for different analysis sections
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "📊 Data Types", 
            "🔍 Missing Values", 
            "🔗 Correlations", 
            "🚨 Outliers", 
            "⚙️ Preprocessing", 
            "🔧 Feature Engineering",
            "📄 Full Report"
        ])
        
        with tab1:
            display_data_types_analysis(df)
        
        with tab2:
            display_missing_values_analysis(df)
        
        with tab3:
            display_comprehensive_correlation_analysis(df)
        
        with tab4:
            display_outlier_analysis(df)
        
        with tab5:
            display_preprocessing_recommendations(df)
        
        with tab6:
            display_feature_engineering_suggestions(df)
        
        with tab7:
            generate_comprehensive_report(df)
    
    else:
        st.info("Please upload and process your dataset on the main page first to begin EDA.")

if __name__ == "__main__":
    main()