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
from src.components.data_ingestion import DataIngestion
from src.logger import logging
from src.exception import CustomException
import sys

st.set_page_config(page_title="EDA", page_icon="🔍", layout="wide")

def load_data():
    """Load data from session state or file"""
    try:
        if 'data_path' in st.session_state and st.session_state.data_path:
            df = pd.read_csv(st.session_state.data_path)
            return df
        else:
            return None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def display_data_overview(df):
    """Display basic data information"""
    st.subheader("📊 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", f"{df.shape[0]:,}")
    
    with col2:
        st.metric("Total Features", df.shape[1])
    
    with col3:
        st.metric("Missing Values", f"{df.isnull().sum().sum():,}")
    
    with col4:
        st.metric("Memory Usage", f"{df.memory_usage().sum() / 1024:.1f} KB")

def display_target_distribution(df):
    """Display target variable distribution"""
    if 'status_group' in df.columns:
        st.subheader("🎯 Target Variable Distribution")
        
        # Count plot
        status_counts = df['status_group'].value_counts().reset_index()
        status_counts.columns = ['status', 'count']
        fig = px.bar(
            status_counts,
            x='status',
            y='count',
            title='Water Pump Status Distribution',
            labels={'status': 'Status', 'count': 'Count'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Pie chart
        col1, col2 = st.columns(2)
        
        with col1:
            fig_pie = px.pie(
                values=df['status_group'].value_counts().values,
                names=df['status_group'].value_counts().index,
                title='Status Distribution (Pie Chart)'
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            st.subheader("Status Statistics")
            status_stats = df['status_group'].value_counts()
            for status, count in status_stats.items():
                percentage = (count / len(df)) * 100
                st.write(f"**{status}**: {count:,} ({percentage:.1f}%)")

def display_numerical_features(df):
    """Display numerical features analysis"""
    st.subheader("🔢 Numerical Features Analysis")
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numerical_cols) > 0:
        # Feature selection
        selected_features = st.multiselect(
            "Select numerical features to analyze:",
            numerical_cols,
            default=numerical_cols[:5] if len(numerical_cols) >= 5 else numerical_cols
        )
        
        if selected_features:
            # Statistical summary
            st.subheader("Statistical Summary")
            st.dataframe(df[selected_features].describe())
            
            # Distribution plots
            st.subheader("Feature Distributions")
            
            for feature in selected_features:
                fig = px.histogram(
                    df, 
                    x=feature, 
                    title=f'Distribution of {feature}',
                    marginal="box"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Correlation matrix
            if len(selected_features) > 1:
                st.subheader("Correlation Matrix")
                corr_matrix = df[selected_features].corr()
                
                fig = px.imshow(
                    corr_matrix,
                    title="Feature Correlation Matrix",
                    color_continuous_scale='RdBu'
                )
                st.plotly_chart(fig, use_container_width=True)

def display_categorical_features(df):
    """Display categorical features analysis"""
    st.subheader("📝 Categorical Features Analysis")
    
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    if len(categorical_cols) > 0:
        # Feature selection
        selected_feature = st.selectbox(
            "Select categorical feature to analyze:",
            categorical_cols
        )
        
        if selected_feature:
            # Value counts
            st.subheader(f"Value Counts for {selected_feature}")
            
            value_counts = df[selected_feature].value_counts().head(20)
            
            fig = px.bar(
                x=value_counts.index,
                y=value_counts.values,
                title=f'Top 20 values in {selected_feature}',
                labels={'x': selected_feature, 'y': 'Count'}
            )
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistics
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Unique Values", df[selected_feature].nunique())
            
            with col2:
                st.metric("Missing Values", df[selected_feature].isnull().sum())
            
            # Cross-tabulation with target
            if 'status_group' in df.columns:
                st.subheader(f"Cross-tabulation: {selected_feature} vs Status")
                
                # Create cross-tabulation
                crosstab = pd.crosstab(df[selected_feature], df['status_group'])
                
                # Display top 10 categories
                top_categories = df[selected_feature].value_counts().head(10).index
                crosstab_subset = crosstab.loc[top_categories]
                
                fig = px.bar(
                    crosstab_subset,
                    title=f'{selected_feature} by Status (Top 10 categories)',
                    barmode='group'
                )
                st.plotly_chart(fig, use_container_width=True)

def display_geospatial_analysis(df):
    """Display geospatial analysis"""
    st.subheader("🗺️ Geospatial Analysis")
    
    if 'latitude' in df.columns and 'longitude' in df.columns:
        # Filter out invalid coordinates
        valid_coords = df[
            (df['latitude'] != 0) & 
            (df['longitude'] != 0) & 
            (df['latitude'].notna()) & 
            (df['longitude'].notna())
        ]
        
        if len(valid_coords) > 0:
            # Sample data for performance
            sample_size = min(1000, len(valid_coords))
            sample_data = valid_coords.sample(n=sample_size, random_state=42)
            
            st.write(f"Showing {sample_size} randomly sampled water pumps out of {len(valid_coords)} with valid coordinates")
            
            # Create folium map
            center_lat = sample_data['latitude'].mean()
            center_lon = sample_data['longitude'].mean()
            
            m = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=6,
                tiles='OpenStreetMap'
            )
            
            # Add points to map
            if 'status_group' in sample_data.columns:
                # Color code by status
                color_map = {
                    'functional': 'green',
                    'functional needs repair': 'orange',
                    'non functional': 'red'
                }
                
                for idx, row in sample_data.iterrows():
                    folium.CircleMarker(
                        location=[row['latitude'], row['longitude']],
                        radius=3,
                        color=color_map.get(row['status_group'], 'blue'),
                        fill=True,
                        popup=f"Status: {row['status_group']}<br>Region: {row.get('region', 'Unknown')}",
                        tooltip=f"Status: {row['status_group']}"
                    ).add_to(m)
            else:
                for idx, row in sample_data.iterrows():
                    folium.CircleMarker(
                        location=[row['latitude'], row['longitude']],
                        radius=3,
                        color='blue',
                        fill=True,
                        popup=f"Lat: {row['latitude']}<br>Lon: {row['longitude']}"
                    ).add_to(m)
            
            # Display map
            st_folium(m, width=700, height=500)
            
            # Regional distribution
            if 'region' in df.columns:
                st.subheader("Regional Distribution")
                region_counts = df['region'].value_counts()
                
                fig = px.bar(
                    x=region_counts.index,
                    y=region_counts.values,
                    title='Water Pumps by Region',
                    labels={'x': 'Region', 'y': 'Count'}
                )
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No valid coordinates found in the dataset.")
    else:
        st.warning("Latitude and longitude columns not found in the dataset.")

def display_missing_values_analysis(df):
    """Display missing values analysis"""
    st.subheader("❓ Missing Values Analysis")
    
    missing_values = df.isnull().sum()
    missing_percent = (missing_values / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'Column': missing_values.index,
        'Missing Count': missing_values.values,
        'Missing Percentage': missing_percent.values
    })
    
    missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
    
    if len(missing_df) > 0:
        # Bar chart
        fig = px.bar(
            missing_df,
            x='Column',
            y='Missing Percentage',
            title='Missing Values by Column (%)',
            labels={'Missing Percentage': 'Missing %'}
        )
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
        
        # Data table
        st.subheader("Missing Values Summary")
        st.dataframe(missing_df, use_container_width=True)
    else:
        st.success("✅ No missing values found in the dataset!")

def main():
    """Main EDA page function"""
    st.title("🔍 Exploratory Data Analysis")
    st.markdown("---")
    
    # Load data
    df = load_data()
    
    if df is not None:
        # Display data overview
        display_data_overview(df)
        
        # Data preview
        st.subheader("📋 Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Analysis sections
        tabs = st.tabs([
            "Target Analysis", 
            "Numerical Features", 
            "Categorical Features", 
            "Geospatial Analysis", 
            "Missing Values"
        ])
        
        with tabs[0]:
            display_target_distribution(df)
        
        with tabs[1]:
            display_numerical_features(df)
        
        with tabs[2]:
            display_categorical_features(df)
        
        with tabs[3]:
            display_geospatial_analysis(df)
        
        with tabs[4]:
            display_missing_values_analysis(df)
        
        # Data export
        st.subheader("📥 Export Data")
        if st.button("Download EDA Report"):
            # Create summary report
            report = f"""
            # EDA Report
            
            ## Dataset Overview
            - Total Records: {df.shape[0]:,}
            - Total Features: {df.shape[1]}
            - Missing Values: {df.isnull().sum().sum():,}
            
            ## Feature Types
            - Numerical: {len(df.select_dtypes(include=[np.number]).columns)}
            - Categorical: {len(df.select_dtypes(include=['object']).columns)}
            
            ## Target Distribution
            {df['status_group'].value_counts().to_string() if 'status_group' in df.columns else 'No target column found'}
            """
            
            st.download_button(
                label="Download Report",
                data=report,
                file_name="eda_report.txt",
                mime="text/plain"
            )
    
    else:
        st.warning("⚠️ No data loaded. Please upload data from the main page first.")
        
        if st.button("Go to Main Page"):
            st.switch_page("app.py")

if __name__ == "__main__":
    main()
