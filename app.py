import streamlit as st
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException
import sys
import os

# Set page config
st.set_page_config(
    page_title="Water Pump Functionality Prediction",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    """Main application function"""
    try:
        # Title and description
        st.title("💧 Water Pump Functionality Prediction")
        st.markdown("""
        ## Tanzania Water Pump Status Prediction System
        
        This application predicts the functionality status of water pumps in Tanzania using machine learning.
        The system can classify pumps into three categories:
        - **Functional**: Working properly
        - **Functional needs repair**: Working but requires maintenance
        - **Non-functional**: Not working
        
        ### Features:
        - **Exploratory Data Analysis**: Comprehensive data exploration and visualization
        - **Data Preprocessing**: Automated data cleaning and feature engineering
        - **Model Training**: Train and compare multiple ML models with hyperparameter tuning
        - **Model Evaluation**: Detailed performance metrics and comparisons
        - **Predictions**: Make predictions on new data
        - **Geospatial Analysis**: Interactive maps and location-based insights
        
        ### Navigation:
        Use the sidebar to navigate between different sections of the application.
        """)
        
        # Sidebar for navigation
        st.sidebar.title("Navigation")
        st.sidebar.markdown("Select a page from the dropdown or use the pages above.")
        
        # Dataset information
        st.header("📊 Dataset Information")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Target Classes", "3")
            st.caption("Functional, Needs Repair, Non-functional")
        
        with col2:
            st.metric("Features", "40+")
            st.caption("Geographic, Technical, Management data")
        
        with col3:
            st.metric("Data Source", "DrivenData")
            st.caption("Tanzania Ministry of Water")
        
        # Quick start guide
        st.header("🚀 Quick Start Guide")
        st.markdown("""
        1. **Upload Data**: Go to the EDA page to upload your dataset
        2. **Explore**: Use the EDA tools to understand your data
        3. **Preprocess**: Clean and prepare your data for modeling
        4. **Train**: Build and train machine learning models
        5. **Evaluate**: Compare model performance
        6. **Predict**: Make predictions on new data
        7. **Analyze**: Explore geospatial patterns
        """)
        
        # Data upload section
        st.header("📁 Data Upload")
        
        # Multiple file upload for separate training features, labels, and test files
        st.subheader("Option 1: Upload Multiple Files (Recommended)")
        st.markdown("Upload separate files for training features, training labels, and test features:")
        
        uploaded_files = st.file_uploader(
            "Upload training features, training labels, and test features CSV files",
            type=['csv'],
            accept_multiple_files=True,
            help="Upload 3 CSV files: training features (X), training labels (y), and test features"
        )
        
        if uploaded_files and len(uploaded_files) == 3:
            if st.button("Process Uploaded Files", type="primary"):
                try:
                    # Save uploaded files temporarily
                    temp_paths = []
                    for i, uploaded_file in enumerate(uploaded_files):
                        temp_path = f"temp_upload_{i}_{uploaded_file.name}"
                        with open(temp_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        temp_paths.append(temp_path)
                    
                    # Process files using FileHandler
                    from src.components.file_handler import FileHandler
                    file_handler = FileHandler()
                    
                    with st.spinner("Processing uploaded files..."):
                        results = file_handler.process_uploaded_files(temp_paths)
                    
                    if results['success']:
                        st.success("Files processed successfully!")
                        
                        # Store in session state
                        st.session_state['file_processing_results'] = results
                        st.session_state['data_uploaded'] = True
                        st.session_state['train_data_path'] = results['saved_files']['merged_train']
                        if 'test' in results['saved_files']:
                            st.session_state['test_data_path'] = results['saved_files']['test']
                        
                        # Display summary
                        st.subheader("Processing Summary")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Training Data", f"{results['summary']['train_data_info']['shape'][0]:,} rows")
                        
                        with col2:
                            st.metric("Features", f"{len(results['summary']['train_data_info']['numerical_columns']) + len(results['summary']['train_data_info']['categorical_columns'])}")
                        
                        with col3:
                            if 'test_data_info' in results['summary']:
                                st.metric("Test Data", f"{results['summary']['test_data_info']['shape'][0]:,} rows")
                            else:
                                st.metric("Test Data", "Not available")
                        
                        # Show identified files
                        st.subheader("Identified Files")
                        for file_type, path in results['identified_files'].items():
                            st.write(f"**{file_type.replace('_', ' ').title()}**: {os.path.basename(path)}")
                        
                        # Show target distribution
                        if 'target_distribution' in results['summary']['train_data_info']:
                            st.subheader("Target Distribution")
                            target_dist = results['summary']['train_data_info']['target_distribution']
                            for status, count in target_dist.items():
                                st.write(f"**{status}**: {count:,} ({count/sum(target_dist.values())*100:.1f}%)")
                    
                    else:
                        st.error(f"Error processing files: {results['error']}")
                    
                    # Clean up temporary files
                    for temp_path in temp_paths:
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                            
                except Exception as e:
                    st.error(f"Error processing files: {str(e)}")
        
        elif uploaded_files and len(uploaded_files) != 3:
            st.warning(f"Please upload exactly 3 CSV files. You uploaded {len(uploaded_files)} files.")
        
        st.markdown("---")
        
        # Single file upload option
        st.subheader("Option 2: Upload Single Combined File")
        uploaded_file = st.file_uploader(
            "Upload a single CSV file with both features and target variable",
            type=['csv'],
            key="single_file",
            help="Upload a CSV file containing water pump data with features and target variable"
        )
        
        if uploaded_file is not None:
            try:
                # Save uploaded file
                os.makedirs("artifacts", exist_ok=True)
                file_path = os.path.join("artifacts", "uploaded_data.csv")
                
                # Read and save the file
                df = pd.read_csv(uploaded_file)
                df.to_csv(file_path, index=False)
                
                st.success(f"✅ File uploaded successfully! {df.shape[0]} rows, {df.shape[1]} columns")
                
                # Display basic info
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Dataset Preview")
                    st.dataframe(df.head())
                
                with col2:
                    st.subheader("Dataset Info")
                    st.write(f"**Shape:** {df.shape}")
                    st.write(f"**Memory Usage:** {df.memory_usage().sum() / 1024:.2f} KB")
                    st.write(f"**Missing Values:** {df.isnull().sum().sum()}")
                
                # Store file path in session state
                st.session_state.data_path = file_path
                st.session_state.data_uploaded = True
                
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
                logging.error(f"Error reading uploaded file: {str(e)}")
        
        # Sample data option
        st.header("🎯 Sample Data")
        if st.button("Load Sample Dataset"):
            try:
                sample_path = "sample_data/sample_water_pumps.csv"
                if os.path.exists(sample_path):
                    st.session_state.data_path = sample_path
                    st.session_state.data_uploaded = True
                    st.success("✅ Sample dataset loaded successfully!")
                    st.rerun()
                else:
                    st.warning("Sample dataset not found. Please upload your own data.")
            except Exception as e:
                st.error(f"Error loading sample data: {str(e)}")
        
        # Footer
        st.markdown("---")
        st.markdown("**Built with Streamlit** | **Data Source:** DrivenData Competition")
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        logging.error(f"Error in main app: {str(e)}")

if __name__ == "__main__":
    main()
