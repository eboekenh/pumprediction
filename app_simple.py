import streamlit as st
import pandas as pd
import numpy as np
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
        
        # Single file upload option
        st.subheader("Upload Dataset")
        uploaded_file = st.file_uploader(
            "Upload a CSV file with water pump data",
            type=['csv'],
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
        
        # Check for existing data
        st.header("🎯 Existing Data")
        if os.path.exists("artifacts/merged_train_data.csv"):
            st.success("✅ Tanzania water pump dataset available!")
            
            # Load sample data for display
            df_sample = pd.read_csv("artifacts/merged_train_data.csv", nrows=1000)
            
            # Get actual row count
            with open("artifacts/merged_train_data.csv", 'r') as f:
                row_count = sum(1 for line in f) - 1  # Subtract header
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Samples", f"{row_count:,}")
            with col2:
                st.metric("Features", f"{len(df_sample.columns)-1}")
            with col3:
                if 'status_group' in df_sample.columns:
                    st.metric("Target Classes", df_sample['status_group'].nunique())
            
            # Show sample data
            st.subheader("Sample Data")
            st.dataframe(df_sample.head())
            
        else:
            st.info("No existing dataset found. Please upload data to get started.")
        
        # Footer
        st.markdown("---")
        st.markdown("**Built with Streamlit** | **Data Source:** DrivenData Competition")
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()