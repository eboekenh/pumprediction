import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from src.components.data_transformation import DataTransformation
from src.components.data_ingestion import DataIngestion
from src.logger import logging
from src.exception import CustomException
import sys
import os

st.set_page_config(page_title="Preprocessing", page_icon="⚙️", layout="wide")

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

def display_data_quality_issues(df):
    """Display data quality issues"""
    st.subheader("🔍 Data Quality Assessment")
    
    # Missing values
    missing_values = df.isnull().sum()
    missing_percent = (missing_values / len(df)) * 100
    
    # Duplicate rows
    duplicate_rows = df.duplicated().sum()
    
    # Data types
    data_types = df.dtypes.value_counts()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Missing Values", f"{missing_values.sum():,}")
        st.metric("Duplicate Rows", f"{duplicate_rows:,}")
    
    with col2:
        st.metric("Numeric Columns", len(df.select_dtypes(include=[np.number]).columns))
        st.metric("Categorical Columns", len(df.select_dtypes(include=['object']).columns))
    
    with col3:
        st.metric("Total Memory", f"{df.memory_usage().sum() / 1024:.1f} KB")
        st.metric("Data Quality Score", f"{((len(df) - missing_values.sum()) / (len(df) * len(df.columns)) * 100):.1f}%")

def display_missing_values_handling(df):
    """Display missing values handling options"""
    st.subheader("❓ Missing Values Handling")
    
    missing_values = df.isnull().sum()
    missing_columns = missing_values[missing_values > 0].sort_values(ascending=False)
    
    if len(missing_columns) > 0:
        st.write("Columns with missing values:")
        
        # Create a DataFrame for better display
        missing_df = pd.DataFrame({
            'Column': missing_columns.index,
            'Missing Count': missing_columns.values,
            'Missing %': (missing_columns / len(df) * 100).round(2)
        })
        
        st.dataframe(missing_df, use_container_width=True)
        
        # Imputation strategy selection
        st.subheader("Imputation Strategy")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Numerical Columns:**")
            num_strategy = st.selectbox(
                "Select strategy for numerical columns:",
                ["mean", "median", "mode", "constant"],
                index=1
            )
            
            if num_strategy == "constant":
                num_fill_value = st.number_input("Fill value for numerical columns:", value=0)
            else:
                num_fill_value = None
        
        with col2:
            st.write("**Categorical Columns:**")
            cat_strategy = st.selectbox(
                "Select strategy for categorical columns:",
                ["mode", "constant"],
                index=1
            )
            
            if cat_strategy == "constant":
                cat_fill_value = st.text_input("Fill value for categorical columns:", value="Unknown")
            else:
                cat_fill_value = None
        
        # Apply imputation
        if st.button("Apply Imputation"):
            try:
                df_imputed = df.copy()
                
                # Numerical imputation
                numerical_cols = df.select_dtypes(include=[np.number]).columns
                missing_numerical = [col for col in numerical_cols if col in missing_columns.index]
                
                if missing_numerical:
                    if num_strategy == "constant":
                        df_imputed[missing_numerical] = df_imputed[missing_numerical].fillna(num_fill_value)
                    else:
                        imputer = SimpleImputer(strategy=num_strategy)
                        df_imputed[missing_numerical] = imputer.fit_transform(df_imputed[missing_numerical])
                
                # Categorical imputation
                categorical_cols = df.select_dtypes(include=['object']).columns
                missing_categorical = [col for col in categorical_cols if col in missing_columns.index]
                
                if missing_categorical:
                    if cat_strategy == "constant":
                        df_imputed[missing_categorical] = df_imputed[missing_categorical].fillna(cat_fill_value)
                    else:
                        for col in missing_categorical:
                            mode_value = df_imputed[col].mode()[0] if not df_imputed[col].mode().empty else "Unknown"
                            df_imputed[col] = df_imputed[col].fillna(mode_value)
                
                # Store processed data
                st.session_state.processed_data = df_imputed
                st.success("✅ Imputation applied successfully!")
                
                # Show before/after comparison
                st.subheader("Before/After Comparison")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Before Imputation:**")
                    st.write(f"Missing values: {df.isnull().sum().sum():,}")
                
                with col2:
                    st.write("**After Imputation:**")
                    st.write(f"Missing values: {df_imputed.isnull().sum().sum():,}")
                
            except Exception as e:
                st.error(f"Error applying imputation: {str(e)}")
    
    else:
        st.success("✅ No missing values found in the dataset!")

def display_outlier_detection(df):
    """Display outlier detection and handling"""
    st.subheader("📊 Outlier Detection")
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numerical_cols) > 0:
        selected_col = st.selectbox(
            "Select column for outlier analysis:",
            numerical_cols
        )
        
        if selected_col:
            # Calculate outliers using IQR method
            Q1 = df[selected_col].quantile(0.25)
            Q3 = df[selected_col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[selected_col] < lower_bound) | (df[selected_col] > upper_bound)]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Total Outliers", len(outliers))
                st.metric("Outlier Percentage", f"{len(outliers) / len(df) * 100:.2f}%")
            
            with col2:
                st.metric("Lower Bound", f"{lower_bound:.2f}")
                st.metric("Upper Bound", f"{upper_bound:.2f}")
            
            # Visualize outliers
            fig = px.box(df, y=selected_col, title=f"Box Plot of {selected_col}")
            st.plotly_chart(fig, use_container_width=True)
            
            # Outlier handling options
            st.subheader("Outlier Handling")
            
            handling_method = st.selectbox(
                "Select outlier handling method:",
                ["Keep outliers", "Remove outliers", "Cap outliers"]
            )
            
            if st.button("Apply Outlier Handling"):
                try:
                    if handling_method == "Remove outliers":
                        df_processed = df[~((df[selected_col] < lower_bound) | (df[selected_col] > upper_bound))]
                        st.success(f"✅ Removed {len(outliers)} outlier(s)")
                    elif handling_method == "Cap outliers":
                        df_processed = df.copy()
                        df_processed[selected_col] = np.where(df_processed[selected_col] < lower_bound, lower_bound, df_processed[selected_col])
                        df_processed[selected_col] = np.where(df_processed[selected_col] > upper_bound, upper_bound, df_processed[selected_col])
                        st.success(f"✅ Capped {len(outliers)} outlier(s)")
                    else:
                        df_processed = df.copy()
                        st.info("Outliers kept in the dataset")
                    
                    st.session_state.processed_data = df_processed
                    
                except Exception as e:
                    st.error(f"Error handling outliers: {str(e)}")
    
    else:
        st.info("No numerical columns found for outlier detection.")

def display_feature_engineering(df):
    """Display feature engineering options"""
    st.subheader("🔧 Feature Engineering")
    
    # Create new features
    st.write("**Create New Features:**")
    
    # Age of water pump
    if 'construction_year' in df.columns:
        if st.checkbox("Create 'pump_age' feature"):
            current_year = 2024
            df['pump_age'] = current_year - df['construction_year']
            st.success("✅ Created 'pump_age' feature")
    
    # Coordinate-based features
    if 'latitude' in df.columns and 'longitude' in df.columns:
        if st.checkbox("Create location-based features"):
            # Distance from center
            center_lat = df['latitude'].mean()
            center_lon = df['longitude'].mean()
            
            df['distance_from_center'] = np.sqrt(
                (df['latitude'] - center_lat)**2 + (df['longitude'] - center_lon)**2
            )
            st.success("✅ Created 'distance_from_center' feature")
    
    # Population density
    if 'population' in df.columns:
        if st.checkbox("Create population categories"):
            df['population_category'] = pd.cut(
                df['population'],
                bins=[0, 100, 500, 1000, float('inf')],
                labels=['Low', 'Medium', 'High', 'Very High']
            )
            st.success("✅ Created 'population_category' feature")
    
    # Feature selection
    st.subheader("Feature Selection")
    
    all_columns = df.columns.tolist()
    if 'status_group' in all_columns:
        all_columns.remove('status_group')
    
    selected_features = st.multiselect(
        "Select features to keep:",
        all_columns,
        default=all_columns[:10] if len(all_columns) > 10 else all_columns
    )
    
    if selected_features:
        # Add target column back if it exists
        if 'status_group' in df.columns:
            selected_features.append('status_group')
        
        df_selected = df[selected_features]
        st.session_state.processed_data = df_selected
        st.success(f"✅ Selected {len(selected_features)} features")

def display_preprocessing_pipeline(df):
    """Display automated preprocessing pipeline"""
    st.subheader("🔄 Automated Preprocessing Pipeline")
    
    if st.button("Run Complete Preprocessing Pipeline"):
        try:
            with st.spinner("Running preprocessing pipeline..."):
                # Initialize data transformation
                data_transformation = DataTransformation()
                
                # Save current data
                temp_path = "artifacts/temp_data.csv"
                os.makedirs("artifacts", exist_ok=True)
                df.to_csv(temp_path, index=False)
                
                # Run preprocessing
                if 'status_group' in df.columns:
                    # Split data for preprocessing
                    data_ingestion = DataIngestion()
                    train_path, test_path = data_ingestion.initiate_data_ingestion(temp_path)
                    
                    # Transform data
                    train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(
                        train_path, test_path
                    )
                    
                    st.success("✅ Preprocessing pipeline completed successfully!")
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Training Samples", train_arr.shape[0])
                        st.metric("Features", train_arr.shape[1] - 1)  # Exclude target
                    
                    with col2:
                        st.metric("Test Samples", test_arr.shape[0])
                        st.metric("Preprocessor Saved", "✅" if os.path.exists(preprocessor_path) else "❌")
                    
                    # Store preprocessed data
                    st.session_state.preprocessed_train = train_arr
                    st.session_state.preprocessed_test = test_arr
                    st.session_state.preprocessor_path = preprocessor_path
                    
                else:
                    st.warning("No target column found. Cannot split data for supervised learning.")
                
        except Exception as e:
            st.error(f"Error in preprocessing pipeline: {str(e)}")
            logging.error(f"Preprocessing error: {str(e)}")

def main():
    """Main preprocessing page function"""
    st.title("⚙️ Data Preprocessing")
    st.markdown("---")
    
    # Load data
    df = load_data()
    
    if df is not None:
        # Display data quality assessment
        display_data_quality_issues(df)
        
        # Preprocessing tabs
        tabs = st.tabs([
            "Missing Values", 
            "Outlier Detection", 
            "Feature Engineering", 
            "Automated Pipeline"
        ])
        
        with tabs[0]:
            display_missing_values_handling(df)
        
        with tabs[1]:
            display_outlier_detection(df)
        
        with tabs[2]:
            display_feature_engineering(df)
        
        with tabs[3]:
            display_preprocessing_pipeline(df)
        
        # Show processed data
        if 'processed_data' in st.session_state:
            st.subheader("📋 Processed Data Preview")
            st.dataframe(st.session_state.processed_data.head(), use_container_width=True)
            
            # Download processed data
            if st.button("Download Processed Data"):
                csv_data = st.session_state.processed_data.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv_data,
                    file_name="processed_data.csv",
                    mime="text/csv"
                )
    
    else:
        st.warning("⚠️ No data loaded. Please upload data from the main page first.")
        
        if st.button("Go to Main Page"):
            st.switch_page("app.py")

if __name__ == "__main__":
    main()
