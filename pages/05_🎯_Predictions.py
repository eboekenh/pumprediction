import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from src.pipeline.predict_pipeline import PredictPipeline, CustomData
from src.utils import load_object
from src.logger import logging
from src.exception import CustomException
import os
import sys

st.set_page_config(page_title="Predictions", page_icon="🎯", layout="wide")

def check_model_availability():
    """Check if trained model and preprocessor are available"""
    model_path = "artifacts/model.pkl"
    preprocessor_path = "artifacts/preprocessor.pkl"
    
    model_available = os.path.exists(model_path)
    preprocessor_available = os.path.exists(preprocessor_path)
    
    return model_available and preprocessor_available

def load_sample_data_for_prediction():
    """Load sample data for prediction examples"""
    try:
        sample_path = "sample_data/sample_water_pumps.csv"
        if os.path.exists(sample_path):
            df = pd.read_csv(sample_path)
            # Remove target column if present
            if 'status_group' in df.columns:
                df = df.drop('status_group', axis=1)
            return df
        else:
            return None
    except Exception as e:
        st.error(f"Error loading sample data: {str(e)}")
        return None

def display_single_prediction_form():
    """Display form for single prediction"""
    st.subheader("🎯 Single Prediction")
    
    with st.form("single_prediction_form"):
        st.write("Enter water pump information:")
        
        # Create columns for form layout
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Geographic information
            st.write("**Geographic Information**")
            longitude = st.number_input("Longitude", value=0.0, format="%.6f")
            latitude = st.number_input("Latitude", value=0.0, format="%.6f")
            region = st.selectbox("Region", [
                "Dodoma", "Arusha", "Kilimanjaro", "Tanga", "Morogoro",
                "Pwani", "Lindi", "Mtwara", "Ruvuma", "Iringa", "Mbeya",
                "Singida", "Tabora", "Rukwa", "Kigoma", "Shinyanga",
                "Kagera", "Mwanza", "Mara", "Manyara", "Dar es Salaam"
            ])
            gps_height = st.number_input("GPS Height", value=0)
            
        with col2:
            # Technical specifications
            st.write("**Technical Specifications**")
            extraction_type = st.selectbox("Extraction Type", [
                "gravity", "handpump", "submersible", "motorpump", "rope pump",
                "wind-powered", "other"
            ])
            pump_type = st.selectbox("Pump Type", [
                "afridev", "india mark ii", "india mark iii", "ksb", "other"
            ])
            waterpoint_type = st.selectbox("Waterpoint Type", [
                "communal standpipe", "hand pump", "improved spring",
                "cattle trough", "dam", "other"
            ])
            
        with col3:
            # Management and payment
            st.write("**Management & Payment**")
            management = st.selectbox("Management", [
                "vwc", "wug", "water authority", "school", "parastatal",
                "private operator", "other"
            ])
            payment = st.selectbox("Payment", [
                "pay per bucket", "pay monthly", "pay annually", "never pay", "other"
            ])
            water_quality = st.selectbox("Water Quality", [
                "soft", "salty", "milky", "colored", "fluoride", "unknown"
            ])
            quantity = st.selectbox("Quantity", [
                "enough", "insufficient", "dry", "seasonal", "unknown"
            ])
        
        # Additional information
        st.write("**Additional Information**")
        col4, col5, col6 = st.columns(3)
        
        with col4:
            construction_year = st.number_input("Construction Year", min_value=1900, max_value=2024, value=2000)
            population = st.number_input("Population", min_value=0, value=0)
            
        with col5:
            installer = st.text_input("Installer", value="unknown")
            funder = st.text_input("Funder", value="unknown")
            
        with col6:
            basin = st.selectbox("Basin", [
                "Lake Victoria", "Pangani", "Internal", "Lake Nyasa", "Ruvuma / Southern Coast",
                "Rufiji", "Lake Rukwa", "Lake Tanganyika", "Wami / Ruvu"
            ])
            source = st.selectbox("Source", [
                "borehole", "shallow well", "spring", "rainwater harvesting",
                "dam", "river", "other"
            ])
        
        # Submit button
        submitted = st.form_submit_button("🔮 Make Prediction", type="primary")
        
        if submitted:
            try:
                # Create custom data object
                custom_data = CustomData(
                    longitude=longitude,
                    latitude=latitude,
                    region=region,
                    gps_height=gps_height,
                    extraction_type=extraction_type,
                    pump_type=pump_type,
                    waterpoint_type=waterpoint_type,
                    management=management,
                    payment=payment,
                    water_quality=water_quality,
                    quantity=quantity,
                    construction_year=construction_year,
                    population=population,
                    installer=installer,
                    funder=funder,
                    basin=basin,
                    source=source
                )
                
                # Convert to DataFrame
                pred_df = custom_data.get_data_as_data_frame()
                
                # Make prediction
                predict_pipeline = PredictPipeline()
                prediction = predict_pipeline.predict(pred_df)
                probabilities = predict_pipeline.predict_proba(pred_df)
                
                # Display results
                st.success("✅ Prediction completed!")
                
                # Display prediction
                prediction_result = prediction[0]
                
                # Color code based on prediction
                if prediction_result == 'functional':
                    st.success(f"🟢 **Prediction: {prediction_result.upper()}**")
                elif prediction_result == 'functional needs repair':
                    st.warning(f"🟡 **Prediction: {prediction_result.upper()}**")
                else:
                    st.error(f"🔴 **Prediction: {prediction_result.upper()}**")
                
                # Display probabilities
                st.subheader("📊 Prediction Probabilities")
                
                # Get class labels
                class_labels = ['functional', 'functional needs repair', 'non functional']
                prob_df = pd.DataFrame({
                    'Status': class_labels,
                    'Probability': probabilities[0]
                })
                
                # Create probability chart
                fig = px.bar(
                    prob_df,
                    x='Status',
                    y='Probability',
                    title='Prediction Probabilities',
                    color='Probability',
                    color_continuous_scale='viridis'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Display probability table
                st.dataframe(prob_df, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
                logging.error(f"Prediction error: {str(e)}")

def display_batch_prediction():
    """Display batch prediction interface"""
    st.subheader("📊 Batch Prediction")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload CSV file for batch prediction",
        type=['csv'],
        help="Upload a CSV file with water pump data (without target column)"
    )
    
    if uploaded_file is not None:
        try:
            # Load data
            df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ File uploaded successfully! {df.shape[0]} rows, {df.shape[1]} columns")
            
            # Display preview
            st.subheader("📋 Data Preview")
            st.dataframe(df.head(), use_container_width=True)
            
            # Make predictions
            if st.button("🚀 Run Batch Prediction"):
                with st.spinner("Making predictions..."):
                    # Remove target column if present
                    if 'status_group' in df.columns:
                        df = df.drop('status_group', axis=1)
                    
                    # Make predictions
                    predict_pipeline = PredictPipeline()
                    predictions = predict_pipeline.predict(df)
                    probabilities = predict_pipeline.predict_proba(df)
                    
                    # Add predictions to dataframe
                    df['predicted_status'] = predictions
                    
                    # Add probability columns
                    class_labels = ['functional', 'functional needs repair', 'non functional']
                    for i, label in enumerate(class_labels):
                        df[f'prob_{label}'] = probabilities[:, i]
                    
                    st.success("✅ Batch prediction completed!")
                    
                    # Display results
                    st.subheader("📊 Prediction Results")
                    
                    # Summary statistics
                    prediction_counts = pd.Series(predictions).value_counts()
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        functional_count = prediction_counts.get('functional', 0)
                        st.metric("Functional", functional_count)
                    
                    with col2:
                        repair_count = prediction_counts.get('functional needs repair', 0)
                        st.metric("Needs Repair", repair_count)
                    
                    with col3:
                        non_functional_count = prediction_counts.get('non functional', 0)
                        st.metric("Non-functional", non_functional_count)
                    
                    # Visualization
                    fig = px.pie(
                        values=prediction_counts.values,
                        names=prediction_counts.index,
                        title='Prediction Distribution'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display results table
                    st.subheader("📋 Detailed Results")
                    st.dataframe(df, use_container_width=True)
                    
                    # Download results
                    csv_data = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Predictions",
                        data=csv_data,
                        file_name="batch_predictions.csv",
                        mime="text/csv"
                    )
                    
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            logging.error(f"Batch prediction error: {str(e)}")

def display_prediction_examples():
    """Display prediction examples using sample data"""
    st.subheader("💡 Prediction Examples")
    
    # Load sample data
    sample_df = load_sample_data_for_prediction()
    
    if sample_df is not None:
        st.write("Here are some example predictions using sample data:")
        
        # Select random samples
        sample_size = min(5, len(sample_df))
        sample_examples = sample_df.sample(n=sample_size, random_state=42)
        
        if st.button("🎲 Generate Example Predictions"):
            try:
                with st.spinner("Making example predictions..."):
                    # Make predictions
                    predict_pipeline = PredictPipeline()
                    predictions = predict_pipeline.predict(sample_examples)
                    probabilities = predict_pipeline.predict_proba(sample_examples)
                    
                    # Display results
                    for i, (idx, row) in enumerate(sample_examples.iterrows()):
                        with st.expander(f"Example {i+1}: {predictions[i].title()}"):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write("**Input Features:**")
                                feature_display = {
                                    'Region': row.get('region', 'Unknown'),
                                    'Extraction Type': row.get('extraction_type', 'Unknown'),
                                    'Water Quality': row.get('water_quality', 'Unknown'),
                                    'Quantity': row.get('quantity', 'Unknown'),
                                    'Management': row.get('management', 'Unknown'),
                                    'Construction Year': row.get('construction_year', 'Unknown'),
                                    'Population': row.get('population', 'Unknown')
                                }
                                
                                for key, value in feature_display.items():
                                    st.write(f"• {key}: {value}")
                            
                            with col2:
                                st.write("**Prediction:**")
                                prediction_result = predictions[i]
                                
                                if prediction_result == 'functional':
                                    st.success(f"🟢 {prediction_result.upper()}")
                                elif prediction_result == 'functional needs repair':
                                    st.warning(f"🟡 {prediction_result.upper()}")
                                else:
                                    st.error(f"🔴 {prediction_result.upper()}")
                                
                                st.write("**Probabilities:**")
                                class_labels = ['functional', 'functional needs repair', 'non functional']
                                for j, label in enumerate(class_labels):
                                    st.write(f"• {label}: {probabilities[i][j]:.3f}")
                            
            except Exception as e:
                st.error(f"Error generating examples: {str(e)}")
    else:
        st.info("Sample data not available for examples.")

def display_model_info():
    """Display information about the trained model"""
    st.subheader("🤖 Model Information")
    
    try:
        # Load model report
        model_report_path = "artifacts/model_report.pkl"
        if os.path.exists(model_report_path):
            model_report = load_object(model_report_path)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Model Type", model_report.get('model_name', 'Unknown'))
            
            with col2:
                st.metric("Accuracy", f"{model_report.get('accuracy', 0):.4f}")
            
            with col3:
                st.metric("F1 Score", f"{model_report.get('f1_score', 0):.4f}")
            
            # Display model parameters
            st.write("**Best Parameters:**")
            best_params = model_report.get('best_params', {})
            if best_params:
                params_df = pd.DataFrame(list(best_params.items()), columns=['Parameter', 'Value'])
                st.dataframe(params_df, use_container_width=True)
            
        else:
            st.info("Model report not available.")
            
    except Exception as e:
        st.error(f"Error loading model information: {str(e)}")

def main():
    """Main predictions page function"""
    st.title("🎯 Water Pump Predictions")
    st.markdown("---")
    
    # Check if model is available
    if not check_model_availability():
        st.error("❌ Trained model not found. Please train a model first.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Go to Model Training"):
                st.switch_page("pages/03_🤖_Model_Training.py")
        
        with col2:
            if st.button("🏠 Go to Main Page"):
                st.switch_page("app.py")
        
        return
    
    # Display model information
    display_model_info()
    
    st.markdown("---")
    
    # Prediction tabs
    tabs = st.tabs(["Single Prediction", "Batch Prediction", "Examples"])
    
    with tabs[0]:
        display_single_prediction_form()
    
    with tabs[1]:
        display_batch_prediction()
    
    with tabs[2]:
        display_prediction_examples()
    
    # Instructions
    st.markdown("---")
    st.subheader("📝 Instructions")
    
    st.markdown("""
    ### How to Use:
    
    1. **Single Prediction**: Fill in the form with water pump details to get a prediction for one pump.
    2. **Batch Prediction**: Upload a CSV file with multiple water pump records to get predictions for all.
    3. **Examples**: View example predictions to understand how the model works.
    
    ### Prediction Classes:
    - 🟢 **Functional**: Water pump is working properly
    - 🟡 **Functional needs repair**: Water pump works but requires maintenance
    - 🔴 **Non-functional**: Water pump is not working
    
    ### Tips:
    - Ensure all required fields are filled for accurate predictions
    - For batch predictions, make sure your CSV has the same column structure as the training data
    - Higher confidence scores indicate more reliable predictions
    """)

if __name__ == "__main__":
    main()
