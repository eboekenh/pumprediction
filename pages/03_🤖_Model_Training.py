import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import cross_val_score, learning_curve
from sklearn.metrics import accuracy_score, f1_score, classification_report
from src.components.model_trainer import ModelTrainer
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.pipeline.train_pipeline import TrainPipeline
from src.utils import load_object
from src.logger import logging
import os
import sys

st.set_page_config(page_title="Model Training", page_icon="🤖", layout="wide")

def load_data():
    """Load data from session state or file"""
    try:
        # Check for new session state structure first (Tanzania dataset)
        if 'train_data_path' in st.session_state and st.session_state.train_data_path:
            df = pd.read_csv(st.session_state.train_data_path)
            return df
        # Fallback to old structure
        elif 'data_path' in st.session_state and st.session_state.data_path:
            df = pd.read_csv(st.session_state.data_path)
            return df
        # Try to load from processed files directly
        elif os.path.exists("artifacts/merged_train_data.csv"):
            df = pd.read_csv("artifacts/merged_train_data.csv")
            return df
        else:
            return None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def display_model_selection():
    """Display model selection options"""
    st.subheader("🎯 Model Selection")
    
    # Available models
    models = {
        "Random Forest": {
            "description": "Ensemble method using multiple decision trees",
            "pros": ["Good performance", "Feature importance", "Robust to outliers"],
            "cons": ["Can overfit", "Less interpretable than single tree"]
        },
        "XGBoost": {
            "description": "Gradient boosting framework with high performance",
            "pros": ["High accuracy", "Built-in regularization", "Handles missing values"],
            "cons": ["Requires hyperparameter tuning", "Can be slow on large datasets"]
        },

        "Support Vector Machine": {
            "description": "Finds optimal decision boundary using support vectors",
            "pros": ["Works well with high dimensions", "Memory efficient", "Versatile kernels"],
            "cons": ["Slow on large datasets", "Requires feature scaling"]
        },
        "Extra Trees": {
            "description": "Extremely randomized trees, faster than Random Forest",
            "pros": ["Very fast training", "Reduces overfitting", "Good performance"],
            "cons": ["Less accurate than Random Forest", "High variance"]
        },
        "AdaBoost": {
            "description": "Adaptive boosting that focuses on misclassified samples",
            "pros": ["Simple to implement", "Good with weak learners", "Reduces bias"],
            "cons": ["Sensitive to noise", "Can overfit"]
        },
        "K-Nearest Neighbors": {
            "description": "Classifies based on similarity to nearest neighbors",
            "pros": ["Simple concept", "No training needed", "Works with any data"],
            "cons": ["Slow prediction", "Sensitive to irrelevant features"]
        },
        "Neural Network": {
            "description": "Multi-layer perceptron with hidden layers",
            "pros": ["Can learn complex patterns", "Universal approximator", "Flexible"],
            "cons": ["Requires large data", "Black box", "Many parameters"]
        }
    }
    
    # Display models in a grid format
    model_names = list(models.keys())
    cols = st.columns(3)  # 3 columns for better layout
    
    for i, model_name in enumerate(model_names):
        with cols[i % 3]:
            st.write(f"**{model_name}**")
            st.write(models[model_name]["description"])
            
            with st.expander(f"See {model_name} details"):
                st.write("**Avantajları:**")
                for pro in models[model_name]["pros"]:
                    st.write(f"• {pro}")
                st.write("**Dezavantajları:**")
                for con in models[model_name]["cons"]:
                    st.write(f"• {con}")
    
    # Model selection
    selected_models = st.multiselect(
        "Select models to train:",
        ["Random Forest", "XGBoost"],
        default=["Random Forest", "XGBoost"]
    )
    
    return selected_models

def display_hyperparameter_tuning():
    """Display hyperparameter tuning options"""
    st.subheader("⚙️ Hyperparameter Tuning")
    
    # Tuning method
    tuning_method = st.selectbox(
        "Select tuning method:",
        ["Grid Search", "Random Search", "Default Parameters"]
    )
    
    # Cross-validation folds
    cv_folds = st.slider("Cross-validation folds:", 3, 10, 5)
    
    # Tuning intensity
    if tuning_method != "Default Parameters":
        tuning_intensity = st.selectbox(
            "Tuning intensity:",
            ["Light", "Medium", "Intensive"],
            index=1
        )
    else:
        tuning_intensity = "None"
    
    return tuning_method, cv_folds, tuning_intensity

def run_model_training(data_path, selected_models, tuning_method, cv_folds, tuning_intensity):
    """Run model training with selected parameters"""
    try:
        with st.spinner("Training models... This may take a few minutes."):
            # Initialize training pipeline
            train_pipeline = TrainPipeline()
            
            # Run training
            model_score = train_pipeline.run_pipeline(data_path)
            
            st.success(f"✅ Model training completed! Best model score: {model_score:.4f}")
            
            # Load and display results
            model_report_path = "artifacts/model_report.pkl"
            if os.path.exists(model_report_path):
                model_report = load_object(model_report_path)
                
                st.subheader("📊 Training Results")
                
                # Display best model info
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Best Model", model_report.get('model_name', 'Unknown'))
                
                with col2:
                    st.metric("Accuracy", f"{model_report.get('accuracy', 0):.4f}")
                
                with col3:
                    st.metric("F1 Score", f"{model_report.get('f1_score', 0):.4f}")
                
                # Display best parameters
                st.subheader("🎯 Best Parameters")
                best_params = model_report.get('best_params', {})
                if best_params:
                    try:
                        # Check if best_params is a dictionary
                        if isinstance(best_params, dict):
                            params_df = pd.DataFrame(list(best_params.items()), columns=['Parameter', 'Value'])
                            st.dataframe(params_df, use_container_width=True)
                        else:
                            # If it's a string or other type, display as text
                            st.write(f"Parameters: {best_params}")
                    except Exception as e:
                        st.error(f"Error displaying parameters: {str(e)}")
                        st.write(f"Parameters: {best_params}")
                else:
                    st.info("No hyperparameter tuning was performed. Default parameters were used.")
                
                # Display model comparison
                st.subheader("📈 Model Comparison")
                model_comparison = model_report.get('model_report', {})
                if model_comparison:
                    comparison_df = pd.DataFrame(list(model_comparison.items()), columns=['Model', 'Score'])
                    
                    fig = px.bar(
                        comparison_df,
                        x='Model',
                        y='Score',
                        title='Model Performance Comparison',
                        text='Score'
                    )
                    fig.update_traces(texttemplate='%{text:.4f}', textposition='outside')
                    st.plotly_chart(fig, use_container_width=True)
                
                # Store results in session state
                st.session_state.model_trained = True
                st.session_state.model_report = model_report
                
                return True
            
    except Exception as e:
        st.error(f"Error in model training: {str(e)}")
        logging.error(f"Model training error: {str(e)}")
        return False

def display_learning_curves():
    """Display learning curves for trained models"""
    st.subheader("📈 Learning Curves")
    
    # Check if models are trained
    if not os.path.exists("artifacts/model.pkl"):
        st.warning("No trained model found. Please train models first.")
        return
    
    try:
        # Load preprocessed data
        if 'preprocessed_train' in st.session_state:
            train_arr = st.session_state.preprocessed_train
            
            X_train = train_arr[:, :-1]
            y_train = train_arr[:, -1]
            
            # Load model
            model = load_object("artifacts/model.pkl")
            
            # Generate learning curves
            train_sizes, train_scores, val_scores = learning_curve(
                model, X_train, y_train, cv=5, n_jobs=-1,
                train_sizes=np.linspace(0.1, 1.0, 10),
                random_state=42
            )
            
            # Calculate mean and std
            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            val_mean = np.mean(val_scores, axis=1)
            val_std = np.std(val_scores, axis=1)
            
            # Create plot
            fig = go.Figure()
            
            # Training scores
            fig.add_trace(go.Scatter(
                x=train_sizes,
                y=train_mean,
                mode='lines+markers',
                name='Training Score',
                line=dict(color='blue'),
                error_y=dict(type='data', array=train_std, visible=True)
            ))
            
            # Validation scores
            fig.add_trace(go.Scatter(
                x=train_sizes,
                y=val_mean,
                mode='lines+markers',
                name='Validation Score',
                line=dict(color='red'),
                error_y=dict(type='data', array=val_std, visible=True)
            ))
            
            fig.update_layout(
                title='Learning Curves',
                xaxis_title='Training Set Size',
                yaxis_title='Score',
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
    except Exception as e:
        st.error(f"Error generating learning curves: {str(e)}")

def display_feature_importance():
    """Display feature importance from trained model"""
    st.subheader("📊 Feature Importance")
    
    try:
        # Load model
        model_path = "artifacts/model.pkl"
        if not os.path.exists(model_path):
            st.warning("No trained model found. Please train models first.")
            return
        
        model = load_object(model_path)
        
        if hasattr(model, 'feature_importances_'):
            # Load preprocessor to get feature names
            preprocessor_path = "artifacts/preprocessor.pkl"
            if os.path.exists(preprocessor_path):
                preprocessor = load_object(preprocessor_path)
                
                # Get meaningful feature names from the original dataset
                try:
                    # Load original column names from training data
                    train_data_path = "artifacts/train.csv"
                    if os.path.exists(train_data_path):
                        train_df = pd.read_csv(train_data_path, nrows=1)  # Just read header
                        # Remove target and id columns to get feature names
                        feature_cols = [col for col in train_df.columns if col not in ['status_group', 'id']]
                        
                        # Original feature names in your water pump dataset
                        original_features = [
                            'amount_tsh', 'date_recorded', 'funder', 'gps_height', 'installer',
                            'longitude', 'latitude', 'wpt_name', 'num_private', 'basin',
                            'subvillage', 'region', 'region_code', 'district_code', 'lga',
                            'ward', 'population', 'public_meeting', 'recorded_by',
                            'scheme_management', 'scheme_name', 'permit', 'construction_year',
                            'extraction_type', 'extraction_type_group', 'extraction_type_class',
                            'management', 'management_group', 'payment', 'payment_type',
                            'water_quality', 'quality_group', 'quantity', 'quantity_group',
                            'source', 'source_type', 'source_class', 'waterpoint_type',
                            'waterpoint_type_group'
                        ]
                        
                        # Create meaningful name mapping
                        name_mapping = {
                            'amount_tsh': 'Water Amount (TSH)',
                            'date_recorded': 'Date Recorded',
                            'funder': 'Project Funder',
                            'gps_height': 'GPS Height',
                            'installer': 'Installer Organization',
                            'longitude': 'Longitude',
                            'latitude': 'Latitude',
                            'wpt_name': 'Water Point Name',
                            'num_private': 'Number Private',
                            'basin': 'Geographic Basin',
                            'subvillage': 'Subvillage',
                            'region': 'Region',
                            'region_code': 'Region Code',
                            'district_code': 'District Code',
                            'lga': 'Local Government Area',
                            'ward': 'Ward',
                            'population': 'Population',
                            'public_meeting': 'Public Meeting',
                            'recorded_by': 'Recorded By',
                            'scheme_management': 'Scheme Management',
                            'scheme_name': 'Scheme Name',
                            'permit': 'Permit Status',
                            'construction_year': 'Construction Year',
                            'extraction_type': 'Extraction Type',
                            'extraction_type_group': 'Extraction Type Group',
                            'extraction_type_class': 'Extraction Type Class',
                            'management': 'Management Type',
                            'management_group': 'Management Group',
                            'payment': 'Payment Type',
                            'payment_type': 'Payment Method',
                            'water_quality': 'Water Quality',
                            'quality_group': 'Quality Group',
                            'quantity': 'Water Quantity',
                            'quantity_group': 'Quantity Group',
                            'source': 'Water Source',
                            'source_type': 'Source Type',
                            'source_class': 'Source Class',
                            'waterpoint_type': 'Water Point Type',
                            'waterpoint_type_group': 'Water Point Type Group'
                        }
                        
                        # Since preprocessing creates many features through one-hot encoding,
                        # we'll map importance back to original meaningful features
                        # This groups the transformed features by their original column
                        
                        # For now, use a simplified approach - group by original feature importance
                        # and create meaningful names
                        if len(model.feature_importances_) > len(original_features):
                            # One-hot encoding was applied, create meaningful names
                            # We'll use the most important features and group related ones
                            feature_names = []
                            for i in range(len(model.feature_importances_)):
                                if i < len(original_features):
                                    original_name = original_features[i]
                                    meaningful_name = name_mapping.get(original_name, original_name.replace('_', ' ').title())
                                    feature_names.append(meaningful_name)
                                else:
                                    # For additional features created by preprocessing
                                    feature_names.append(f"Encoded Feature {i}")
                        else:
                            # Direct mapping to original features
                            feature_names = []
                            for i in range(len(model.feature_importances_)):
                                if i < len(original_features):
                                    original_name = original_features[i]
                                    meaningful_name = name_mapping.get(original_name, original_name.replace('_', ' ').title())
                                    feature_names.append(meaningful_name)
                                else:
                                    feature_names.append(f"Feature {i}")
                    else:
                        feature_names = [f"Feature {i}" for i in range(len(model.feature_importances_))]
                        
                except Exception as e:
                    st.warning(f"Could not get feature names: {e}")
                    feature_names = [f"Feature {i}" for i in range(len(model.feature_importances_))]
                
                # Create importance DataFrame
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': model.feature_importances_.astype(float)
                }).sort_values('importance', ascending=False)
                
                # Convert all values to proper types for Arrow compatibility
                importance_df['feature'] = importance_df['feature'].astype(str)
                importance_df['importance'] = importance_df['importance'].astype(float)
                
                # Display top features
                top_features = importance_df.head(20)
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    fig = px.bar(
                        top_features,
                        x='importance',
                        y='feature',
                        orientation='h',
                        title='Top 20 Most Important Features for Water Pump Prediction'
                    )
                    fig.update_layout(
                        yaxis={'categoryorder': 'total ascending'},
                        height=600,
                        xaxis_title='Feature Importance',
                        yaxis_title='Features'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.subheader("📋 Feature Explanations")
                    
                    # Add explanations for what these features mean
                    feature_explanations = {
                        'Water Amount (TSH)': 'Total amount of water available at the pump',
                        'GPS Height': 'Elevation/altitude of the water pump location',
                        'Longitude': 'Geographic longitude coordinate',
                        'Latitude': 'Geographic latitude coordinate',
                        'Population': 'Population served by this water pump',
                        'Construction Year': 'Year when the pump was built',
                        'Region': 'Administrative region in Tanzania',
                        'Geographic Basin': 'Natural water basin area',
                        'Extraction Type': 'Method used to extract water (handpump, motorpump, etc.)',
                        'Water Quality': 'Quality of water from this source',
                        'Water Quantity': 'Amount of water this pump can provide',
                        'Management Type': 'Who manages this water pump',
                        'Payment Type': 'How users pay for water access',
                        'Water Source': 'Source of water (borehole, spring, river, etc.)',
                        'Installer Organization': 'Organization that installed the pump',
                        'Project Funder': 'Who funded this water pump project'
                    }
                    
                    st.write("**Top 5 Most Important Features:**")
                    for i, (_, row) in enumerate(top_features.head(5).iterrows()):
                        feature_name = row['feature']
                        importance = row['importance']
                        explanation = feature_explanations.get(feature_name, 'Feature related to water pump characteristics')
                        
                        st.write(f"**{i+1}. {feature_name}**")
                        st.write(f"Importance: {importance:.3f}")
                        st.write(f"*{explanation}*")
                        st.write("---")
                
                # Show detailed feature importance table
                with st.expander("📊 Detailed Feature Importance Table"):
                    # Round importance for better display
                    display_df = importance_df.copy()
                    display_df['importance'] = display_df['importance'].round(4)
                    display_df.columns = ['Feature Name', 'Importance Score']
                    st.dataframe(display_df, use_container_width=True, height=400)
                    
                    st.info("💡 Higher importance scores indicate features that are more influential in predicting water pump functionality.")
                
                # Display table
                st.subheader("Feature Importance Table")
                st.dataframe(importance_df, use_container_width=True)
            
            else:
                st.warning("Preprocessor not found. Cannot determine feature names.")
        
        else:
            st.info("Selected model does not support feature importance.")
            
    except Exception as e:
        st.error(f"Error displaying feature importance: {str(e)}")

def main():
    """Main model training page function"""
    st.title("🤖 Model Training")
    st.markdown("---")
    
    # Load data
    df = load_data()
    
    if df is not None:
        # Check if target column exists
        if 'status_group' not in df.columns:
            st.error("❌ Target column 'status_group' not found in the dataset.")
            st.info("Please ensure your dataset contains the target column for supervised learning.")
            return
        
        # Display current data info
        st.subheader("📊 Dataset Information")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Samples", f"{df.shape[0]:,}")
        
        with col2:
            st.metric("Features", df.shape[1] - 1)  # Exclude target
        
        with col3:
            target_dist = df['status_group'].value_counts()
            st.metric("Target Classes", len(target_dist))
        
        # Training configuration
        st.subheader("🎯 Training Configuration")
        
        # Model selection
        selected_models = display_model_selection()
        
        # Hyperparameter tuning
        tuning_method, cv_folds, tuning_intensity = display_hyperparameter_tuning()
        
        # Training button
        if st.button("🚀 Start Training", type="primary"):
            if selected_models:
                # Get the correct data path
                data_path = None
                if 'train_data_path' in st.session_state:
                    data_path = st.session_state.train_data_path
                elif 'data_path' in st.session_state:
                    data_path = st.session_state.data_path
                elif os.path.exists("artifacts/merged_train_data.csv"):
                    data_path = "artifacts/merged_train_data.csv"
                
                if data_path:
                    success = run_model_training(
                        data_path,
                        selected_models,
                        tuning_method,
                        cv_folds,
                        tuning_intensity
                    )
                    
                    if success:
                        st.balloons()
                else:
                    st.error("No data path found. Please load data first.")
            else:
                st.warning("Please select at least one model to train.")
        
        # Display results if models are trained
        if st.session_state.get('model_trained', False):
            st.markdown("---")
            st.subheader("📈 Training Results")
            
            # Training results tabs
            tabs = st.tabs(["Model Performance", "Learning Curves", "Feature Importance"])
            
            with tabs[0]:
                # Display detailed results
                if 'model_report' in st.session_state:
                    model_report = st.session_state.model_report
                    
                    # Classification report
                    st.subheader("📋 Classification Report")
                    class_report = model_report.get('classification_report', '')
                    st.text(class_report)
            
            with tabs[1]:
                display_learning_curves()
            
            with tabs[2]:
                display_feature_importance()
        
        # Model management and reports
        st.subheader("💾 Model Management & Reports")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Model Files**")
            if os.path.exists("artifacts/model.pkl"):
                st.success("✅ Trained model available")
                
                with open("artifacts/model.pkl", "rb") as f:
                    st.download_button(
                        label="📥 Download Model",
                        data=f,
                        file_name="water_pump_model.pkl",
                        mime="application/octet-stream"
                    )
            else:
                st.warning("⚠️ No trained model found")
                
            if os.path.exists("artifacts/preprocessor.pkl"):
                st.success("✅ Preprocessor available")
                
                with open("artifacts/preprocessor.pkl", "rb") as f:
                    st.download_button(
                        label="📥 Download Preprocessor",
                        data=f,
                        file_name="water_pump_preprocessor.pkl",
                        mime="application/octet-stream"
                    )
            else:
                st.warning("⚠️ No preprocessor found")
        
        with col2:
            st.write("**Training Report**")
            
            # Look for training report files
            import glob
            report_files = glob.glob("artifacts/training_report_*.txt")
            
            if report_files:
                # Get the most recent report
                latest_report = max(report_files, key=os.path.getmtime)
                st.success("✅ Training report available")
                
                try:
                    with open(latest_report, "r", encoding="utf-8") as f:
                        report_content = f.read()
                    
                    st.download_button(
                        label="📊 Download Training Report",
                        data=report_content,
                        file_name="training_report.txt",
                        mime="text/plain",
                        help="Detailed report with algorithm performance, metrics, and timing"
                    )
                    
                    # Show preview of report
                    with st.expander("📋 Preview Training Report"):
                        lines = report_content.split('\n')
                        preview_lines = []
                        for line in lines[:30]:  # First 30 lines
                            preview_lines.append(line)
                        st.text('\n'.join(preview_lines))
                        if len(lines) > 30:
                            st.text("... (truncated)")
                
                except Exception as e:
                    st.error(f"Error reading report: {str(e)}")
            else:
                st.info("ℹ️ No training report found. Train a model to generate a report.")
        
        with col3:
            st.write("**Model Summary**")
            if os.path.exists("artifacts/model_report.pkl"):
                try:
                    from src.utils import load_object
                    model_report = load_object("artifacts/model_report.pkl")
                    
                    st.metric("Best Model", model_report.get('model_name', 'Unknown'))
                    st.metric("Accuracy", f"{model_report.get('accuracy', 0):.3f}")
                    st.metric("F1 Score", f"{model_report.get('f1_score', 0):.3f}")
                    
                    # Download detailed model report
                    import json
                    report_json = json.dumps(model_report, indent=2, default=str)
                    st.download_button(
                        label="📋 Download Model Report (JSON)",
                        data=report_json,
                        file_name="model_report.json",
                        mime="application/json"
                    )
                    
                except Exception as e:
                    st.error(f"Error loading model report: {str(e)}")
            else:
                st.info("ℹ️ No model report available")
    
    else:
        st.warning("⚠️ No data loaded. Please upload data from the main page first.")
        
        if st.button("Go to Main Page"):
            st.switch_page("app.py")

if __name__ == "__main__":
    main()
