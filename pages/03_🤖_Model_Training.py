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
        if 'data_path' in st.session_state and st.session_state.data_path:
            df = pd.read_csv(st.session_state.data_path)
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
        }
    }
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Random Forest**")
        st.write(models["Random Forest"]["description"])
        st.write("**Pros:**")
        for pro in models["Random Forest"]["pros"]:
            st.write(f"• {pro}")
        st.write("**Cons:**")
        for con in models["Random Forest"]["cons"]:
            st.write(f"• {con}")
    
    with col2:
        st.write("**XGBoost**")
        st.write(models["XGBoost"]["description"])
        st.write("**Pros:**")
        for pro in models["XGBoost"]["pros"]:
            st.write(f"• {pro}")
        st.write("**Cons:**")
        for con in models["XGBoost"]["cons"]:
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
                    params_df = pd.DataFrame(list(best_params.items()), columns=['Parameter', 'Value'])
                    st.dataframe(params_df, use_container_width=True)
                
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
                
                # Get feature names (this is a simplified approach)
                # In practice, you'd need to track feature names through the pipeline
                feature_names = [f"feature_{i}" for i in range(len(model.feature_importances_))]
                
                # Create importance DataFrame
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                # Display top features
                top_features = importance_df.head(20)
                
                fig = px.bar(
                    top_features,
                    x='importance',
                    y='feature',
                    orientation='h',
                    title='Top 20 Feature Importances'
                )
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
                
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
                success = run_model_training(
                    st.session_state.data_path,
                    selected_models,
                    tuning_method,
                    cv_folds,
                    tuning_intensity
                )
                
                if success:
                    st.balloons()
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
        
        # Model management
        st.subheader("💾 Model Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if os.path.exists("artifacts/model.pkl"):
                st.success("✅ Trained model available")
                
                if st.button("Download Model"):
                    with open("artifacts/model.pkl", "rb") as f:
                        st.download_button(
                            label="Download Model File",
                            data=f,
                            file_name="water_pump_model.pkl",
                            mime="application/octet-stream"
                        )
            else:
                st.warning("⚠️ No trained model found")
        
        with col2:
            if os.path.exists("artifacts/preprocessor.pkl"):
                st.success("✅ Preprocessor available")
            else:
                st.warning("⚠️ No preprocessor found")
    
    else:
        st.warning("⚠️ No data loaded. Please upload data from the main page first.")
        
        if st.button("Go to Main Page"):
            st.switch_page("app.py")

if __name__ == "__main__":
    main()
