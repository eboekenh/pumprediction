import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
from sklearn.model_selection import cross_val_score
from src.utils import load_object, calculate_metrics
from src.logger import logging
import os

st.set_page_config(page_title="Model Evaluation", page_icon="📊", layout="wide")

def load_preprocessed_data():
    """Load preprocessed data from files or session state"""
    try:
        # First try to load from session state
        if 'preprocessed_train' in st.session_state and 'preprocessed_test' in st.session_state:
            return st.session_state.preprocessed_train, st.session_state.preprocessed_test
        
        # If not in session state, load from files and preprocess
        train_path = "artifacts/train.csv"
        test_path = "artifacts/test.csv"
        preprocessor_path = "artifacts/preprocessor.pkl"
        
        if os.path.exists(train_path) and os.path.exists(test_path) and os.path.exists(preprocessor_path):
            # Load the data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            # Load preprocessor with direct pickle loading
            import pickle
            with open(preprocessor_path, 'rb') as f:
                preprocessor = pickle.load(f)
            
            # Separate features and target
            X_train = train_df.drop(['status_group', 'id'], axis=1, errors='ignore')
            y_train = train_df['status_group'] if 'status_group' in train_df.columns else None
            
            X_test = test_df.drop(['status_group', 'id'], axis=1, errors='ignore') 
            y_test = test_df['status_group'] if 'status_group' in test_df.columns else None
            
            if y_train is None or y_test is None:
                st.error("Target column 'status_group' not found in data files.")
                return None, None
            
            # Transform the data
            X_train_processed = preprocessor.transform(X_train)
            X_test_processed = preprocessor.transform(X_test)
            
            # Convert target to numeric if needed
            label_encoder_path = "artifacts/label_encoder.pkl"
            if os.path.exists(label_encoder_path):
                with open(label_encoder_path, 'rb') as f:
                    label_encoder = pickle.load(f)
                y_train_encoded = label_encoder.transform(y_train)
                y_test_encoded = label_encoder.transform(y_test)
            else:
                # Handle string labels directly
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                y_train_encoded = le.fit_transform(y_train)
                y_test_encoded = le.transform(y_test)
            
            # Combine features and targets
            train_arr = np.column_stack([X_train_processed, y_train_encoded])
            test_arr = np.column_stack([X_test_processed, y_test_encoded])
            
            # Store in session state for future use
            st.session_state.preprocessed_train = train_arr
            st.session_state.preprocessed_test = test_arr
            
            return train_arr, test_arr
        else:
            return None, None
            
    except Exception as e:
        st.error(f"Error loading preprocessed data: {str(e)}")
        return None, None

def display_model_performance():
    """Display comprehensive model performance metrics"""
    st.subheader("📈 Model Performance Metrics")
    
    # Check if model and data are available
    model_path = "artifacts/model.pkl"
    if not os.path.exists(model_path):
        st.warning("⚠️ No trained model found. Please train a model first.")
        return None, None, None
    
    train_arr, test_arr = load_preprocessed_data()
    if train_arr is None or test_arr is None:
        st.warning("⚠️ No preprocessed data found. Please preprocess data first.")
        return None, None, None
    
    try:
        # Load model with direct pickle loading
        import pickle
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Prepare data
        X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
        X_test, y_test = test_arr[:, :-1], test_arr[:, -1]
        
        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Calculate metrics
        train_metrics = calculate_metrics(y_train, y_pred_train)
        test_metrics = calculate_metrics(y_test, y_pred_test)
        
        # Display metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Training Set Performance")
            st.metric("Accuracy", f"{train_metrics['accuracy']:.4f}")
            st.metric("Precision", f"{train_metrics['precision']:.4f}")
            st.metric("Recall", f"{train_metrics['recall']:.4f}")
            st.metric("F1 Score", f"{train_metrics['f1_score']:.4f}")
        
        with col2:
            st.subheader("🎯 Test Set Performance")
            st.metric("Accuracy", f"{test_metrics['accuracy']:.4f}")
            st.metric("Precision", f"{test_metrics['precision']:.4f}")
            st.metric("Recall", f"{test_metrics['recall']:.4f}")
            st.metric("F1 Score", f"{test_metrics['f1_score']:.4f}")
        
        # Performance comparison
        st.subheader("📊 Performance Comparison")
        
        comparison_data = {
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
            'Training': [
                train_metrics['accuracy'],
                train_metrics['precision'],
                train_metrics['recall'],
                train_metrics['f1_score']
            ],
            'Test': [
                test_metrics['accuracy'],
                test_metrics['precision'],
                test_metrics['recall'],
                test_metrics['f1_score']
            ]
        }
        
        comparison_df = pd.DataFrame(comparison_data)
        
        fig = px.bar(
            comparison_df.melt(id_vars='Metric', value_vars=['Training', 'Test']),
            x='Metric',
            y='value',
            color='variable',
            title='Training vs Test Performance',
            barmode='group'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        return model, y_test, y_pred_test
        
    except Exception as e:
        st.error(f"Error calculating model performance: {str(e)}")
        return None, None, None

def display_confusion_matrix(y_true, y_pred):
    """Display confusion matrix visualization"""
    st.subheader("🔍 Confusion Matrix")
    
    try:
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Get class labels
        label_encoder_path = "artifacts/label_encoder.pkl"
        if os.path.exists(label_encoder_path):
            import pickle
            with open(label_encoder_path, 'rb') as f:
                label_encoder = pickle.load(f)
            class_names = label_encoder.classes_
        else:
            class_names = ['functional', 'functional needs repair', 'non functional']
        
        # Create confusion matrix plot
        fig = px.imshow(
            cm,
            x=class_names,
            y=class_names,
            color_continuous_scale='Blues',
            title='Confusion Matrix',
            text_auto=True
        )
        
        fig.update_layout(
            xaxis_title='Predicted',
            yaxis_title='Actual',
            width=600,
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display normalized confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        fig_norm = px.imshow(
            cm_normalized,
            x=class_names,
            y=class_names,
            color_continuous_scale='Blues',
            title='Normalized Confusion Matrix',
            text_auto='.2f'
        )
        
        fig_norm.update_layout(
            xaxis_title='Predicted',
            yaxis_title='Actual',
            width=600,
            height=500
        )
        
        st.plotly_chart(fig_norm, use_container_width=True)
        
        # Class-wise performance
        st.subheader("📋 Class-wise Performance")
        
        precision_per_class = precision_score(y_true, y_pred, average=None)
        recall_per_class = recall_score(y_true, y_pred, average=None)
        f1_per_class = f1_score(y_true, y_pred, average=None)
        
        class_performance = pd.DataFrame({
            'Class': class_names,
            'Precision': precision_per_class,
            'Recall': recall_per_class,
            'F1 Score': f1_per_class
        })
        
        st.dataframe(class_performance, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error displaying confusion matrix: {str(e)}")

def display_classification_report(y_true, y_pred):
    """Display detailed classification report"""
    st.subheader("📋 Classification Report")
    
    try:
        # Get class labels
        label_encoder_path = "artifacts/label_encoder.pkl"
        if os.path.exists(label_encoder_path):
            label_encoder = load_object(label_encoder_path)
            target_names = label_encoder.classes_
        else:
            target_names = ['functional', 'functional needs repair', 'non functional']
        
        # Generate classification report
        report = classification_report(
            y_true, y_pred, 
            target_names=target_names,
            output_dict=True
        )
        
        # Convert to DataFrame
        report_df = pd.DataFrame(report).transpose()
        
        # Display as table
        st.dataframe(report_df, use_container_width=True)
        
        # Visualize per-class metrics
        class_metrics = report_df.iloc[:-3]  # Exclude accuracy, macro avg, weighted avg
        
        fig = px.bar(
            class_metrics.reset_index(),
            x='index',
            y=['precision', 'recall', 'f1-score'],
            title='Per-Class Performance Metrics',
            barmode='group',
            labels={'index': 'Class'}
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error displaying classification report: {str(e)}")

def display_cross_validation_results():
    """Display cross-validation results"""
    st.subheader("🔄 Cross-Validation Results")
    
    try:
        # Load model and data
        model_path = "artifacts/model.pkl"
        if not os.path.exists(model_path):
            st.warning("No trained model found.")
            return
        
        model = load_object(model_path)
        train_arr, _ = load_preprocessed_data()
        
        if train_arr is None:
            st.warning("No preprocessed data found.")
            return
        
        X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
        
        # Perform cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        
        # Display results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Mean CV Score", f"{cv_scores.mean():.4f}")
        
        with col2:
            st.metric("Std CV Score", f"{cv_scores.std():.4f}")
        
        with col3:
            st.metric("CV Range", f"{cv_scores.min():.4f} - {cv_scores.max():.4f}")
        
        # Visualize CV scores
        fig = px.box(
            y=cv_scores,
            title='Cross-Validation Scores Distribution'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Display individual fold scores
        cv_df = pd.DataFrame({
            'Fold': range(1, len(cv_scores) + 1),
            'Score': cv_scores
        })
        
        fig_bar = px.bar(
            cv_df,
            x='Fold',
            y='Score',
            title='Cross-Validation Scores by Fold'
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error in cross-validation: {str(e)}")

def display_model_comparison():
    """Display comparison with different models if available"""
    st.subheader("⚖️ Model Comparison")
    
    try:
        # Load model report
        model_report_path = "artifacts/model_report.pkl"
        if os.path.exists(model_report_path):
            model_report = load_object(model_report_path)
            
            # Get model comparison data
            model_comparison = model_report.get('model_report', {})
            
            if model_comparison:
                # Create comparison DataFrame
                comparison_df = pd.DataFrame(
                    list(model_comparison.items()),
                    columns=['Model', 'Score']
                )
                
                # Visualize comparison
                fig = px.bar(
                    comparison_df,
                    x='Model',
                    y='Score',
                    title='Model Performance Comparison',
                    text='Score'
                )
                fig.update_traces(texttemplate='%{text:.4f}', textposition='outside')
                st.plotly_chart(fig, use_container_width=True)
                
                # Display table
                st.dataframe(comparison_df, use_container_width=True)
                
                # Best model info
                best_model = model_report.get('model_name', 'Unknown')
                best_score = model_report.get('best_score', 0)
                
                st.success(f"🏆 Best Model: {best_model} (Score: {best_score:.4f})")
            
            else:
                st.info("No model comparison data available.")
        
        else:
            st.info("No model report found.")
            
    except Exception as e:
        st.error(f"Error displaying model comparison: {str(e)}")

def display_model_diagnostics():
    """Display model diagnostics and recommendations"""
    st.subheader("🔧 Model Diagnostics")
    
    try:
        # Load model and data
        model_path = "artifacts/model.pkl"
        if not os.path.exists(model_path):
            st.warning("No trained model found.")
            return
        
        model = load_object(model_path)
        train_arr, test_arr = load_preprocessed_data()
        
        if train_arr is None or test_arr is None:
            st.warning("No preprocessed data found.")
            return
        
        # Prepare data
        X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
        X_test, y_test = test_arr[:, :-1], test_arr[:, -1]
        
        # Calculate metrics
        train_accuracy = accuracy_score(y_train, model.predict(X_train))
        test_accuracy = accuracy_score(y_test, model.predict(X_test))
        
        # Overfitting check
        overfitting_threshold = 0.05
        overfitting_gap = train_accuracy - test_accuracy
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Overfitting Gap", f"{overfitting_gap:.4f}")
            
            if overfitting_gap > overfitting_threshold:
                st.warning("⚠️ Potential overfitting detected")
            else:
                st.success("✅ No significant overfitting")
        
        with col2:
            st.metric("Model Complexity", "Medium" if hasattr(model, 'n_estimators') else "Unknown")
        
        # Recommendations
        st.subheader("💡 Recommendations")
        
        recommendations = []
        
        if overfitting_gap > overfitting_threshold:
            recommendations.append("Consider reducing model complexity or adding regularization")
        
        if test_accuracy < 0.7:
            recommendations.append("Consider feature engineering or collecting more data")
        
        if test_accuracy > 0.9:
            recommendations.append("Excellent performance! Consider deploying the model")
        
        if not recommendations:
            recommendations.append("Model performance looks good. Consider testing on new data")
        
        for i, rec in enumerate(recommendations, 1):
            st.write(f"{i}. {rec}")
            
    except Exception as e:
        st.error(f"Error in model diagnostics: {str(e)}")

def main():
    """Main model evaluation page function"""
    st.title("📊 Model Evaluation")
    st.markdown("---")
    
    # Check if model is available
    model_path = "artifacts/model.pkl"
    if not os.path.exists(model_path):
        st.warning("⚠️ No trained model found. Please train a model first.")
        
        if st.button("Go to Model Training"):
            st.switch_page("pages/03_🤖_Model_Training.py")
        return
    
    # Load and display model performance
    result = display_model_performance()
    
    if result is not None and len(result) == 3:
        model, y_test, y_pred_test = result
        
        if model is not None and y_test is not None and y_pred_test is not None:
            # Evaluation tabs
            tabs = st.tabs([
                "Confusion Matrix",
                "Classification Report",
                "Cross-Validation",
                "Model Comparison",
                "Diagnostics"
            ])
            
            with tabs[0]:
                display_confusion_matrix(y_test, y_pred_test)
            
            with tabs[1]:
                display_classification_report(y_test, y_pred_test)
            
            with tabs[2]:
                display_cross_validation_results()
            
            with tabs[3]:
                display_model_comparison()
            
            with tabs[4]:
                display_model_diagnostics()
        else:
            st.warning("⚠️ Model evaluation data not available.")
    else:
        st.warning("⚠️ Unable to load model evaluation data.")
    
    # Export evaluation results
    st.subheader("📥 Export Results")
    
    if st.button("Generate Evaluation Report"):
        try:
            # Create evaluation report
            model_report_path = "artifacts/model_report.pkl"
            if os.path.exists(model_report_path):
                model_report = load_object(model_report_path)
                
                report_content = f"""
# Model Evaluation Report

## Model Information
- Model Type: {model_report.get('model_name', 'Unknown')}
- Best Score: {model_report.get('best_score', 0):.4f}
- Accuracy: {model_report.get('accuracy', 0):.4f}
- F1 Score: {model_report.get('f1_score', 0):.4f}

## Best Parameters
{model_report.get('best_params', {})}

## Classification Report
{model_report.get('classification_report', 'N/A')}

## Model Comparison
{model_report.get('model_report', {})}
                """
                
                st.download_button(
                    label="Download Evaluation Report",
                    data=report_content,
                    file_name="model_evaluation_report.txt",
                    mime="text/plain"
                )
        
        except Exception as e:
            st.error(f"Error generating report: {str(e)}")

if __name__ == "__main__":
    main()
