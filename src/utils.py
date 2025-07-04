import os
import sys
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
    """
    Save object to file using pickle
    
    Args:
        file_path: Path to save the object
        obj: Object to save
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
            
    except Exception as e:
        logging.error(f"Error saving object: {str(e)}")
        raise CustomException(e, sys)

def load_object(file_path):
    """
    Load object from file using pickle
    
    Args:
        file_path: Path to load the object from
        
    Returns:
        Loaded object
    """
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
            
    except Exception as e:
        logging.error(f"Error loading object: {str(e)}")
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    """
    Evaluate multiple models with hyperparameter tuning
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        models: Dictionary of models
        param: Dictionary of parameters for each model
        
    Returns:
        Dictionary of model scores
    """
    try:
        report = {}
        
        for model_name, model in models.items():
            # Get parameters for current model
            params = param.get(model_name, {})
            
            # Perform grid search
            gs = GridSearchCV(
                estimator=model,
                param_grid=params,
                cv=3,
                scoring='accuracy',
                n_jobs=-1
            )
            
            # Fit the model
            gs.fit(X_train, y_train)
            
            # Get best model
            best_model = gs.best_estimator_
            
            # Make predictions
            y_pred = best_model.predict(X_test)
            
            # Calculate score
            test_score = accuracy_score(y_test, y_pred)
            
            report[model_name] = test_score
            
            logging.info(f"{model_name} - Best params: {gs.best_params_}")
            logging.info(f"{model_name} - Test score: {test_score}")
        
        return report
        
    except Exception as e:
        logging.error(f"Error evaluating models: {str(e)}")
        raise CustomException(e, sys)

def get_feature_importance(model, feature_names):
    """
    Get feature importance from trained model
    
    Args:
        model: Trained model
        feature_names: List of feature names
        
    Returns:
        DataFrame with feature importance
    """
    try:
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return importance_df
        else:
            return None
            
    except Exception as e:
        logging.error(f"Error getting feature importance: {str(e)}")
        return None

def calculate_metrics(y_true, y_pred):
    """
    Calculate various classification metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary of metrics
    """
    try:
        from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
        from sklearn.metrics import confusion_matrix, classification_report
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted'),
            'confusion_matrix': confusion_matrix(y_true, y_pred),
            'classification_report': classification_report(y_true, y_pred)
        }
        
        return metrics
        
    except Exception as e:
        logging.error(f"Error calculating metrics: {str(e)}")
        raise CustomException(e, sys)
