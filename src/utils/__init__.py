# Utils package

import pickle
import os
import sys
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
import pandas as pd

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
        raise Exception(f"Error saving object to {file_path}: {str(e)}")

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
        raise Exception(f"Error loading object from {file_path}: {str(e)}")

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
        
        for i in range(len(list(models))):
            model = list(models.values())[i]
            params = param[list(models.keys())[i]]
            
            if params:
                gs = GridSearchCV(model, params, cv=3, scoring='accuracy', n_jobs=-1)
                gs.fit(X_train, y_train)
                
                model = gs.best_estimator_
            else:
                model.fit(X_train, y_train)
            
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            train_model_score = accuracy_score(y_train, y_train_pred)
            test_model_score = accuracy_score(y_test, y_test_pred)
            
            report[list(models.keys())[i]] = test_model_score
            
        return report
        
    except Exception as e:
        raise Exception(f"Error evaluating models: {str(e)}")

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
        raise Exception(f"Error getting feature importance: {str(e)}")

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
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred, average='weighted'),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        
        return metrics
        
    except Exception as e:
        raise Exception(f"Error calculating metrics: {str(e)}")