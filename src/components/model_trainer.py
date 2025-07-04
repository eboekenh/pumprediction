import os
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, f1_score
from xgboost import XGBClassifier
from dataclasses import dataclass
import pickle
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    """Configuration for model training"""
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")
    model_report_file_path: str = os.path.join("artifacts", "model_report.pkl")

class ModelTrainer:
    """Model training component"""
    
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self, train_array, test_array):
        """
        Initiate model training with hyperparameter tuning
        
        Args:
            train_array: Training data array
            test_array: Test data array
            
        Returns:
            Best model score
        """
        try:
            logging.info("Starting model training")
            
            # Split features and target
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            
            # Define models and their parameters (more conservative to prevent overfitting)
            models = {
                "Random Forest": RandomForestClassifier(random_state=42, n_jobs=-1),
                "XGBoost": XGBClassifier(random_state=42, eval_metric='mlogloss', n_jobs=-1)
            }
            
            params = {
                "Random Forest": {
                    'n_estimators': [50, 100, 200],  # Reduced to prevent overfitting
                    'max_depth': [3, 5, 7, 10],  # More conservative max depth
                    'min_samples_split': [5, 10, 20],  # Higher minimum to prevent overfitting
                    'min_samples_leaf': [2, 4, 8]  # Higher minimum to prevent overfitting
                },
                "XGBoost": {
                    'n_estimators': [50, 100, 200],  # Reduced to prevent overfitting
                    'max_depth': [3, 4, 5],  # More conservative max depth
                    'learning_rate': [0.01, 0.05, 0.1],  # Lower learning rates
                    'subsample': [0.8, 0.9],  # Add regularization
                    'colsample_bytree': [0.8, 0.9]  # Add feature sampling
                }
            }
            
            # Evaluate models
            model_report = evaluate_models(
                X_train=X_train, 
                y_train=y_train,
                X_test=X_test, 
                y_test=y_test,
                models=models,
                param=params
            )
            
            # Get best model
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]
            
            logging.info(f"Best model: {best_model_name}")
            logging.info(f"Best model score: {best_model_score}")
            
            if best_model_score < 0.6:
                raise CustomException("No best model found with acceptable performance")
            
            # Train best model with best parameters
            best_params = params[best_model_name]
            
            # Perform grid search for best model
            grid_search = GridSearchCV(
                estimator=best_model,
                param_grid=best_params,
                cv=5,
                scoring='accuracy',
                n_jobs=-1
            )
            
            grid_search.fit(X_train, y_train)
            best_model_final = grid_search.best_estimator_
            
            # Make predictions
            y_pred = best_model_final.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            logging.info(f"Final model accuracy: {accuracy}")
            logging.info(f"Final model F1 score: {f1}")
            
            # Save model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model_final
            )
            
            # Save model report
            detailed_report = {
                'model_name': best_model_name,
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'accuracy': accuracy,
                'f1_score': f1,
                'model_report': model_report,
                'classification_report': classification_report(y_test, y_pred)
            }
            
            save_object(
                file_path=self.model_trainer_config.model_report_file_path,
                obj=detailed_report
            )
            
            return accuracy
            
        except Exception as e:
            logging.error(f"Error in model training: {str(e)}")
            raise CustomException(e, sys)
    
    def get_model_comparison(self, X_train, y_train, X_test, y_test):
        """
        Compare multiple models without extensive hyperparameter tuning
        
        Args:
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target
            
        Returns:
            Dictionary of model performances
        """
        try:
            models = {
                "Random Forest": RandomForestClassifier(random_state=42),
                "XGBoost": XGBClassifier(random_state=42, eval_metric='mlogloss')
            }
            
            results = {}
            
            for name, model in models.items():
                # Train model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='weighted')
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
                
                results[name] = {
                    'accuracy': accuracy,
                    'f1_score': f1,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'classification_report': classification_report(y_test, y_pred)
                }
            
            return results
            
        except Exception as e:
            logging.error(f"Error in model comparison: {str(e)}")
            raise CustomException(e, sys)
