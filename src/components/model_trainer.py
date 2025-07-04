import os
import sys
import pandas as pd
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, f1_score
from xgboost import XGBClassifier
# from lightgbm import LGBMClassifier  # Disabled due to system library requirements
# from catboost import CatBoostClassifier  # Disabled due to system library requirements
from dataclasses import dataclass
import pickle
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models
from src.components.training_report import TrainingReportGenerator

@dataclass
class ModelTrainerConfig:
    """Configuration for model training"""
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")
    model_report_file_path: str = os.path.join("artifacts", "model_report.pkl")
    training_report_file_path: str = os.path.join("artifacts", "training_report.txt")

class ModelTrainer:
    """Model training component"""
    
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self, train_array, test_array, selected_models=None):
        """
        Initiate model training with hyperparameter tuning
        
        Args:
            train_array: Training data array
            test_array: Test data array
            selected_models: List of model names to train (if None, trains all models)
            
        Returns:
            Best model score
        """
        try:
            logging.info("Starting model training")
            
            # Initialize training report generator
            report_generator = TrainingReportGenerator()
            
            # Dataset information for report
            dataset_info = {
                'training_samples': len(train_array),
                'test_samples': len(test_array),
                'total_features': train_array.shape[1] - 1,  # Excluding target
                'target_classes': len(np.unique(train_array[:, -1]))
            }
            
            report_generator.start_training_session(dataset_info)
            
            # Split features and target
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            
            # Define models with optimized parameters for large dataset (faster training)
            models = {
                "Random Forest": RandomForestClassifier(
                    n_estimators=50,    # Reduced from 100 for faster training
                    max_depth=10,       # Reduced from 15 for faster training
                    min_samples_split=20,  # Increased for faster training
                    min_samples_leaf=10,   # Increased for faster training
                    random_state=42, 
                    n_jobs=-1
                ),
                "XGBoost": XGBClassifier(
                    n_estimators=50,    # Reduced from 100 for faster training
                    max_depth=4,        # Reduced from 6 for faster training
                    learning_rate=0.2,  # Increased for faster convergence
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42, 
                    eval_metric='mlogloss', 
                    n_jobs=-1
                ),

                "Support Vector Machine": SVC(
                    kernel='rbf',
                    C=1.0,
                    gamma='scale',
                    random_state=42,
                    probability=True  # Enable probability estimates
                ),
                "Extra Trees": ExtraTreesClassifier(
                    n_estimators=50,
                    max_depth=10,
                    min_samples_split=20,
                    min_samples_leaf=10,
                    random_state=42,
                    n_jobs=-1
                ),
                "AdaBoost": AdaBoostClassifier(
                    n_estimators=50,
                    learning_rate=1.0,
                    random_state=42
                ),
                "K-Nearest Neighbors": KNeighborsClassifier(
                    n_neighbors=5,
                    weights='distance',
                    n_jobs=-1
                ),
                "Neural Network": MLPClassifier(
                    hidden_layer_sizes=(100, 50),
                    max_iter=100,  # Reduced for faster training
                    random_state=42,
                    early_stopping=True,
                    validation_fraction=0.1
                )
            }
            
            # Optimized parameters for large dataset - no grid search, use fixed optimal values
            params = {
                "Random Forest": {},  # Use default parameters, no grid search
                "XGBoost": {},  # Use default parameters, no grid search
                "Support Vector Machine": {},  # Use default parameters, no grid search
                "Extra Trees": {},  # Use default parameters, no grid search
                "AdaBoost": {},  # Use default parameters, no grid search
                "K-Nearest Neighbors": {},  # Use default parameters, no grid search
                "Neural Network": {}  # Use default parameters, no grid search
            }
            
            # Filter models based on selection
            if selected_models:
                models = {name: model for name, model in models.items() if name in selected_models}
                params = {name: param for name, param in params.items() if name in selected_models}
            
            # Train and evaluate each model with timing
            model_report = {}
            
            for model_name, model in models.items():
                logging.info(f"Training {model_name} with default parameters")
                
                # Time the training
                start_time = time.time()
                model.fit(X_train, y_train)
                training_time = time.time() - start_time
                
                # Evaluate model
                test_score = model.score(X_test, y_test)
                logging.info(f"{model_name} - Default params used")
                logging.info(f"{model_name} - Test score: {test_score}")
                
                model_report[model_name] = test_score
                
                # Add to training report
                hyperparams = {
                    param: getattr(model, param) 
                    for param in ['n_estimators', 'max_depth', 'random_state'] 
                    if hasattr(model, param)
                }
                
                report_generator.add_model_result(
                    model_name=model_name,
                    model=model,
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    training_time=training_time,
                    hyperparams=hyperparams
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
                raise CustomException("No best model found with acceptable performance", sys)
            
            # Since we're using empty params (no grid search), just use the best model directly
            best_params = params[best_model_name]
            
            # If no parameters for grid search, use the model directly
            if not best_params:
                logging.info(f"Using {best_model_name} with default optimized parameters")
                best_model_final = best_model
                # Train the model
                best_model_final.fit(X_train, y_train)
                # Get the actual parameters used by the model
                best_params_used = best_model_final.get_params()
            else:
                # Perform grid search for best model
                grid_search = GridSearchCV(
                    estimator=best_model,
                    param_grid=best_params,
                    cv=3,  # Reduced CV for faster training
                    scoring='accuracy',
                    n_jobs=-1
                )
                
                grid_search.fit(X_train, y_train)
                best_model_final = grid_search.best_estimator_
                best_params_used = grid_search.best_params_
            
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
                'best_params': best_params_used,
                'best_score': best_model_score,
                'accuracy': accuracy,
                'f1_score': f1,
                'model_report': model_report,
                'classification_report': classification_report(y_test, y_pred)
            }
            
            save_object(
                file_path=self.model_trainer_config.model_report_file_path,
                obj=detailed_report
            )
            
            # Finalize training report
            report_generator.set_best_model(best_model_name, best_model_score)
            report_generator.finish_training_session()
            
            # Save training report
            training_report_path = report_generator.save_report()
            if training_report_path:
                logging.info(f"Training report saved to {training_report_path}")
            
            return accuracy
            
        except Exception as e:
            logging.error(f"Error in model training: {str(e)}")
            raise CustomException(str(e), sys)
    
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
            raise CustomException(str(e), sys)
