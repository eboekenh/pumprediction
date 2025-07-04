"""
Training Report Generator for Model Training Results
"""

import pandas as pd
import numpy as np
import time
from datetime import datetime
import os
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    classification_report, confusion_matrix
)
from src.logger import logging

class TrainingReportGenerator:
    """Generate comprehensive training reports"""
    
    def __init__(self):
        self.report_data = {
            'training_start_time': None,
            'training_end_time': None,
            'models_tested': [],
            'best_model': None,
            'dataset_info': {},
            'performance_metrics': {}
        }
    
    def start_training_session(self, dataset_info):
        """Mark the start of training session"""
        self.report_data['training_start_time'] = datetime.now()
        self.report_data['dataset_info'] = dataset_info
        logging.info("Training session started")
    
    def add_model_result(self, model_name, model, X_train, y_train, X_test, y_test, 
                        training_time, hyperparams=None):
        """Add model training results"""
        try:
            # Make predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Calculate metrics
            train_accuracy = accuracy_score(y_train, y_pred_train)
            test_accuracy = accuracy_score(y_test, y_pred_test)
            
            # Multi-class metrics with weighted average
            precision = precision_score(y_test, y_pred_test, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred_test, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred_test, average='weighted', zero_division=0)
            
            # Per-class metrics
            classification_rep = classification_report(y_test, y_pred_test, output_dict=True)
            
            model_result = {
                'model_name': model_name,
                'training_time_seconds': training_time,
                'hyperparameters': hyperparams or {},
                'metrics': {
                    'train_accuracy': train_accuracy,
                    'test_accuracy': test_accuracy,
                    'precision_weighted': precision,
                    'recall_weighted': recall,
                    'f1_weighted': f1
                },
                'per_class_metrics': classification_rep,
                'confusion_matrix': confusion_matrix(y_test, y_pred_test).tolist()
            }
            
            self.report_data['models_tested'].append(model_result)
            logging.info(f"Added results for {model_name}")
            
        except Exception as e:
            logging.error(f"Error adding model result for {model_name}: {str(e)}")
    
    def set_best_model(self, model_name, final_score):
        """Set the best performing model"""
        self.report_data['best_model'] = {
            'name': model_name,
            'score': final_score
        }
    
    def finish_training_session(self):
        """Mark the end of training session"""
        self.report_data['training_end_time'] = datetime.now()
        logging.info("Training session completed")
    
    def generate_report_text(self):
        """Generate comprehensive text report"""
        try:
            report_lines = []
            
            # Header
            report_lines.append("=" * 80)
            report_lines.append("WATER PUMP FUNCTIONALITY PREDICTION - TRAINING REPORT")
            report_lines.append("=" * 80)
            report_lines.append("")
            
            # Training session info
            start_time = self.report_data['training_start_time']
            end_time = self.report_data['training_end_time']
            
            if start_time and end_time:
                duration = end_time - start_time
                report_lines.append(f"Training Date: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
                report_lines.append(f"Training Duration: {duration.total_seconds():.1f} seconds ({duration.total_seconds()/60:.1f} minutes)")
            report_lines.append("")
            
            # Dataset information
            dataset_info = self.report_data['dataset_info']
            if dataset_info:
                report_lines.append("DATASET INFORMATION")
                report_lines.append("-" * 40)
                for key, value in dataset_info.items():
                    report_lines.append(f"{key}: {value}")
                report_lines.append("")
            
            # Model results
            report_lines.append("MODEL PERFORMANCE COMPARISON")
            report_lines.append("-" * 40)
            
            models = self.report_data['models_tested']
            if models:
                # Summary table
                report_lines.append("Summary Table:")
                report_lines.append(f"{'Model':<20} {'Accuracy':<10} {'F1-Score':<10} {'Precision':<10} {'Recall':<10} {'Time(s)':<10}")
                report_lines.append("-" * 70)
                
                for model in models:
                    name = model['model_name']
                    metrics = model['metrics']
                    time_taken = model['training_time_seconds']
                    
                    report_lines.append(
                        f"{name:<20} "
                        f"{metrics['test_accuracy']:<10.4f} "
                        f"{metrics['f1_weighted']:<10.4f} "
                        f"{metrics['precision_weighted']:<10.4f} "
                        f"{metrics['recall_weighted']:<10.4f} "
                        f"{time_taken:<10.2f}"
                    )
                
                report_lines.append("")
                
                # Detailed results for each model
                for model in models:
                    report_lines.append(f"DETAILED RESULTS - {model['model_name'].upper()}")
                    report_lines.append("-" * 50)
                    
                    metrics = model['metrics']
                    report_lines.append(f"Training Accuracy: {metrics['train_accuracy']:.4f}")
                    report_lines.append(f"Test Accuracy: {metrics['test_accuracy']:.4f}")
                    report_lines.append(f"Precision (weighted): {metrics['precision_weighted']:.4f}")
                    report_lines.append(f"Recall (weighted): {metrics['recall_weighted']:.4f}")
                    report_lines.append(f"F1-Score (weighted): {metrics['f1_weighted']:.4f}")
                    report_lines.append(f"Training Time: {model['training_time_seconds']:.2f} seconds")
                    
                    # Hyperparameters
                    if model['hyperparameters']:
                        report_lines.append("Hyperparameters:")
                        for param, value in model['hyperparameters'].items():
                            report_lines.append(f"  {param}: {value}")
                    
                    # Per-class metrics
                    per_class = model['per_class_metrics']
                    if per_class and isinstance(per_class, dict):
                        report_lines.append("Per-Class Performance:")
                        for class_name, class_metrics in per_class.items():
                            if isinstance(class_metrics, dict) and 'precision' in class_metrics:
                                report_lines.append(
                                    f"  {class_name}: "
                                    f"Precision={class_metrics['precision']:.4f}, "
                                    f"Recall={class_metrics['recall']:.4f}, "
                                    f"F1={class_metrics['f1-score']:.4f}, "
                                    f"Support={class_metrics['support']}"
                                )
                    
                    # Confusion Matrix
                    if model['confusion_matrix']:
                        report_lines.append("Confusion Matrix:")
                        cm = model['confusion_matrix']
                        for row in cm:
                            report_lines.append(f"  {row}")
                    
                    report_lines.append("")
            
            # Best model summary
            best_model = self.report_data['best_model']
            if best_model:
                report_lines.append("BEST MODEL SELECTION")
                report_lines.append("-" * 40)
                report_lines.append(f"Best Model: {best_model['name']}")
                report_lines.append(f"Best Score: {best_model['score']:.4f}")
                report_lines.append("")
            
            # Recommendations
            report_lines.append("RECOMMENDATIONS")
            report_lines.append("-" * 40)
            
            if models:
                best_accuracy = max(model['metrics']['test_accuracy'] for model in models)
                if best_accuracy >= 0.8:
                    report_lines.append("• Excellent model performance achieved (>80% accuracy)")
                elif best_accuracy >= 0.7:
                    report_lines.append("• Good model performance achieved (70-80% accuracy)")
                else:
                    report_lines.append("• Model performance could be improved (<70% accuracy)")
                    report_lines.append("• Consider feature engineering or hyperparameter tuning")
                
                # Training time analysis
                avg_time = np.mean([model['training_time_seconds'] for model in models])
                if avg_time > 300:  # 5 minutes
                    report_lines.append("• Training time is high - consider data sampling for development")
                
                report_lines.append("• Model is ready for deployment")
                report_lines.append("• Regular retraining recommended with new data")
            
            report_lines.append("")
            report_lines.append("=" * 80)
            report_lines.append("End of Report")
            report_lines.append("=" * 80)
            
            return "\n".join(report_lines)
            
        except Exception as e:
            logging.error(f"Error generating report text: {str(e)}")
            return f"Error generating report: {str(e)}"
    
    def save_report(self, filename=None):
        """Save report to file"""
        try:
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"artifacts/training_report_{timestamp}.txt"
            
            # Ensure artifacts directory exists
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            report_text = self.generate_report_text()
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(report_text)
            
            logging.info(f"Training report saved to {filename}")
            return filename
            
        except Exception as e:
            logging.error(f"Error saving report: {str(e)}")
            return None
    
    def get_summary_stats(self):
        """Get summary statistics for quick display"""
        try:
            models = self.report_data['models_tested']
            if not models:
                return {}
            
            accuracies = [model['metrics']['test_accuracy'] for model in models]
            f1_scores = [model['metrics']['f1_weighted'] for model in models]
            training_times = [model['training_time_seconds'] for model in models]
            
            return {
                'models_tested': len(models),
                'best_accuracy': max(accuracies),
                'avg_accuracy': np.mean(accuracies),
                'best_f1': max(f1_scores),
                'avg_f1': np.mean(f1_scores),
                'total_training_time': sum(training_times),
                'avg_training_time': np.mean(training_times)
            }
            
        except Exception as e:
            logging.error(f"Error getting summary stats: {str(e)}")
            return {}