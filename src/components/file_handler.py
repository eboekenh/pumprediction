import pandas as pd
import os
import logging
import sys
from typing import Dict, Tuple, Optional
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.exception import CustomException

class FileHandler:
    """Handle multiple file upload and identification for Tanzania water pump dataset"""
    
    def __init__(self):
        self.identified_files = {}
        self.merged_train_data = None
        self.test_data = None
        
    def identify_files(self, file_paths: list) -> Dict[str, str]:
        """
        Identify which file contains what data based on structure
        
        Args:
            file_paths: List of file paths to analyze
            
        Returns:
            Dictionary mapping file types to paths
        """
        try:
            file_info = {}
            
            for path in file_paths:
                if os.path.exists(path):
                    # Read first few rows to check structure
                    df_sample = pd.read_csv(path, nrows=5)
                    
                    # Check if it's labels file (has target column, few columns)
                    if 'status_group' in df_sample.columns and len(df_sample.columns) <= 3:
                        file_info['train_labels'] = path
                        logging.info(f"Identified training labels file: {os.path.basename(path)}")
                    
                    # Check if it's features file (many columns, no target)
                    elif 'status_group' not in df_sample.columns and len(df_sample.columns) > 10:
                        # Check if it's larger (training) or smaller (test) file
                        df_full = pd.read_csv(path)
                        if len(df_full) > 20000:  # Threshold for training vs test
                            file_info['train_features'] = path
                            logging.info(f"Identified training features file: {os.path.basename(path)}")
                        else:
                            file_info['test_features'] = path
                            logging.info(f"Identified test features file: {os.path.basename(path)}")
                    
                    # Alternative check based on file size
                    elif len(df_sample.columns) > 10:
                        df_full = pd.read_csv(path)
                        if len(df_full) > 20000:
                            file_info['train_features'] = path
                            logging.info(f"Identified training features file: {os.path.basename(path)}")
                        else:
                            file_info['test_features'] = path
                            logging.info(f"Identified test features file: {os.path.basename(path)}")
            
            self.identified_files = file_info
            return file_info
            
        except Exception as e:
            logging.error(f"Error identifying files: {str(e)}")
            raise CustomException(e, sys)
    
    def merge_training_data(self, train_features_path: str, train_labels_path: str) -> pd.DataFrame:
        """
        Merge training features and labels on common ID
        
        Args:
            train_features_path: Path to training features
            train_labels_path: Path to training labels
            
        Returns:
            Merged DataFrame
        """
        try:
            # Load both files
            train_features = pd.read_csv(train_features_path)
            train_labels = pd.read_csv(train_labels_path)
            
            logging.info(f"Training features shape: {train_features.shape}")
            logging.info(f"Training labels shape: {train_labels.shape}")
            
            # Merge on ID column
            merged_df = pd.merge(train_features, train_labels, on='id', how='inner')
            
            logging.info(f"Merged training data shape: {merged_df.shape}")
            logging.info(f"Columns in merged data: {merged_df.columns.tolist()}")
            
            # Validate merge
            if len(merged_df) == 0:
                raise ValueError("No matching IDs found between training features and labels")
            
            self.merged_train_data = merged_df
            return merged_df
            
        except Exception as e:
            logging.error(f"Error merging training data: {str(e)}")
            raise CustomException(e, sys)
    
    def load_test_data(self, test_features_path: str) -> pd.DataFrame:
        """
        Load test features
        
        Args:
            test_features_path: Path to test features
            
        Returns:
            Test DataFrame
        """
        try:
            test_df = pd.read_csv(test_features_path)
            logging.info(f"Test data shape: {test_df.shape}")
            logging.info(f"Test data columns: {test_df.columns.tolist()}")
            
            self.test_data = test_df
            return test_df
            
        except Exception as e:
            logging.error(f"Error loading test data: {str(e)}")
            raise CustomException(e, sys)
    
    def validate_data_consistency(self) -> Dict[str, any]:
        """
        Validate consistency between training and test data
        
        Returns:
            Dictionary with validation results
        """
        try:
            validation_results = {}
            
            if self.merged_train_data is not None and self.test_data is not None:
                # Check column consistency (excluding target)
                train_features = [col for col in self.merged_train_data.columns if col != 'status_group']
                test_features = self.test_data.columns.tolist()
                
                missing_in_test = set(train_features) - set(test_features)
                extra_in_test = set(test_features) - set(train_features)
                
                validation_results['column_consistency'] = {
                    'missing_in_test': list(missing_in_test),
                    'extra_in_test': list(extra_in_test),
                    'consistent': len(missing_in_test) == 0 and len(extra_in_test) == 0
                }
                
                # Check data types consistency
                common_cols = set(train_features) & set(test_features)
                dtype_mismatches = []
                
                for col in common_cols:
                    if str(self.merged_train_data[col].dtype) != str(self.test_data[col].dtype):
                        dtype_mismatches.append({
                            'column': col,
                            'train_dtype': str(self.merged_train_data[col].dtype),
                            'test_dtype': str(self.test_data[col].dtype)
                        })
                
                validation_results['dtype_consistency'] = {
                    'mismatches': dtype_mismatches,
                    'consistent': len(dtype_mismatches) == 0
                }
                
                # Check value ranges for numerical columns
                numerical_cols = self.merged_train_data.select_dtypes(include=['number']).columns.tolist()
                if 'status_group' in numerical_cols:
                    numerical_cols.remove('status_group')
                
                range_issues = []
                for col in numerical_cols:
                    if col in self.test_data.columns:
                        train_min, train_max = self.merged_train_data[col].min(), self.merged_train_data[col].max()
                        test_min, test_max = self.test_data[col].min(), self.test_data[col].max()
                        
                        if test_min < train_min or test_max > train_max:
                            range_issues.append({
                                'column': col,
                                'train_range': (train_min, train_max),
                                'test_range': (test_min, test_max)
                            })
                
                validation_results['range_consistency'] = {
                    'issues': range_issues,
                    'consistent': len(range_issues) == 0
                }
            
            logging.info("Data validation completed")
            return validation_results
            
        except Exception as e:
            logging.error(f"Error validating data consistency: {str(e)}")
            raise CustomException(e, sys)
    
    def get_data_summary(self) -> Dict[str, any]:
        """
        Get comprehensive summary of loaded data
        
        Returns:
            Dictionary with data summary
        """
        try:
            summary = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'files_identified': self.identified_files,
                'data_loaded': {
                    'train_data': self.merged_train_data is not None,
                    'test_data': self.test_data is not None
                }
            }
            
            if self.merged_train_data is not None:
                summary['train_data_info'] = {
                    'shape': self.merged_train_data.shape,
                    'columns': self.merged_train_data.columns.tolist(),
                    'target_distribution': self.merged_train_data['status_group'].value_counts().to_dict() if 'status_group' in self.merged_train_data.columns else {},
                    'missing_values': self.merged_train_data.isnull().sum().to_dict(),
                    'numerical_columns': self.merged_train_data.select_dtypes(include=['number']).columns.tolist(),
                    'categorical_columns': self.merged_train_data.select_dtypes(include=['object']).columns.tolist()
                }
            
            if self.test_data is not None:
                summary['test_data_info'] = {
                    'shape': self.test_data.shape,
                    'columns': self.test_data.columns.tolist(),
                    'missing_values': self.test_data.isnull().sum().to_dict(),
                    'numerical_columns': self.test_data.select_dtypes(include=['number']).columns.tolist(),
                    'categorical_columns': self.test_data.select_dtypes(include=['object']).columns.tolist()
                }
            
            return summary
            
        except Exception as e:
            logging.error(f"Error generating data summary: {str(e)}")
            raise CustomException(e, sys)
    
    def save_processed_data(self, output_dir: str = "artifacts") -> Dict[str, str]:
        """
        Save processed data to files
        
        Args:
            output_dir: Directory to save files
            
        Returns:
            Dictionary with saved file paths
        """
        try:
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            saved_files = {}
            
            # Save merged training data
            if self.merged_train_data is not None:
                train_path = os.path.join(output_dir, "merged_train_data.csv")
                self.merged_train_data.to_csv(train_path, index=False)
                saved_files['merged_train'] = train_path
                logging.info(f"Merged training data saved to: {train_path}")
            
            # Save test data
            if self.test_data is not None:
                test_path = os.path.join(output_dir, "test_data.csv")
                self.test_data.to_csv(test_path, index=False)
                saved_files['test'] = test_path
                logging.info(f"Test data saved to: {test_path}")
            
            # Save data summary
            summary = self.get_data_summary()
            summary_path = os.path.join(output_dir, "data_summary.txt")
            with open(summary_path, 'w') as f:
                f.write("DATA PROCESSING SUMMARY\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Processing completed at: {summary['timestamp']}\n\n")
                
                f.write("FILES IDENTIFIED:\n")
                for file_type, path in summary['files_identified'].items():
                    f.write(f"  - {file_type}: {os.path.basename(path)}\n")
                f.write("\n")
                
                if 'train_data_info' in summary:
                    f.write(f"TRAINING DATA:\n")
                    f.write(f"  - Shape: {summary['train_data_info']['shape']}\n")
                    f.write(f"  - Target distribution: {summary['train_data_info']['target_distribution']}\n")
                    f.write(f"  - Numerical columns: {len(summary['train_data_info']['numerical_columns'])}\n")
                    f.write(f"  - Categorical columns: {len(summary['train_data_info']['categorical_columns'])}\n\n")
                
                if 'test_data_info' in summary:
                    f.write(f"TEST DATA:\n")
                    f.write(f"  - Shape: {summary['test_data_info']['shape']}\n")
                    f.write(f"  - Numerical columns: {len(summary['test_data_info']['numerical_columns'])}\n")
                    f.write(f"  - Categorical columns: {len(summary['test_data_info']['categorical_columns'])}\n\n")
            
            saved_files['summary'] = summary_path
            logging.info(f"Data summary saved to: {summary_path}")
            
            return saved_files
            
        except Exception as e:
            logging.error(f"Error saving processed data: {str(e)}")
            raise CustomException(e, sys)
    
    def process_uploaded_files(self, file_paths: list) -> Dict[str, any]:
        """
        Complete processing workflow for uploaded files
        
        Args:
            file_paths: List of uploaded file paths
            
        Returns:
            Dictionary with processing results
        """
        try:
            # Step 1: Identify files
            identified = self.identify_files(file_paths)
            
            # Step 2: Merge training data if both files are available
            if 'train_features' in identified and 'train_labels' in identified:
                merged_train = self.merge_training_data(identified['train_features'], identified['train_labels'])
            else:
                raise ValueError("Could not identify both training features and labels files")
            
            # Step 3: Load test data if available
            if 'test_features' in identified:
                test_data = self.load_test_data(identified['test_features'])
            else:
                logging.warning("No test features file identified")
            
            # Step 4: Validate data consistency
            validation_results = self.validate_data_consistency()
            
            # Step 5: Save processed data
            saved_files = self.save_processed_data()
            
            # Step 6: Generate summary
            summary = self.get_data_summary()
            
            return {
                'success': True,
                'identified_files': identified,
                'validation_results': validation_results,
                'saved_files': saved_files,
                'summary': summary
            }
            
        except Exception as e:
            logging.error(f"Error in processing workflow: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'identified_files': self.identified_files,
                'summary': self.get_data_summary() if self.merged_train_data is not None else None
            }