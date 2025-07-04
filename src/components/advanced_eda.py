import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from scipy.stats import chi2_contingency, spearmanr, pearsonr
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import IsolationForest
import warnings
import logging
from datetime import datetime
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.exception import CustomException

warnings.filterwarnings('ignore')

class AdvancedEDA:
    """Advanced Exploratory Data Analysis with comprehensive correlation analysis and preprocessing recommendations"""
    
    def __init__(self):
        self.numerical_cols = []
        self.categorical_cols = []
        self.target_col = 'status_group'
        self.correlation_results = {}
        self.outlier_results = {}
        self.preprocessing_recommendations = []
        
    def analyze_data_types(self, df):
        """Identify and categorize data types"""
        try:
            # Identify numerical and categorical columns
            self.numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            self.categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            
            # Remove target column from features if present
            if self.target_col in self.numerical_cols:
                self.numerical_cols.remove(self.target_col)
            if self.target_col in self.categorical_cols:
                self.categorical_cols.remove(self.target_col)
            
            # Remove ID column if present
            if 'id' in self.numerical_cols:
                self.numerical_cols.remove('id')
            if 'id' in self.categorical_cols:
                self.categorical_cols.remove('id')
                
            logging.info(f"Numerical columns: {len(self.numerical_cols)}")
            logging.info(f"Categorical columns: {len(self.categorical_cols)}")
            
            return {
                'numerical': self.numerical_cols,
                'categorical': self.categorical_cols,
                'target': self.target_col if self.target_col in df.columns else None
            }
            
        except Exception as e:
            logging.error(f"Error in data type analysis: {str(e)}")
            raise CustomException(e, sys)
    
    def calculate_cramers_v(self, x, y):
        """Calculate Cramér's V for categorical variables"""
        try:
            confusion_matrix = pd.crosstab(x, y)
            chi2 = chi2_contingency(confusion_matrix)[0]
            n = confusion_matrix.sum().sum()
            phi2 = chi2 / n
            r, k = confusion_matrix.shape
            phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
            rcorr = r - ((r-1)**2)/(n-1)
            kcorr = k - ((k-1)**2)/(n-1)
            return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))
        except:
            return 0.0
    
    def calculate_correlation_matrix(self, df):
        """Calculate comprehensive correlation matrix using appropriate methods"""
        try:
            results = {}
            
            # Pearson correlation for numerical-numerical
            if len(self.numerical_cols) > 1:
                pearson_corr = df[self.numerical_cols].corr(method='pearson')
                results['pearson'] = pearson_corr
                logging.info("Pearson correlation calculated")
            
            # Spearman correlation for numerical-numerical (rank-based)
            if len(self.numerical_cols) > 1:
                spearman_corr = df[self.numerical_cols].corr(method='spearman')
                results['spearman'] = spearman_corr
                logging.info("Spearman correlation calculated")
            
            # Cramér's V for categorical-categorical
            if len(self.categorical_cols) > 1:
                cramers_matrix = pd.DataFrame(index=self.categorical_cols, columns=self.categorical_cols)
                for col1 in self.categorical_cols:
                    for col2 in self.categorical_cols:
                        if col1 == col2:
                            cramers_matrix.loc[col1, col2] = 1.0
                        else:
                            cramers_matrix.loc[col1, col2] = self.calculate_cramers_v(df[col1], df[col2])
                
                cramers_matrix = cramers_matrix.astype(float)
                results['cramers_v'] = cramers_matrix
                logging.info("Cramér's V calculated")
            
            # Mutual information for feature-target relationships
            if self.target_col in df.columns:
                # Encode target variable
                le = LabelEncoder()
                y_encoded = le.fit_transform(df[self.target_col])
                
                # Calculate mutual information for numerical features
                if self.numerical_cols:
                    mi_numerical = mutual_info_classif(df[self.numerical_cols], y_encoded, random_state=42)
                    results['mutual_info_numerical'] = pd.Series(mi_numerical, index=self.numerical_cols)
                
                # Calculate mutual information for categorical features
                if self.categorical_cols:
                    # Encode categorical variables
                    cat_encoded = pd.DataFrame()
                    for col in self.categorical_cols:
                        le_cat = LabelEncoder()
                        cat_encoded[col] = le_cat.fit_transform(df[col].astype(str))
                    
                    mi_categorical = mutual_info_classif(cat_encoded, y_encoded, random_state=42)
                    results['mutual_info_categorical'] = pd.Series(mi_categorical, index=self.categorical_cols)
                
                logging.info("Mutual information calculated")
            
            self.correlation_results = results
            return results
            
        except Exception as e:
            logging.error(f"Error in correlation analysis: {str(e)}")
            raise CustomException(e, sys)
    
    def detect_outliers(self, df):
        """Detect outliers using multiple methods"""
        try:
            outlier_results = {}
            
            for col in self.numerical_cols:
                col_outliers = {}
                
                # IQR method
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                iqr_outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                col_outliers['iqr'] = {
                    'count': len(iqr_outliers),
                    'percentage': (len(iqr_outliers) / len(df)) * 100,
                    'bounds': (lower_bound, upper_bound)
                }
                
                # Z-score method
                z_scores = np.abs(stats.zscore(df[col].dropna()))
                z_outliers = df[z_scores > 3]
                col_outliers['zscore'] = {
                    'count': len(z_outliers),
                    'percentage': (len(z_outliers) / len(df)) * 100,
                    'threshold': 3
                }
                
                outlier_results[col] = col_outliers
            
            # Isolation Forest for multivariate outliers
            if len(self.numerical_cols) > 1:
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                outlier_labels = iso_forest.fit_predict(df[self.numerical_cols].fillna(df[self.numerical_cols].mean()))
                multivariate_outliers = df[outlier_labels == -1]
                
                outlier_results['multivariate'] = {
                    'count': len(multivariate_outliers),
                    'percentage': (len(multivariate_outliers) / len(df)) * 100,
                    'method': 'Isolation Forest'
                }
            
            self.outlier_results = outlier_results
            logging.info("Outlier detection completed")
            return outlier_results
            
        except Exception as e:
            logging.error(f"Error in outlier detection: {str(e)}")
            raise CustomException(e, sys)
    
    def generate_preprocessing_recommendations(self, df):
        """Generate comprehensive preprocessing recommendations"""
        try:
            recommendations = []
            
            # Missing value analysis
            missing_analysis = df.isnull().sum()
            high_missing = missing_analysis[missing_analysis > len(df) * 0.5].index.tolist()
            medium_missing = missing_analysis[(missing_analysis > len(df) * 0.1) & 
                                           (missing_analysis <= len(df) * 0.5)].index.tolist()
            low_missing = missing_analysis[(missing_analysis > 0) & 
                                         (missing_analysis <= len(df) * 0.1)].index.tolist()
            
            if high_missing:
                recommendations.append({
                    'category': 'Missing Values',
                    'priority': 'High',
                    'recommendation': f"Consider dropping columns with >50% missing values: {high_missing}",
                    'columns': high_missing
                })
            
            if medium_missing:
                recommendations.append({
                    'category': 'Missing Values',
                    'priority': 'Medium',
                    'recommendation': f"Impute missing values for columns with 10-50% missing: {medium_missing}",
                    'columns': medium_missing,
                    'suggested_method': 'KNN imputation or model-based imputation'
                })
            
            if low_missing:
                recommendations.append({
                    'category': 'Missing Values',
                    'priority': 'Low',
                    'recommendation': f"Simple imputation for columns with <10% missing: {low_missing}",
                    'columns': low_missing,
                    'suggested_method': 'Mean/median for numerical, mode for categorical'
                })
            
            # High cardinality categorical variables
            high_cardinality = []
            for col in self.categorical_cols:
                if df[col].nunique() > 20:
                    high_cardinality.append(col)
            
            if high_cardinality:
                recommendations.append({
                    'category': 'Categorical Encoding',
                    'priority': 'High',
                    'recommendation': f"Handle high cardinality categorical variables: {high_cardinality}",
                    'columns': high_cardinality,
                    'suggested_method': 'Target encoding, frequency encoding, or feature hashing'
                })
            
            # Skewed numerical variables
            skewed_vars = []
            for col in self.numerical_cols:
                skewness = df[col].skew()
                if abs(skewness) > 1:
                    skewed_vars.append((col, skewness))
            
            if skewed_vars:
                recommendations.append({
                    'category': 'Feature Scaling',
                    'priority': 'Medium',
                    'recommendation': f"Transform skewed variables: {[col for col, _ in skewed_vars]}",
                    'columns': [col for col, _ in skewed_vars],
                    'suggested_method': 'Log transformation, Box-Cox, or power transformation'
                })
            
            # Highly correlated features
            if 'pearson' in self.correlation_results:
                corr_matrix = self.correlation_results['pearson']
                high_corr_pairs = []
                
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        if abs(corr_matrix.iloc[i, j]) > 0.8:
                            high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
                
                if high_corr_pairs:
                    recommendations.append({
                        'category': 'Feature Selection',
                        'priority': 'Medium',
                        'recommendation': f"Remove highly correlated features (|r| > 0.8): {high_corr_pairs}",
                        'pairs': high_corr_pairs,
                        'suggested_method': 'Remove one feature from each correlated pair'
                    })
            
            # Constant or near-constant features
            constant_features = []
            for col in df.columns:
                if df[col].nunique() == 1:
                    constant_features.append(col)
                elif df[col].nunique() / len(df) < 0.01:  # Less than 1% unique values
                    constant_features.append(col)
            
            if constant_features:
                recommendations.append({
                    'category': 'Feature Selection',
                    'priority': 'High',
                    'recommendation': f"Remove constant/near-constant features: {constant_features}",
                    'columns': constant_features,
                    'suggested_method': 'Drop these features as they provide no information'
                })
            
            # Feature importance recommendations
            if 'mutual_info_numerical' in self.correlation_results:
                low_importance_numerical = self.correlation_results['mutual_info_numerical'][
                    self.correlation_results['mutual_info_numerical'] < 0.01
                ].index.tolist()
                
                if low_importance_numerical:
                    recommendations.append({
                        'category': 'Feature Selection',
                        'priority': 'Low',
                        'recommendation': f"Consider removing low-importance numerical features: {low_importance_numerical}",
                        'columns': low_importance_numerical,
                        'suggested_method': 'Feature selection based on mutual information'
                    })
            
            self.preprocessing_recommendations = recommendations
            logging.info("Preprocessing recommendations generated")
            return recommendations
            
        except Exception as e:
            logging.error(f"Error generating preprocessing recommendations: {str(e)}")
            raise CustomException(e, sys)
    
    def generate_feature_engineering_suggestions(self, df):
        """Generate feature engineering suggestions"""
        try:
            suggestions = []
            
            # Date/time features
            date_cols = []
            for col in df.columns:
                if 'date' in col.lower() or 'time' in col.lower():
                    date_cols.append(col)
            
            if date_cols:
                suggestions.append({
                    'category': 'Date/Time Features',
                    'suggestion': f"Extract date components from: {date_cols}",
                    'columns': date_cols,
                    'new_features': ['year', 'month', 'day', 'weekday', 'is_weekend', 'days_since_epoch']
                })
            
            # Geographical features
            geo_cols = []
            for col in df.columns:
                if any(geo_term in col.lower() for geo_term in ['lat', 'lon', 'gps', 'coord']):
                    geo_cols.append(col)
            
            if len(geo_cols) >= 2:
                suggestions.append({
                    'category': 'Geographical Features',
                    'suggestion': f"Create location-based features from: {geo_cols}",
                    'columns': geo_cols,
                    'new_features': ['distance_to_center', 'region_cluster', 'population_density']
                })
            
            # Interaction features
            if len(self.numerical_cols) > 1:
                suggestions.append({
                    'category': 'Interaction Features',
                    'suggestion': f"Create interaction features between highly correlated numerical variables",
                    'columns': self.numerical_cols[:5],  # Top 5 numerical columns
                    'new_features': ['ratios', 'products', 'differences']
                })
            
            # Categorical combinations
            if len(self.categorical_cols) > 1:
                suggestions.append({
                    'category': 'Categorical Combinations',
                    'suggestion': f"Create combined categorical features: {self.categorical_cols[:3]}",
                    'columns': self.categorical_cols[:3],
                    'new_features': ['combined_categories', 'hierarchical_encoding']
                })
            
            # Aggregated features
            if 'id' in df.columns:
                suggestions.append({
                    'category': 'Aggregated Features',
                    'suggestion': "Create aggregated features based on grouping variables",
                    'columns': ['region', 'district', 'ward'] if all(col in df.columns for col in ['region', 'district', 'ward']) else [],
                    'new_features': ['regional_avg', 'district_count', 'ward_median']
                })
            
            logging.info("Feature engineering suggestions generated")
            return suggestions
            
        except Exception as e:
            logging.error(f"Error generating feature engineering suggestions: {str(e)}")
            raise CustomException(e, sys)
    
    def generate_comprehensive_report(self, df):
        """Generate a comprehensive EDA report"""
        try:
            report = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'dataset_info': {
                    'shape': df.shape,
                    'missing_values_total': df.isnull().sum().sum(),
                    'missing_percentage': (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100,
                    'duplicate_rows': df.duplicated().sum()
                },
                'data_types': self.analyze_data_types(df),
                'correlation_analysis': self.calculate_correlation_matrix(df),
                'outlier_analysis': self.detect_outliers(df),
                'preprocessing_recommendations': self.generate_preprocessing_recommendations(df),
                'feature_engineering_suggestions': self.generate_feature_engineering_suggestions(df)
            }
            
            # Statistical summary
            report['statistical_summary'] = {
                'numerical_summary': df[self.numerical_cols].describe().to_dict() if self.numerical_cols else {},
                'categorical_summary': {col: df[col].value_counts().head().to_dict() for col in self.categorical_cols}
            }
            
            # Target variable analysis
            if self.target_col in df.columns:
                report['target_analysis'] = {
                    'distribution': df[self.target_col].value_counts().to_dict(),
                    'balance_ratio': df[self.target_col].value_counts(normalize=True).to_dict()
                }
            
            logging.info("Comprehensive EDA report generated")
            return report
            
        except Exception as e:
            logging.error(f"Error generating comprehensive report: {str(e)}")
            raise CustomException(e, sys)
    
    def save_report_to_markdown(self, report, filename="eda_report.md"):
        """Save the EDA report to a markdown file"""
        try:
            with open(filename, 'w') as f:
                f.write("# Comprehensive EDA Report\n\n")
                f.write(f"**Generated on:** {report['timestamp']}\n\n")
                
                # Dataset Overview
                f.write("## Dataset Overview\n")
                f.write(f"- **Shape:** {report['dataset_info']['shape']}\n")
                f.write(f"- **Missing values:** {report['dataset_info']['missing_values_total']} ({report['dataset_info']['missing_percentage']:.2f}%)\n")
                f.write(f"- **Duplicate rows:** {report['dataset_info']['duplicate_rows']}\n\n")
                
                # Data Types
                f.write("## Data Types Analysis\n")
                f.write(f"- **Numerical columns:** {len(report['data_types']['numerical'])}\n")
                f.write(f"- **Categorical columns:** {len(report['data_types']['categorical'])}\n")
                f.write(f"- **Target variable:** {report['data_types']['target']}\n\n")
                
                # Preprocessing Recommendations
                f.write("## Preprocessing Recommendations\n")
                for rec in report['preprocessing_recommendations']:
                    f.write(f"### {rec['category']} - {rec['priority']} Priority\n")
                    f.write(f"**Recommendation:** {rec['recommendation']}\n")
                    if 'suggested_method' in rec:
                        f.write(f"**Suggested method:** {rec['suggested_method']}\n")
                    f.write("\n")
                
                # Feature Engineering Suggestions
                f.write("## Feature Engineering Suggestions\n")
                for sug in report['feature_engineering_suggestions']:
                    f.write(f"### {sug['category']}\n")
                    f.write(f"**Suggestion:** {sug['suggestion']}\n")
                    if 'new_features' in sug:
                        f.write(f"**New features:** {', '.join(sug['new_features'])}\n")
                    f.write("\n")
                
                # Target Analysis
                if 'target_analysis' in report:
                    f.write("## Target Variable Analysis\n")
                    f.write("### Distribution\n")
                    for status, count in report['target_analysis']['distribution'].items():
                        f.write(f"- **{status}:** {count}\n")
                    f.write("\n")
                    
                    f.write("### Balance Ratio\n")
                    for status, ratio in report['target_analysis']['balance_ratio'].items():
                        f.write(f"- **{status}:** {ratio:.3f}\n")
                    f.write("\n")
                
                f.write("## Summary\n")
                f.write("This report provides comprehensive insights into the dataset structure, quality issues, ")
                f.write("and recommendations for preprocessing and feature engineering. ")
                f.write("Follow the recommendations based on their priority levels to improve model performance.\n")
            
            logging.info(f"EDA report saved to {filename}")
            return filename
            
        except Exception as e:
            logging.error(f"Error saving report to markdown: {str(e)}")
            raise CustomException(e, sys)