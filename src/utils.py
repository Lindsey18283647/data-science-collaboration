"""
Utility functions for the data science collaboration project.

This module contains helper functions that are used across different
parts of the project.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional, Tuple
import os
import json
import yaml
import logging
from datetime import datetime
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


def setup_plotting_style():
    """
    Set up consistent plotting style for all visualizations.
    """
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10


def check_data_quality(df: pd.DataFrame, output_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Perform comprehensive data quality checks.
    
    Args:
        df (pd.DataFrame): Input dataframe
        output_path (Optional[str]): Path to save the report
        
    Returns:
        Dict: Data quality report
    """
    logger.info("Performing data quality checks...")
    
    report = {
        'basic_info': {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum()
        },
        'missing_values': {
            'total_missing': df.isnull().sum().to_dict(),
            'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict()
        },
        'duplicates': {
            'duplicate_rows': df.duplicated().sum(),
            'duplicate_percentage': df.duplicated().sum() / len(df) * 100
        },
        'numeric_stats': {},
        'categorical_stats': {},
        'potential_issues': []
    }
    
    # Numeric columns analysis
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        stats = {
            'mean': df[col].mean(),
            'median': df[col].median(),
            'std': df[col].std(),
            'min': df[col].min(),
            'max': df[col].max(),
            'zeros': (df[col] == 0).sum(),
            'negative_values': (df[col] < 0).sum()
        }
        report['numeric_stats'][col] = stats
        
        # Check for potential issues
        if stats['std'] == 0:
            report['potential_issues'].append(f"Column '{col}' has zero variance")
        if stats['zeros'] / len(df) > 0.5:
            report['potential_issues'].append(f"Column '{col}' has >50% zeros")
    
    # Categorical columns analysis
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        unique_values = df[col].nunique()
        value_counts = df[col].value_counts()
        
        stats = {
            'unique_values': unique_values,
            'most_frequent': value_counts.index[0] if len(value_counts) > 0 else None,
            'most_frequent_count': value_counts.iloc[0] if len(value_counts) > 0 else 0,
            'least_frequent': value_counts.index[-1] if len(value_counts) > 0 else None,
            'least_frequent_count': value_counts.iloc[-1] if len(value_counts) > 0 else 0
        }
        report['categorical_stats'][col] = stats
        
        # Check for potential issues
        if unique_values == len(df):
            report['potential_issues'].append(f"Column '{col}' has all unique values (potential ID column)")
        if unique_values == 1:
            report['potential_issues'].append(f"Column '{col}' has only one unique value")
    
    # Save report if path provided
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Data quality report saved to {output_path}")
    
    logger.info("Data quality check completed")
    return report


def create_data_profile(df: pd.DataFrame, title: str = "Data Profile") -> None:
    """
    Create a comprehensive data profile with visualizations.
    
    Args:
        df (pd.DataFrame): Input dataframe
        title (str): Title for the profile
    """
    setup_plotting_style()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Missing values heatmap
    missing_data = df.isnull()
    axes[0, 0].imshow(missing_data, cmap='viridis', aspect='auto')
    axes[0, 0].set_title('Missing Values Pattern')
    axes[0, 0].set_xlabel('Columns')
    axes[0, 0].set_ylabel('Rows')
    
    # Missing values bar plot
    missing_counts = df.isnull().sum()
    missing_counts = missing_counts[missing_counts > 0]
    if len(missing_counts) > 0:
        missing_counts.plot(kind='bar', ax=axes[0, 1])
        axes[0, 1].set_title('Missing Values Count')
        axes[0, 1].tick_params(axis='x', rotation=45)
    else:
        axes[0, 1].text(0.5, 0.5, 'No Missing Values', ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('Missing Values Count')
    
    # Numeric columns distribution
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        df[numeric_cols].hist(bins=20, ax=axes[1, 0], alpha=0.7)
        axes[1, 0].set_title('Numeric Columns Distribution')
    else:
        axes[1, 0].text(0.5, 0.5, 'No Numeric Columns', ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Numeric Columns Distribution')
    
    # Categorical columns value counts
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        cat_col = categorical_cols[0]  # Show first categorical column
        df[cat_col].value_counts().head(10).plot(kind='bar', ax=axes[1, 1])
        axes[1, 1].set_title(f'Value Counts: {cat_col}')
        axes[1, 1].tick_params(axis='x', rotation=45)
    else:
        axes[1, 1].text(0.5, 0.5, 'No Categorical Columns', ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Categorical Columns')
    
    plt.tight_layout()
    plt.show()


def detect_outliers(df: pd.DataFrame, columns: List[str], method: str = 'iqr') -> Dict[str, np.ndarray]:
    """
    Detect outliers in specified columns using various methods.
    
    Args:
        df (pd.DataFrame): Input dataframe
        columns (List[str]): Columns to check for outliers
        method (str): Method for outlier detection ('iqr', 'zscore', 'isolation')
        
    Returns:
        Dict: Dictionary with column names as keys and outlier indices as values
    """
    outliers = {}
    
    for column in columns:
        if column not in df.columns:
            logger.warning(f"Column {column} not found in dataframe")
            continue
            
        if method == 'iqr':
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outlier_mask = (df[column] < lower_bound) | (df[column] > upper_bound)
            
        elif method == 'zscore':
            z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
            outlier_mask = z_scores > 3
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        outlier_indices = df[outlier_mask].index.to_numpy()
        outliers[column] = outlier_indices
        
        logger.info(f"Found {len(outlier_indices)} outliers in column {column}")
    
    return outliers


def correlation_analysis(df: pd.DataFrame, threshold: float = 0.8, plot: bool = True) -> pd.DataFrame:
    """
    Perform correlation analysis and identify highly correlated features.
    
    Args:
        df (pd.DataFrame): Input dataframe
        threshold (float): Correlation threshold for identifying high correlations
        plot (bool): Whether to plot correlation heatmap
        
    Returns:
        pd.DataFrame: High correlation pairs
    """
    numeric_df = df.select_dtypes(include=[np.number])
    
    if len(numeric_df.columns) < 2:
        logger.warning("Not enough numeric columns for correlation analysis")
        return pd.DataFrame()
    
    corr_matrix = numeric_df.corr()
    
    if plot:
        setup_plotting_style()
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.2f', cbar_kws={'label': 'Correlation'})
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.show()
    
    # Find high correlations
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_value = abs(corr_matrix.iloc[i, j])
            if corr_value >= threshold:
                high_corr_pairs.append({
                    'feature_1': corr_matrix.columns[i],
                    'feature_2': corr_matrix.columns[j],
                    'correlation': corr_matrix.iloc[i, j]
                })
    
    high_corr_df = pd.DataFrame(high_corr_pairs)
    
    if len(high_corr_df) > 0:
        logger.info(f"Found {len(high_corr_df)} highly correlated feature pairs (threshold: {threshold})")
    else:
        logger.info(f"No highly correlated feature pairs found (threshold: {threshold})")
    
    return high_corr_df


def save_config(config: Dict[str, Any], file_path: str) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config (Dict): Configuration dictionary
        file_path (str): Path to save the config file
    """
    try:
        with open(file_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        logger.info(f"Configuration saved to {file_path}")
    except Exception as e:
        logger.error(f"Error saving configuration: {str(e)}")
        raise


def load_config(file_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        file_path (str): Path to the config file
        
    Returns:
        Dict: Configuration dictionary
    """
    try:
        with open(file_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {file_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        raise


def create_directory_structure(base_path: str, structure: Dict[str, Any]) -> None:
    """
    Create directory structure based on nested dictionary.
    
    Args:
        base_path (str): Base directory path
        structure (Dict): Nested dictionary representing directory structure
    """
    for name, content in structure.items():
        path = os.path.join(base_path, name)
        
        if isinstance(content, dict):
            os.makedirs(path, exist_ok=True)
            create_directory_structure(path, content)
        else:
            # Create file with content
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w') as f:
                f.write(content if isinstance(content, str) else str(content))
    
    logger.info(f"Directory structure created at {base_path}")


def log_experiment(experiment_name: str, params: Dict[str, Any], 
                  metrics: Dict[str, float], model_path: str = None) -> None:
    """
    Log experiment parameters and results.
    
    Args:
        experiment_name (str): Name of the experiment
        params (Dict): Model parameters
        metrics (Dict): Performance metrics
        model_path (str): Path to saved model (optional)
    """
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'experiment_name': experiment_name,
        'parameters': params,
        'metrics': metrics,
        'model_path': model_path
    }
    
    log_file = 'experiment_log.json'
    
    # Load existing logs
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            logs = json.load(f)
    else:
        logs = []
    
    # Add new log entry
    logs.append(log_entry)
    
    # Save updated logs
    with open(log_file, 'w') as f:
        json.dump(logs, f, indent=2, default=str)
    
    logger.info(f"Experiment '{experiment_name}' logged successfully")


def generate_model_report(model, X_test: pd.DataFrame, y_test: pd.Series, 
                         model_name: str = "Model") -> str:
    """
    Generate a comprehensive model performance report.
    
    Args:
        model: Trained model
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test targets
        model_name (str): Name of the model
        
    Returns:
        str: Formatted report string
    """
    predictions = model.predict(X_test)
    
    report = f"""
    ========================================
    {model_name} Performance Report
    ========================================
    Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    
    Dataset Info:
    - Test samples: {len(X_test)}
    - Features: {X_test.shape[1]}
    
    Model Parameters:
    {model.get_params() if hasattr(model, 'get_params') else 'Parameters not available'}
    
    """
    
    # Add specific metrics based on problem type
    if hasattr(model, 'predict_proba'):  # Classification
        from sklearn.metrics import accuracy_score, classification_report
        accuracy = accuracy_score(y_test, predictions)
        class_report = classification_report(y_test, predictions)
        
        report += f"""
    Classification Metrics:
    - Accuracy: {accuracy:.4f}
    
    Detailed Classification Report:
    {class_report}
        """
    else:  # Regression
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        report += f"""
    Regression Metrics:
    - Mean Squared Error: {mse:.4f}
    - Root Mean Squared Error: {rmse:.4f}
    - Mean Absolute Error: {mae:.4f}
    - R² Score: {r2:.4f}
        """
    
    report += "\n    ========================================"
    
    return report


# Example usage
if __name__ == "__main__":
    # Example of how to use utility functions
    print("Data Science Utilities Module")
    print("This module provides various utility functions for data science projects.")
    print("\nAvailable functions:")
    print("- check_data_quality: Comprehensive data quality analysis")
    print("- create_data_profile: Visual data profiling")
    print("- detect_outliers: Outlier detection using various methods")
    print("- correlation_analysis: Feature correlation analysis")
    print("- save_config/load_config: Configuration management")
    print("- log_experiment: Experiment tracking")
    print("- generate_model_report: Model performance reporting")

def calculate_statistics(data):
    """Calculate basic statistics for dataset."""
    return {
        'mean': data.mean(),
        'std': data.std(),
        'count': len(data)
    }
