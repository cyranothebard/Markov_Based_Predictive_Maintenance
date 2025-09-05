"""
Helper functions for Markov Predictive Maintenance

This module provides utility functions for data processing, visualization,
and model evaluation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')


def create_engine_sequences(data: pd.DataFrame, engine_id: int) -> pd.DataFrame:
    """
    Extract sequence data for a specific engine.
    
    Args:
        data (pd.DataFrame): Full dataset
        engine_id (int): Engine unit number
        
    Returns:
        pd.DataFrame: Engine sequence data sorted by cycle
    """
    engine_data = data[data['unit'] == engine_id].copy()
    engine_data = engine_data.sort_values('cycle').reset_index(drop=True)
    return engine_data


def calculate_engine_rul(data: pd.DataFrame, engine_id: int) -> pd.DataFrame:
    """
    Calculate RUL for a specific engine.
    
    Args:
        data (pd.DataFrame): Full dataset
        engine_id (int): Engine unit number
        
    Returns:
        pd.DataFrame: Engine data with RUL column
    """
    engine_data = create_engine_sequences(data, engine_id)
    max_cycle = engine_data['cycle'].max()
    engine_data['RUL'] = max_cycle - engine_data['cycle']
    return engine_data


def normalize_sensor_data(data: pd.DataFrame, sensor_columns: List[str], 
                         method: str = 'standard') -> pd.DataFrame:
    """
    Normalize sensor data using specified method.
    
    Args:
        data (pd.DataFrame): Data containing sensor columns
        sensor_columns (List[str]): List of sensor column names
        method (str): Normalization method ('standard', 'minmax', 'robust')
        
    Returns:
        pd.DataFrame: Data with normalized sensor columns
    """
    data_normalized = data.copy()
    
    for sensor in sensor_columns:
        if sensor in data.columns:
            if method == 'standard':
                # Z-score normalization
                data_normalized[f'{sensor}_norm'] = (
                    data[sensor] - data[sensor].mean()
                ) / data[sensor].std()
            elif method == 'minmax':
                # Min-max normalization
                data_normalized[f'{sensor}_norm'] = (
                    data[sensor] - data[sensor].min()
                ) / (data[sensor].max() - data[sensor].min())
            elif method == 'robust':
                # Robust normalization using median and IQR
                median = data[sensor].median()
                iqr = data[sensor].quantile(0.75) - data[sensor].quantile(0.25)
                data_normalized[f'{sensor}_norm'] = (data[sensor] - median) / iqr
    
    return data_normalized


def plot_engine_degradation(data: pd.DataFrame, engine_id: int, 
                           sensors: List[str], figsize: Tuple[int, int] = (15, 10)):
    """
    Plot degradation patterns for a specific engine.
    
    Args:
        data (pd.DataFrame): Full dataset
        engine_id (int): Engine unit number
        sensors (List[str]): List of sensor names to plot
        figsize (Tuple[int, int]): Figure size
    """
    engine_data = create_engine_sequences(data, engine_id)
    
    n_sensors = len(sensors)
    n_cols = min(3, n_sensors)
    n_rows = (n_sensors + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_sensors == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, sensor in enumerate(sensors):
        row = i // n_cols
        col = i % n_cols
        
        if n_rows > 1:
            ax = axes[row, col]
        else:
            ax = axes[col]
        
        ax.plot(engine_data['cycle'], engine_data[sensor], linewidth=2)
        ax.set_xlabel('Cycle')
        ax.set_ylabel(f'{sensor} Value')
        ax.set_title(f'Engine {engine_id} - {sensor}')
        ax.grid(True, alpha=0.3)
    
    # Remove empty subplots
    for i in range(n_sensors, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        if n_rows > 1:
            fig.delaxes(axes[row, col])
        else:
            fig.delaxes(axes[col])
    
    plt.tight_layout()
    plt.show()


def create_sensor_correlation_plot(data: pd.DataFrame, sensor_columns: List[str], 
                                  figsize: Tuple[int, int] = (12, 10)):
    """
    Create correlation heatmap for sensors.
    
    Args:
        data (pd.DataFrame): Data containing sensor columns
        sensor_columns (List[str]): List of sensor column names
        figsize (Tuple[int, int]): Figure size
    """
    sensor_data = data[sensor_columns]
    correlation_matrix = sensor_data.corr()
    
    plt.figure(figsize=figsize)
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
                center=0, square=True, linewidths=0.5, cbar_kws={"shrink": .8}, 
                fmt='.2f')
    plt.title('Sensor Correlation Matrix', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


def calculate_rolling_statistics(data: pd.DataFrame, column: str, 
                                window: int, engine_id: int = None) -> pd.DataFrame:
    """
    Calculate rolling statistics for a specific column.
    
    Args:
        data (pd.DataFrame): Input data
        column (str): Column name to calculate statistics for
        window (int): Rolling window size
        engine_id (int): Specific engine ID (optional)
        
    Returns:
        pd.DataFrame: Data with rolling statistics
    """
    if engine_id is not None:
        data = data[data['unit'] == engine_id].copy()
    
    data = data.sort_values('cycle').reset_index(drop=True)
    
    rolling_stats = pd.DataFrame({
        'cycle': data['cycle'],
        f'{column}_rolling_mean': data[column].rolling(window=window, min_periods=1).mean(),
        f'{column}_rolling_std': data[column].rolling(window=window, min_periods=1).std(),
        f'{column}_rolling_min': data[column].rolling(window=window, min_periods=1).min(),
        f'{column}_rolling_max': data[column].rolling(window=window, min_periods=1).max()
    })
    
    return rolling_stats


def save_results_to_file(results: Dict[str, Any], filename: str, 
                        results_dir: str = 'results'):
    """
    Save results dictionary to file.
    
    Args:
        results (Dict[str, Any]): Results dictionary
        filename (str): Output filename
        results_dir (str): Results directory
    """
    import json
    import os
    
    os.makedirs(results_dir, exist_ok=True)
    filepath = os.path.join(results_dir, filename)
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        return obj
    
    # Recursively convert numpy objects
    def recursive_convert(d):
        if isinstance(d, dict):
            return {k: recursive_convert(v) for k, v in d.items()}
        elif isinstance(d, list):
            return [recursive_convert(item) for item in d]
        else:
            return convert_numpy(d)
    
    converted_results = recursive_convert(results)
    
    with open(filepath, 'w') as f:
        json.dump(converted_results, f, indent=2)
    
    print(f"Results saved to {filepath}")


def load_results_from_file(filename: str, results_dir: str = 'results') -> Dict[str, Any]:
    """
    Load results dictionary from file.
    
    Args:
        filename (str): Input filename
        results_dir (str): Results directory
        
    Returns:
        Dict[str, Any]: Loaded results dictionary
    """
    import json
    import os
    
    filepath = os.path.join(results_dir, filename)
    
    with open(filepath, 'r') as f:
        results = json.load(f)
    
    return results


def print_model_summary(model_name: str, metrics: Dict[str, float]):
    """
    Print formatted model summary.
    
    Args:
        model_name (str): Name of the model
        metrics (Dict[str, float]): Model performance metrics
    """
    print(f"\n{'='*50}")
    print(f"MODEL: {model_name.upper()}")
    print(f"{'='*50}")
    
    for metric, value in metrics.items():
        if isinstance(value, float):
            print(f"{metric.upper():<25}: {value:.4f}")
        else:
            print(f"{metric.upper():<25}: {value}")
    
    print(f"{'='*50}")


def create_performance_comparison_plot(model_results: Dict[str, Dict[str, float]], 
                                     metric: str = 'rmse', figsize: Tuple[int, int] = (10, 6)):
    """
    Create performance comparison plot for multiple models.
    
    Args:
        model_results (Dict[str, Dict[str, float]]): Results for multiple models
        metric (str): Metric to compare
        figsize (Tuple[int, int]): Figure size
    """
    model_names = list(model_results.keys())
    metric_values = [model_results[model][metric] for model in model_names]
    
    plt.figure(figsize=figsize)
    bars = plt.bar(model_names, metric_values, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
    
    plt.xlabel('Model')
    plt.ylabel(metric.upper())
    plt.title(f'Model Performance Comparison - {metric.upper()}')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, metric_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()


