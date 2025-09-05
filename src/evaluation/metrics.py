"""
Evaluation Metrics for RUL Prediction

This module provides comprehensive evaluation metrics specifically designed
for Remaining Useful Life (RUL) prediction tasks.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')


def calculate_rul_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate comprehensive RUL prediction metrics.
    
    Args:
        y_true (np.ndarray): True RUL values
        y_pred (np.ndarray): Predicted RUL values
        
    Returns:
        Dict[str, float]: Dictionary containing all metrics
    """
    # Basic regression metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Mean Absolute Percentage Error (MAPE)
    # Add small epsilon to avoid division by zero
    epsilon = 1e-8
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
    
    # Directional accuracy (trend prediction)
    directional_accuracy = _calculate_directional_accuracy(y_true, y_pred)
    
    # Late prediction penalty
    late_penalty = late_prediction_penalty(y_true, y_pred)
    
    # Prognostic horizon analysis
    horizon_metrics = prognostic_horizon_analysis(y_true, y_pred)
    
    metrics = {
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'r2_score': r2,
        'directional_accuracy': directional_accuracy,
        'late_prediction_penalty': late_penalty,
        'prognostic_horizon': horizon_metrics
    }
    
    return metrics


def _calculate_directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate directional accuracy for trend prediction.
    
    Args:
        y_true (np.ndarray): True RUL values
        y_pred (np.ndarray): Predicted RUL values
        
    Returns:
        float: Percentage of correct trend predictions
    """
    if len(y_true) < 2:
        return 0.0
    
    # Calculate differences (decreasing RUL indicates degradation)
    true_diff = np.diff(y_true)
    pred_diff = np.diff(y_pred)
    
    # Check if trends match (both decreasing or both increasing)
    trend_match = np.sign(true_diff) == np.sign(pred_diff)
    
    return np.mean(trend_match) * 100


def prognostic_horizon_analysis(y_true: np.ndarray, y_pred: np.ndarray, 
                               thresholds: List[int] = [10, 20, 30]) -> Dict[str, float]:
    """
    Analyze prediction accuracy at different prognostic horizons.
    
    Args:
        y_true (np.ndarray): True RUL values
        y_pred (np.ndarray): Predicted RUL values
        thresholds (List[int]): RUL thresholds for analysis
        
    Returns:
        Dict[str, float]: Accuracy metrics for each threshold
    """
    horizon_metrics = {}
    
    for threshold in thresholds:
        # Find predictions within threshold range
        mask = (y_true <= threshold) & (y_true > 0)
        
        if np.sum(mask) > 0:
            true_subset = y_true[mask]
            pred_subset = y_pred[mask]
            
            # Calculate metrics for this horizon
            rmse = np.sqrt(mean_squared_error(true_subset, pred_subset))
            mae = mean_absolute_error(true_subset, pred_subset)
            mape = np.mean(np.abs((true_subset - pred_subset) / (true_subset + 1e-8))) * 100
            
            horizon_metrics[f'horizon_{threshold}'] = {
                'rmse': rmse,
                'mae': mae,
                'mape': mape,
                'n_samples': np.sum(mask)
            }
        else:
            horizon_metrics[f'horizon_{threshold}'] = {
                'rmse': np.nan,
                'mae': np.nan,
                'mape': np.nan,
                'n_samples': 0
            }
    
    return horizon_metrics


def late_prediction_penalty(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate penalty for late predictions (predicting failure after it occurs).
    
    Late predictions are more costly than early predictions in maintenance.
    
    Args:
        y_true (np.ndarray): True RUL values
        y_pred (np.ndarray): Predicted RUL values
        
    Returns:
        float: Late prediction penalty score
    """
    # Identify late predictions (predicted RUL > true RUL when true RUL is low)
    low_rul_mask = y_true <= 20  # Focus on critical RUL range
    late_predictions = (y_pred > y_true) & low_rul_mask
    
    if np.sum(late_predictions) == 0:
        return 0.0
    
    # Calculate penalty based on how late the prediction is
    late_true = y_true[late_predictions]
    late_pred = y_pred[late_predictions]
    
    # Penalty increases exponentially with lateness
    lateness = late_pred - late_true
    penalty = np.sum(np.exp(lateness / 10))  # Exponential penalty
    
    # Normalize by number of late predictions
    normalized_penalty = penalty / np.sum(late_predictions)
    
    return normalized_penalty


def calculate_confidence_intervals(y_true: np.ndarray, y_pred: np.ndarray, 
                                 confidence_level: float = 0.95) -> Dict[str, Tuple[float, float]]:
    """
    Calculate confidence intervals for prediction errors.
    
    Args:
        y_true (np.ndarray): True RUL values
        y_pred (np.ndarray): Predicted RUL values
        confidence_level (float): Confidence level (default: 0.95)
        
    Returns:
        Dict[str, Tuple[float, float]]: Confidence intervals for different metrics
    """
    errors = y_true - y_pred
    abs_errors = np.abs(errors)
    relative_errors = abs_errors / (y_true + 1e-8)
    
    # Calculate percentiles
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    intervals = {
        'absolute_error': (
            np.percentile(abs_errors, lower_percentile),
            np.percentile(abs_errors, upper_percentile)
        ),
        'relative_error': (
            np.percentile(relative_errors, lower_percentile),
            np.percentile(relative_errors, upper_percentile)
        ),
        'prediction_error': (
            np.percentile(errors, lower_percentile),
            np.percentile(errors, upper_percentile)
        )
    }
    
    return intervals


def evaluate_model_robustness(y_true: np.ndarray, y_pred: np.ndarray, 
                            noise_levels: List[float] = [0.01, 0.05, 0.1]) -> Dict[str, Dict[str, float]]:
    """
    Evaluate model robustness by adding noise to predictions.
    
    Args:
        y_true (np.ndarray): True RUL values
        y_pred (np.ndarray): Predicted RUL values
        noise_levels (List[float]): Noise levels to test
        
    Returns:
        Dict[str, Dict[str, float]]: Robustness metrics for each noise level
    """
    robustness_results = {}
    
    for noise_level in noise_levels:
        # Add Gaussian noise to predictions
        noise = np.random.normal(0, noise_level * np.std(y_pred), len(y_pred))
        y_pred_noisy = y_pred + noise
        
        # Calculate metrics with noisy predictions
        metrics = calculate_rul_metrics(y_true, y_pred_noisy)
        robustness_results[f'noise_{noise_level}'] = metrics
    
    return robustness_results


def calculate_engine_level_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                                 engine_ids: np.ndarray) -> Dict[str, float]:
    """
    Calculate metrics aggregated by engine.
    
    Args:
        y_true (np.ndarray): True RUL values
        y_pred (np.ndarray): Predicted RUL values
        engine_ids (np.ndarray): Engine identifiers
        
    Returns:
        Dict[str, float]: Engine-level aggregated metrics
    """
    unique_engines = np.unique(engine_ids)
    engine_metrics = []
    
    for engine_id in unique_engines:
        engine_mask = engine_ids == engine_id
        engine_true = y_true[engine_mask]
        engine_pred = y_pred[engine_mask]
        
        if len(engine_true) > 0:
            engine_rmse = np.sqrt(mean_squared_error(engine_true, engine_pred))
            engine_mae = mean_absolute_error(engine_true, engine_pred)
            engine_metrics.append({
                'engine_id': engine_id,
                'rmse': engine_rmse,
                'mae': engine_mae,
                'n_cycles': len(engine_true)
            })
    
    # Aggregate across engines
    if engine_metrics:
        engine_df = pd.DataFrame(engine_metrics)
        aggregated_metrics = {
            'mean_engine_rmse': engine_df['rmse'].mean(),
            'std_engine_rmse': engine_df['rmse'].std(),
            'mean_engine_mae': engine_df['mae'].mean(),
            'std_engine_mae': engine_df['mae'].std(),
            'n_engines': len(unique_engines),
            'total_cycles': engine_df['n_cycles'].sum()
        }
    else:
        aggregated_metrics = {
            'mean_engine_rmse': np.nan,
            'std_engine_rmse': np.nan,
            'mean_engine_mae': np.nan,
            'std_engine_mae': np.nan,
            'n_engines': 0,
            'total_cycles': 0
        }
    
    return aggregated_metrics


def generate_evaluation_report(y_true: np.ndarray, y_pred: np.ndarray, 
                             engine_ids: np.ndarray = None, 
                             model_name: str = "Model") -> Dict[str, any]:
    """
    Generate comprehensive evaluation report.
    
    Args:
        y_true (np.ndarray): True RUL values
        y_pred (np.ndarray): Predicted RUL values
        engine_ids (np.ndarray): Engine identifiers (optional)
        model_name (str): Name of the model being evaluated
        
    Returns:
        Dict[str, any]: Comprehensive evaluation report
    """
    # Basic metrics
    basic_metrics = calculate_rul_metrics(y_true, y_pred)
    
    # Confidence intervals
    confidence_intervals = calculate_confidence_intervals(y_true, y_pred)
    
    # Robustness analysis
    robustness_results = evaluate_model_robustness(y_true, y_pred)
    
    # Engine-level metrics (if engine IDs provided)
    engine_metrics = {}
    if engine_ids is not None:
        engine_metrics = calculate_engine_level_metrics(y_true, y_pred, engine_ids)
    
    # Create comprehensive report
    report = {
        'model_name': model_name,
        'basic_metrics': basic_metrics,
        'confidence_intervals': confidence_intervals,
        'robustness_analysis': robustness_results,
        'engine_level_metrics': engine_metrics,
        'data_summary': {
            'n_samples': len(y_true),
            'true_rul_range': (y_true.min(), y_true.max()),
            'pred_rul_range': (y_pred.min(), y_pred.max()),
            'true_rul_mean': y_true.mean(),
            'pred_rul_mean': y_pred.mean()
        }
    }
    
    return report


