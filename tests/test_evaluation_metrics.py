"""
Unit tests for evaluation metrics module.

Tests the evaluation metrics functions and RUL-specific metrics.
"""

import pytest
import numpy as np
import pandas as pd

from evaluation.metrics import (
    calculate_rul_metrics,
    _calculate_directional_accuracy,
    prognostic_horizon_analysis,
    late_prediction_penalty,
    calculate_confidence_intervals,
    evaluate_model_robustness,
    calculate_engine_level_metrics,
    generate_evaluation_report
)


class TestCalculateRULMetrics:
    """Test cases for calculate_rul_metrics function."""
    
    def test_calculate_rul_metrics_basic(self, sample_predictions):
        """Test basic RUL metrics calculation."""
        y_true, y_pred = sample_predictions
        
        metrics = calculate_rul_metrics(y_true, y_pred)
        
        # Verify all expected metrics are present
        expected_metrics = ['rmse', 'mae', 'mape', 'r2_score', 'directional_accuracy', 
                          'late_prediction_penalty', 'prognostic_horizon']
        for metric in expected_metrics:
            assert metric in metrics
        
        # Verify metric values are reasonable
        assert metrics['rmse'] >= 0
        assert metrics['mae'] >= 0
        assert metrics['mape'] >= 0
        assert metrics['r2_score'] <= 1  # R² can be negative but typically <= 1
        assert 0 <= metrics['directional_accuracy'] <= 100
        assert metrics['late_prediction_penalty'] >= 0
    
    def test_calculate_rul_metrics_perfect_predictions(self):
        """Test metrics with perfect predictions."""
        y_true = np.array([100, 150, 200, 50, 75])
        y_pred = y_true.copy()  # Perfect predictions
        
        metrics = calculate_rul_metrics(y_true, y_pred)
        
        # Perfect predictions should give ideal metrics
        assert metrics['rmse'] == 0
        assert metrics['mae'] == 0
        assert metrics['mape'] == 0
        assert metrics['r2_score'] == 1
        assert metrics['directional_accuracy'] == 100
    
    def test_calculate_rul_metrics_identical_values(self):
        """Test metrics with identical true values."""
        y_true = np.array([100, 100, 100, 100, 100])
        y_pred = np.array([90, 110, 95, 105, 100])
        
        metrics = calculate_rul_metrics(y_true, y_pred)
        
        # Should handle identical values gracefully
        assert not np.isnan(metrics['rmse'])
        assert not np.isnan(metrics['mae'])
        assert not np.isnan(metrics['r2_score'])
    
    def test_calculate_rul_metrics_zero_values(self):
        """Test metrics with zero values."""
        y_true = np.array([0, 10, 20, 0, 5])
        y_pred = np.array([5, 15, 25, 2, 8])
        
        metrics = calculate_rul_metrics(y_true, y_pred)
        
        # Should handle zero values gracefully
        assert not np.isnan(metrics['rmse'])
        assert not np.isnan(metrics['mae'])
        assert not np.isnan(metrics['r2_score'])
    
    def test_calculate_rul_metrics_empty_arrays(self):
        """Test metrics with empty arrays."""
        y_true = np.array([])
        y_pred = np.array([])
        
        with pytest.raises(ValueError):
            calculate_rul_metrics(y_true, y_pred)
    
    def test_calculate_rul_metrics_mismatched_lengths(self):
        """Test metrics with mismatched array lengths."""
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1, 2])
        
        with pytest.raises(ValueError):
            calculate_rul_metrics(y_true, y_pred)


class TestDirectionalAccuracy:
    """Test cases for directional accuracy calculation."""
    
    def test_directional_accuracy_perfect(self):
        """Test directional accuracy with perfect trend prediction."""
        y_true = np.array([100, 90, 80, 70, 60])  # Decreasing
        y_pred = np.array([95, 85, 75, 65, 55])   # Also decreasing
        
        accuracy = _calculate_directional_accuracy(y_true, y_pred)
        assert accuracy == 100.0
    
    def test_directional_accuracy_wrong_trend(self):
        """Test directional accuracy with wrong trend prediction."""
        y_true = np.array([100, 90, 80, 70, 60])  # Decreasing
        y_pred = np.array([95, 100, 85, 90, 65])  # Mixed trends
        
        accuracy = _calculate_directional_accuracy(y_true, y_pred)
        assert 0 <= accuracy <= 100
    
    def test_directional_accuracy_single_value(self):
        """Test directional accuracy with single value."""
        y_true = np.array([100])
        y_pred = np.array([90])
        
        accuracy = _calculate_directional_accuracy(y_true, y_pred)
        assert accuracy == 0.0  # No trend to compare
    
    def test_directional_accuracy_constant_values(self):
        """Test directional accuracy with constant values."""
        y_true = np.array([100, 100, 100, 100])
        y_pred = np.array([90, 90, 90, 90])
        
        accuracy = _calculate_directional_accuracy(y_true, y_pred)
        assert accuracy == 100.0  # No change = perfect match


class TestPrognosticHorizonAnalysis:
    """Test cases for prognostic horizon analysis."""
    
    def test_prognostic_horizon_analysis_basic(self, sample_predictions):
        """Test basic prognostic horizon analysis."""
        y_true, y_pred = sample_predictions
        
        horizon_metrics = prognostic_horizon_analysis(y_true, y_pred)
        
        # Verify structure
        assert 'horizon_10' in horizon_metrics
        assert 'horizon_20' in horizon_metrics
        assert 'horizon_30' in horizon_metrics
        
        # Verify each horizon has expected metrics
        for horizon in ['horizon_10', 'horizon_20', 'horizon_30']:
            assert 'rmse' in horizon_metrics[horizon]
            assert 'mae' in horizon_metrics[horizon]
            assert 'mape' in horizon_metrics[horizon]
            assert 'n_samples' in horizon_metrics[horizon]
    
    def test_prognostic_horizon_analysis_custom_thresholds(self, sample_predictions):
        """Test prognostic horizon analysis with custom thresholds."""
        y_true, y_pred = sample_predictions
        
        custom_thresholds = [5, 15, 25, 50]
        horizon_metrics = prognostic_horizon_analysis(y_true, y_pred, custom_thresholds)
        
        # Verify custom thresholds
        for threshold in custom_thresholds:
            assert f'horizon_{threshold}' in horizon_metrics
    
    def test_prognostic_horizon_analysis_no_low_rul(self):
        """Test prognostic horizon analysis with no low RUL values."""
        y_true = np.array([200, 250, 300, 350, 400])  # All high RUL
        y_pred = np.array([190, 240, 290, 340, 390])
        
        horizon_metrics = prognostic_horizon_analysis(y_true, y_pred)
        
        # Should handle no low RUL values gracefully
        for horizon in ['horizon_10', 'horizon_20', 'horizon_30']:
            assert horizon_metrics[horizon]['n_samples'] == 0


class TestLatePredictionPenalty:
    """Test cases for late prediction penalty calculation."""
    
    def test_late_prediction_penalty_basic(self):
        """Test basic late prediction penalty calculation."""
        y_true = np.array([5, 10, 15, 20, 25])  # Low RUL values
        y_pred = np.array([8, 12, 18, 22, 28])  # Some late predictions
        
        penalty = late_prediction_penalty(y_true, y_pred)
        
        assert penalty >= 0
        assert not np.isnan(penalty)
    
    def test_late_prediction_penalty_no_late_predictions(self):
        """Test late prediction penalty with no late predictions."""
        y_true = np.array([5, 10, 15, 20, 25])
        y_pred = np.array([3, 8, 12, 18, 22])  # All early predictions
        
        penalty = late_prediction_penalty(y_true, y_pred)
        
        assert penalty == 0.0
    
    def test_late_prediction_penalty_no_low_rul(self):
        """Test late prediction penalty with no low RUL values."""
        y_true = np.array([100, 150, 200, 250, 300])  # All high RUL
        y_pred = np.array([110, 160, 210, 260, 310])
        
        penalty = late_prediction_penalty(y_true, y_pred)
        
        assert penalty == 0.0


class TestConfidenceIntervals:
    """Test cases for confidence interval calculation."""
    
    def test_calculate_confidence_intervals_basic(self, sample_predictions):
        """Test basic confidence interval calculation."""
        y_true, y_pred = sample_predictions
        
        intervals = calculate_confidence_intervals(y_true, y_pred)
        
        # Verify structure
        assert 'absolute_error' in intervals
        assert 'relative_error' in intervals
        assert 'prediction_error' in intervals
        
        # Verify each interval is a tuple
        for interval_name, interval in intervals.items():
            assert isinstance(interval, tuple)
            assert len(interval) == 2
            assert interval[0] <= interval[1]  # Lower <= Upper
    
    def test_calculate_confidence_intervals_custom_confidence(self, sample_predictions):
        """Test confidence intervals with custom confidence level."""
        y_true, y_pred = sample_predictions
        
        intervals = calculate_confidence_intervals(y_true, y_pred, confidence_level=0.99)
        
        # Should still have same structure
        assert 'absolute_error' in intervals
        assert 'relative_error' in intervals
        assert 'prediction_error' in intervals


class TestModelRobustness:
    """Test cases for model robustness evaluation."""
    
    def test_evaluate_model_robustness_basic(self, sample_predictions):
        """Test basic model robustness evaluation."""
        y_true, y_pred = sample_predictions
        
        robustness_results = evaluate_model_robustness(y_true, y_pred)
        
        # Verify structure
        assert 'noise_0.01' in robustness_results
        assert 'noise_0.05' in robustness_results
        assert 'noise_0.1' in robustness_results
        
        # Verify each noise level has metrics
        for noise_level in ['noise_0.01', 'noise_0.05', 'noise_0.1']:
            assert 'rmse' in robustness_results[noise_level]
            assert 'mae' in robustness_results[noise_level]
            assert 'r2_score' in robustness_results[noise_level]
    
    def test_evaluate_model_robustness_custom_noise(self, sample_predictions):
        """Test model robustness with custom noise levels."""
        y_true, y_pred = sample_predictions
        
        custom_noise = [0.02, 0.08, 0.15]
        robustness_results = evaluate_model_robustness(y_true, y_pred, custom_noise)
        
        # Verify custom noise levels
        for noise in custom_noise:
            assert f'noise_{noise}' in robustness_results


class TestEngineLevelMetrics:
    """Test cases for engine-level metrics calculation."""
    
    def test_calculate_engine_level_metrics_basic(self, sample_predictions):
        """Test basic engine-level metrics calculation."""
        y_true, y_pred = sample_predictions
        
        # Create engine IDs
        engine_ids = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3, 4] * 100)  # 1000 samples
        
        metrics = calculate_engine_level_metrics(y_true, y_pred, engine_ids)
        
        # Verify structure
        expected_metrics = ['mean_engine_rmse', 'std_engine_rmse', 'mean_engine_mae', 
                          'std_engine_mae', 'n_engines', 'total_cycles']
        for metric in expected_metrics:
            assert metric in metrics
        
        # Verify values
        assert metrics['n_engines'] == 4
        assert metrics['total_cycles'] == 1000
        assert metrics['mean_engine_rmse'] >= 0
        assert metrics['mean_engine_mae'] >= 0
    
    def test_calculate_engine_level_metrics_single_engine(self, sample_predictions):
        """Test engine-level metrics with single engine."""
        y_true, y_pred = sample_predictions
        engine_ids = np.ones(len(y_true))  # All same engine
        
        metrics = calculate_engine_level_metrics(y_true, y_pred, engine_ids)
        
        assert metrics['n_engines'] == 1
        assert metrics['total_cycles'] == len(y_true)
        assert metrics['std_engine_rmse'] == 0  # No variation with single engine
        assert metrics['std_engine_mae'] == 0


class TestEvaluationReport:
    """Test cases for comprehensive evaluation report generation."""
    
    def test_generate_evaluation_report_basic(self, sample_predictions):
        """Test basic evaluation report generation."""
        y_true, y_pred = sample_predictions
        
        report = generate_evaluation_report(y_true, y_pred, model_name="Test Model")
        
        # Verify structure
        assert 'model_name' in report
        assert 'basic_metrics' in report
        assert 'confidence_intervals' in report
        assert 'robustness_analysis' in report
        assert 'engine_level_metrics' in report
        assert 'data_summary' in report
        
        # Verify model name
        assert report['model_name'] == "Test Model"
        
        # Verify basic metrics
        assert 'rmse' in report['basic_metrics']
        assert 'mae' in report['basic_metrics']
        assert 'r2_score' in report['basic_metrics']
    
    def test_generate_evaluation_report_with_engine_ids(self, sample_predictions):
        """Test evaluation report with engine IDs."""
        y_true, y_pred = sample_predictions
        engine_ids = np.array([1, 1, 2, 2, 3] * 200)  # 1000 samples
        
        report = generate_evaluation_report(y_true, y_pred, engine_ids, "Test Model")
        
        # Should include engine-level metrics
        assert 'engine_level_metrics' in report
        assert report['engine_level_metrics'] is not None
        assert 'n_engines' in report['engine_level_metrics']
    
    def test_generate_evaluation_report_data_summary(self, sample_predictions):
        """Test evaluation report data summary."""
        y_true, y_pred = sample_predictions
        
        report = generate_evaluation_report(y_true, y_pred)
        
        # Verify data summary
        data_summary = report['data_summary']
        assert 'n_samples' in data_summary
        assert 'true_rul_range' in data_summary
        assert 'pred_rul_range' in data_summary
        assert 'true_rul_mean' in data_summary
        assert 'pred_rul_mean' in data_summary
        
        # Verify values
        assert data_summary['n_samples'] == len(y_true)
        assert data_summary['true_rul_range'][0] <= data_summary['true_rul_range'][1]
        assert data_summary['pred_rul_range'][0] <= data_summary['pred_rul_range'][1]


class TestEvaluationMetricsIntegration:
    """Integration tests for evaluation metrics functionality."""
    
    def test_metrics_consistency(self):
        """Test that metrics are consistent across different inputs."""
        # Create test data
        np.random.seed(42)
        y_true = np.random.uniform(10, 200, 1000)
        y_pred = y_true + np.random.normal(0, 10, 1000)
        
        # Calculate metrics
        metrics = calculate_rul_metrics(y_true, y_pred)
        
        # Verify metrics are reasonable
        assert 0 <= metrics['rmse'] <= 100  # Reasonable RMSE range
        assert 0 <= metrics['mae'] <= 100   # Reasonable MAE range
        assert 0 <= metrics['mape'] <= 1000 # Reasonable MAPE range
        assert -10 <= metrics['r2_score'] <= 1  # R² can be negative
        assert 0 <= metrics['directional_accuracy'] <= 100
    
    def test_metrics_with_extreme_values(self):
        """Test metrics with extreme values."""
        # Very large values
        y_true = np.array([10000, 20000, 30000])
        y_pred = np.array([10100, 20100, 30100])
        
        metrics = calculate_rul_metrics(y_true, y_pred)
        
        # Should handle large values gracefully
        assert not np.isnan(metrics['rmse'])
        assert not np.isnan(metrics['mae'])
        assert not np.isnan(metrics['r2_score'])
        
        # Very small values
        y_true = np.array([0.1, 0.2, 0.3])
        y_pred = np.array([0.11, 0.21, 0.31])
        
        metrics = calculate_rul_metrics(y_true, y_pred)
        
        # Should handle small values gracefully
        assert not np.isnan(metrics['rmse'])
        assert not np.isnan(metrics['mae'])
        assert not np.isnan(metrics['r2_score'])
    
    def test_comprehensive_evaluation_workflow(self):
        """Test comprehensive evaluation workflow."""
        # Create realistic test data
        np.random.seed(42)
        n_samples = 2000
        
        # Create RUL data with some structure
        y_true = np.random.uniform(10, 200, n_samples)
        y_pred = y_true + np.random.normal(0, 15, n_samples)
        
        # Create engine IDs
        engine_ids = np.random.choice([1, 2, 3, 4, 5], n_samples)
        
        # Generate comprehensive report
        report = generate_evaluation_report(y_true, y_pred, engine_ids, "Integration Test Model")
        
        # Verify all components are present and reasonable
        assert report['model_name'] == "Integration Test Model"
        assert report['basic_metrics']['rmse'] > 0
        assert report['basic_metrics']['mae'] > 0
        assert report['engine_level_metrics']['n_engines'] == 5
        assert report['data_summary']['n_samples'] == n_samples
        
        # Verify no NaN or infinite values in key metrics
        assert not np.isnan(report['basic_metrics']['rmse'])
        assert not np.isnan(report['basic_metrics']['mae'])
        assert not np.isnan(report['basic_metrics']['r2_score'])
        assert not np.isinf(report['basic_metrics']['rmse'])
        assert not np.isinf(report['basic_metrics']['mae'])

