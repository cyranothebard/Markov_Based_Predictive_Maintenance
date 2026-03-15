"""
Unit tests for baseline models module.

Tests the BaselineModels class and baseline model implementations.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
import torch

from models.baseline_models import BaselineModels


class TestBaselineModels:
    """Test cases for BaselineModels class."""
    
    def test_init(self):
        """Test BaselineModels initialization."""
        models = BaselineModels()
        assert models is not None
        assert hasattr(models, 'train_random_forest')
        assert hasattr(models, 'train_linear_regression')
        assert hasattr(models, 'train_lstm')
    
    def test_train_random_forest_basic(self, sample_feature_matrix, sample_rul_data):
        """Test basic Random Forest training."""
        models = BaselineModels()
        
        # Train Random Forest
        rf_model = models.train_random_forest(sample_feature_matrix, sample_rul_data)
        
        # Verify model is trained
        assert rf_model is not None
        assert hasattr(rf_model, 'predict')
        
        # Test prediction
        predictions = rf_model.predict(sample_feature_matrix[:10])
        assert len(predictions) == 10
        assert np.all(predictions >= 0)  # RUL should be non-negative
    
    def test_train_random_forest_with_test_data(self, sample_feature_matrix, sample_rul_data):
        """Test Random Forest training with test data."""
        models = BaselineModels()
        
        # Split data
        n_train = 800
        X_train = sample_feature_matrix[:n_train]
        y_train = sample_rul_data[:n_train]
        X_test = sample_feature_matrix[n_train:]
        y_test = sample_rul_data[n_train:]
        
        # Train with test data
        rf_model = models.train_random_forest(X_train, y_train, X_test, y_test)
        
        assert rf_model is not None
        
        # Test prediction
        predictions = rf_model.predict(X_test)
        assert len(predictions) == len(X_test)
        assert np.all(predictions >= 0)
    
    def test_train_linear_regression_basic(self, sample_feature_matrix, sample_rul_data):
        """Test basic Linear Regression training."""
        models = BaselineModels()
        
        # Train Linear Regression
        lr_model = models.train_linear_regression(sample_feature_matrix, sample_rul_data)
        
        # Verify model is trained
        assert lr_model is not None
        assert hasattr(lr_model, 'predict')
        
        # Test prediction
        predictions = lr_model.predict(sample_feature_matrix[:10])
        assert len(predictions) == 10
        # Linear regression can predict negative values, so we don't check >= 0
    
    def test_train_linear_regression_with_test_data(self, sample_feature_matrix, sample_rul_data):
        """Test Linear Regression training with test data."""
        models = BaselineModels()
        
        # Split data
        n_train = 800
        X_train = sample_feature_matrix[:n_train]
        y_train = sample_rul_data[:n_train]
        X_test = sample_feature_matrix[n_train:]
        y_test = sample_rul_data[n_train:]
        
        # Train with test data
        lr_model = models.train_linear_regression(X_train, y_train, X_test, y_test)
        
        assert lr_model is not None
        
        # Test prediction
        predictions = lr_model.predict(X_test)
        assert len(predictions) == len(X_test)
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="PyTorch not available")
    def test_train_lstm_basic(self, sample_feature_matrix, sample_rul_data):
        """Test basic LSTM training."""
        models = BaselineModels()
        
        # Train LSTM
        lstm_model = models.train_lstm(
            sample_feature_matrix, 
            sample_rul_data,
            sequence_length=10
        )
        
        # Verify model is trained
        assert lstm_model is not None
        assert hasattr(lstm_model, 'predict')
        
        # Test prediction
        predictions = lstm_model.predict(sample_feature_matrix[:20])  # Need enough for sequence
        assert len(predictions) > 0
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="PyTorch not available")
    def test_train_lstm_with_test_data(self, sample_feature_matrix, sample_rul_data):
        """Test LSTM training with test data."""
        models = BaselineModels()
        
        # Split data
        n_train = 800
        X_train = sample_feature_matrix[:n_train]
        y_train = sample_rul_data[:n_train]
        X_test = sample_feature_matrix[n_train:]
        y_test = sample_rul_data[n_train:]
        
        # Train with test data
        lstm_model = models.train_lstm(
            X_train, 
            y_train,
            sequence_length=10,
            X_test=X_test,
            y_test=y_test
        )
        
        assert lstm_model is not None
        
        # Test prediction
        predictions = lstm_model.predict(X_test)
        assert len(predictions) > 0
    
    def test_train_random_forest_empty_data(self):
        """Test Random Forest training with empty data."""
        models = BaselineModels()
        
        with pytest.raises(ValueError):
            models.train_random_forest(np.array([]), np.array([]))
    
    def test_train_linear_regression_empty_data(self):
        """Test Linear Regression training with empty data."""
        models = BaselineModels()
        
        with pytest.raises(ValueError):
            models.train_linear_regression(np.array([]), np.array([]))
    
    def test_train_random_forest_mismatched_data(self, sample_feature_matrix, sample_rul_data):
        """Test Random Forest training with mismatched data."""
        models = BaselineModels()
        
        with pytest.raises(ValueError):
            models.train_random_forest(sample_feature_matrix, sample_rul_data[:100])
    
    def test_train_linear_regression_mismatched_data(self, sample_feature_matrix, sample_rul_data):
        """Test Linear Regression training with mismatched data."""
        models = BaselineModels()
        
        with pytest.raises(ValueError):
            models.train_linear_regression(sample_feature_matrix, sample_rul_data[:100])
    
    def test_train_random_forest_single_feature(self, sample_rul_data):
        """Test Random Forest training with single feature."""
        models = BaselineModels()
        
        # Create single feature data
        X_single = np.random.randn(len(sample_rul_data), 1)
        
        rf_model = models.train_random_forest(X_single, sample_rul_data)
        
        assert rf_model is not None
        
        # Test prediction
        predictions = rf_model.predict(X_single[:10])
        assert len(predictions) == 10
    
    def test_train_linear_regression_single_feature(self, sample_rul_data):
        """Test Linear Regression training with single feature."""
        models = BaselineModels()
        
        # Create single feature data
        X_single = np.random.randn(len(sample_rul_data), 1)
        
        lr_model = models.train_linear_regression(X_single, sample_rul_data)
        
        assert lr_model is not None
        
        # Test prediction
        predictions = lr_model.predict(X_single[:10])
        assert len(predictions) == 10
    
    def test_train_random_forest_constant_target(self, sample_feature_matrix):
        """Test Random Forest training with constant target."""
        models = BaselineModels()
        
        # Create constant target
        y_constant = np.full(len(sample_feature_matrix), 100)
        
        rf_model = models.train_random_forest(sample_feature_matrix, y_constant)
        
        assert rf_model is not None
        
        # Test prediction
        predictions = rf_model.predict(sample_feature_matrix[:10])
        assert len(predictions) == 10
        # Should predict constant value
        assert np.allclose(predictions, 100, rtol=1e-10)
    
    def test_train_linear_regression_constant_target(self, sample_feature_matrix):
        """Test Linear Regression training with constant target."""
        models = BaselineModels()
        
        # Create constant target
        y_constant = np.full(len(sample_feature_matrix), 100)
        
        lr_model = models.train_linear_regression(sample_feature_matrix, y_constant)
        
        assert lr_model is not None
        
        # Test prediction
        predictions = lr_model.predict(sample_feature_matrix[:10])
        assert len(predictions) == 10
        # Should predict constant value
        assert np.allclose(predictions, 100, rtol=1e-10)


class TestBaselineModelsIntegration:
    """Integration tests for baseline models functionality."""
    
    def test_evaluate_all_models(self, sample_feature_matrix, sample_rul_data):
        """Test evaluation of all baseline models."""
        models = BaselineModels()
        
        # Split data
        n_train = 800
        X_train = sample_feature_matrix[:n_train]
        y_train = sample_rul_data[:n_train]
        X_test = sample_feature_matrix[n_train:]
        y_test = sample_rul_data[n_train:]
        
        # Evaluate all models
        results = models.evaluate_all_models(X_train, y_train, X_test, y_test)
        
        # Verify results structure
        assert 'random_forest' in results
        assert 'linear_regression' in results
        # LSTM might be skipped if PyTorch not available
        
        # Verify each model has metrics
        for model_name, model_results in results.items():
            assert 'model' in model_results
            assert 'predictions' in model_results
            assert 'metrics' in model_results
            
            # Verify metrics structure
            metrics = model_results['metrics']
            assert 'rmse' in metrics
            assert 'mae' in metrics
            assert 'r2_score' in metrics
    
    def test_model_comparison(self, sample_feature_matrix, sample_rul_data):
        """Test comparison between different baseline models."""
        models = BaselineModels()
        
        # Split data
        n_train = 800
        X_train = sample_feature_matrix[:n_train]
        y_train = sample_rul_data[:n_train]
        X_test = sample_feature_matrix[n_train:]
        y_test = sample_rul_data[n_train:]
        
        # Train models
        rf_model = models.train_random_forest(X_train, y_train, X_test, y_test)
        lr_model = models.train_linear_regression(X_train, y_train, X_test, y_test)
        
        # Get predictions
        rf_predictions = rf_model.predict(X_test)
        lr_predictions = lr_model.predict(X_test)
        
        # Verify predictions are different (models should behave differently)
        assert not np.allclose(rf_predictions, lr_predictions, rtol=1e-10)
        
        # Verify both models predict reasonable values
        assert len(rf_predictions) == len(X_test)
        assert len(lr_predictions) == len(X_test)
    
    def test_models_with_different_data_sizes(self):
        """Test models with different data sizes."""
        models = BaselineModels()
        
        # Test with small dataset
        X_small = np.random.randn(50, 5)
        y_small = np.random.uniform(10, 200, 50)
        
        rf_small = models.train_random_forest(X_small, y_small)
        lr_small = models.train_linear_regression(X_small, y_small)
        
        assert rf_small is not None
        assert lr_small is not None
        
        # Test with large dataset
        X_large = np.random.randn(5000, 20)
        y_large = np.random.uniform(10, 200, 5000)
        
        rf_large = models.train_random_forest(X_large, y_large)
        lr_large = models.train_linear_regression(X_large, y_large)
        
        assert rf_large is not None
        assert lr_large is not None
    
    def test_models_with_extreme_values(self):
        """Test models with extreme values."""
        models = BaselineModels()
        
        # Create data with extreme values
        X_extreme = np.array([
            [1e6, -1e6, 0, 1e-6, -1e-6],
            [1e6, -1e6, 0, 1e-6, -1e-6],
            [1e6, -1e6, 0, 1e-6, -1e-6],
            [1e6, -1e6, 0, 1e-6, -1e-6],
            [1e6, -1e6, 0, 1e-6, -1e-6]
        ])
        y_extreme = np.array([1e6, 1e-6, 0, -1e6, 1e3])
        
        # Should handle extreme values gracefully
        rf_model = models.train_random_forest(X_extreme, y_extreme)
        lr_model = models.train_linear_regression(X_extreme, y_extreme)
        
        assert rf_model is not None
        assert lr_model is not None
        
        # Test predictions
        rf_predictions = rf_model.predict(X_extreme)
        lr_predictions = lr_model.predict(X_extreme)
        
        assert len(rf_predictions) == len(X_extreme)
        assert len(lr_predictions) == len(X_extreme)
    
    def test_models_with_na_values(self, sample_feature_matrix, sample_rul_data):
        """Test models with NA values in data."""
        models = BaselineModels()
        
        # Introduce NA values
        X_with_na = sample_feature_matrix.copy()
        X_with_na[10:15, 0] = np.nan
        X_with_na[20:25, 1] = np.nan
        
        # Should handle NA values gracefully
        rf_model = models.train_random_forest(X_with_na, sample_rul_data)
        lr_model = models.train_linear_regression(X_with_na, sample_rul_data)
        
        assert rf_model is not None
        assert lr_model is not None
        
        # Test predictions
        rf_predictions = rf_model.predict(X_with_na[:10])
        lr_predictions = lr_model.predict(X_with_na[:10])
        
        assert len(rf_predictions) == 10
        assert len(lr_predictions) == 10
    
    def test_model_performance_consistency(self, sample_feature_matrix, sample_rul_data):
        """Test that model performance is consistent across multiple runs."""
        models = BaselineModels()
        
        # Split data
        n_train = 800
        X_train = sample_feature_matrix[:n_train]
        y_train = sample_rul_data[:n_train]
        X_test = sample_feature_matrix[n_train:]
        y_test = sample_rul_data[n_train:]
        
        # Train models multiple times
        rf_models = []
        lr_models = []
        
        for _ in range(3):
            rf_model = models.train_random_forest(X_train, y_train, X_test, y_test)
            lr_model = models.train_linear_regression(X_train, y_train, X_test, y_test)
            
            rf_models.append(rf_model)
            lr_models.append(lr_model)
        
        # Get predictions from all models
        rf_predictions = [model.predict(X_test) for model in rf_models]
        lr_predictions = [model.predict(X_test) for model in lr_models]
        
        # Random Forest should be consistent (deterministic)
        for i in range(1, len(rf_predictions)):
            assert np.allclose(rf_predictions[0], rf_predictions[i], rtol=1e-10)
        
        # Linear Regression should be consistent (deterministic)
        for i in range(1, len(lr_predictions)):
            assert np.allclose(lr_predictions[0], lr_predictions[i], rtol=1e-10)

