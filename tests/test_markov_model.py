"""
Unit tests for Markov model module.

Tests the MarkovChainRUL class and Markov chain functionality.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

from models.markov_model import MarkovChainRUL


class TestMarkovChainRUL:
    """Test cases for MarkovChainRUL class."""
    
    def test_init(self):
        """Test MarkovChainRUL initialization."""
        model = MarkovChainRUL(n_states=4)
        
        assert model.n_states == 4
        assert model.transition_matrix is None
        assert model.emission_probabilities is None
        assert model.state_emission_models is None
        assert model.is_fitted == False
    
    def test_init_invalid_states(self):
        """Test initialization with invalid number of states."""
        with pytest.raises(ValueError):
            MarkovChainRUL(n_states=0)
        
        with pytest.raises(ValueError):
            MarkovChainRUL(n_states=-1)
    
    def test_fit_basic(self, sample_feature_matrix, sample_health_states):
        """Test basic model fitting."""
        model = MarkovChainRUL(n_states=4)
        
        # Fit the model
        model.fit(sample_feature_matrix, sample_health_states)
        
        # Verify model is fitted
        assert model.is_fitted == True
        assert model.transition_matrix is not None
        assert model.emission_probabilities is not None
        
        # Verify transition matrix shape
        assert model.transition_matrix.shape == (4, 4)
        
        # Verify transition matrix properties
        assert np.allclose(model.transition_matrix.sum(axis=1), 1.0)  # Rows sum to 1
        assert np.all(model.transition_matrix >= 0)  # All probabilities >= 0
        assert np.all(model.transition_matrix <= 1)  # All probabilities <= 1
    
    def test_fit_with_different_states(self, sample_feature_matrix):
        """Test fitting with different number of states."""
        # Test with 3 states
        model_3 = MarkovChainRUL(n_states=3)
        states_3 = np.random.choice([0, 1, 2], 1000, p=[0.5, 0.3, 0.2])
        model_3.fit(sample_feature_matrix, states_3)
        
        assert model_3.transition_matrix.shape == (3, 3)
        assert model_3.is_fitted == True
        
        # Test with 5 states
        model_5 = MarkovChainRUL(n_states=5)
        states_5 = np.random.choice([0, 1, 2, 3, 4], 1000, p=[0.3, 0.25, 0.2, 0.15, 0.1])
        model_5.fit(sample_feature_matrix, states_5)
        
        assert model_5.transition_matrix.shape == (5, 5)
        assert model_5.is_fitted == True
    
    def test_fit_empty_data(self):
        """Test fitting with empty data."""
        model = MarkovChainRUL(n_states=4)
        
        with pytest.raises(ValueError):
            model.fit(np.array([]), np.array([]))
    
    def test_fit_mismatched_data(self, sample_feature_matrix, sample_health_states):
        """Test fitting with mismatched data dimensions."""
        model = MarkovChainRUL(n_states=4)
        
        # Mismatched lengths
        with pytest.raises(ValueError):
            model.fit(sample_feature_matrix, sample_health_states[:100])
    
    def test_predict_rul_basic(self, sample_feature_matrix, sample_health_states):
        """Test basic RUL prediction."""
        model = MarkovChainRUL(n_states=4)
        model.fit(sample_feature_matrix, sample_health_states)
        
        # Predict RUL (method requires current_cycle parameter)
        predictions = []
        for i in range(100):
            pred = model.predict_rul(sample_feature_matrix[i:i+1], current_cycle=i)
            predictions.append(pred)
        predictions = np.array(predictions)
        
        # Verify predictions
        assert len(predictions) == 100
        assert isinstance(predictions, np.ndarray)
        assert np.all(predictions >= 0)  # RUL should be non-negative
        assert not np.isnan(predictions).any()  # No NaN values
        assert not np.isinf(predictions).any()  # No infinite values
    
    def test_predict_rul_not_fitted(self, sample_feature_matrix):
        """Test prediction without fitting."""
        model = MarkovChainRUL(n_states=4)
        
        with pytest.raises(ValueError):
            model.predict_rul(sample_feature_matrix, current_cycle=1)
    
    def test_predict_rul_single_sample(self, sample_feature_matrix, sample_health_states):
        """Test RUL prediction for single sample."""
        model = MarkovChainRUL(n_states=4)
        model.fit(sample_feature_matrix, sample_health_states)
        
        # Predict for single sample
        single_sample = sample_feature_matrix[0:1]
        prediction = model.predict_rul(single_sample, current_cycle=1)
        
        assert isinstance(prediction, (int, float, np.number))
        assert prediction >= 0
    
    def test_get_transition_probabilities(self, sample_feature_matrix, sample_health_states):
        """Test getting transition probabilities."""
        model = MarkovChainRUL(n_states=4)
        model.fit(sample_feature_matrix, sample_health_states)
        
        transition_probs = model.get_transition_probabilities()
        
        # Verify transition probabilities
        assert transition_probs.shape == (4, 4)
        assert np.allclose(transition_probs.sum(axis=1), 1.0)
        assert np.all(transition_probs >= 0)
        assert np.all(transition_probs <= 1)
        
        # Should be same as internal transition matrix
        assert np.allclose(transition_probs, model.transition_matrix)
    
    def test_get_transition_probabilities_not_fitted(self):
        """Test getting transition probabilities without fitting."""
        model = MarkovChainRUL(n_states=4)
        
        with pytest.raises(ValueError):
            model.get_transition_probabilities()
    
    def test_state_means_calculation(self, sample_feature_matrix, sample_health_states):
        """Test state means calculation."""
        model = MarkovChainRUL(n_states=4)
        model.fit(sample_feature_matrix, sample_health_states)
        
        # Verify emission probabilities
        assert model.emission_probabilities is not None
        assert model.emission_probabilities.shape[0] == 4
    
    def test_fit_with_constant_states(self, sample_feature_matrix):
        """Test fitting with constant states."""
        model = MarkovChainRUL(n_states=4)
        
        # All states are the same
        constant_states = np.full(1000, 1)
        model.fit(sample_feature_matrix, constant_states)
        
        # Should handle constant states gracefully
        assert model.is_fitted == True
        assert model.transition_matrix is not None
    
    def test_fit_with_single_state(self, sample_feature_matrix):
        """Test fitting with only one state present."""
        model = MarkovChainRUL(n_states=4)
        
        # Only state 0 present
        single_state = np.zeros(1000)
        model.fit(sample_feature_matrix, single_state)
        
        # Should handle single state gracefully
        assert model.is_fitted == True
        assert model.transition_matrix is not None
    
    def test_predict_rul_edge_cases(self, sample_feature_matrix, sample_health_states):
        """Test RUL prediction with edge cases."""
        model = MarkovChainRUL(n_states=4)
        model.fit(sample_feature_matrix, sample_health_states)
        
        # Test with zero features
        zero_features = np.zeros((10, sample_feature_matrix.shape[1]))
        predictions = model.predict_rul(zero_features)
        
        assert len(predictions) == 10
        assert np.all(predictions >= 0)
        
        # Test with extreme values
        extreme_features = np.full((10, sample_feature_matrix.shape[1]), 1000)
        predictions = model.predict_rul(extreme_features)
        
        assert len(predictions) == 10
        assert np.all(predictions >= 0)
    
    def test_model_serialization(self, sample_feature_matrix, sample_health_states):
        """Test model serialization and deserialization."""
        model = MarkovChainRUL(n_states=4)
        model.fit(sample_feature_matrix, sample_health_states)
        
        # Test that model attributes can be accessed
        assert model.n_states == 4
        assert model.is_fitted == True
        assert model.transition_matrix is not None
        assert model.state_means is not None


class TestMarkovChainRULIntegration:
    """Integration tests for MarkovChainRUL functionality."""
    
    def test_full_workflow(self):
        """Test complete workflow from fitting to prediction."""
        # Create realistic data
        np.random.seed(42)
        n_samples = 2000
        n_features = 10
        
        # Create feature matrix
        X = np.random.randn(n_samples, n_features)
        
        # Create realistic health states with progression
        states = []
        current_state = 0
        for i in range(n_samples):
            # Simulate state progression
            if current_state < 3:
                if np.random.random() < 0.1:  # 10% chance to progress
                    current_state += 1
            states.append(current_state)
        
        states = np.array(states)
        
        # Create model and fit
        model = MarkovChainRUL(n_states=4)
        model.fit(X, states)
        
        # Verify model is properly fitted
        assert model.is_fitted == True
        assert model.transition_matrix.shape == (4, 4)
        assert len(model.state_means) == 4
        
        # Make predictions
        test_X = np.random.randn(100, n_features)
        predictions = model.predict_rul(test_X)
        
        # Verify predictions
        assert len(predictions) == 100
        assert np.all(predictions >= 0)
        assert not np.isnan(predictions).any()
        
        # Verify transition matrix properties
        transition_probs = model.get_transition_probabilities()
        assert np.allclose(transition_probs.sum(axis=1), 1.0)
        assert np.all(transition_probs >= 0)
        assert np.all(transition_probs <= 1)
    
    def test_model_consistency(self, sample_feature_matrix, sample_health_states):
        """Test model consistency across multiple fits."""
        model = MarkovChainRUL(n_states=4)
        
        # Fit multiple times with same data
        model.fit(sample_feature_matrix, sample_health_states)
        transition_matrix_1 = model.transition_matrix.copy()
        state_means_1 = model.state_means.copy()
        
        model.fit(sample_feature_matrix, sample_health_states)
        transition_matrix_2 = model.transition_matrix.copy()
        state_means_2 = model.state_means.copy()
        
        # Results should be consistent
        assert np.allclose(transition_matrix_1, transition_matrix_2)
        assert np.allclose(state_means_1, state_means_2)
    
    def test_prediction_stability(self, sample_feature_matrix, sample_health_states):
        """Test prediction stability."""
        model = MarkovChainRUL(n_states=4)
        model.fit(sample_feature_matrix, sample_health_states)
        
        # Make multiple predictions with same input
        test_X = sample_feature_matrix[:10]
        predictions_1 = model.predict_rul(test_X)
        predictions_2 = model.predict_rul(test_X)
        
        # Predictions should be identical
        assert np.allclose(predictions_1, predictions_2)
    
    def test_model_with_different_data_sizes(self):
        """Test model with different data sizes."""
        model = MarkovChainRUL(n_states=4)
        
        # Test with small dataset
        X_small = np.random.randn(50, 5)
        states_small = np.random.choice([0, 1, 2, 3], 50)
        model.fit(X_small, states_small)
        
        assert model.is_fitted == True
        predictions_small = model.predict_rul(X_small[:10])
        assert len(predictions_small) == 10
        
        # Test with large dataset
        X_large = np.random.randn(5000, 20)
        states_large = np.random.choice([0, 1, 2, 3], 5000)
        model.fit(X_large, states_large)
        
        assert model.is_fitted == True
        predictions_large = model.predict_rul(X_large[:100])
        assert len(predictions_large) == 100
