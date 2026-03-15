"""
Pytest configuration and shared fixtures for the test suite.

This module provides common test fixtures and configuration for all test modules.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import os

# Add src to path for imports
project_root = Path(__file__).parent.parent
src_path = project_root / 'src'
sys.path.insert(0, str(src_path))

@pytest.fixture
def sample_sensor_data():
    """Create sample sensor data for testing."""
    np.random.seed(42)
    n_engines = 5
    n_cycles = 100
    
    data = []
    for engine_id in range(1, n_engines + 1):
        for cycle in range(1, n_cycles + 1):
            # Simulate sensor degradation over time
            degradation_factor = cycle / n_cycles
            
            row = {
                'engine_id': engine_id,
                'cycle': cycle,
                'sensor_1': 100 + np.random.normal(0, 5) + degradation_factor * 10,
                'sensor_2': 200 + np.random.normal(0, 10) + degradation_factor * 20,
                'sensor_3': 300 + np.random.normal(0, 15) + degradation_factor * 30,
                'sensor_4': 400 + np.random.normal(0, 20) + degradation_factor * 40,
                'sensor_5': 500 + np.random.normal(0, 25) + degradation_factor * 50,
                'sensor_6': 600 + np.random.normal(0, 30) + degradation_factor * 60,
                'sensor_7': 700 + np.random.normal(0, 35) + degradation_factor * 70,
                'sensor_8': 800 + np.random.normal(0, 40) + degradation_factor * 80,
                'sensor_9': 900 + np.random.normal(0, 45) + degradation_factor * 90,
                'sensor_10': 1000 + np.random.normal(0, 50) + degradation_factor * 100,
                'sensor_11': 1100 + np.random.normal(0, 55) + degradation_factor * 110,
                'sensor_12': 1200 + np.random.normal(0, 60) + degradation_factor * 120,
                'sensor_13': 1300 + np.random.normal(0, 65) + degradation_factor * 130,
                'sensor_14': 1400 + np.random.normal(0, 70) + degradation_factor * 140,
                'sensor_15': 1500 + np.random.normal(0, 75) + degradation_factor * 150,
                'sensor_16': 1600 + np.random.normal(0, 80) + degradation_factor * 160,
                'sensor_17': 1700 + np.random.normal(0, 85) + degradation_factor * 170,
                'sensor_18': 1800 + np.random.normal(0, 90) + degradation_factor * 180,
                'sensor_19': 1900 + np.random.normal(0, 95) + degradation_factor * 190,
                'sensor_20': 2000 + np.random.normal(0, 100) + degradation_factor * 200,
                'sensor_21': 2100 + np.random.normal(0, 105) + degradation_factor * 210,
            }
            data.append(row)
    
    return pd.DataFrame(data)

@pytest.fixture
def sample_rul_data():
    """Create sample RUL data for testing."""
    np.random.seed(42)
    n_samples = 1000
    
    # Create RUL values that decrease over time (simulating degradation)
    rul_values = np.random.uniform(10, 200, n_samples)
    
    return rul_values

@pytest.fixture
def sample_health_states():
    """Create sample health states for testing."""
    np.random.seed(42)
    n_samples = 1000
    
    # Create health states (0=Healthy, 1=Degrading, 2=Critical, 3=Failure)
    states = np.random.choice([0, 1, 2, 3], n_samples, p=[0.4, 0.3, 0.2, 0.1])
    
    return states

@pytest.fixture
def sample_transition_matrix():
    """Create sample transition matrix for testing."""
    return np.array([
        [0.9, 0.1, 0.0, 0.0],  # Healthy -> [Healthy, Degrading, Critical, Failure]
        [0.0, 0.8, 0.2, 0.0],  # Degrading -> [Healthy, Degrading, Critical, Failure]
        [0.0, 0.0, 0.7, 0.3],  # Critical -> [Healthy, Degrading, Critical, Failure]
        [0.0, 0.0, 0.0, 1.0]   # Failure -> [Healthy, Degrading, Critical, Failure]
    ])

@pytest.fixture
def sample_feature_matrix():
    """Create sample feature matrix for testing."""
    np.random.seed(42)
    n_samples = 1000
    n_features = 14
    
    # Create feature matrix with some correlation
    X = np.random.randn(n_samples, n_features)
    
    # Add some structure to make it more realistic
    X[:, 0] = X[:, 0] * 2 + 1  # Scale first feature
    X[:, 1] = X[:, 1] * 0.5 - 0.5  # Scale second feature
    
    return X

@pytest.fixture
def sample_predictions():
    """Create sample predictions for testing."""
    np.random.seed(42)
    n_samples = 1000
    
    # Create predictions with some noise
    true_values = np.random.uniform(10, 200, n_samples)
    noise = np.random.normal(0, 10, n_samples)
    predictions = true_values + noise
    
    return true_values, predictions

@pytest.fixture
def temp_data_dir(tmp_path):
    """Create temporary directory for test data files."""
    return tmp_path / "test_data"

@pytest.fixture
def sample_config():
    """Create sample configuration for testing."""
    return {
        'data': {
            'train_file': 'train_FD001.txt',
            'test_file': 'test_FD001.txt',
            'rul_file': 'RUL_FD001.txt'
        },
        'model': {
            'markov_states': 4,
            'hmm_states': 4,
            'sequence_length': 20
        },
        'feature_engineering': {
            'window_sizes': [5, 10, 20],
            'sensor_columns': [f'sensor_{i}' for i in range(1, 22)]
        }
    }

