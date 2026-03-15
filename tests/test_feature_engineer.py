"""
Unit tests for feature engineering module.

Tests the FeatureEngineer class and feature engineering functionality.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from data.feature_engineer import FeatureEngineer


class TestFeatureEngineer:
    """Test cases for FeatureEngineer class."""
    
    def test_init(self):
        """Test FeatureEngineer initialization."""
        engineer = FeatureEngineer()
        assert engineer is not None
        assert hasattr(engineer, 'create_rolling_features')
        assert hasattr(engineer, 'create_degradation_indicators')
        assert hasattr(engineer, 'normalize_features')
    
    def test_create_rolling_features_basic(self, sample_sensor_data):
        """Test basic rolling features creation."""
        engineer = FeatureEngineer()
        
        # Test with small window size
        df = sample_sensor_data.copy()
        result = engineer.create_rolling_features(df, window_sizes=[5])
        
        # Verify rolling features were created
        rolling_cols = [col for col in result.columns if 'rolling' in col]
        assert len(rolling_cols) > 0
        
        # Verify rolling mean was created
        mean_cols = [col for col in rolling_cols if 'rolling_mean' in col]
        assert len(mean_cols) > 0
        
        # Verify rolling std was created
        std_cols = [col for col in rolling_cols if 'rolling_std' in col]
        assert len(std_cols) > 0
    
    def test_create_rolling_features_multiple_windows(self, sample_sensor_data):
        """Test rolling features with multiple window sizes."""
        engineer = FeatureEngineer()
        
        df = sample_sensor_data.copy()
        window_sizes = [5, 10, 20]
        result = engineer.create_rolling_features(df, window_sizes=window_sizes)
        
        # Verify features for each window size
        for window in window_sizes:
            window_cols = [col for col in result.columns if f'rolling_mean_{window}' in col]
            assert len(window_cols) > 0
    
    def test_create_rolling_features_handles_na(self, sample_sensor_data):
        """Test that rolling features handle NA values correctly."""
        engineer = FeatureEngineer()
        
        df = sample_sensor_data.copy()
        
        # Introduce some NA values
        df.loc[10:15, 'sensor_1'] = np.nan
        df.loc[20:25, 'sensor_2'] = np.nan
        
        result = engineer.create_rolling_features(df, window_sizes=[5])
        
        # Verify no errors occurred and features were created
        rolling_cols = [col for col in result.columns if 'rolling' in col]
        assert len(rolling_cols) > 0
        
        # Verify that NA values are handled (either filled or preserved)
        assert result is not None
    
    def test_create_degradation_indicators(self, sample_sensor_data):
        """Test degradation indicators creation."""
        engineer = FeatureEngineer()
        
        df = sample_sensor_data.copy()
        result = engineer.create_degradation_indicators(df)
        
        # Verify degradation indicators were created
        assert 'health_state' in result.columns
        assert 'degradation_rate' in result.columns
        
        # Verify health states are valid
        valid_states = [0, 1, 2, 3]  # Healthy, Degrading, Critical, Failure
        assert result['health_state'].isin(valid_states).all()
    
    def test_create_degradation_indicators_handles_na(self, sample_sensor_data):
        """Test degradation indicators with NA values."""
        engineer = FeatureEngineer()
        
        df = sample_sensor_data.copy()
        
        # Introduce NA values
        df.loc[10:15, 'sensor_1'] = np.nan
        
        result = engineer.create_degradation_indicators(df)
        
        # Verify no errors occurred
        assert 'health_state' in result.columns
        assert 'degradation_rate' in result.columns
    
    def test_normalize_features(self, sample_sensor_data):
        """Test feature normalization."""
        engineer = FeatureEngineer()
        
        df = sample_sensor_data.copy()
        sensor_cols = [f'sensor_{i}' for i in range(1, 6)]  # Use first 5 sensors
        
        result = engineer.normalize_features(df, sensor_cols)
        
        # Verify normalized features were created
        normalized_cols = [col for col in result.columns if 'normalized' in col]
        assert len(normalized_cols) == len(sensor_cols)
        
        # Verify normalization (mean ≈ 0, std ≈ 1)
        for col in normalized_cols:
            assert abs(result[col].mean()) < 0.1  # Mean close to 0
            assert abs(result[col].std() - 1.0) < 0.1  # Std close to 1
    
    def test_normalize_features_handles_na(self, sample_sensor_data):
        """Test normalization with NA values."""
        engineer = FeatureEngineer()
        
        df = sample_sensor_data.copy()
        sensor_cols = [f'sensor_{i}' for i in range(1, 6)]
        
        # Introduce NA values
        df.loc[10:15, 'sensor_1'] = np.nan
        
        result = engineer.normalize_features(df, sensor_cols)
        
        # Verify no errors occurred
        normalized_cols = [col for col in result.columns if 'normalized' in col]
        assert len(normalized_cols) == len(sensor_cols)
    
    def test_create_engineered_features_integration(self, sample_sensor_data):
        """Test the complete feature engineering pipeline."""
        engineer = FeatureEngineer()
        
        df = sample_sensor_data.copy()
        sensor_cols = [f'sensor_{i}' for i in range(1, 6)]
        
        result = engineer.create_engineered_features(
            df, 
            sensor_cols=sensor_cols,
            window_sizes=[5, 10]
        )
        
        # Verify all feature types were created
        assert 'health_state' in result.columns
        assert 'degradation_rate' in result.columns
        
        # Verify rolling features
        rolling_cols = [col for col in result.columns if 'rolling' in col]
        assert len(rolling_cols) > 0
        
        # Verify normalized features
        normalized_cols = [col for col in result.columns if 'normalized' in col]
        assert len(normalized_cols) == len(sensor_cols)
    
    def test_create_engineered_features_empty_dataframe(self):
        """Test feature engineering with empty DataFrame."""
        engineer = FeatureEngineer()
        
        df = pd.DataFrame()
        sensor_cols = ['sensor_1', 'sensor_2']
        
        with pytest.raises((ValueError, KeyError)):
            engineer.create_engineered_features(df, sensor_cols)
    
    def test_create_engineered_features_missing_columns(self, sample_sensor_data):
        """Test feature engineering with missing sensor columns."""
        engineer = FeatureEngineer()
        
        df = sample_sensor_data.copy()
        sensor_cols = ['sensor_1', 'sensor_2', 'missing_sensor']
        
        with pytest.raises(KeyError):
            engineer.create_engineered_features(df, sensor_cols)
    
    def test_rolling_features_edge_cases(self, sample_sensor_data):
        """Test rolling features with edge cases."""
        engineer = FeatureEngineer()
        
        df = sample_sensor_data.copy()
        
        # Test with window size larger than data
        result = engineer.create_rolling_features(df, window_sizes=[1000])
        
        # Should not crash, but may have NA values
        assert result is not None
        
        # Test with window size 1
        result = engineer.create_rolling_features(df, window_sizes=[1])
        
        # Should work but rolling features should be same as original
        assert result is not None
    
    def test_degradation_indicators_edge_cases(self, sample_sensor_data):
        """Test degradation indicators with edge cases."""
        engineer = FeatureEngineer()
        
        df = sample_sensor_data.copy()
        
        # Test with single engine
        single_engine_df = df[df['engine_id'] == 1].copy()
        result = engineer.create_degradation_indicators(single_engine_df)
        
        assert 'health_state' in result.columns
        assert 'degradation_rate' in result.columns
        
        # Test with very short sequences
        short_df = df[df['engine_id'] == 1].head(3).copy()
        result = engineer.create_degradation_indicators(short_df)
        
        assert 'health_state' in result.columns
        assert 'degradation_rate' in result.columns
    
    def test_normalize_features_edge_cases(self, sample_sensor_data):
        """Test normalization with edge cases."""
        engineer = FeatureEngineer()
        
        df = sample_sensor_data.copy()
        sensor_cols = [f'sensor_{i}' for i in range(1, 6)]
        
        # Test with constant values
        df['sensor_1'] = 100  # Constant value
        result = engineer.normalize_features(df, sensor_cols)
        
        # Should handle constant values gracefully
        assert result is not None
        
        # Test with single value
        single_row_df = df.head(1).copy()
        result = engineer.normalize_features(single_row_df, sensor_cols)
        
        # Should handle single value gracefully
        assert result is not None


class TestFeatureEngineerIntegration:
    """Integration tests for feature engineering functionality."""
    
    def test_full_pipeline_with_realistic_data(self):
        """Test full feature engineering pipeline with realistic data."""
        engineer = FeatureEngineer()
        
        # Create realistic sensor data
        np.random.seed(42)
        n_engines = 10
        n_cycles = 200
        
        data = []
        for engine_id in range(1, n_engines + 1):
            for cycle in range(1, n_cycles + 1):
                # Simulate sensor degradation
                degradation = cycle / n_cycles
                
                row = {
                    'engine_id': engine_id,
                    'cycle': cycle,
                    'sensor_1': 100 + degradation * 50 + np.random.normal(0, 5),
                    'sensor_2': 200 + degradation * 100 + np.random.normal(0, 10),
                    'sensor_3': 300 + degradation * 150 + np.random.normal(0, 15),
                    'sensor_4': 400 + degradation * 200 + np.random.normal(0, 20),
                    'sensor_5': 500 + degradation * 250 + np.random.normal(0, 25),
                }
                data.append(row)
        
        df = pd.DataFrame(data)
        sensor_cols = [f'sensor_{i}' for i in range(1, 6)]
        
        # Run full pipeline
        result = engineer.create_engineered_features(
            df,
            sensor_cols=sensor_cols,
            window_sizes=[5, 10, 20]
        )
        
        # Verify comprehensive feature set
        assert len(result.columns) > len(df.columns)  # More features created
        
        # Verify all feature types present
        assert 'health_state' in result.columns
        assert 'degradation_rate' in result.columns
        
        rolling_cols = [col for col in result.columns if 'rolling' in col]
        assert len(rolling_cols) > 0
        
        normalized_cols = [col for col in result.columns if 'normalized' in col]
        assert len(normalized_cols) == len(sensor_cols)
        
        # Verify no infinite or extreme values
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            assert not np.isinf(result[col]).any()
            assert not np.isnan(result[col]).all()  # Not all NaN
    
    def test_feature_engineering_performance(self, sample_sensor_data):
        """Test feature engineering performance with larger dataset."""
        engineer = FeatureEngineer()
        
        # Create larger dataset
        large_df = pd.concat([sample_sensor_data] * 10, ignore_index=True)
        sensor_cols = [f'sensor_{i}' for i in range(1, 6)]
        
        import time
        start_time = time.time()
        
        result = engineer.create_engineered_features(
            large_df,
            sensor_cols=sensor_cols,
            window_sizes=[5, 10]
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Should complete within reasonable time (adjust threshold as needed)
        assert processing_time < 10.0  # Less than 10 seconds
        
        # Verify results
        assert result is not None
        assert len(result) == len(large_df)

