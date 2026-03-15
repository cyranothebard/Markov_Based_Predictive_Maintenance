# Configuration Template for Feature Engineering

# This file documents the expected configuration structure for the FeatureEngineer class
# and provides sensible defaults for the NASA CMAPSS dataset

# Default configuration for CMAPSS dataset
CMAPSS_CONFIG = {
    # Rolling window parameters
    'rolling_window': 10,
    
    # Health state thresholds (RUL as percentage of total engine life)
    'health_thresholds': [0.8, 0.6, 0.4, 0.0],  # Legacy format
    
    # Model parameters for health state classification
    'model': {
        'health_threshold': 0.8,     # Above 80% RUL = Healthy (state 0)
        'warning_threshold': 0.6,    # 60-80% RUL = Warning (state 1)
        'critical_threshold': 0.4,   # 40-60% RUL = Critical (state 2)
                                    # Below 40% RUL = Failure (state 3)
    },
    
    # Sensor groupings for the CMAPSS dataset
    # Based on NASA technical documentation
    'sensors': {
        'temperature_sensors': [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21],
        'pressure_sensors': [1, 5, 6, 10, 16, 18, 19],
        'flow_sensors': [1, 6, 10, 16, 18, 19],  # Some overlap with pressure
        'all_sensors': list(range(1, 22))  # Sensors 1-21
    },
    
    # Feature engineering settings
    'feature_engineering': {
        'rolling_windows': [5, 10, 20],
        'include_rolling_features': True,
        'include_health_indicators': True,
        'normalize_sensors': True
    }
}

# Minimal configuration for testing
MINIMAL_CONFIG = {
    'model': {
        'health_threshold': 0.8,
        'warning_threshold': 0.6, 
        'critical_threshold': 0.4,
    },
    'sensors': {
        'temperature_sensors': [1, 2],
        'pressure_sensors': [1],
        'flow_sensors': [1],
    }
}