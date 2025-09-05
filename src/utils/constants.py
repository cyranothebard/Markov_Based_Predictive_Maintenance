"""
Constants for Markov Predictive Maintenance Project

This module contains project-wide constants and configuration values.
"""

# Dataset constants
CMAPSS_DATASETS = ['FD001', 'FD002', 'FD003', 'FD004']
DEFAULT_DATASET = 'FD001'

# Model constants
DEFAULT_N_STATES = 4
STATE_NAMES = ['Healthy', 'Warning', 'Critical', 'Failure']
STATE_COLORS = ['green', 'yellow', 'orange', 'red']

# Sensor groups
TEMPERATURE_SENSORS = [2, 3, 4, 7, 12, 13, 14, 15]
PRESSURE_SENSORS = [8, 9]
FLOW_SENSORS = [11, 17]
VIBRATION_SENSORS = [1, 5, 6, 10, 16, 18, 19, 20, 21]

# Evaluation thresholds
TARGET_RMSE = 20.0
TARGET_MAPE = 15.0
TARGET_DIRECTIONAL_ACCURACY = 80.0

# File paths
DATA_DIR = 'data'
RAW_DATA_DIR = 'data/raw'
PROCESSED_DATA_DIR = 'data/processed'
FEATURES_DATA_DIR = 'data/features'
RESULTS_DIR = 'results'
FIGURES_DIR = 'results/figures'

# Visualization settings
FIGURE_SIZE = (12, 8)
DPI = 300
PLOT_STYLE = 'seaborn-v0_8'


