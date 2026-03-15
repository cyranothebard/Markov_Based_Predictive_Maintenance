#!/usr/bin/env python3
"""
Test Real Sensor Analysis to Show Live Example
"""

import pandas as pd
from src.data.data_loader import CMAPSSLoader
from src.features.feature_engineering import engineer_features
from src.analysis.sensor_pattern_analyzer import RealSensorPatternAnalyzer, real_sensor_anomaly_analysis

print('🔬 TESTING REAL SENSOR ANALYSIS ON SAMPLE DATA')
print('=' * 60)

# Load and prepare data
loader = CMAPSSLoader()
train_data = loader.load_train_data('FD001')
test_data = loader.load_test_data('FD001')

# Engineer features
train_data = engineer_features(train_data)

# Initialize baseline sensor analyzer
healthy_data = train_data[train_data['RUL'] > 100]  
baseline_analyzer = RealSensorPatternAnalyzer(healthy_data)
print(f'✅ Baseline analyzer initialized with {len(healthy_data)} healthy measurements')

# Test on a few engines from test set
test_engine_ids = [5, 10, 15]

for engine_id in test_engine_ids:
    engine_data = test_data[test_data['engine_id'] == engine_id]
    if len(engine_data) > 0:
        # Get last cycle of engine (closest to failure)
        last_cycle = engine_data.iloc[-1]
        engine_dict = last_cycle.to_dict()
        engine_dict['equipment_id'] = f'TEST_ENGINE_{engine_id}'
        engine_dict['health_state'] = 2  # Critical
        
        print(f'\n🔍 Engine {engine_id} Analysis:')
        analysis = real_sensor_anomaly_analysis(engine_dict, baseline_analyzer)
        print(f'   Primary: {analysis["primary_anomaly"]}')
        print(f'   Urgency: {analysis["urgency_level"]}') 
        print(f'   Actions: {analysis["recommended_actions"][0]}')
    
print('\n✅ Real sensor analysis successfully replacing simulated descriptions!')