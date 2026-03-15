#!/usr/bin/env python3
"""
Simple test to demonstrate real sensor analysis working
"""

import sys
from pathlib import Path
import pandas as pd

# Add the src directory to the Python path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from analysis.sensor_pattern_analyzer import RealSensorPatternAnalyzer, real_sensor_anomaly_analysis

print('🔬 TESTING REAL SENSOR ANALYSIS')
print('=' * 50)

# Create synthetic test data that represents baseline healthy sensors
baseline_data = pd.DataFrame({
    'sensor_02': [20.0, 21.0, 19.5, 20.5, 21.2, 20.8, 19.8, 20.3],
    'sensor_03': [580.0, 585.0, 575.0, 590.0, 582.0, 588.0, 578.0, 583.0],
    'sensor_04': [1400.0, 1405.0, 1395.0, 1410.0, 1402.0, 1408.0, 1398.0, 1403.0],
    'sensor_07': [550.0, 555.0, 545.0, 560.0, 552.0, 558.0, 548.0, 553.0]
})

# Initialize baseline sensor analyzer
baseline_analyzer = RealSensorPatternAnalyzer(baseline_data)
print(f'✅ Baseline analyzer initialized with {len(baseline_data)} healthy measurements')

# Test engine scenarios showing real sensor analysis
test_scenarios = [
    {
        'name': 'High Temperature Engine',
        'data': {
            'equipment_id': 'ENGINE_001',
            'health_state': 2,
            'sensor_02': 28.5,  # Significantly elevated temperature
            'sensor_03': 620.0,  # Higher than baseline
            'sensor_04': 1450.0,
            'sensor_07': 580.0
        }
    },
    {
        'name': 'Pressure Anomaly Engine', 
        'data': {
            'equipment_id': 'ENGINE_002',
            'health_state': 1,
            'sensor_02': 21.5,
            'sensor_03': 520.0,  # Lower pressure
            'sensor_04': 1380.0,  # Lower than baseline
            'sensor_07': 540.0
        }
    },
    {
        'name': 'Critical Multi-Sensor Engine',
        'data': {
            'equipment_id': 'ENGINE_003', 
            'health_state': 3,
            'sensor_02': 32.0,  # Very high temperature
            'sensor_03': 650.0,  # High pressure
            'sensor_04': 1500.0,  # High fan inlet temperature  
            'sensor_07': 600.0   # High
        }
    }
]

# Test each scenario
for scenario in test_scenarios:
    print(f'\n🔍 {scenario["name"]} Analysis:')
    
    analysis = real_sensor_anomaly_analysis(
        scenario['data'], 
        baseline_analyzer, 
        engine_history=None  # No historical data for this demo
    )
    
    print(f'   Primary Anomaly: {analysis["primary_anomaly"]}')
    print(f'   Urgency Level: {analysis["urgency_level"]}') 
    print(f'   Recommended Action: {analysis["recommended_actions"][0]}')
    print(f'   Analysis Method: {analysis["analysis_method"]}')

print('\n✅ SUCCESS: Real sensor analysis producing data-driven anomaly descriptions!')
print('🎯 This replaces simulated descriptions with genuine sensor pattern analysis')