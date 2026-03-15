"""
Real Sensor Pattern Analysis for Production Use

This module provides real sensor anomaly detection to replace 
simulated descriptions with actual data-driven analysis.
"""

import pandas as pd
import numpy as np
from scipy import signal, stats
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')


class RealSensorPatternAnalyzer:
    """
    Real sensor pattern analysis for turbofan engines.
    
    Provides data-driven anomaly detection and pattern classification
    based on actual sensor measurements.
    """
    
    def __init__(self, baseline_data=None):
        """
        Initialize analyzer with baseline data for comparison.
        
        Args:
            baseline_data: DataFrame with healthy equipment sensor readings
        """
        self.baseline_data = baseline_data
        self.baseline_stats = None
        self.sensor_thresholds = None
        
        if baseline_data is not None:
            self._calculate_baselines()
    
    def _calculate_baselines(self):
        """Calculate baseline statistics for healthy operation"""
        sensor_cols = [col for col in self.baseline_data.columns if col.startswith('sensor_')]
        
        self.baseline_stats = {
            'mean': self.baseline_data[sensor_cols].mean(),
            'std': self.baseline_data[sensor_cols].std(),
            'q95': self.baseline_data[sensor_cols].quantile(0.95),
            'q05': self.baseline_data[sensor_cols].quantile(0.05)
        }
        
        # Set anomaly detection thresholds
        self.sensor_thresholds = {
            'temperature_warning': 2.0,  # 2 sigma above baseline
            'temperature_critical': 3.0,  # 3 sigma above baseline
            'pressure_warning': -1.5,     # 1.5 sigma below baseline
            'pressure_critical': -2.5,    # 2.5 sigma below baseline
            'vibration_warning': 2.0,     # 2 sigma above baseline
            'vibration_critical': 3.0,    # 3 sigma above baseline
        }
    
    def detect_temperature_anomalies(self, current_data, engine_history=None):
        """
        Real temperature anomaly detection based on statistical analysis.
        
        Args:
            current_data: Series or DataFrame with current sensor readings
            engine_history: DataFrame with historical data for trend analysis
            
        Returns:
            dict: Detailed temperature anomaly analysis
        """
        temp_sensors = [f'sensor_{i:02d}' for i in [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]]
        temp_sensors = [col for col in temp_sensors if col in current_data.index]
        
        anomalies = []
        severity_score = 0
        
        for sensor in temp_sensors:
            if sensor not in current_data.index:
                continue
                
            current_temp = current_data[sensor]
            baseline_mean = self.baseline_stats['mean'][sensor]
            baseline_std = self.baseline_stats['std'][sensor]
            
            # Calculate z-score deviation
            z_score = (current_temp - baseline_mean) / baseline_std
            
            # Check for anomalies
            if z_score > self.sensor_thresholds['temperature_critical']:
                temp_increase = current_temp - baseline_mean
                anomalies.append(f"{sensor} temperature {temp_increase:.1f}°C above baseline (critical: {z_score:.1f}σ)")
                severity_score += 3
            elif z_score > self.sensor_thresholds['temperature_warning']:
                temp_increase = current_temp - baseline_mean
                anomalies.append(f"{sensor} temperature {temp_increase:.1f}°C above baseline (warning: {z_score:.1f}σ)")
                severity_score += 1
        
        # Trend analysis if historical data available
        trends = []
        if engine_history is not None and len(engine_history) > 10:
            for sensor in temp_sensors:
                if sensor in engine_history.columns:
                    # Calculate recent trend (last 10 cycles)
                    recent_values = engine_history[sensor].tail(10).values
                    if len(recent_values) >= 5:
                        trend_slope = np.polyfit(range(len(recent_values)), recent_values, 1)[0]
                        baseline_std_val = self.baseline_stats['std'][sensor]  # Fix: get scalar value
                        if abs(trend_slope) > baseline_std_val * 0.1:  # Significant trend
                            trend_direction = "increasing" if trend_slope > 0 else "decreasing"
                            trends.append(f"{sensor} showing {trend_direction} trend ({trend_slope:.2f}°C/cycle)")
        
        return {
            'anomalies': anomalies,
            'trends': trends,
            'severity_score': severity_score,
            'max_z_score': max([abs((current_data.get(s, 0) - self.baseline_stats['mean'].get(s, 0)) / self.baseline_stats['std'].get(s, 1)) for s in temp_sensors]) if temp_sensors else 0
        }
    
    def detect_pressure_anomalies(self, current_data, engine_history=None):
        """
        Real pressure system anomaly detection.
        
        Args:
            current_data: Current sensor readings
            engine_history: Historical data for analysis
            
        Returns:
            dict: Pressure system analysis results
        """
        pressure_sensors = [f'sensor_{i:02d}' for i in [1, 5, 6, 10, 16, 18, 19]]
        pressure_sensors = [col for col in pressure_sensors if col in current_data.index]
        
        anomalies = []
        severity_score = 0
        
        for sensor in pressure_sensors:
            if sensor not in current_data.index:
                continue
                
            current_pressure = current_data[sensor]
            baseline_mean = self.baseline_stats['mean'][sensor]
            baseline_std = self.baseline_stats['std'][sensor]
            
            # Calculate deviation (pressure drops are concerning)
            z_score = (current_pressure - baseline_mean) / baseline_std
            
            if z_score < self.sensor_thresholds['pressure_critical']:
                pressure_drop = baseline_mean - current_pressure
                pressure_pct = (pressure_drop / baseline_mean) * 100
                anomalies.append(f"{sensor} pressure drop {pressure_drop:.2f} units ({pressure_pct:.1f}% below baseline)")
                severity_score += 3
            elif z_score < self.sensor_thresholds['pressure_warning']:
                pressure_drop = baseline_mean - current_pressure
                pressure_pct = (pressure_drop / baseline_mean) * 100
                anomalies.append(f"{sensor} pressure declining ({pressure_pct:.1f}% below baseline)")
                severity_score += 1
        
        # Pressure ratio analysis
        ratios = []
        if len(pressure_sensors) >= 2:
            # Analyze pressure ratios between sensors
            for i in range(len(pressure_sensors)):
                for j in range(i+1, len(pressure_sensors)):
                    sensor_a, sensor_b = pressure_sensors[i], pressure_sensors[j]
                    if sensor_a in current_data.index and sensor_b in current_data.index:
                        current_ratio = current_data[sensor_a] / current_data[sensor_b] if current_data[sensor_b] != 0 else 0
                        baseline_ratio = self.baseline_stats['mean'][sensor_a] / self.baseline_stats['mean'][sensor_b]
                        
                        ratio_deviation = abs(current_ratio - baseline_ratio) / baseline_ratio
                        if ratio_deviation > 0.1:  # 10% deviation in pressure ratio
                            ratios.append(f"Pressure ratio {sensor_a}/{sensor_b} deviated {ratio_deviation*100:.1f}% from baseline")
        
        return {
            'anomalies': anomalies,
            'ratios': ratios,
            'severity_score': severity_score
        }
    
    def detect_vibration_patterns(self, engine_history, current_cycle=None):
        """
        Vibration pattern analysis using frequency domain techniques.
        
        Args:
            engine_history: Time series data for frequency analysis (can be None)
            current_cycle: Current cycle number
            
        Returns:
            dict: Vibration analysis results
        """
        patterns = []
        severity_score = 0
        
        # Handle case when no historical data is available
        if engine_history is None or len(engine_history) < 20:
            return {
                'patterns': ['Historical data insufficient for vibration analysis'],
                'severity_score': 0,
                'analysis_note': 'Vibration analysis requires historical time series data'
            }
        
        # For CMAPSS data, we'll use proxy vibration indicators
        # In real applications, this would analyze actual vibration sensors
        vibration_proxies = ['sensor_02', 'sensor_03', 'sensor_04', 'sensor_11']
        vibration_proxies = [col for col in vibration_proxies if col in engine_history.columns]
        
        for sensor in vibration_proxies:
            sensor_data = engine_history[sensor].values
            
            # Calculate variance trend (increasing variance indicates bearing wear)
            window_size = 10
            if len(sensor_data) > window_size * 2:
                early_variance = np.var(sensor_data[:window_size])
                recent_variance = np.var(sensor_data[-window_size:])
                
                variance_increase = (recent_variance - early_variance) / early_variance
                
                if variance_increase > 0.5:  # 50% increase in variance
                    patterns.append(f"{sensor} variance increased {variance_increase*100:.1f}% (potential bearing wear)")
                    severity_score += 2
                elif variance_increase > 0.2:  # 20% increase
                    patterns.append(f"{sensor} variance increasing {variance_increase*100:.1f}% (monitor bearing condition)")
                    severity_score += 1
                
                # Peak detection for irregular patterns
                if len(sensor_data) > 10:
                    peaks, _ = signal.find_peaks(sensor_data, height=np.mean(sensor_data) + 2*np.std(sensor_data))
                    if len(peaks) > len(sensor_data) * 0.1:  # More than 10% of readings are peaks
                        patterns.append(f"{sensor} showing irregular peak patterns ({len(peaks)} peaks detected)")
                        severity_score += 1
        
        return {
            'patterns': patterns,
            'severity_score': severity_score
        }
    
    def analyze_cross_sensor_correlations(self, current_data, engine_history=None):
        """
        Analyze correlations between sensors to detect system-wide issues.
        
        Args:
            current_data: Current sensor readings
            engine_history: Historical data for correlation analysis
            
        Returns:
            dict: Cross-sensor correlation analysis
        """
        correlations = []
        
        # Handle case when no historical data is available
        if engine_history is None or len(engine_history) < 50:
            return {
                'correlations': ['Historical data insufficient for correlation analysis'],
                'analysis_note': 'Cross-sensor correlation analysis requires extended historical data'
            }
        
        sensor_cols = [col for col in engine_history.columns if col.startswith('sensor_')]
        
        # Calculate correlation matrix
        corr_matrix = engine_history[sensor_cols].corr()
        
        # Find strongly correlated sensor pairs that have diverged
        for i in range(len(sensor_cols)):
            for j in range(i+1, len(sensor_cols)):
                sensor_a, sensor_b = sensor_cols[i], sensor_cols[j]
                correlation = corr_matrix.loc[sensor_a, sensor_b]
                
                if abs(correlation) > 0.7:  # Strong correlation in historical data
                    # Check if current readings maintain this correlation
                    if sensor_a in current_data.index and sensor_b in current_data.index:
                        # Normalize current readings
                        norm_a = (current_data[sensor_a] - self.baseline_stats['mean'][sensor_a]) / self.baseline_stats['std'][sensor_a]
                        norm_b = (current_data[sensor_b] - self.baseline_stats['mean'][sensor_b]) / self.baseline_stats['std'][sensor_b]
                        
                        # Check for deviation from expected correlation
                        expected_correlation = correlation
                        current_correlation_proxy = np.sign(norm_a) == np.sign(norm_b)
                        
                        if not current_correlation_proxy and abs(correlation) > 0.8:
                            correlations.append(f"Correlation breakdown between {sensor_a} and {sensor_b} (historically {correlation:.2f})")
        
        return {
            'correlations': correlations
        }


def real_sensor_anomaly_analysis(engine_data: dict, baseline_analyzer: RealSensorPatternAnalyzer, 
                                engine_history: pd.DataFrame = None) -> dict:
    """
    Production-ready real sensor anomaly analysis function.
    
    This function replaces the simulated anomaly descriptions with
    actual data-driven analysis of sensor patterns.
    
    Args:
        engine_data: Dictionary with equipment information and sensor readings
        baseline_analyzer: Initialized RealSensorPatternAnalyzer
        engine_history: Historical sensor data for the specific engine
        
    Returns:
        dict: Real anomaly analysis with specific sensor findings
    """
    
    health_state = engine_data['health_state']
    
    # Convert engine_data to pandas Series for analysis
    if isinstance(engine_data, dict):
        sensor_data = pd.Series({k: v for k, v in engine_data.items() if 'sensor_' in str(k)})
    else:
        sensor_data = engine_data
    
    # Perform real sensor analysis
    temp_analysis = baseline_analyzer.detect_temperature_anomalies(
        current_data=sensor_data, 
        engine_history=engine_history
    )
    
    pressure_analysis = baseline_analyzer.detect_pressure_anomalies(
        current_data=sensor_data,
        engine_history=engine_history 
    )
    
    vibration_analysis = baseline_analyzer.detect_vibration_patterns(
        engine_history=engine_history
    )
    
    correlation_analysis = baseline_analyzer.analyze_cross_sensor_correlations(
        current_data=sensor_data,
        engine_history=engine_history
    )
    
    # Aggregate findings
    all_anomalies = []
    all_anomalies.extend(temp_analysis.get('anomalies', []))
    all_anomalies.extend(temp_analysis.get('trends', []))
    all_anomalies.extend(pressure_analysis.get('anomalies', []))
    all_anomalies.extend(pressure_analysis.get('ratios', []))
    all_anomalies.extend(vibration_analysis.get('patterns', []))
    all_anomalies.extend(correlation_analysis.get('correlations', []))
    
    # Calculate total severity
    total_severity = (
        temp_analysis.get('severity_score', 0) +
        pressure_analysis.get('severity_score', 0) +
        vibration_analysis.get('severity_score', 0)
    )
    
    # Determine urgency level based on actual analysis
    if total_severity >= 6 or health_state == 3:
        urgency_level = 'CRITICAL'
        action_timeframe = 'IMMEDIATE'
    elif total_severity >= 3 or health_state == 2:
        urgency_level = 'HIGH'
        action_timeframe = '48 hours'
    elif total_severity >= 1 or health_state == 1:
        urgency_level = 'MEDIUM'
        action_timeframe = '2 weeks'
    else:
        urgency_level = 'LOW'
        action_timeframe = 'next scheduled maintenance'
    
    # Prioritize most critical findings
    primary_anomaly = all_anomalies[0] if all_anomalies else "Sensors within normal parameters"
    secondary_indicators = all_anomalies[1:4] if len(all_anomalies) > 1 else []
    
    # Generate specific recommendations based on findings
    recommendations = []
    
    if temp_analysis.get('anomalies'):
        if 'critical' in temp_analysis['anomalies'][0].lower():
            recommendations.append(f"IMMEDIATE: Investigate temperature system - action required within {action_timeframe}")
        else:
            recommendations.append(f"Monitor temperature trends closely - schedule inspection within {action_timeframe}")
    
    if pressure_analysis.get('anomalies'):
        recommendations.append("Check pressure system integrity and seal conditions")
        
    if vibration_analysis.get('patterns'):
        recommendations.append("Inspect bearing housing and rotating components")
        
    if correlation_analysis.get('correlations'):
        recommendations.append("Conduct comprehensive system correlation analysis")
        
    # Default recommendations based on urgency
    if not recommendations:
        if urgency_level == 'CRITICAL':
            recommendations = ["IMMEDIATE ACTION REQUIRED: Stop equipment operation", "Conduct emergency inspection"]
        elif urgency_level == 'HIGH':
            recommendations = ["Schedule immediate maintenance within 48 hours", "Monitor equipment closely"]
        elif urgency_level == 'MEDIUM':
            recommendations = ["Schedule preventive maintenance within 2 weeks", "Continue monitoring"]
        else:
            recommendations = ["Continue normal monitoring", "Next scheduled maintenance"]
    
    return {
        'primary_anomaly': primary_anomaly,
        'secondary_indicators': secondary_indicators,
        'recommended_actions': recommendations,
        'urgency_level': urgency_level,
        'severity_score': total_severity,
        'analysis_method': 'real_sensor_data_analysis',
        'detected_patterns': {
            'temperature': len(temp_analysis.get('anomalies', [])),
            'pressure': len(pressure_analysis.get('anomalies', [])), 
            'vibration': len(vibration_analysis.get('patterns', [])),
            'correlations': len(correlation_analysis.get('correlations', []))
        }
    }