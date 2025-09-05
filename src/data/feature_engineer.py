"""
Feature Engineering for Predictive Maintenance

This module provides functionality to engineer features from raw sensor data
for Markov-based predictive maintenance models.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """
    Feature engineering class for predictive maintenance data.
    
    Handles RUL calculation, rolling features, health indicators,
    and state classification for Markov models.
    """
    
    def __init__(self, config: dict):
        """
        Initialize feature engineer with configuration.
        
        Args:
            config (dict): Configuration dictionary containing thresholds and parameters
        """
        self.config = config
        self.scaler = StandardScaler()
        self.fitted_scaler = False
    
    def calculate_rul_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate RUL (Remaining Useful Life) labels for training data.
        
        For each engine, RUL = max_cycle - current_cycle
        
        Args:
            df (pd.DataFrame): Training data with unit and cycle columns
            
        Returns:
            pd.DataFrame: Data with RUL column added
        """
        df = df.copy()
        
        # Calculate maximum cycle for each unit
        max_cycles = df.groupby('unit')['cycle'].max().reset_index()
        max_cycles.columns = ['unit', 'max_cycle']
        
        # Merge with original data
        df = df.merge(max_cycles, on='unit', how='left')
        
        # Calculate RUL
        df['RUL'] = df['max_cycle'] - df['cycle']
        
        # Drop the temporary max_cycle column
        df = df.drop('max_cycle', axis=1)
        
        return df
    
    def create_rolling_features(self, df: pd.DataFrame, windows: List[int] = [5, 10, 20]) -> pd.DataFrame:
        """
        Create rolling window features for sensor measurements.
        
        Args:
            df (pd.DataFrame): Data with sensor columns
            windows (List[int]): List of window sizes for rolling calculations
            
        Returns:
            pd.DataFrame: Data with rolling features added
        """
        df = df.copy()
        
        # Get sensor columns
        sensor_columns = [col for col in df.columns if col.startswith('sensor_')]
        
        # Create rolling features for each sensor and window size
        for sensor in sensor_columns:
            for window in windows:
                # Rolling mean
                rolling_mean = (
                    df.groupby('unit')[sensor]
                    .rolling(window=window, min_periods=1)
                    .mean()
                )
                df[f'{sensor}_rolling_mean_{window}'] = rolling_mean.reset_index(level=0, drop=True)
                
                # Rolling standard deviation
                rolling_std = (
                    df.groupby('unit')[sensor]
                    .rolling(window=window, min_periods=1)
                    .std()
                )
                df[f'{sensor}_rolling_std_{window}'] = rolling_std.reset_index(level=0, drop=True)
                
                # Rolling minimum
                rolling_min = (
                    df.groupby('unit')[sensor]
                    .rolling(window=window, min_periods=1)
                    .min()
                )
                df[f'{sensor}_rolling_min_{window}'] = rolling_min.reset_index(level=0, drop=True)
                
                # Rolling maximum
                rolling_max = (
                    df.groupby('unit')[sensor]
                    .rolling(window=window, min_periods=1)
                    .max()
                )
                df[f'{sensor}_rolling_max_{window}'] = rolling_max.reset_index(level=0, drop=True)
        
        # Handle NaN values in rolling features
        rolling_columns = [col for col in df.columns if 'rolling' in col]
        for col in rolling_columns:
            # Fill NaN values with forward fill, then backward fill
            df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
            
            # If still NaN (shouldn't happen), fill with 0
            df[col] = df[col].fillna(0)
        
        return df
    
    def create_degradation_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create health indicators combining multiple sensors.
        
        Args:
            df (pd.DataFrame): Data with sensor measurements
            
        Returns:
            pd.DataFrame: Data with health indicator columns added
        """
        df = df.copy()
        
        # Get sensor groups from config
        temp_sensors = [f'sensor_{i:02d}' for i in self.config['sensors']['temperature_sensors']]
        pressure_sensors = [f'sensor_{i:02d}' for i in self.config['sensors']['pressure_sensors']]
        flow_sensors = [f'sensor_{i:02d}' for i in self.config['sensors']['flow_sensors']]
        
        # Temperature health index (normalized average)
        if temp_sensors:
            available_temp = [s for s in temp_sensors if s in df.columns]
            if available_temp:
                df['temp_health_index'] = df[available_temp].mean(axis=1)
                # Safe min-max normalization to avoid divide-by-zero
                temp_min = df['temp_health_index'].min()
                temp_max = df['temp_health_index'].max()
                denom = (temp_max - temp_min)
                if pd.isna(denom) or denom == 0:
                    # If constant, set to 0.0
                    df['temp_health_index'] = 0.0
                else:
                    df['temp_health_index'] = (df['temp_health_index'] - temp_min) / denom
        
        # Pressure health index
        if pressure_sensors:
            available_pressure = [s for s in pressure_sensors if s in df.columns]
            if available_pressure:
                df['pressure_health_index'] = df[available_pressure].mean(axis=1)
                press_min = df['pressure_health_index'].min()
                press_max = df['pressure_health_index'].max()
                denom = (press_max - press_min)
                if pd.isna(denom) or denom == 0:
                    df['pressure_health_index'] = 0.0
                else:
                    df['pressure_health_index'] = (df['pressure_health_index'] - press_min) / denom
        
        # Flow health index
        if flow_sensors:
            available_flow = [s for s in flow_sensors if s in df.columns]
            if available_flow:
                df['flow_health_index'] = df[available_flow].mean(axis=1)
                flow_min = df['flow_health_index'].min()
                flow_max = df['flow_health_index'].max()
                denom = (flow_max - flow_min)
                if pd.isna(denom) or denom == 0:
                    df['flow_health_index'] = 0.0
                else:
                    df['flow_health_index'] = (df['flow_health_index'] - flow_min) / denom
        
        # Overall health index (weighted combination)
        health_indicators = []
        if 'temp_health_index' in df.columns:
            health_indicators.append(df['temp_health_index'])
        if 'pressure_health_index' in df.columns:
            health_indicators.append(df['pressure_health_index'])
        if 'flow_health_index' in df.columns:
            health_indicators.append(df['flow_health_index'])
        
        if health_indicators:
            df['overall_health_index'] = pd.concat(health_indicators, axis=1).mean(axis=1)
        
        return df
    
    def classify_health_states(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify each time point into health states based on RUL.
        
        States: 0=Healthy, 1=Warning, 2=Critical, 3=Failure
        
        Args:
            df (pd.DataFrame): Data with RUL column
            
        Returns:
            pd.DataFrame: Data with health_state column added
        """
        df = df.copy()
        
        if 'RUL' not in df.columns:
            raise ValueError("RUL column not found. Run calculate_rul_labels first.")
        
        # Calculate RUL percentage for each engine
        max_rul = df.groupby('unit')['RUL'].max().reset_index()
        max_rul.columns = ['unit', 'max_rul']
        df = df.merge(max_rul, on='unit', how='left')
        df['rul_percentage'] = df['RUL'] / df['max_rul']
        
        # Classify states based on thresholds
        health_threshold = self.config['model']['health_threshold']
        warning_threshold = self.config['model']['warning_threshold']
        critical_threshold = self.config['model']['critical_threshold']
        
        conditions = [
            df['rul_percentage'] >= health_threshold,  # Healthy
            (df['rul_percentage'] >= warning_threshold) & (df['rul_percentage'] < health_threshold),  # Warning
            (df['rul_percentage'] >= critical_threshold) & (df['rul_percentage'] < warning_threshold),  # Critical
            df['rul_percentage'] < critical_threshold  # Failure
        ]
        
        choices = [0, 1, 2, 3]  # Healthy, Warning, Critical, Failure
        df['health_state'] = np.select(conditions, choices, default=0)
        
        # Drop temporary columns
        df = df.drop(['max_rul', 'rul_percentage'], axis=1)
        
        return df
    
    def normalize_features(self, df: pd.DataFrame, fit_scaler: bool = True) -> pd.DataFrame:
        """
        Apply StandardScaler to sensor measurements.
        
        Args:
            df (pd.DataFrame): Data with sensor columns
            fit_scaler (bool): Whether to fit the scaler (True for training data)
            
        Returns:
            pd.DataFrame: Data with normalized sensor columns
        """
        df = df.copy()
        
        # Get sensor columns
        sensor_columns = [col for col in df.columns if col.startswith('sensor_')]
        
        if not sensor_columns:
            return df
        
        # Fill NaNs in sensor columns prior to scaling (per series, preserving order)
        for col in sensor_columns:
            df[col] = df[col].fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Fit scaler on training data
        if fit_scaler:
            self.scaler.fit(df[sensor_columns])
            self.fitted_scaler = True
        
        # Transform data
        if self.fitted_scaler:
            normalized_data = self.scaler.transform(df[sensor_columns])
            normalized_df = pd.DataFrame(
                normalized_data, 
                columns=[f'{col}_norm' for col in sensor_columns],
                index=df.index
            )
            
            # Concatenate with original data
            df = pd.concat([df, normalized_df], axis=1)
            
            # Final safety: ensure no NaNs in normalized columns
            for col in normalized_df.columns:
                if df[col].isnull().any():
                    df[col] = df[col].fillna(0)
        
        return df
    
    def create_engineered_features(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """
        Complete feature engineering pipeline.
        
        Args:
            df (pd.DataFrame): Raw data
            is_training (bool): Whether this is training data (affects RUL calculation)
            
        Returns:
            pd.DataFrame: Fully engineered features
        """
        print("Starting feature engineering pipeline...")
        
        # Step 0: Handle any initial NaN values in raw data
        initial_nan_count = df.isnull().sum().sum()
        if initial_nan_count > 0:
            print(f"Found {initial_nan_count} NaN values in raw data, handling...")
            # Fill NaN values with forward fill, then backward fill
            df = df.fillna(method='ffill').fillna(method='bfill')
            # If still NaN, fill with 0
            df = df.fillna(0)
            print("✓ Initial NaN values handled")
        
        # Step 1: Calculate RUL labels (only for training data)
        if is_training:
            df = self.calculate_rul_labels(df)
            print("✓ RUL labels calculated")
        
        # Step 2: Create rolling features
        df = self.create_rolling_features(df)
        print("✓ Rolling features created")
        
        # Step 3: Create degradation indicators
        df = self.create_degradation_indicators(df)
        print("✓ Degradation indicators created")
        
        # Step 4: Classify health states (only for training data)
        if is_training and 'RUL' in df.columns:
            df = self.classify_health_states(df)
            print("✓ Health states classified")
        
        # Step 5: Normalize features
        df = self.normalize_features(df, fit_scaler=is_training)
        print("✓ Features normalized")
        
        # Step 6: Final NaN check and handling
        final_nan_count = df.isnull().sum().sum()
        if final_nan_count > 0:
            print(f"Found {final_nan_count} NaN values after feature engineering, handling...")
            # Fill any remaining NaN values with 0
            df = df.fillna(0)
            print("✓ Final NaN values handled")
        
        print("Feature engineering pipeline completed!")
        return df
    
    def get_feature_list(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Get categorized list of features.
        
        Args:
            df (pd.DataFrame): Engineered dataset
            
        Returns:
            Dict[str, List[str]]: Dictionary of feature categories
        """
        features = {
            'basic': ['unit', 'cycle'],
            'settings': [col for col in df.columns if col.startswith('setting')],
            'sensors': [col for col in df.columns if col.startswith('sensor_') and not col.endswith('_norm')],
            'sensors_normalized': [col for col in df.columns if col.endswith('_norm')],
            'rolling': [col for col in df.columns if 'rolling' in col],
            'health_indicators': [col for col in df.columns if 'health_index' in col],
            'targets': [col for col in df.columns if col in ['RUL', 'health_state']]
        }
        
        return features
