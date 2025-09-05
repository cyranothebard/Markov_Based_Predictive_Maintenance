"""
NASA CMAPSS Dataset Loader for Predictive Maintenance

This module provides functionality to load and validate NASA's CMAPSS
(Commercial Modular Aero-Propulsion System Simulation) dataset for
turbofan engine degradation modeling.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
import os


class CMAPSSLoader:
    """
    Loader class for NASA CMAPSS turbofan engine dataset.
    
    Handles loading of training data, test data, and RUL labels
    with proper column naming and data validation.
    """
    
    def __init__(self, data_path: str):
        """
        Initialize the CMAPSS data loader.
        
        Args:
            data_path (str): Path to the directory containing CMAPSS data files
        """
        self.data_path = data_path
        self.column_names = self._define_column_names()
    
    def _define_column_names(self) -> List[str]:
        """
        Define column names for CMAPSS dataset.
        
        Returns:
            List[str]: List of column names for the dataset
        """
        # Operational settings: altitude, Mach number, throttle
        setting_columns = ['setting1', 'setting2', 'setting3']
        
        # Sensor measurements (21 sensors)
        sensor_columns = [f'sensor_{i:02d}' for i in range(1, 22)]
        
        # Unit number and time cycles
        return ['unit', 'cycle'] + setting_columns + sensor_columns
    
    def load_train_data(self, dataset_name: str) -> pd.DataFrame:
        """
        Load training data for specified dataset.
        
        Args:
            dataset_name (str): Dataset identifier (e.g., 'FD001', 'FD002')
            
        Returns:
            pd.DataFrame: Training data with proper column names
            
        Raises:
            FileNotFoundError: If training data file is not found
        """
        filename = f"train_{dataset_name}.txt"
        filepath = os.path.join(self.data_path, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Training data file not found: {filepath}")
        
        # Load data without header (CMAPSS files have no column names)
        df = pd.read_csv(filepath, sep=' ', header=None)
        
        # Remove any trailing whitespace columns (columns that are all NaN or 0)
        df = df.loc[:, (df != 0).any(axis=0)]
        df = df.dropna(axis=1, how='all')
        
        # Ensure we have exactly 26 columns as expected
        if len(df.columns) > 26:
            df = df.iloc[:, :26]
        elif len(df.columns) < 26:
            # Pad with zeros if we have fewer columns
            for i in range(len(df.columns), 26):
                df[f'col_{i}'] = 0
        
        # Assign column names
        df.columns = self.column_names
        
        return df
    
    def load_test_data(self, dataset_name: str) -> pd.DataFrame:
        """
        Load test data for specified dataset.
        
        Args:
            dataset_name (str): Dataset identifier (e.g., 'FD001', 'FD002')
            
        Returns:
            pd.DataFrame: Test data with proper column names
            
        Raises:
            FileNotFoundError: If test data file is not found
        """
        filename = f"test_{dataset_name}.txt"
        filepath = os.path.join(self.data_path, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Test data file not found: {filepath}")
        
        # Load data without header
        df = pd.read_csv(filepath, sep=' ', header=None)
        
        # Remove any trailing whitespace columns (columns that are all NaN or 0)
        df = df.loc[:, (df != 0).any(axis=0)]
        df = df.dropna(axis=1, how='all')
        
        # Ensure we have exactly 26 columns as expected
        if len(df.columns) > 26:
            df = df.iloc[:, :26]
        elif len(df.columns) < 26:
            # Pad with zeros if we have fewer columns
            for i in range(len(df.columns), 26):
                df[f'col_{i}'] = 0
        
        # Assign column names
        df.columns = self.column_names
        
        return df
    
    def load_rul_labels(self, dataset_name: str) -> pd.DataFrame:
        """
        Load RUL (Remaining Useful Life) labels for test data.
        
        Args:
            dataset_name (str): Dataset identifier (e.g., 'FD001', 'FD002')
            
        Returns:
            pd.DataFrame: RUL labels with unit numbers
            
        Raises:
            FileNotFoundError: If RUL labels file is not found
        """
        filename = f"RUL_{dataset_name}.txt"
        filepath = os.path.join(self.data_path, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"RUL labels file not found: {filepath}")
        
        # Load RUL values
        rul_values = pd.read_csv(filepath, header=None, names=['RUL'])
        
        # Add unit numbers (1 to number of test engines)
        rul_values['unit'] = range(1, len(rul_values) + 1)
        
        return rul_values
    
    def validate_data_integrity(self, df: pd.DataFrame) -> bool:
        """
        Validate data integrity for loaded dataset.
        
        Args:
            df (pd.DataFrame): Dataset to validate
            
        Returns:
            bool: True if data passes validation, False otherwise
        """
        try:
            # Check for missing values
            if df.isnull().any().any():
                print("Warning: Dataset contains missing values")
                return False
            
            # Check data types
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) != len(df.columns):
                print("Warning: Non-numeric columns found in sensor data")
                return False
            
            # Check for negative values in sensors (should be positive)
            sensor_columns = [col for col in df.columns if col.startswith('sensor_')]
            for col in sensor_columns:
                if (df[col] < 0).any():
                    print(f"Warning: Negative values found in {col}")
                    return False
            
            # Check unit and cycle ranges
            if 'unit' in df.columns:
                if df['unit'].min() < 1:
                    print("Warning: Unit numbers should start from 1")
                    return False
            
            if 'cycle' in df.columns:
                if df['cycle'].min() < 1:
                    print("Warning: Cycle numbers should start from 1")
                    return False
            
            print("Data validation passed successfully")
            return True
            
        except Exception as e:
            print(f"Data validation failed: {str(e)}")
            return False
    
    def get_dataset_info(self, df: pd.DataFrame) -> dict:
        """
        Get basic information about the dataset.
        
        Args:
            df (pd.DataFrame): Dataset to analyze
            
        Returns:
            dict: Dictionary containing dataset statistics
        """
        info = {
            'total_engines': df['unit'].nunique() if 'unit' in df.columns else 0,
            'total_cycles': len(df),
            'avg_cycles_per_engine': df.groupby('unit').size().mean() if 'unit' in df.columns else 0,
            'min_cycles': df.groupby('unit').size().min() if 'unit' in df.columns else 0,
            'max_cycles': df.groupby('unit').size().max() if 'unit' in df.columns else 0,
            'sensor_columns': len([col for col in df.columns if col.startswith('sensor_')]),
            'setting_columns': len([col for col in df.columns if col.startswith('setting')])
        }
        
        return info
