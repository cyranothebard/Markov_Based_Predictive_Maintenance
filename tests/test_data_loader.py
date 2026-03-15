"""
Unit tests for data loading module.

Tests the CMAPSSLoader class and data loading functionality.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os

from data.data_loader import CMAPSSLoader


class TestCMAPSSLoader:
    """Test cases for CMAPSSLoader class."""
    
    def test_init(self):
        """Test CMAPSSLoader initialization."""
        loader = CMAPSSLoader()
        assert loader is not None
        assert hasattr(loader, 'load_data')
    
    def test_load_data_file_not_found(self):
        """Test load_data with non-existent file."""
        loader = CMAPSSLoader()
        
        with pytest.raises(FileNotFoundError):
            loader.load_data('non_existent_file.txt')
    
    def test_load_data_invalid_format(self, temp_data_dir):
        """Test load_data with invalid file format."""
        loader = CMAPSSLoader()
        
        # Create invalid data file
        invalid_file = temp_data_dir / 'invalid.txt'
        with open(invalid_file, 'w') as f:
            f.write("invalid,data,format\n")
            f.write("1,2,3\n")
        
        with pytest.raises((ValueError, pd.errors.EmptyDataError)):
            loader.load_data(str(invalid_file))
    
    def test_load_data_valid_format(self, temp_data_dir):
        """Test load_data with valid CMAPSS format."""
        loader = CMAPSSLoader()
        
        # Create valid CMAPSS data file
        valid_file = temp_data_dir / 'valid.txt'
        with open(valid_file, 'w') as f:
            # Write header
            f.write("engine_id,cycle,sensor_1,sensor_2,sensor_3,sensor_4,sensor_5,sensor_6,sensor_7,sensor_8,sensor_9,sensor_10,sensor_11,sensor_12,sensor_13,sensor_14,sensor_15,sensor_16,sensor_17,sensor_18,sensor_19,sensor_20,sensor_21\n")
            # Write sample data
            for engine_id in range(1, 4):
                for cycle in range(1, 11):
                    sensor_values = [np.random.randn() for _ in range(21)]
                    f.write(f"{engine_id},{cycle}," + ",".join([f"{val:.6f}" for val in sensor_values]) + "\n")
        
        # Load data
        df = loader.load_data(str(valid_file))
        
        # Verify structure
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 30  # 3 engines * 10 cycles
        assert 'engine_id' in df.columns
        assert 'cycle' in df.columns
        assert len([col for col in df.columns if col.startswith('sensor_')]) == 21
    
    def test_load_data_with_na_values(self, temp_data_dir):
        """Test load_data with NA values in the data."""
        loader = CMAPSSLoader()
        
        # Create data file with NA values
        na_file = temp_data_dir / 'na_data.txt'
        with open(na_file, 'w') as f:
            f.write("engine_id,cycle,sensor_1,sensor_2,sensor_3\n")
            f.write("1,1,1.0,2.0,3.0\n")
            f.write("1,2,NA,2.5,3.5\n")  # NA value
            f.write("1,3,1.5,3.0,4.0\n")
        
        # Load data
        df = loader.load_data(str(na_file))
        
        # Verify NA handling
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert df['sensor_1'].isna().any()  # Should have NA values
    
    def test_load_data_column_names(self, temp_data_dir):
        """Test that column names are correctly assigned."""
        loader = CMAPSSLoader()
        
        # Create data file
        data_file = temp_data_dir / 'column_test.txt'
        with open(data_file, 'w') as f:
            f.write("engine_id,cycle,sensor_1,sensor_2,sensor_3\n")
            f.write("1,1,1.0,2.0,3.0\n")
            f.write("1,2,1.5,2.5,3.5\n")
        
        # Load data
        df = loader.load_data(str(data_file))
        
        # Verify column names
        expected_columns = ['engine_id', 'cycle', 'sensor_1', 'sensor_2', 'sensor_3']
        assert list(df.columns) == expected_columns
    
    def test_load_data_data_types(self, temp_data_dir):
        """Test that data types are correctly inferred."""
        loader = CMAPSSLoader()
        
        # Create data file
        data_file = temp_data_dir / 'dtype_test.txt'
        with open(data_file, 'w') as f:
            f.write("engine_id,cycle,sensor_1,sensor_2\n")
            f.write("1,1,1.0,2.0\n")
            f.write("2,2,1.5,2.5\n")
        
        # Load data
        df = loader.load_data(str(data_file))
        
        # Verify data types
        assert df['engine_id'].dtype in ['int64', 'int32']
        assert df['cycle'].dtype in ['int64', 'int32']
        assert df['sensor_1'].dtype in ['float64', 'float32']
        assert df['sensor_2'].dtype in ['float64', 'float32']
    
    def test_load_data_empty_file(self, temp_data_dir):
        """Test load_data with empty file."""
        loader = CMAPSSLoader()
        
        # Create empty file
        empty_file = temp_data_dir / 'empty.txt'
        empty_file.touch()
        
        with pytest.raises(pd.errors.EmptyDataError):
            loader.load_data(str(empty_file))
    
    def test_load_data_single_row(self, temp_data_dir):
        """Test load_data with single row of data."""
        loader = CMAPSSLoader()
        
        # Create single row file
        single_file = temp_data_dir / 'single.txt'
        with open(single_file, 'w') as f:
            f.write("engine_id,cycle,sensor_1,sensor_2\n")
            f.write("1,1,1.0,2.0\n")
        
        # Load data
        df = loader.load_data(str(single_file))
        
        # Verify single row
        assert len(df) == 1
        assert df.iloc[0]['engine_id'] == 1
        assert df.iloc[0]['cycle'] == 1
        assert df.iloc[0]['sensor_1'] == 1.0
        assert df.iloc[0]['sensor_2'] == 2.0


class TestDataLoaderIntegration:
    """Integration tests for data loading functionality."""
    
    def test_load_multiple_files(self, temp_data_dir):
        """Test loading multiple data files."""
        loader = CMAPSSLoader()
        
        # Create multiple data files
        files = []
        for i in range(3):
            file_path = temp_data_dir / f'data_{i}.txt'
            with open(file_path, 'w') as f:
                f.write("engine_id,cycle,sensor_1,sensor_2\n")
                for engine_id in range(1, 3):
                    for cycle in range(1, 4):
                        f.write(f"{engine_id},{cycle},{i+1}.0,{i+2}.0\n")
            files.append(str(file_path))
        
        # Load each file
        dataframes = []
        for file_path in files:
            df = loader.load_data(file_path)
            dataframes.append(df)
        
        # Verify all files loaded correctly
        assert len(dataframes) == 3
        for df in dataframes:
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 6  # 2 engines * 3 cycles
    
    def test_load_data_memory_usage(self, temp_data_dir):
        """Test that data loading doesn't consume excessive memory."""
        loader = CMAPSSLoader()
        
        # Create large data file
        large_file = temp_data_dir / 'large.txt'
        with open(large_file, 'w') as f:
            f.write("engine_id,cycle,sensor_1,sensor_2,sensor_3\n")
            for engine_id in range(1, 101):  # 100 engines
                for cycle in range(1, 101):  # 100 cycles each
                    f.write(f"{engine_id},{cycle},{engine_id}.0,{cycle}.0,{engine_id+cycle}.0\n")
        
        # Load data
        df = loader.load_data(str(large_file))
        
        # Verify data loaded correctly
        assert len(df) == 10000  # 100 engines * 100 cycles
        assert df.memory_usage(deep=True).sum() < 100 * 1024 * 1024  # Less than 100MB

