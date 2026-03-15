"""
Comprehensive Repo Review Script

This script tests the core functionality of the Markov-based predictive maintenance project
to identify any non-functional AI-generated code.
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path

# Add the src directory to the Python path
project_root = Path(__file__).parent
src_path = project_root / "src" 
sys.path.insert(0, str(src_path))

def test_imports():
    """Test if all modules can be imported successfully"""
    print("🧪 Testing imports...")
    
    try:
        from models.markov_model import MarkovChainRUL
        print("✅ MarkovChainRUL imported successfully")
    except Exception as e:
        print(f"❌ MarkovChainRUL import failed: {e}")
    
    try:
        from data.data_loader import CMAPSSLoader
        print("✅ CMAPSSLoader imported successfully")
    except Exception as e:
        print(f"❌ CMAPSSLoader import failed: {e}")
    
    try:
        from data.feature_engineer import FeatureEngineer
        print("✅ FeatureEngineer imported successfully")
    except Exception as e:
        print(f"❌ FeatureEngineer import failed: {e}")

def test_markov_model_basic():
    """Test basic MarkovChainRUL functionality with synthetic data"""
    print("\n🧪 Testing MarkovChainRUL basic functionality...")
    
    try:
        from models.markov_model import MarkovChainRUL
        
        # Create test data
        np.random.seed(42)
        n_samples = 1000
        n_features = 10
        X = np.random.randn(n_samples, n_features)
        states = np.random.choice([0, 1, 2, 3], n_samples, p=[0.4, 0.3, 0.2, 0.1])
        
        # Test model creation
        model = MarkovChainRUL(n_states=4)
        print("✅ Model instance created")
        
        # Test fitting
        model.fit(X, states)
        print("✅ Model fitting completed")
        
        # Test prediction
        test_X = np.random.randn(5, n_features)
        predictions = [model.predict_rul(x, 100) for x in test_X]
        print(f"✅ Predictions generated: {predictions[:3]}")
        
        # Test transition matrix
        P = model.get_transition_probabilities()
        print(f"✅ Transition matrix shape: {P.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ MarkovChainRUL test failed: {e}")
        return False

def test_data_loader():
    """Test data loader with sample data"""
    print("\n🧪 Testing CMAPSSLoader...")
    
    try:
        from data.data_loader import CMAPSSLoader
        
        # Test initialization - now auto-detects data location
        loader = CMAPSSLoader()
        print("✅ CMAPSSLoader instance created")
        
        # Test column names
        columns = loader._define_column_names()
        expected_length = 2 + 3 + 21  # unit, cycle, 3 settings, 21 sensors
        if len(columns) == expected_length:
            print(f"✅ Column names correct length: {len(columns)}")
        else:
            print(f"❌ Column names wrong length: {len(columns)}, expected {expected_length}")
        
        return True
        
    except Exception as e:
        print(f"❌ CMAPSSLoader test failed: {e}")
        return False

def test_feature_engineering():
    """Test feature engineering functionality"""
    print("\n🧪 Testing FeatureEngineer...")
    
    try:
        from data.feature_engineer import FeatureEngineer
        
        # Test initialization with proper config structure (matches what the class expects)
        config = {
            'model': {
                'health_threshold': 0.8,
                'warning_threshold': 0.6,
                'critical_threshold': 0.4
            },
            'sensors': {
                'temperature_sensors': [1, 2],
                'pressure_sensors': [1],
                'flow_sensors': [1]
            }
        }
        
        engineer = FeatureEngineer(config)
        print("✅ FeatureEngineer instance created")
        
        # Create sample dataframe
        df = pd.DataFrame({
            'unit': [1, 1, 1, 2, 2, 2],
            'cycle': [1, 2, 3, 1, 2, 3],
            'sensor_01': [0.5, 0.6, 0.7, 0.4, 0.5, 0.6],
            'sensor_02': [0.1, 0.2, 0.3, 0.1, 0.2, 0.25]
        })
        
        # Test RUL calculation (note: class uses 'RUL' uppercase)
        df_with_rul = engineer.calculate_rul_labels(df)
        if 'RUL' in df_with_rul.columns:
            print("✅ RUL labels calculated")
            print(f"    Sample RUL values: {df_with_rul['RUL'].tolist()}")
        else:
            print(f"❌ RUL labels not found. Available columns: {df_with_rul.columns.tolist()}")
            
        # Test health state classification
        df_with_states = engineer.classify_health_states(df_with_rul)
        if 'health_state' in df_with_states.columns:
            print("✅ Health states classified")
            print(f"    Sample states: {df_with_states['health_state'].tolist()}")
        else:
            print("❌ Health states not found")
        
        return True
        
    except Exception as e:
        print(f"❌ FeatureEngineer test failed: {e}")
        import traceback
        print(f"    Full error: {traceback.format_exc()}")
        return False

def check_data_directory():
    """Check if data directory exists and what's in it"""
    print("\n🧪 Checking data directory...")
    
    data_dir = project_root / "data"
    if data_dir.exists():
        print(f"✅ Data directory exists: {data_dir}")
        
        # List contents
        for item in data_dir.iterdir():
            if item.is_file():
                print(f"  📄 {item.name}")
            elif item.is_dir():
                print(f"  📁 {item.name}/")
                # List contents of subdirectories
                for subitem in item.iterdir():
                    print(f"    📄 {subitem.name}")
    else:
        print("❌ Data directory not found")

def run_all_tests():
    """Run all functionality tests"""
    print("🚀 Starting comprehensive repo review...")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_markov_model_basic,
        test_data_loader,
        test_feature_engineering,
        check_data_directory
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    print("📊 REVIEW SUMMARY:")
    print("=" * 60)
    
    passed = sum(1 for r in results if r)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Code appears functional.")
    elif passed >= total * 0.7:
        print("⚠️  Most functionality works, minor issues to address.")
    else:
        print("🚨 Major functionality issues detected!")

if __name__ == "__main__":
    run_all_tests()