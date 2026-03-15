#!/usr/bin/env python3

import sys
import os
from pathlib import Path

# Add the src directory to the Python path
project_root = Path(__file__).parent
src_path = project_root / "src" 

print(f"Project root: {project_root}")
print(f"Src path: {src_path}")
print(f"Src exists: {src_path.exists()}")
print(f"Contents of src: {list(src_path.iterdir()) if src_path.exists() else 'Not found'}")

# Show Python path before
print(f"Python path before: {sys.path[:3]}")

# Add to path
sys.path.insert(0, str(src_path))
print(f"Python path after: {sys.path[:3]}")

print("\nTesting imports:")
print("-" * 40)

# Try the imports
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

try:
    from models.markov_model import MarkovChainRUL
    print("✅ MarkovChainRUL imported successfully")
except Exception as e:
    print(f"❌ MarkovChainRUL import failed: {e}")

# Check what's in data and models directories
print(f"\nContents of {src_path}/data/:")
data_path = src_path / "data"
if data_path.exists():
    for item in data_path.iterdir():
        print(f"  {item.name}")
else:
    print("  Data directory not found")

print(f"\nContents of {src_path}/models/:")
models_path = src_path / "models"
if models_path.exists():
    for item in models_path.iterdir():
        print(f"  {item.name}")
else:
    print("  Models directory not found")