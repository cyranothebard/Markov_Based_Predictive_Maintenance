#!/usr/bin/env python3
"""
Data Download Script for NASA CMAPSS Dataset

This script downloads the NASA CMAPSS (Commercial Modular Aero-Propulsion System Simulation) 
dataset for predictive maintenance research.

Dataset URL: https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository
"""

import os
import urllib.request
import zipfile
from pathlib import Path

def download_cmapss_data():
    """Download and extract NASA CMAPSS dataset"""
    
    # Create data directory
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Dataset URLs (these are example URLs - replace with actual NASA CMAPSS URLs)
    dataset_urls = {
        "CMAPSS_Data.zip": "https://ti.arc.nasa.gov/c/6/",
        "Damage_Propagation_Modeling.pdf": "https://ti.arc.nasa.gov/m/project/prognostic-data-repository/CMAPSSData.zip"
    }
    
    print("Downloading NASA CMAPSS dataset...")
    print("Note: This script provides the structure for downloading the dataset.")
    print("Please visit https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository")
    print("to download the actual dataset files.")
    
    # Create a README file with download instructions
    readme_content = """
# NASA CMAPSS Dataset Download Instructions

## Dataset Information
- **Name**: Commercial Modular Aero-Propulsion System Simulation (CMAPSS)
- **Source**: NASA Prognostics Center of Excellence
- **Purpose**: Turbofan engine degradation simulation data for predictive maintenance

## Download Instructions

1. Visit the NASA Prognostics Center of Excellence data repository:
   https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository

2. Download the CMAPSS dataset files:
   - train_FD001.txt
   - train_FD002.txt  
   - train_FD003.txt
   - train_FD004.txt
   - test_FD001.txt
   - test_FD002.txt
   - test_FD003.txt
   - test_FD004.txt
   - RUL_FD001.txt
   - RUL_FD002.txt
   - RUL_FD003.txt
   - RUL_FD004.txt
   - readme.txt
   - Damage Propagation Modeling.pdf

3. Place all files in the `data/raw/` directory

## File Descriptions

### Training Data (train_FD*.txt)
- Contains sensor measurements for training engines
- Each row represents one cycle of operation
- Columns: engine ID, cycle number, 3 operational settings, 21 sensor measurements

### Test Data (test_FD*.txt)  
- Contains sensor measurements for test engines
- Same format as training data
- Used for final model evaluation

### RUL Data (RUL_FD*.txt)
- Contains actual remaining useful life for test engines
- Used for model evaluation and performance metrics

### Documentation
- readme.txt: Detailed dataset description
- Damage Propagation Modeling.pdf: Technical documentation

## Usage

After downloading the files, you can run the analysis notebooks:
- `notebooks/01-data-exploration.ipynb`
- `notebooks/02-markov-modeling.ipynb`
- `notebooks/03-business-case-analysis.ipynb`

## Citation

If you use this dataset in your research, please cite:

Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008). Damage propagation modeling for aircraft engine run-to-failure simulation. In International Conference on Prognostics and Health Management (pp. 1-9). IEEE.
"""
    
    with open(data_dir / "DOWNLOAD_INSTRUCTIONS.md", "w") as f:
        f.write(readme_content)
    
    print(f"\nDownload instructions saved to: {data_dir / 'DOWNLOAD_INSTRUCTIONS.md'}")
    print("\nPlease follow the instructions to download the dataset files manually.")
    print("The dataset is free to use for research purposes.")

if __name__ == "__main__":
    download_cmapss_data()
