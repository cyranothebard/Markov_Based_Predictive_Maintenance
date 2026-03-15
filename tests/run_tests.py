#!/usr/bin/env python3
"""
Test runner script for the Markov-Based Predictive Maintenance project.

This script provides a convenient way to run all tests with different configurations.
"""

import sys
import subprocess
import argparse
from pathlib import Path


def run_tests(test_path=None, verbose=False, coverage=False, parallel=False):
    """
    Run tests with specified configuration.
    
    Args:
        test_path (str): Specific test file or directory to run
        verbose (bool): Run tests in verbose mode
        coverage (bool): Generate coverage report
        parallel (bool): Run tests in parallel
    """
    # Base pytest command
    cmd = ["python", "-m", "pytest"]
    
    # Add test path
    if test_path:
        cmd.append(test_path)
    else:
        cmd.append("tests/")
    
    # Add verbosity
    if verbose:
        cmd.append("-v")
    
    # Add coverage
    if coverage:
        cmd.extend(["--cov=src", "--cov-report=html", "--cov-report=term"])
    
    # Add parallel execution
    if parallel:
        cmd.extend(["-n", "auto"])
    
    # Add other useful options
    cmd.extend([
        "--tb=short",  # Shorter traceback format
        "--strict-markers",  # Strict marker checking
        "--disable-warnings",  # Disable warnings for cleaner output
    ])
    
    print(f"Running command: {' '.join(cmd)}")
    print("=" * 60)
    
    # Run tests
    try:
        result = subprocess.run(cmd, check=True)
        print("=" * 60)
        print("✅ All tests passed!")
        return 0
    except subprocess.CalledProcessError as e:
        print("=" * 60)
        print(f"❌ Tests failed with exit code {e.returncode}")
        return e.returncode


def run_specific_test_suite(suite_name):
    """Run a specific test suite."""
    test_files = {
        "data": "tests/test_data_loader.py",
        "features": "tests/test_feature_engineer.py",
        "markov": "tests/test_markov_model.py",
        "metrics": "tests/test_evaluation_metrics.py",
        "baseline": "tests/test_baseline_models.py",
        "all": "tests/"
    }
    
    if suite_name not in test_files:
        print(f"❌ Unknown test suite: {suite_name}")
        print(f"Available suites: {', '.join(test_files.keys())}")
        return 1
    
    return run_tests(test_files[suite_name], verbose=True)


def main():
    """Main function to handle command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run tests for the Markov-Based Predictive Maintenance project"
    )
    
    parser.add_argument(
        "--suite", 
        choices=["data", "features", "markov", "metrics", "baseline", "all"],
        default="all",
        help="Test suite to run (default: all)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Run tests in verbose mode"
    )
    
    parser.add_argument(
        "--coverage", "-c",
        action="store_true",
        help="Generate coverage report"
    )
    
    parser.add_argument(
        "--parallel", "-p",
        action="store_true",
        help="Run tests in parallel"
    )
    
    parser.add_argument(
        "--file", "-f",
        help="Run specific test file"
    )
    
    args = parser.parse_args()
    
    # Check if we're in the right directory
    if not Path("tests").exists():
        print("❌ Error: tests directory not found. Please run from project root.")
        return 1
    
    # Run tests
    if args.file:
        return run_tests(args.file, args.verbose, args.coverage, args.parallel)
    else:
        return run_specific_test_suite(args.suite)


if __name__ == "__main__":
    sys.exit(main())

