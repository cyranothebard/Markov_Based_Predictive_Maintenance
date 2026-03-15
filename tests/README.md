# Test Suite Documentation

This directory contains comprehensive unit tests for the Markov-Based Predictive Maintenance project.

## 🧪 Test Structure

```
tests/
├── __init__.py                 # Test package initialization
├── conftest.py                 # Pytest configuration and fixtures
├── run_tests.py                # Test runner script
├── README.md                   # This documentation
├── test_data_loader.py         # Tests for data loading functionality
├── test_feature_engineer.py    # Tests for feature engineering
├── test_markov_model.py        # Tests for Markov Chain model
├── test_evaluation_metrics.py  # Tests for evaluation metrics
└── test_baseline_models.py     # Tests for baseline models
```

## 🚀 Running Tests

### Quick Start

```bash
# Run all tests
python -m pytest tests/

# Run with verbose output
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_markov_model.py -v

# Run specific test class
python -m pytest tests/test_markov_model.py::TestMarkovChainRUL -v

# Run specific test method
python -m pytest tests/test_markov_model.py::TestMarkovChainRUL::test_fit_basic -v
```

### Using the Test Runner Script

```bash
# Run all tests
python tests/run_tests.py

# Run specific test suite
python tests/run_tests.py --suite markov

# Run with coverage report
python tests/run_tests.py --coverage

# Run in parallel
python tests/run_tests.py --parallel

# Run specific test file
python tests/run_tests.py --file tests/test_markov_model.py
```

### Test Suites

- **`data`**: Data loading and preprocessing tests
- **`features`**: Feature engineering tests
- **`markov`**: Markov Chain model tests
- **`metrics`**: Evaluation metrics tests
- **`baseline`**: Baseline models tests
- **`all`**: All tests (default)

## 📊 Test Coverage

The test suite provides comprehensive coverage of:

### Data Loading (`test_data_loader.py`)
- ✅ CMAPSSLoader initialization
- ✅ Data file loading and validation
- ✅ Error handling for invalid files
- ✅ Data type inference
- ✅ NA value handling
- ✅ Memory usage optimization

### Feature Engineering (`test_feature_engineer.py`)
- ✅ Rolling features creation
- ✅ Degradation indicators
- ✅ Feature normalization
- ✅ Complete pipeline integration
- ✅ Edge case handling
- ✅ Performance testing

### Markov Model (`test_markov_model.py`)
- ✅ Model initialization and configuration
- ✅ Training with various data types
- ✅ RUL prediction functionality
- ✅ Transition matrix validation
- ✅ State means calculation
- ✅ Edge cases and error handling

### Evaluation Metrics (`test_evaluation_metrics.py`)
- ✅ RUL-specific metrics calculation
- ✅ Directional accuracy
- ✅ Prognostic horizon analysis
- ✅ Late prediction penalty
- ✅ Confidence intervals
- ✅ Model robustness evaluation
- ✅ Engine-level metrics

### Baseline Models (`test_baseline_models.py`)
- ✅ Random Forest training and prediction
- ✅ Linear Regression training and prediction
- ✅ LSTM training and prediction (when PyTorch available)
- ✅ Model comparison and evaluation
- ✅ Error handling and edge cases

## 🔧 Test Fixtures

The test suite includes comprehensive fixtures in `conftest.py`:

### Data Fixtures
- `sample_sensor_data`: Realistic sensor data with degradation patterns
- `sample_rul_data`: RUL values for testing
- `sample_health_states`: Health state classifications
- `sample_transition_matrix`: Markov transition matrix
- `sample_feature_matrix`: Feature matrix for model testing
- `sample_predictions`: Sample predictions for evaluation

### Utility Fixtures
- `temp_data_dir`: Temporary directory for test data files
- `sample_config`: Sample configuration for testing

## 🎯 Test Categories

### Unit Tests
- Test individual functions and methods
- Mock external dependencies
- Fast execution
- Isolated testing

### Integration Tests
- Test component interactions
- End-to-end workflows
- Real data processing
- Performance validation

### Edge Case Tests
- Empty data handling
- Invalid input validation
- Extreme value processing
- Error condition testing

## 📈 Performance Testing

The test suite includes performance tests to ensure:

- **Memory efficiency**: Large dataset processing
- **Speed optimization**: Feature engineering pipelines
- **Scalability**: Model training with various data sizes
- **Resource usage**: CPU and memory consumption

## 🐛 Error Handling Tests

Comprehensive error handling tests cover:

- **Invalid inputs**: Wrong data types, shapes, values
- **Missing data**: NA values, empty datasets
- **File I/O errors**: Missing files, permission issues
- **Model errors**: Training failures, prediction errors
- **Resource limits**: Memory exhaustion, timeout handling

## 🔍 Test Quality Metrics

### Coverage Targets
- **Line Coverage**: > 90%
- **Branch Coverage**: > 85%
- **Function Coverage**: > 95%

### Test Quality
- **Test Isolation**: Each test is independent
- **Deterministic**: Tests produce consistent results
- **Fast Execution**: Most tests complete in < 1 second
- **Clear Assertions**: Meaningful error messages

## 🚨 Continuous Integration

The test suite is designed for CI/CD integration:

```yaml
# Example GitHub Actions workflow
- name: Run Tests
  run: |
    python -m pytest tests/ --cov=src --cov-report=xml
    
- name: Upload Coverage
  uses: codecov/codecov-action@v3
  with:
    file: ./coverage.xml
```

## 📝 Writing New Tests

### Test Naming Convention
- Test files: `test_*.py`
- Test classes: `Test*`
- Test methods: `test_*`

### Test Structure
```python
def test_function_name_scenario():
    """Test description."""
    # Arrange
    input_data = create_test_data()
    
    # Act
    result = function_under_test(input_data)
    
    # Assert
    assert result is not None
    assert result.shape == expected_shape
```

### Best Practices
- Use descriptive test names
- Include docstrings explaining test purpose
- Test both success and failure cases
- Use fixtures for common test data
- Mock external dependencies
- Keep tests fast and isolated

## 🛠️ Troubleshooting

### Common Issues

#### Import Errors
```bash
# Ensure src is in Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

#### Missing Dependencies
```bash
# Install test dependencies
pip install -r requirements.txt
```

#### PyTorch Issues
```bash
# Skip LSTM tests if PyTorch not available
python -m pytest tests/ -m "not gpu"
```

#### Memory Issues
```bash
# Run tests with memory limit
python -m pytest tests/ --maxfail=1
```

### Debug Mode
```bash
# Run with debug output
python -m pytest tests/ -v -s --tb=long

# Run single test with debug
python -m pytest tests/test_markov_model.py::test_fit_basic -v -s
```

## 📚 Additional Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Python Testing Best Practices](https://docs.python.org/3/library/unittest.html)
- [Test-Driven Development](https://en.wikipedia.org/wiki/Test-driven_development)

---

**Test Suite Status**: ✅ Comprehensive coverage of all core modules  
**Last Updated**: [Current Date]  
**Maintainer**: Project Development Team

