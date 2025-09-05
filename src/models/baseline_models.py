"""
Baseline Models for RUL Prediction Comparison

This module implements baseline models (Random Forest, LSTM, Linear Regression)
for comparing performance against Markov-based approaches.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')


class LSTMModel(nn.Module):
    """
    PyTorch LSTM model for RUL prediction.
    """
    
    def __init__(self, input_size: int, hidden_size: int = 50, 
                 num_layers: int = 2, output_size: int = 1, dropout: float = 0.2):
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout)
        
        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        # LSTM forward pass
        out, _ = self.lstm(x, (h0, c0))
        
        # Take the last output
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        
        return out


class BaselineModels:
    """
    Collection of baseline models for RUL prediction comparison.
    
    Includes Random Forest, LSTM, and Linear Regression models
    for benchmarking Markov-based approaches.
    """
    
    def __init__(self):
        """Initialize baseline models collection."""
        self.models = {}
        self.model_history = {}
        
    def train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray, 
                          X_test: np.ndarray = None, y_test: np.ndarray = None) -> RandomForestRegressor:
        """
        Train Random Forest model for RUL prediction.
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training RUL labels
            X_test (np.ndarray): Test features (optional)
            y_test (np.ndarray): Test RUL labels (optional)
            
        Returns:
            RandomForestRegressor: Trained Random Forest model
        """
        print("Training Random Forest model...")
        
        # Initialize Random Forest with specified parameters
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        # Train the model
        rf_model.fit(X_train, y_train)
        
        # Store the model
        self.models['random_forest'] = rf_model
        
        # Evaluate if test data provided
        if X_test is not None and y_test is not None:
            y_pred = rf_model.predict(X_test)
            metrics = self._calculate_metrics(y_test, y_pred)
            self.model_history['random_forest'] = metrics
            print(f"✓ Random Forest trained - RMSE: {metrics['rmse']:.2f}, R²: {metrics['r2_score']:.3f}")
        else:
            print("✓ Random Forest trained")
        
        return rf_model
    
    def train_lstm(self, X_train: np.ndarray, y_train: np.ndarray, 
                   sequence_length: int = 20, X_test: np.ndarray = None, 
                   y_test: np.ndarray = None) -> 'LSTMModel':
        """
        Train LSTM model for RUL prediction using PyTorch.
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training RUL labels
            sequence_length (int): Length of input sequences
            X_test (np.ndarray): Test features (optional)
            y_test (np.ndarray): Test RUL labels (optional)
            
        Returns:
            LSTMModel: Trained PyTorch LSTM model
        """
        print("Training LSTM model with PyTorch...")
        
        # Reshape data for LSTM input (samples, timesteps, features)
        X_train_reshaped = self._create_sequences(X_train, sequence_length)
        y_train_reshaped = y_train[sequence_length-1:]  # Adjust target length
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train_reshaped)
        y_train_tensor = torch.FloatTensor(y_train_reshaped).unsqueeze(1)
        
        # Create PyTorch LSTM model
        lstm_model = LSTMModel(
            input_size=X_train.shape[1],
            hidden_size=50,
            num_layers=2,
            output_size=1
        )
        
        # Set up training
        criterion = nn.MSELoss()
        optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)
        
        # Create data loader
        dataset = TensorDataset(X_train_tensor, y_train_tensor)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Training loop
        lstm_model.train()
        train_losses = []
        
        for epoch in range(50):
            epoch_loss = 0.0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = lstm_model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(dataloader)
            train_losses.append(avg_loss)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")
        
        # Store the model and history
        self.models['lstm'] = lstm_model
        self.model_history['lstm'] = {'loss': train_losses}
        
        # Evaluate if test data provided
        if X_test is not None and y_test is not None:
            X_test_reshaped = self._create_sequences(X_test, sequence_length)
            y_test_reshaped = y_test[sequence_length-1:]
            
            X_test_tensor = torch.FloatTensor(X_test_reshaped)
            lstm_model.eval()
            with torch.no_grad():
                y_pred_tensor = lstm_model(X_test_tensor)
                y_pred = y_pred_tensor.numpy().flatten()
            
            metrics = self._calculate_metrics(y_test_reshaped, y_pred)
            self.model_history['lstm_metrics'] = metrics
            print(f"✓ LSTM trained - RMSE: {metrics['rmse']:.2f}, R²: {metrics['r2_score']:.3f}")
        else:
            print("✓ LSTM trained")
        
        return lstm_model
    
    def train_linear_regression(self, X_train: np.ndarray, y_train: np.ndarray,
                               X_test: np.ndarray = None, y_test: np.ndarray = None) -> LinearRegression:
        """
        Train Linear Regression model for RUL prediction.
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training RUL labels
            X_test (np.ndarray): Test features (optional)
            y_test (np.ndarray): Test RUL labels (optional)
            
        Returns:
            LinearRegression: Trained Linear Regression model
        """
        print("Training Linear Regression model...")
        
        # Initialize and train Linear Regression
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        
        # Store the model
        self.models['linear_regression'] = lr_model
        
        # Evaluate if test data provided
        if X_test is not None and y_test is not None:
            y_pred = lr_model.predict(X_test)
            metrics = self._calculate_metrics(y_test, y_pred)
            self.model_history['linear_regression'] = metrics
            print(f"✓ Linear Regression trained - RMSE: {metrics['rmse']:.2f}, R²: {metrics['r2_score']:.3f}")
        else:
            print("✓ Linear Regression trained")
        
        return lr_model
    
    def _create_sequences(self, data: np.ndarray, sequence_length: int) -> np.ndarray:
        """
        Create sequences for LSTM input.
        
        Args:
            data (np.ndarray): Input data
            sequence_length (int): Length of sequences
            
        Returns:
            np.ndarray: Reshaped data for LSTM
        """
        sequences = []
        for i in range(sequence_length, len(data)):
            sequences.append(data[i-sequence_length:i])
        return np.array(sequences)
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate evaluation metrics.
        
        Args:
            y_true (np.ndarray): True RUL values
            y_pred (np.ndarray): Predicted RUL values
            
        Returns:
            Dict[str, float]: Dictionary of metrics
        """
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2_score': r2_score(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        }
        
        # Calculate directional accuracy
        if len(y_true) > 1:
            true_direction = np.diff(y_true) < 0  # Decreasing RUL
            pred_direction = np.diff(y_pred) < 0
            metrics['directional_accuracy'] = np.mean(true_direction == pred_direction) * 100
        else:
            metrics['directional_accuracy'] = 0.0
        
        return metrics
    
    def evaluate_all_models(self, X_test: np.ndarray, y_test: np.ndarray, 
                           sequence_length: int = 20) -> Dict[str, Dict[str, float]]:
        """
        Evaluate all trained models on test data.
        
        Args:
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test RUL labels
            sequence_length (int): Sequence length for LSTM
            
        Returns:
            Dict[str, Dict[str, float]]: Evaluation results for all models
        """
        results = {}
        
        # Evaluate Random Forest
        if 'random_forest' in self.models:
            y_pred_rf = self.models['random_forest'].predict(X_test)
            results['random_forest'] = self._calculate_metrics(y_test, y_pred_rf)
        
        # Evaluate Linear Regression
        if 'linear_regression' in self.models:
            y_pred_lr = self.models['linear_regression'].predict(X_test)
            results['linear_regression'] = self._calculate_metrics(y_test, y_pred_lr)
        
        # Evaluate LSTM
        if 'lstm' in self.models:
            X_test_reshaped = self._create_sequences(X_test, sequence_length)
            y_test_reshaped = y_test[sequence_length-1:]
            
            if len(X_test_reshaped) > 0:
                y_pred_lstm = self.models['lstm'].predict(X_test_reshaped, verbose=0).flatten()
                results['lstm'] = self._calculate_metrics(y_test_reshaped, y_pred_lstm)
        
        return results
    
    def get_feature_importance(self, model_name: str = 'random_forest') -> Optional[np.ndarray]:
        """
        Get feature importance from Random Forest model.
        
        Args:
            model_name (str): Name of the model
            
        Returns:
            Optional[np.ndarray]: Feature importance array
        """
        if model_name in self.models and hasattr(self.models[model_name], 'feature_importances_'):
            return self.models[model_name].feature_importances_
        return None
    
    def predict_rul(self, X: np.ndarray, model_name: str = 'random_forest') -> np.ndarray:
        """
        Predict RUL using specified model.
        
        Args:
            X (np.ndarray): Input features
            model_name (str): Name of the model to use
            
        Returns:
            np.ndarray: Predicted RUL values
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")
        
        model = self.models[model_name]
        
        if model_name == 'lstm':
            # For LSTM, we need to create sequences
            sequence_length = 20  # Default sequence length
            if len(X) >= sequence_length:
                X_reshaped = self._create_sequences(X, sequence_length)
                if len(X_reshaped) > 0:
                    return model.predict(X_reshaped, verbose=0).flatten()
            return np.array([])
        else:
            return model.predict(X)
    
    def get_model_summary(self) -> Dict[str, any]:
        """
        Get summary of all trained models.
        
        Returns:
            Dict[str, any]: Summary of models and their performance
        """
        summary = {
            'trained_models': list(self.models.keys()),
            'model_history': self.model_history,
            'model_types': {
                'random_forest': 'Random Forest Regressor',
                'lstm': 'LSTM Neural Network',
                'linear_regression': 'Linear Regression'
            }
        }
        
        return summary


