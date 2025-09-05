"""
Markov Chain Model for RUL Prediction

This module implements a Markov Chain-based approach for predicting
Remaining Useful Life (RUL) of turbofan engines using health state transitions.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from scipy import stats
from sklearn.mixture import GaussianMixture
import warnings
warnings.filterwarnings('ignore')


class MarkovChainRUL:
    """
    Markov Chain model for RUL prediction based on health state transitions.
    
    Uses Maximum Likelihood Estimation to estimate transition probabilities
    and emission probabilities for each health state.
    """
    
    def __init__(self, n_states: int = 4):
        """
        Initialize Markov Chain model.
        
        Args:
            n_states (int): Number of health states (default: 4)
        """
        self.n_states = n_states
        self.state_names = ['Healthy', 'Warning', 'Critical', 'Failure']
        self.transition_matrix = None
        self.emission_probabilities = None
        self.state_emission_models = None
        self.is_fitted = False
        
    def fit(self, X: np.ndarray, states: np.ndarray) -> 'MarkovChainRUL':
        """
        Fit the Markov Chain model to training data.
        
        Args:
            X (np.ndarray): Feature matrix (n_samples, n_features)
            states (np.ndarray): Health state labels (n_samples,)
            
        Returns:
            MarkovChainRUL: Fitted model
        """
        print("Fitting Markov Chain model...")
        
        # Estimate transition matrix using Maximum Likelihood Estimation
        self.transition_matrix = self._estimate_transition_matrix(states)
        print("✓ Transition matrix estimated")
        
        # Calculate emission probabilities for each state
        self.emission_probabilities = self._calculate_emission_probabilities(X, states)
        print("✓ Emission probabilities calculated")
        
        # Fit Gaussian emission models for each state
        self.state_emission_models = self._fit_emission_models(X, states)
        print("✓ Emission models fitted")
        
        self.is_fitted = True
        print("Markov Chain model fitting completed!")
        
        return self
    
    def _estimate_transition_matrix(self, states: np.ndarray) -> np.ndarray:
        """
        Estimate transition matrix using Maximum Likelihood Estimation.
        
        Args:
            states (np.ndarray): Sequence of health states
            
        Returns:
            np.ndarray: Transition probability matrix (n_states x n_states)
        """
        # Initialize transition count matrix
        transition_counts = np.zeros((self.n_states, self.n_states))
        
        # Count transitions
        for i in range(len(states) - 1):
            current_state = int(states[i])
            next_state = int(states[i + 1])
            transition_counts[current_state, next_state] += 1
        
        # Convert counts to probabilities
        transition_matrix = np.zeros_like(transition_counts)
        for i in range(self.n_states):
            row_sum = transition_counts[i].sum()
            if row_sum > 0:
                transition_matrix[i] = transition_counts[i] / row_sum
            else:
                # If no transitions from state i, assume self-transition
                transition_matrix[i, i] = 1.0
        
        return transition_matrix
    
    def _calculate_emission_probabilities(self, X: np.ndarray, states: np.ndarray) -> np.ndarray:
        """
        Calculate emission probabilities for each state.
        
        Args:
            X (np.ndarray): Feature matrix
            states (np.ndarray): Health state labels
            
        Returns:
            np.ndarray: Emission probability matrix
        """
        emission_probs = np.zeros((self.n_states, X.shape[1]))
        
        for state in range(self.n_states):
            state_mask = states == state
            if np.any(state_mask):
                # Calculate mean emission probability for each feature in this state
                state_features = X[state_mask]
                emission_probs[state] = np.mean(state_features, axis=0)
        
        return emission_probs
    
    def _fit_emission_models(self, X: np.ndarray, states: np.ndarray) -> List[GaussianMixture]:
        """
        Fit Gaussian emission models for each state.
        
        Args:
            X (np.ndarray): Feature matrix
            states (np.ndarray): Health state labels
            
        Returns:
            List[GaussianMixture]: List of fitted Gaussian models for each state
        """
        emission_models = []
        
        for state in range(self.n_states):
            state_mask = states == state
            if np.any(state_mask) and np.sum(state_mask) > 1:
                state_features = X[state_mask]
                
                # Fit Gaussian mixture model for this state
                n_components = min(3, len(state_features) // 10)  # Adaptive components
                if n_components < 1:
                    n_components = 1
                
                gmm = GaussianMixture(n_components=n_components, random_state=42)
                gmm.fit(state_features)
                emission_models.append(gmm)
            else:
                # If insufficient data, create a simple Gaussian model
                gmm = GaussianMixture(n_components=1, random_state=42)
                if np.any(state_mask):
                    gmm.fit(X[state_mask].reshape(-1, 1))
                else:
                    # Default model with zero mean and unit variance
                    gmm.means_ = np.array([[0.0]])
                    gmm.covariances_ = np.array([[[1.0]]])
                    gmm.weights_ = np.array([1.0])
                emission_models.append(gmm)
        
        return emission_models
    
    def predict_rul(self, X: np.ndarray, current_cycle: int) -> float:
        """
        Predict RUL using Viterbi algorithm and expected transition times.
        
        Args:
            X (np.ndarray): Current feature vector
            current_cycle (int): Current cycle number
            
        Returns:
            float: Predicted RUL
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Get state probability distribution for current observation
        state_probs = self.get_state_probabilities(X)[0]

        # Compute expected cycles to failure for each state
        expected_lives = np.array([self.calculate_expected_life(s) for s in range(self.n_states)])

        # Probability-weighted expected RUL (no heuristic subtraction)
        predicted_rul = float(np.dot(state_probs, expected_lives))

        # Ensure non-negative
        return max(0.0, predicted_rul)
    
    def get_transition_probabilities(self) -> np.ndarray:
        """
        Get the transition probability matrix.
        
        Returns:
            np.ndarray: Transition probability matrix
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before accessing transition probabilities")
        
        return self.transition_matrix.copy()
    
    def get_state_probabilities(self, X: np.ndarray) -> np.ndarray:
        """
        Get probability distribution over states for given observations.
        
        Args:
            X (np.ndarray): Feature vector or matrix
            
        Returns:
            np.ndarray: Probability distribution over states
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Ensure X is 2D
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        state_probs = np.zeros((X.shape[0], self.n_states))
        
        for i, x in enumerate(X):
            for state in range(self.n_states):
                # Calculate emission probability for this state
                if self.state_emission_models[state] is not None:
                    try:
                        # Get log probability and convert to probability
                        log_prob = self.state_emission_models[state].score_samples(x.reshape(1, -1))
                        state_probs[i, state] = np.exp(log_prob[0])
                    except:
                        # Fallback to simple distance-based probability
                        state_probs[i, state] = 1.0 / self.n_states
                else:
                    state_probs[i, state] = 1.0 / self.n_states
        
        # Normalize probabilities
        row_sums = state_probs.sum(axis=1)
        state_probs = state_probs / row_sums[:, np.newaxis]
        
        return state_probs
    
    def calculate_expected_life(self, current_state: int) -> float:
        """
        Calculate expected cycles to failure from current state.
        
        Uses absorbing Markov chain analysis where failure state (3) is absorbing.
        
        Args:
            current_state (int): Current health state
            
        Returns:
            float: Expected cycles to failure
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before calculating expected life")
        
        # Create absorbing Markov chain matrix
        # States 0, 1, 2 are transient, state 3 (failure) is absorbing
        Q = self.transition_matrix[:3, :3]  # Transient states
        R = self.transition_matrix[:3, 3:4]  # Transient to absorbing
        
        # Calculate fundamental matrix: (I - Q)^(-1)
        I = np.eye(3)
        try:
            fundamental_matrix = np.linalg.inv(I - Q)
            
            # Expected time to absorption from each transient state
            expected_times = fundamental_matrix.sum(axis=1)
            
            # Return expected time from current state
            if 0 <= current_state < 3:
                return expected_times[current_state]
            else:
                return 0.0  # Already in failure state
                
        except np.linalg.LinAlgError:
            # Fallback: simple heuristic based on state
            return max(0, (3 - current_state) * 50)  # Rough estimate
    
    def predict_state_sequence(self, X: np.ndarray) -> np.ndarray:
        """
        Predict most likely state sequence using Viterbi algorithm.
        
        Args:
            X (np.ndarray): Sequence of feature vectors
            
        Returns:
            np.ndarray: Most likely state sequence
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        n_observations = X.shape[0]
        
        # Initialize Viterbi algorithm
        viterbi = np.zeros((n_observations, self.n_states))
        backpointer = np.zeros((n_observations, self.n_states), dtype=int)
        
        # Initial probabilities (uniform)
        initial_probs = np.ones(self.n_states) / self.n_states
        
        # First observation
        for state in range(self.n_states):
            emission_prob = self._get_emission_probability(X[0], state)
            viterbi[0, state] = initial_probs[state] * emission_prob
        
        # Forward pass
        for t in range(1, n_observations):
            for state in range(self.n_states):
                emission_prob = self._get_emission_probability(X[t], state)
                
                # Find best previous state
                best_prob = 0
                best_state = 0
                for prev_state in range(self.n_states):
                    prob = viterbi[t-1, prev_state] * self.transition_matrix[prev_state, state] * emission_prob
                    if prob > best_prob:
                        best_prob = prob
                        best_state = prev_state
                
                viterbi[t, state] = best_prob
                backpointer[t, state] = best_state
        
        # Backward pass to find best sequence
        best_sequence = np.zeros(n_observations, dtype=int)
        best_sequence[-1] = np.argmax(viterbi[-1])
        
        for t in range(n_observations - 2, -1, -1):
            best_sequence[t] = backpointer[t + 1, best_sequence[t + 1]]
        
        return best_sequence
    
    def _get_emission_probability(self, x: np.ndarray, state: int) -> float:
        """
        Get emission probability for observation x in given state.
        
        Args:
            x (np.ndarray): Feature vector
            state (int): Health state
            
        Returns:
            float: Emission probability
        """
        try:
            if self.state_emission_models[state] is not None:
                log_prob = self.state_emission_models[state].score_samples(x.reshape(1, -1))
                return np.exp(log_prob[0])
            else:
                return 1.0 / self.n_states
        except:
            return 1.0 / self.n_states
    
    def get_model_summary(self) -> dict:
        """
        Get summary statistics of the fitted model.
        
        Returns:
            dict: Model summary including transition probabilities and state statistics
        """
        if not self.is_fitted:
            return {"error": "Model not fitted"}
        
        summary = {
            "n_states": self.n_states,
            "state_names": self.state_names,
            "transition_matrix": self.transition_matrix.tolist(),
            "is_fitted": self.is_fitted
        }
        
        # Add state transition statistics
        for i, state_name in enumerate(self.state_names):
            summary[f"{state_name}_transitions"] = {
                "to_healthy": self.transition_matrix[i, 0],
                "to_warning": self.transition_matrix[i, 1],
                "to_critical": self.transition_matrix[i, 2],
                "to_failure": self.transition_matrix[i, 3]
            }
        
        return summary


