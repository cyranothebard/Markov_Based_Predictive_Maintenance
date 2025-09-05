"""
Hidden Markov Model for RUL Prediction

This module implements a Hidden Markov Model (HMM) approach for predicting
Remaining Useful Life (RUL) using the hmmlearn library.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from hmmlearn import hmm
import warnings
warnings.filterwarnings('ignore')


class HiddenMarkovRUL:
    """
    Hidden Markov Model for RUL prediction.
    
    Uses Gaussian HMM to model hidden health states and predict
    remaining useful life based on sensor observations.
    """
    
    def __init__(self, n_states: int = 4, covariance_type: str = "diag", n_iter: int = 100):
        """
        Initialize Hidden Markov Model.
        
        Args:
            n_states (int): Number of hidden states (default: 4)
            covariance_type (str): Type of covariance matrix ("full", "diag", "spherical")
        """
        self.n_states = n_states
        self.covariance_type = covariance_type
        self.state_names = ['Healthy', 'Warning', 'Critical', 'Failure']
        self.model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type=covariance_type,
            n_iter=n_iter,
            tol=1e-2,
            random_state=42
        )
        self.is_fitted = False
        self.sequence_lengths = None
        self.failure_state_index = self.n_states - 1
        
    def fit(self, X: np.ndarray, lengths: List[int]) -> 'HiddenMarkovRUL':
        """
        Fit the HMM model to training data.
        
        Args:
            X (np.ndarray): Concatenated sequences of sensor measurements
            lengths (List[int]): Length of each engine trajectory
            
        Returns:
            HiddenMarkovRUL: Fitted model
        """
        print("Fitting Hidden Markov Model...")
        
        # Store sequence lengths for later use
        self.sequence_lengths = lengths
        
        # Fit HMM using Baum-Welch algorithm
        self.model.fit(X, lengths)
        print("✓ HMM fitted using Baum-Welch algorithm")
        
        self.is_fitted = True
        print("Hidden Markov Model fitting completed!")
        
        return self

    def align_states_with_rul(self, X: np.ndarray, rul: np.ndarray, lengths: List[int]) -> None:
        """
        Align hidden states to RUL by assigning the failure state as the one with
        the lowest average RUL across observations most likely belonging to it.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before aligning states")
        # Posterior probabilities per state
        posteriors = self.model.predict_proba(X)
        # Expected RUL per state via responsibility-weighted average
        state_rul = np.zeros(self.n_states)
        for s in range(self.n_states):
            weights = posteriors[:, s]
            denom = np.sum(weights)
            state_rul[s] = np.sum(weights * rul) / denom if denom > 0 else np.inf
        self.failure_state_index = int(np.argmin(state_rul))
        # Optionally reorder state_names to reflect progression (not strictly needed)
    
    def predict_states(self, X: np.ndarray) -> np.ndarray:
        """
        Predict most likely state sequence using Viterbi algorithm.
        
        Args:
            X (np.ndarray): Sequence of sensor measurements
            
        Returns:
            np.ndarray: Most likely state sequence
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Use Viterbi algorithm to decode most likely state sequence
        states = self.model.predict(X)
        return states
    
    def predict_rul(self, X: np.ndarray) -> List[float]:
        """
        Predict RUL for each time step in sequence.
        
        Args:
            X (np.ndarray): Sequence of sensor measurements
            
        Returns:
            List[float]: Predicted RUL values for each time step
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Use posterior state probabilities at each time to get expected RUL
        posteriors = self.model.predict_proba(X)
        # Precompute expected life for each state
        expected_lives = np.array([self._calculate_expected_life_from_state(s) for s in range(self.n_states)])
        rul_predictions = posteriors.dot(expected_lives)
        return [max(0.0, float(v)) for v in rul_predictions]
    
    def _calculate_expected_life_from_state(self, state: int) -> float:
        """
        Calculate expected cycles to failure from given state.
        
        Args:
            state (int): Current health state
            
        Returns:
            float: Expected cycles to failure
        """
        # Get transition matrix
        transition_matrix = self.model.transmat_
        
        # Create absorbing Markov chain analysis
        # Assume last state (n_states-1) is failure state
        failure_state = self.failure_state_index
        
        if state == failure_state:
            return 0.0
        
        # Extract transient states (all except failure)
        transient_states = [i for i in range(self.n_states) if i != failure_state]
        
        if state not in transient_states:
            return 0.0
        
        # Create Q matrix (transient to transient transitions)
        Q = transition_matrix[np.ix_(transient_states, transient_states)]
        
        # Calculate fundamental matrix: (I - Q)^(-1)
        I = np.eye(len(transient_states))
        try:
            fundamental_matrix = np.linalg.inv(I - Q)
            
            # Expected time to absorption from each transient state
            expected_times = fundamental_matrix.sum(axis=1)
            
            # Return expected time from current state
            state_index = transient_states.index(state)
            return expected_times[state_index]
            
        except np.linalg.LinAlgError:
            # Fallback: simple heuristic
            return max(0, (self.n_states - 1 - state) * 50)
    
    def get_model_parameters(self) -> dict:
        """
        Get HMM model parameters.
        
        Returns:
            dict: Dictionary containing transition matrix, emission means, and covariances
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before accessing parameters")
        
        parameters = {
            "transition_matrix": self.model.transmat_.tolist(),
            "emission_means": self.model.means_.tolist(),
            "emission_covariances": self.model.covars_.tolist(),
            "initial_state_probs": self.model.startprob_.tolist(),
            "n_states": self.n_states,
            "covariance_type": self.covariance_type,
            "state_names": self.state_names
        }
        
        return parameters
    
    def get_state_probabilities(self, X: np.ndarray) -> np.ndarray:
        """
        Get probability distribution over states for given observations.
        
        Args:
            X (np.ndarray): Sequence of sensor measurements
            
        Returns:
            np.ndarray: Probability distribution over states for each observation
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Use forward-backward algorithm to get state probabilities
        log_probs = self.model.score_samples(X)
        state_probs = np.exp(log_probs)
        
        # Get posterior probabilities for each state
        posteriors = self.model.predict_proba(X)
        
        return posteriors
    
    def predict_sequence_rul(self, X: np.ndarray, lengths: List[int]) -> List[List[float]]:
        """
        Predict RUL for multiple sequences.
        
        Args:
            X (np.ndarray): Concatenated sequences
            lengths (List[int]): Length of each sequence
            
        Returns:
            List[List[float]]: RUL predictions for each sequence
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        all_rul_predictions = []
        start_idx = 0
        
        for length in lengths:
            end_idx = start_idx + length
            sequence = X[start_idx:end_idx]
            
            # Predict RUL for this sequence
            sequence_rul = self.predict_rul(sequence)
            all_rul_predictions.append(sequence_rul)
            
            start_idx = end_idx
        
        return all_rul_predictions
    
    def get_transition_statistics(self) -> dict:
        """
        Get transition statistics from the fitted model.
        
        Returns:
            dict: Transition statistics and insights
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before accessing transition statistics")
        
        transition_matrix = self.model.transmat_
        
        stats = {
            "transition_matrix": transition_matrix.tolist(),
            "state_names": self.state_names,
            "self_transition_probs": np.diag(transition_matrix).tolist(),
            "failure_transition_probs": transition_matrix[:, -1].tolist()
        }
        
        # Calculate average time in each state
        avg_times = []
        for i in range(self.n_states - 1):  # Exclude failure state
            self_transition_prob = transition_matrix[i, i]
            if self_transition_prob < 1.0:
                avg_time = 1 / (1 - self_transition_prob)
            else:
                avg_time = float('inf')
            avg_times.append(avg_time)
        
        stats["average_time_in_state"] = avg_times
        
        return stats
    
    def simulate_engine_life(self, initial_state: int = 0, max_cycles: int = 500) -> dict:
        """
        Simulate engine life trajectory using the fitted HMM.
        
        Args:
            initial_state (int): Starting health state
            max_cycles (int): Maximum simulation cycles
            
        Returns:
            dict: Simulation results including state sequence and RUL
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before simulation")
        
        # Simulate state sequence
        states, observations = self.model.sample(max_cycles)
        
        # Calculate RUL for each time step
        rul_sequence = []
        for i, state in enumerate(states):
            expected_life = self._calculate_expected_life_from_state(state)
            remaining_cycles = max(0, expected_life - i)
            rul_sequence.append(remaining_cycles)
        
        simulation_results = {
            "states": states.tolist(),
            "observations": observations.tolist(),
            "rul_sequence": rul_sequence,
            "final_state": int(states[-1]),
            "cycles_to_failure": len(states) if states[-1] == self.n_states - 1 else None
        }
        
        return simulation_results
    
    def get_model_summary(self) -> dict:
        """
        Get comprehensive model summary.
        
        Returns:
            dict: Complete model summary including parameters and statistics
        """
        if not self.is_fitted:
            return {"error": "Model not fitted"}
        
        summary = {
            "model_type": "Hidden Markov Model",
            "n_states": self.n_states,
            "state_names": self.state_names,
            "covariance_type": self.covariance_type,
            "is_fitted": self.is_fitted,
            "parameters": self.get_model_parameters(),
            "transition_statistics": self.get_transition_statistics()
        }
        
        return summary


